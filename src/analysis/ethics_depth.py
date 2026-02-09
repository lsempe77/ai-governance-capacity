"""
Phase C: Verbatim Depth Extraction
====================================
For each policy in the ethics inventory, extracts the EXACT verbatim text
from the original document for every identified value, principle, and mechanism.

Uses Claude Opus 4.6 via OpenRouter to locate and extract literal passages,
then classifies the depth of each mention:
  - word:      single keyword or term only
  - phrase:    a short clause within a sentence
  - sentence:  one complete sentence
  - paragraph: multiple sentences / a full paragraph
  - section:   a dedicated section, article, or chapter

This adds "texture" to the Phase B inventory — we move from knowing WHAT
is mentioned to understanding HOW DEEPLY it is treated.

Inputs:
  data/analysis/ethics_inventory/phase_b_inventory.jsonl  – identified items per policy
  data/corpus/corpus_enriched.json                        – original policy texts

Outputs:
  data/analysis/ethics_inventory/phase_c_depth.jsonl      – verbatim quotes + depth per item
  data/analysis/ethics_inventory/depth_summary.json       – aggregated depth statistics
  data/analysis/ethics_inventory/depth_report.json        – run statistics

Usage:
  python src/analysis/ethics_depth.py --limit 5              # test on 5 entries
  python src/analysis/ethics_depth.py --resume --workers 3   # full run, resume-safe
  python src/analysis/ethics_depth.py --summarise            # aggregate results
"""

import json
import os
import re
import sys
import time
import argparse
import logging
import threading
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ─── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH       = ROOT / 'data' / 'corpus' / 'corpus_enriched.json'
PHASE_B_OUTPUT    = ROOT / 'data' / 'analysis' / 'ethics_inventory' / 'phase_b_inventory.jsonl'
OUTPUT_DIR        = ROOT / 'data' / 'analysis' / 'ethics_inventory'
PHASE_C_OUTPUT    = OUTPUT_DIR / 'phase_c_depth.jsonl'
DEPTH_SUMMARY     = OUTPUT_DIR / 'depth_summary.json'
DEPTH_REPORT      = OUTPUT_DIR / 'depth_report.json'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Claude Opus 4.6 — reliable JSON, excellent verbatim extraction, 1M context
DEPTH_MODEL = "anthropic/claude-opus-4.6"

MAX_TEXT_CHARS = 50_000  # ~12K tokens — Opus 4.6 has 1M context

# ─── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in AI governance and policy analysis.
Your task is to extract EXACT VERBATIM QUOTES from policy documents.
You must copy text EXACTLY as it appears — same words, same punctuation, same capitalisation.
Never paraphrase. If you cannot find a direct quote, say so explicitly.
Return ONLY valid JSON."""

EXTRACTION_PROMPT = """## TASK

I have a policy document and a list of ethical values, principles, and mechanisms
that were previously identified in it. For EACH identified item, I need you to:

1. **Find the EXACT verbatim passage(s)** in the document where this item is discussed
2. **Copy the literal text** — word for word, character for character
3. **Classify the depth** of treatment:
   - `"word"` — the concept appears only as a single keyword or term (e.g., "transparency")
   - `"phrase"` — appears as a short clause within a sentence (e.g., "ensuring transparency in AI systems")
   - `"sentence"` — one complete sentence dedicated to or substantially about this concept
   - `"paragraph"` — multiple sentences or a full paragraph discussing this concept
   - `"section"` — a dedicated section, article, chapter, or substantial portion of the document

## PREVIOUSLY IDENTIFIED ITEMS

### Values:
{values_list}

### Principles:
{principles_list}

### Mechanisms:
{mechanisms_list}

## POLICY DOCUMENT

**Title**: {title}
**Jurisdiction**: {jurisdiction}

---
{text}
---

## REQUIRED OUTPUT

Return ONLY valid JSON with this structure:

{{
  "values": [
    {{
      "name": "Human rights",
      "depth": "paragraph",
      "verbatim": "The exact text from the document...",
      "location_hint": "Article 3" or "Section 2.1" or "paragraph 5" (if identifiable)
    }},
    ...
  ],
  "principles": [
    {{
      "name": "Transparency",
      "depth": "section",
      "verbatim": "The exact text copied from the document...",
      "location_hint": "..."
    }},
    ...
  ],
  "mechanisms": [
    {{
      "name": "Impact assessment",
      "depth": "sentence",
      "verbatim": "The exact text...",
      "location_hint": "..."
    }},
    ...
  ]
}}

## CRITICAL RULES:
- The "verbatim" field MUST contain text copied EXACTLY from the document above
- For "word" depth: quote the sentence containing the keyword, underline the keyword with >>keyword<<
- For "phrase"/"sentence": quote the complete sentence(s)
- For "paragraph": quote the full paragraph (truncate with [...] if over 500 words)
- For "section": quote the section heading and first 2-3 key sentences, then [...] for remainder
- If an item was identified but you genuinely cannot find supporting text, include it with
  "depth": "not_found" and "verbatim": "No direct textual evidence found in the provided excerpt"
- Keep the SAME item names as provided in the lists above
- Preserve every item — do not drop any"""


# ─── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json_robust(content: str) -> dict:
    """Parse JSON from LLM output, handling fences, newlines, and smart quotes."""
    content = content.strip()

    # 1. Strip markdown code fences
    content = re.sub(r'^```\w*\s*', '', content)
    content = re.sub(r'\s*```\s*$', '', content)
    content = content.strip()

    # 2. Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 3. Fix smart quotes that use ASCII " as closing character
    #    German „Wort" uses U+201E opening + ASCII U+0022 closing → breaks JSON
    #    Replace: „ + text + " → „ + text + \u201d (right double quotation mark)
    fixed = re.sub(r'„([^"„\u201c\u201d]{1,200}?)"', '„\\1\u201d', content)

    # 4. Escape raw control characters inside JSON string values
    #    Walk char-by-char, only toggle in_string on ASCII " (U+0022)
    result = []
    in_string = False
    i = 0
    while i < len(fixed):
        ch = fixed[i]
        if ch == '"' and (i == 0 or fixed[i-1] != '\\'):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\r':
            pass  # skip
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
        i += 1
    fixed = ''.join(result)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. Extract JSON object between first { and last }
    brace_start = fixed.find('{')
    brace_end = fixed.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        extracted = fixed[brace_start:brace_end + 1]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        # 6. Fix trailing commas
        no_trailing = re.sub(r',\s*([}\]])', r'\1', extracted)
        try:
            return json.loads(no_trailing)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from LLM output", content[:200], 0)


# ─── API Call ──────────────────────────────────────────────────────────────────

def call_claude_depth(text: str, title: str, jurisdiction: str,
                      values: list, principles: list, mechanisms: list,
                      max_retries: int = 3) -> dict | None:
    """Call Claude Opus 4.6 via OpenRouter for verbatim depth extraction."""
    import requests

    # Format the item lists
    values_list = "\n".join(
        f"  - {v['name']} (previously classified as: {v.get('strength', '?')})"
        for v in values
    ) or "  (none identified)"

    principles_list = "\n".join(
        f"  - {p['name']} (previously classified as: {p.get('strength', '?')})"
        for p in principles
    ) or "  (none identified)"

    mechanisms_list = "\n".join(
        f"  - {m['name']} (previously classified as: {m.get('strength', '?')})"
        for m in mechanisms
    ) or "  (none identified)"

    prompt = EXTRACTION_PROMPT.format(
        title=title,
        jurisdiction=jurisdiction,
        text=text[:MAX_TEXT_CHARS],
        values_list=values_list,
        principles_list=principles_list,
        mechanisms_list=mechanisms_list,
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lsempe77/ai-governance-capacity",
    }

    payload = {
        "model": DEPTH_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 8000,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            result = resp.json()

            content = result['choices'][0]['message']['content'].strip()
            parsed = _parse_json_robust(content)

            usage = result.get('usage', {})
            return {
                'depth': parsed,
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                },
            }

        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error (attempt {attempt+1}): {e}")
            debug_path = OUTPUT_DIR / 'debug_depth_failed_response.txt'
            if not debug_path.exists():
                try:
                    with open(debug_path, 'w', encoding='utf-8') as df:
                        df.write(f"Title: {title}\nJurisdiction: {jurisdiction}\n")
                        df.write(f"Error: {e}\n\n=== RAW ===\n{content}\n=== END ===")
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except Exception as e:
            log.warning(f"  API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    return None


# ─── Thread-safe counter ──────────────────────────────────────────────────────

class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def increment(self, success=True):
        with self.lock:
            self.done += 1
            if not success:
                self.failed += 1

    def log_progress(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.done / (elapsed / 60) if elapsed > 0 else 0
            remaining = self.total - self.done
            eta = remaining / rate if rate > 0 else 0
            log.info(f"  [{self.done}/{self.total}] (OK) {rate:.1f}/min, "
                     f"ETA {eta:.0f}m (failed: {self.failed})")


# ─── Process one policy ──────────────────────────────────────────────────────

def process_one_policy(entry: dict, corpus_texts: dict) -> dict | None:
    """Extract verbatim depth for a single policy."""
    eid = entry['entry_id']
    text_data = corpus_texts.get(eid, {})
    text = text_data.get('full_text', text_data.get('text', ''))

    if not text or len(text.strip()) < 50:
        return None

    inv = entry.get('inventory', {})
    values = inv.get('values', [])
    principles = inv.get('principles', [])
    mechanisms = inv.get('mechanisms', [])

    if not values and not principles and not mechanisms:
        return None

    result = call_claude_depth(
        text=text,
        title=entry.get('title', ''),
        jurisdiction=entry.get('jurisdiction', ''),
        values=values,
        principles=principles,
        mechanisms=mechanisms,
    )

    if result is None:
        return None

    return {
        'entry_id': eid,
        'title': entry.get('title', ''),
        'jurisdiction': entry.get('jurisdiction', ''),
        'year': entry.get('year', 0),
        'ethics_score': entry.get('ethics_score', 0),
        'depth_data': result['depth'],
        'usage': result['usage'],
        'timestamp': datetime.now().isoformat(),
    }


# ─── Main run ─────────────────────────────────────────────────────────────────

def run_depth_extraction(limit: int = 0, resume: bool = False, workers: int = 2):
    """Run Claude Opus depth extraction on all Phase B policies."""
    log.info("═══ PHASE C: Verbatim depth extraction (Claude Opus 4.6) ═══")

    # Load Phase B inventory
    if not PHASE_B_OUTPUT.exists():
        log.error(f"Phase B output not found: {PHASE_B_OUTPUT}")
        log.error("Run ethics_inventory.py --phase B first!")
        sys.exit(1)

    entries = []
    with open(PHASE_B_OUTPUT, encoding='utf-8') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass
    log.info(f"Phase B entries loaded: {len(entries)}")

    # Load corpus texts
    log.info("Loading corpus texts...")
    with open(CORPUS_PATH, encoding='utf-8') as f:
        corpus_data = json.load(f)

    corpus_by_key = {}
    for ce in corpus_data.get('entries', []):
        key = (ce.get('title', ''), ce.get('jurisdiction', ''))
        text = ce.get('text', ce.get('full_text', ''))
        corpus_by_key[key] = {
            'text': text,
            'full_text': ce.get('full_text', text),
        }
    log.info(f"Corpus entries loaded: {len(corpus_by_key)}")

    # Map corpus texts by entry_id
    corpus_texts = {}
    for e in entries:
        key = (e.get('title', ''), e.get('jurisdiction', ''))
        if key in corpus_by_key:
            corpus_texts[e['entry_id']] = corpus_by_key[key]
    log.info(f"Matched corpus texts: {len(corpus_texts)} / {len(entries)}")

    # Resume support
    done_ids = set()
    if resume and PHASE_C_OUTPUT.exists():
        with open(PHASE_C_OUTPUT, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r['entry_id'])
                except:
                    pass
        log.info(f"Resuming: {len(done_ids)} already completed")

    remaining = [e for e in entries
                 if e['entry_id'] not in done_ids and e['entry_id'] in corpus_texts]

    if limit > 0:
        remaining = remaining[:limit]

    log.info(f"Policies to process: {len(remaining)}")

    if not remaining:
        log.info("Nothing to process!")
        return

    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    # Process
    counter = ProgressCounter(len(remaining))
    results = []
    total_tokens = 0

    mode = 'a' if resume else 'w'
    with open(PHASE_C_OUTPUT, mode, encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_one_policy, entry, corpus_texts): entry
                for entry in remaining
            }

            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result = future.result()
                    if result:
                        out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        out_f.flush()
                        results.append(result)
                        total_tokens += result.get('usage', {}).get('total_tokens', 0)
                        counter.increment(success=True)
                    else:
                        counter.increment(success=False)
                except Exception as e:
                    log.error(f"Error processing {entry.get('title', '?')}: {e}")
                    counter.increment(success=False)

                counter.log_progress()

    elapsed = time.time() - counter.start_time

    # Save report
    report = {
        'created': datetime.now().isoformat(),
        'phase': 'C',
        'model': DEPTH_MODEL,
        'total_target': len(entries),
        'processed': len(results),
        'failed': counter.failed,
        'skipped_already_done': len(done_ids),
        'total_tokens': total_tokens,
        'estimated_cost_usd': round(
            total_tokens * 0.000005  # blended rough estimate $5 in + $25 out
            + (total_tokens * 0.000020 * 0.3),  # ~30% output ratio at $25/M
            2
        ),
        'elapsed_seconds': elapsed,
        'rate_per_minute': len(results) / (elapsed / 60) if elapsed > 0 else 0,
    }

    with open(DEPTH_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Phase C complete!")
    log.info(f"  Processed: {len(results)}")
    log.info(f"  Failed: {counter.failed}")
    log.info(f"  Total tokens: {total_tokens:,}")
    log.info(f"  Est. cost: ${report['estimated_cost_usd']:.2f}")
    log.info(f"  Time: {elapsed/60:.1f} minutes")
    log.info(f"  Output: {PHASE_C_OUTPUT}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARISE
# ══════════════════════════════════════════════════════════════════════════════

def run_summarise():
    """Aggregate Phase C depth results."""
    log.info("═══ SUMMARISE: Aggregating depth extraction results ═══")

    if not PHASE_C_OUTPUT.exists():
        log.error(f"Phase C output not found: {PHASE_C_OUTPUT}")
        sys.exit(1)

    entries = []
    with open(PHASE_C_OUTPUT, encoding='utf-8') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass
    log.info(f"Loaded {len(entries)} depth entries")

    # Import income metadata
    sys.path.insert(0, str(ROOT / 'src' / 'analysis'))
    from country_metadata import get_income_binary

    # ── Depth distributions ────────────────────────────────────────────────
    # Track depth per item name for each category
    value_depth = defaultdict(Counter)       # value_name → {word: n, phrase: n, ...}
    principle_depth = defaultdict(Counter)
    mechanism_depth = defaultdict(Counter)

    # Global depth distribution
    global_depth = Counter()

    # Per income group
    income_depth = defaultdict(lambda: defaultdict(int))  # income → "category_depth" → count

    # Verbatim length stats
    verbatim_lengths = defaultdict(list)  # depth_level → [char_lengths]

    # Per-entry detail
    entry_details = []

    for entry in entries:
        dd = entry.get('depth_data', {})
        jurisdiction = entry.get('jurisdiction', '')
        income = get_income_binary(jurisdiction) or 'Unknown'

        entry_depth_counts = Counter()

        for category, items, tracker in [
            ('values', dd.get('values', []), value_depth),
            ('principles', dd.get('principles', []), principle_depth),
            ('mechanisms', dd.get('mechanisms', []), mechanism_depth),
        ]:
            for item in items:
                name = item.get('name', '').strip()
                depth = item.get('depth', 'unknown').lower().strip()
                verbatim = item.get('verbatim', '')

                if not name:
                    continue

                tracker[name][depth] += 1
                global_depth[depth] += 1
                income_depth[income][category + '_' + depth] += 1
                entry_depth_counts[depth] += 1

                if verbatim:
                    verbatim_lengths[depth].append(len(verbatim))

        entry_details.append({
            'entry_id': entry['entry_id'],
            'title': entry['title'],
            'jurisdiction': jurisdiction,
            'income': income,
            'ethics_score': entry.get('ethics_score', 0),
            'depth_distribution': dict(entry_depth_counts),
            'n_items': sum(entry_depth_counts.values()),
        })

    # ── Build depth rankings ───────────────────────────────────────────────
    import statistics

    def depth_profile(tracker):
        """For each item, compute its depth profile."""
        profiles = []
        for name, depths in tracker.items():
            total = sum(depths.values())
            # Compute a "depth score": word=1, phrase=2, sentence=3, paragraph=4, section=5
            depth_weights = {'word': 1, 'phrase': 2, 'sentence': 3, 'paragraph': 4, 'section': 5, 'not_found': 0}
            weighted = sum(depths.get(d, 0) * w for d, w in depth_weights.items())
            avg_depth = weighted / total if total > 0 else 0

            profiles.append({
                'name': name,
                'count': total,
                'avg_depth_score': round(avg_depth, 2),
                'depth_distribution': {d: depths.get(d, 0) for d in
                                       ['word', 'phrase', 'sentence', 'paragraph', 'section', 'not_found']
                                       if depths.get(d, 0) > 0},
                'pct_deep': round(
                    (depths.get('paragraph', 0) + depths.get('section', 0)) / total * 100, 1
                ) if total > 0 else 0,
            })
        return sorted(profiles, key=lambda x: (-x['count'], -x['avg_depth_score']))

    # Median verbatim lengths by depth
    verbatim_stats = {}
    for depth_level, lengths in verbatim_lengths.items():
        if lengths:
            verbatim_stats[depth_level] = {
                'n': len(lengths),
                'median_chars': round(statistics.median(lengths)),
                'mean_chars': round(statistics.mean(lengths)),
                'min_chars': min(lengths),
                'max_chars': max(lengths),
            }

    # ── Income group comparison ────────────────────────────────────────────
    income_summary = {}
    for income_group in sorted(income_depth.keys()):
        cats = income_depth[income_group]
        n_policies = sum(1 for e in entry_details if e['income'] == income_group)
        total_items = sum(cats.values())

        # Depth breakdown for this income group
        group_depth = Counter()
        for k, v in cats.items():
            # k is like "values_paragraph" or "mechanisms_word"
            depth_level = k.rsplit('_', 1)[-1] if '_' in k else k
            group_depth[depth_level] += v

        income_summary[income_group] = {
            'n_policies': n_policies,
            'total_items': total_items,
            'depth_distribution': dict(group_depth),
            'pct_deep': round(
                (group_depth.get('paragraph', 0) + group_depth.get('section', 0)) /
                total_items * 100, 1
            ) if total_items > 0 else 0,
        }

    summary = {
        'created': datetime.now().isoformat(),
        'n_policies': len(entries),
        'global_depth_distribution': dict(global_depth),
        'verbatim_length_stats': verbatim_stats,
        'values_depth': depth_profile(value_depth)[:30],
        'principles_depth': depth_profile(principle_depth)[:30],
        'mechanisms_depth': depth_profile(mechanism_depth)[:30],
        'by_income_group': income_summary,
        'deepest_policies': sorted(
            entry_details,
            key=lambda x: x['depth_distribution'].get('section', 0) +
                          x['depth_distribution'].get('paragraph', 0),
            reverse=True
        )[:30],
    }

    with open(DEPTH_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"Summary saved: {DEPTH_SUMMARY}")
    log.info(f"\n{'='*60}")
    log.info(f"DEPTH ANALYSIS OVERVIEW")
    log.info(f"{'='*60}")
    log.info(f"Policies analysed: {len(entries)}")
    log.info(f"\nGlobal depth distribution:")
    for depth_level in ['section', 'paragraph', 'sentence', 'phrase', 'word', 'not_found']:
        n = global_depth.get(depth_level, 0)
        total = sum(global_depth.values())
        pct = n / total * 100 if total > 0 else 0
        log.info(f"  {depth_level:>12s}: {n:>5,}  ({pct:.1f}%)")

    log.info(f"\nMedian verbatim length by depth:")
    for d, stats in sorted(verbatim_stats.items(),
                           key=lambda x: {'word': 0, 'phrase': 1, 'sentence': 2,
                                          'paragraph': 3, 'section': 4}.get(x[0], 5)):
        log.info(f"  {d:>12s}: {stats['median_chars']:>5} chars (n={stats['n']})")

    log.info(f"\nTop 5 values by depth score:")
    for item in depth_profile(value_depth)[:5]:
        log.info(f"  {item['avg_depth_score']:.2f}  {item['name']} (n={item['count']}, "
                 f"{item['pct_deep']}% deep)")

    log.info(f"\nTop 5 principles by depth score:")
    for item in depth_profile(principle_depth)[:5]:
        log.info(f"  {item['avg_depth_score']:.2f}  {item['name']} (n={item['count']}, "
                 f"{item['pct_deep']}% deep)")

    log.info(f"\nBy income group:")
    for ig, stats in income_summary.items():
        log.info(f"  {ig}: {stats['n_policies']} policies, "
                 f"{stats['pct_deep']}% deep (paragraph+section)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Phase C: Verbatim Depth Extraction')
    parser.add_argument('--limit', type=int, default=0, help='Limit entries to process')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--workers', type=int, default=2, help='Concurrent API workers')
    parser.add_argument('--summarise', action='store_true', help='Aggregate depth results')

    args = parser.parse_args()

    if not args.summarise and args.limit == 0 and not args.resume:
        parser.print_help()
        print("\nExample workflow:")
        print("  1. python src/analysis/ethics_depth.py --limit 5            # test")
        print("  2. python src/analysis/ethics_depth.py --resume --workers 2 # full run")
        print("  3. python src/analysis/ethics_depth.py --summarise          # aggregate")
        sys.exit(0)

    if args.limit > 0 or args.resume:
        run_depth_extraction(limit=args.limit, resume=args.resume, workers=args.workers)

    if args.summarise:
        run_summarise()


if __name__ == '__main__':
    main()
