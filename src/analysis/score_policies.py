"""
Phase 2: LLM-Based Policy Scoring Pipeline
============================================
Scores every policy on implementation capacity (5 dims) and ethics (5 dims)
using a 3-model ensemble via OpenRouter.

Models:
  A: anthropic/claude-sonnet-4
  B: openai/gpt-4o
  C: google/gemini-2.0-flash-001

Capacity dimensions (0–4, Mazmanian-Sabatier / Lipsky / Grindle / Fukuyama):
  1. Clarity & Specificity
  2. Resources & Budget
  3. Authority & Enforcement
  4. Accountability
  5. Coherence & Coordination

Ethics dimensions (0–4, Jobin / Floridi / OECD / UNESCO / EU AI Act):
  1. Ethical Framework Depth
  2. Rights Protection
  3. Governance Mechanisms
  4. Operationalisation
  5. Inclusion & Participation

Output:
  data/analysis/scores_raw.jsonl         – one JSON line per (entry × model)
  data/analysis/scores_ensemble.json     – merged scores (median of 3 models)
  data/analysis/scoring_report.json      – run statistics

Usage:
  python src/analysis/score_policies.py                  # full run, model A
  python src/analysis/score_policies.py --limit 10       # test on 10 entries
  python src/analysis/score_policies.py --model B        # run model B only
  python src/analysis/score_policies.py --model all      # run all 3 models
  python src/analysis/score_policies.py --merge          # merge existing scores
  python src/analysis/score_policies.py --resume         # resume from checkpoint
"""

import json
import hashlib
import os
import re
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
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
CORPUS_PATH = ROOT / 'data' / 'corpus' / 'corpus_enriched.json'
OUTPUT_DIR = ROOT / 'data' / 'analysis'
SCORES_RAW = OUTPUT_DIR / 'scores_raw.jsonl'
SCORES_ENSEMBLE = OUTPUT_DIR / 'scores_ensemble.json'
REPORT_PATH = OUTPUT_DIR / 'scoring_report.json'

# ─── API Config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    'A': 'anthropic/claude-sonnet-4',
    'B': 'openai/gpt-4o',
    'C': 'google/gemini-2.0-flash-001',
}

# Text limits per model (approximate token limits → character limits)
MAX_TEXT_CHARS = 30_000  # ~7,500 tokens — leaves room for prompt + response

# ─── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert policy analyst with deep expertise in:
1. Implementation science (Mazmanian & Sabatier 1983, Lipsky 1980, Grindle 1996, Fukuyama 2013)
2. AI ethics governance (Jobin et al. 2019, Floridi et al. 2018, OECD AI Principles, UNESCO AI Recommendation, EU AI Act)

You will analyse a policy document and return structured scores. Be rigorous and evidence-based.
Score strictly — most policies should score 1–2 on average, not 3–4. Reserve high scores for genuinely exceptional provisions."""

SCORING_PROMPT = """Analyse this AI governance policy and score it on 10 dimensions.

## CAPACITY DIMENSIONS (0–4 each)

**C1 – Clarity & Specificity** (Mazmanian-Sabatier Condition 1)
0 = No specific goals  |  1 = Vague aspirations ("promote AI")  |  2 = Some targets, no timelines  |  3 = Clear targets with timelines  |  4 = SMART objectives with KPIs

**C2 – Resources & Budget** (Grindle fiscal/technical capacity)
0 = No resources mentioned  |  1 = General commitment  |  2 = Some budget/staffing, no specifics  |  3 = Detailed allocations  |  4 = Multi-year budgets with accountability

**C3 – Authority & Enforcement** (Mazmanian-Sabatier Condition 3)
0 = No implementing body  |  1 = Vague ("relevant ministries")  |  2 = Named agency, unclear mandate  |  3 = Clear mandate with legal basis  |  4 = Multi-level structure with defined roles

**C4 – Accountability** (Lipsky)
0 = No M&E mentioned  |  1 = General reporting  |  2 = Some indicators defined  |  3 = Regular evaluation cycles  |  4 = Independent oversight + sanctions

**C5 – Coherence & Coordination** (Institutional theory)
0 = Isolated policy  |  1 = References other policies  |  2 = Explicit alignment with frameworks  |  3 = Cross-ministry coordination  |  4 = Full ecosystem integration

## ETHICS DIMENSIONS (0–4 each)

**E1 – Ethical Framework Depth** (Jobin et al. 2019, Floridi et al. 2018)
0 = No ethics content  |  1 = Mentions ethics keywords  |  2 = Articulates principles  |  3 = Coherent ethical framework with justification  |  4 = Comprehensive, literature-grounded framework

**E2 – Rights Protection** (UNESCO, EU AI Act)
0 = No rights mentioned  |  1 = Generic rights language  |  2 = Specific rights (privacy, non-discrimination)  |  3 = Mechanisms for rights enforcement  |  4 = Comprehensive rights framework with remedies

**E3 – Governance Mechanisms** (OECD AI Principles)
0 = No governance structures  |  1 = Mentions oversight need  |  2 = Proposes specific bodies  |  3 = Impact assessments + auditing required  |  4 = Independent oversight + public accountability

**E4 – Operationalisation** (Stix 2021)
0 = Purely aspirational  |  1 = Voluntary guidelines  |  2 = Standards/certification  |  3 = Binding requirements with compliance  |  4 = Full implementation with enforcement + penalties

**E5 – Inclusion & Participation** (UNESCO)
0 = No stakeholder engagement  |  1 = Mentions consultation  |  2 = Describes consultation process  |  3 = Formal multi-stakeholder governance  |  4 = Marginalised groups explicitly included + ongoing participation

## METADATA

Also classify:
- **policy_type**: Strategy | Legislation | Regulation | Guideline | Framework | Action Plan | Executive Order | Program | Other
- **binding_nature**: Non-binding | Soft law | Binding regulation | Hard law
- **primary_language**: the language the document is written in (ISO 639-1 code)

## POLICY DOCUMENT

**Title**: {title}
**Jurisdiction**: {jurisdiction}
**Year**: {year}
**Text quality**: {text_quality} ({word_count} words)

---
{text}
---

## REQUIRED OUTPUT

Return ONLY valid JSON (no markdown, no code fences) in this exact structure:

{{
  "c1_clarity": {{"score": 0, "evidence": "quote or paraphrase", "rationale": "why"}},
  "c2_resources": {{"score": 0, "evidence": "", "rationale": ""}},
  "c3_authority": {{"score": 0, "evidence": "", "rationale": ""}},
  "c4_accountability": {{"score": 0, "evidence": "", "rationale": ""}},
  "c5_coherence": {{"score": 0, "evidence": "", "rationale": ""}},
  "e1_framework": {{"score": 0, "evidence": "", "rationale": ""}},
  "e2_rights": {{"score": 0, "evidence": "", "rationale": ""}},
  "e3_governance": {{"score": 0, "evidence": "", "rationale": ""}},
  "e4_operationalisation": {{"score": 0, "evidence": "", "rationale": ""}},
  "e5_inclusion": {{"score": 0, "evidence": "", "rationale": ""}},
  "policy_type": "",
  "binding_nature": "",
  "primary_language": "",
  "capacity_summary": "2-sentence summary of implementation capacity",
  "ethics_summary": "2-sentence summary of ethics governance"
}}"""


# ─── API Call ──────────────────────────────────────────────────────────────────

def call_openrouter(model: str, system: str, prompt: str, max_retries: int = 3) -> dict | None:
    """Call OpenRouter API with retries and exponential backoff."""
    import requests

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lsempe77/ai-governance-capacity",
        "X-Title": "AI Governance Capacity Observatory",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2500,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=180)

            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                log.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if 'error' in data:
                log.warning(f"  API error: {data['error']}")
                time.sleep(2 ** attempt)
                continue

            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            return {'content': content, 'usage': usage}

        except Exception as e:
            log.warning(f"  Request failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    return None


def parse_json_response(text: str) -> dict | None:
    """Parse JSON from LLM response, handling common formatting issues."""
    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split('\n')
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = '\n'.join(lines[start:end])

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_scores(parsed: dict) -> bool:
    """Check that all 10 dimension scores are present and valid."""
    dims = ['c1_clarity', 'c2_resources', 'c3_authority', 'c4_accountability', 'c5_coherence',
            'e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion']
    for d in dims:
        if d not in parsed:
            return False
        if not isinstance(parsed[d], dict) or 'score' not in parsed[d]:
            return False
        score = parsed[d]['score']
        if not isinstance(score, (int, float)) or score < 0 or score > 4:
            return False
    return True


# ─── Entry ID ──────────────────────────────────────────────────────────────────

def entry_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ─── Score a single entry ──────────────────────────────────────────────────────

def score_entry(entry: dict, model_key: str) -> dict | None:
    """Score a single policy entry with one model. Returns result dict or None."""
    model = MODELS[model_key]
    text = entry.get('full_text', '') or entry.get('content', '')

    # Truncate if needed
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[... text truncated ...]"

    prompt = SCORING_PROMPT.format(
        title=entry.get('title', 'Unknown'),
        jurisdiction=entry.get('jurisdiction', 'Unknown'),
        year=entry.get('year', 'Unknown'),
        text_quality=entry.get('text_quality', 'unknown'),
        word_count=entry.get('extraction_word_count', 0),
        text=text,
    )

    response = call_openrouter(model, SYSTEM_PROMPT, prompt)
    if not response:
        return None

    parsed = parse_json_response(response['content'])
    if not parsed:
        log.warning(f"  JSON parse failed for {model_key}")
        return None

    if not validate_scores(parsed):
        log.warning(f"  Score validation failed for {model_key}")
        return None

    # Compute composite scores
    cap_scores = [parsed[d]['score'] for d in ['c1_clarity', 'c2_resources', 'c3_authority', 'c4_accountability', 'c5_coherence']]
    eth_scores = [parsed[d]['score'] for d in ['e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion']]

    parsed['capacity_score'] = round(sum(cap_scores) / len(cap_scores), 2)
    parsed['ethics_score'] = round(sum(eth_scores) / len(eth_scores), 2)
    parsed['overall_score'] = round((parsed['capacity_score'] + parsed['ethics_score']) / 2, 2)

    result = {
        'entry_id': entry_id(entry['url']),
        'title': entry.get('title', ''),
        'jurisdiction': entry.get('jurisdiction', ''),
        'year': entry.get('year', ''),
        'model': model_key,
        'model_name': model,
        'text_quality': entry.get('text_quality', 'unknown'),
        'analysis_ready': entry.get('analysis_ready', False),
        'word_count': entry.get('extraction_word_count', 0),
        'scores': parsed,
        'usage': response.get('usage', {}),
        'timestamp': datetime.now().isoformat(),
    }

    return result


# ─── Checkpoint Management ─────────────────────────────────────────────────────

def load_completed_ids(model_key: str) -> set:
    """Load entry IDs already scored for a given model from the raw scores file."""
    completed = set()
    if SCORES_RAW.exists():
        with open(SCORES_RAW, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get('model') == model_key:
                        completed.add(rec['entry_id'])
                except json.JSONDecodeError:
                    continue
    return completed


def append_result(result: dict):
    """Append a single result to the raw scores file (JSONL)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORES_RAW, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


# ─── Merge Ensemble ────────────────────────────────────────────────────────────

def merge_scores():
    """Merge raw per-model scores into ensemble scores (median of available models)."""
    log.info("Merging scores across models...")

    # Load all raw scores
    by_entry = defaultdict(list)
    if not SCORES_RAW.exists():
        log.error("No raw scores found")
        return

    with open(SCORES_RAW, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_entry[rec['entry_id']].append(rec)
            except json.JSONDecodeError:
                continue

    log.info(f"  {len(by_entry)} entries with scores")

    dims = ['c1_clarity', 'c2_resources', 'c3_authority', 'c4_accountability', 'c5_coherence',
            'e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion']

    ensemble_entries = []
    for eid, records in by_entry.items():
        # Take the first record's metadata
        meta = records[0]

        ensemble = {
            'entry_id': eid,
            'title': meta['title'],
            'jurisdiction': meta['jurisdiction'],
            'year': meta['year'],
            'text_quality': meta['text_quality'],
            'analysis_ready': meta['analysis_ready'],
            'word_count': meta['word_count'],
            'n_models': len(records),
            'models_used': [r['model'] for r in records],
        }

        # Compute median score per dimension
        for d in dims:
            scores_list = []
            for r in records:
                s = r['scores'].get(d, {})
                if isinstance(s, dict) and 'score' in s:
                    scores_list.append(s['score'])
            if scores_list:
                scores_list.sort()
                median = scores_list[len(scores_list) // 2]
                ensemble[d] = {
                    'median': median,
                    'scores': {r['model']: r['scores'].get(d, {}).get('score') for r in records},
                    'spread': max(scores_list) - min(scores_list),
                }
            else:
                ensemble[d] = {'median': 0, 'scores': {}, 'spread': 0}

        # Composite capacity / ethics
        cap = [ensemble[d]['median'] for d in dims[:5]]
        eth = [ensemble[d]['median'] for d in dims[5:]]
        ensemble['capacity_score'] = round(sum(cap) / len(cap), 2)
        ensemble['ethics_score'] = round(sum(eth) / len(eth), 2)
        ensemble['overall_score'] = round((ensemble['capacity_score'] + ensemble['ethics_score']) / 2, 2)

        # Policy type / binding nature (majority vote)
        for field in ['policy_type', 'binding_nature', 'primary_language']:
            values = [r['scores'].get(field, '') for r in records if r['scores'].get(field)]
            if values:
                ensemble[field] = max(set(values), key=values.count)
            else:
                ensemble[field] = 'Unknown'

        # Summaries from model A preferably
        for field in ['capacity_summary', 'ethics_summary']:
            for r in sorted(records, key=lambda x: x['model']):
                if r['scores'].get(field):
                    ensemble[field] = r['scores'][field]
                    break

        # Inter-model agreement (mean absolute spread across dimensions)
        spreads = [ensemble[d]['spread'] for d in dims]
        ensemble['mean_spread'] = round(sum(spreads) / len(spreads), 2) if spreads else 0

        ensemble_entries.append(ensemble)

    # Sort by overall score descending
    ensemble_entries.sort(key=lambda x: x['overall_score'], reverse=True)

    # Compute global stats
    cap_scores = [e['capacity_score'] for e in ensemble_entries]
    eth_scores = [e['ethics_score'] for e in ensemble_entries]

    output = {
        'created': datetime.now().isoformat(),
        'stats': {
            'total_entries': len(ensemble_entries),
            'models_used': list(set(m for e in ensemble_entries for m in e['models_used'])),
            'capacity_mean': round(sum(cap_scores) / len(cap_scores), 2) if cap_scores else 0,
            'ethics_mean': round(sum(eth_scores) / len(eth_scores), 2) if eth_scores else 0,
            'mean_spread': round(sum(e['mean_spread'] for e in ensemble_entries) / len(ensemble_entries), 2) if ensemble_entries else 0,
        },
        'entries': ensemble_entries,
    }

    with open(SCORES_ENSEMBLE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=1)

    log.info(f"  Ensemble saved: {SCORES_ENSEMBLE}")
    log.info(f"  Capacity mean: {output['stats']['capacity_mean']}/4")
    log.info(f"  Ethics mean:   {output['stats']['ethics_mean']}/4")
    log.info(f"  Mean spread:   {output['stats']['mean_spread']}")

    return output


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(model_key: str = 'A', limit: int | None = None, resume: bool = True):
    """Score policies with a single model."""
    log.info("=" * 70)
    log.info(f"PHASE 2: POLICY SCORING — Model {model_key} ({MODELS[model_key]})")
    log.info("=" * 70)

    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY not found in .env")
        sys.exit(1)

    # Load enriched corpus
    log.info(f"Loading corpus from {CORPUS_PATH}")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    entries = corpus['entries']
    log.info(f"  Corpus: {len(entries)} entries")

    # Load checkpoint
    completed = load_completed_ids(model_key) if resume else set()
    if completed:
        log.info(f"  Resuming: {len(completed)} already scored for model {model_key}")

    # Filter to entries to process
    to_process = []
    for entry in entries:
        eid = entry_id(entry['url'])
        if eid in completed:
            continue
        to_process.append(entry)

    if limit:
        to_process = to_process[:limit]

    log.info(f"  To process: {len(to_process)} entries")

    # Counters
    stats = Counter()
    start_time = time.time()
    total_cost = 0.0

    for i, entry in enumerate(to_process):
        eid = entry_id(entry['url'])
        title = entry.get('title', 'Unknown')[:55]
        quality = entry.get('text_quality', '?')

        log.info(f"  [{i+1}/{len(to_process)}] {title}... ({quality})")

        try:
            result = score_entry(entry, model_key)
        except KeyboardInterrupt:
            log.info("\n  Interrupted. Progress saved (JSONL append-only).")
            break
        except Exception as exc:
            log.warning(f"    Exception: {exc}")
            result = None

        if result:
            append_result(result)
            cap = result['scores']['capacity_score']
            eth = result['scores']['ethics_score']
            log.info(f"    ✓ Cap={cap:.1f} Eth={eth:.1f}")
            stats['success'] += 1

            # Estimate cost from usage
            usage = result.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            # Rough cost estimate (varies by model)
            total_cost += (prompt_tokens * 0.003 + completion_tokens * 0.015) / 1000
        else:
            log.warning(f"    ✗ Failed")
            stats['failed'] += 1

        # Rate limiting — be gentle
        time.sleep(1.0)

        # Progress report every 50
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 3600
            eta_hours = (len(to_process) - i - 1) / rate * 3600 / 3600 if rate > 0 else 0
            log.info(f"\n  Progress: {i+1}/{len(to_process)} | "
                     f"Success: {stats['success']} | Failed: {stats['failed']} | "
                     f"Rate: {rate:.0f}/hr | ETA: {eta_hours:.1f}h | "
                     f"Est. cost: ${total_cost:.2f}\n")

    elapsed = time.time() - start_time

    # Save report
    report = {
        'model': model_key,
        'model_name': MODELS[model_key],
        'completed': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'entries_processed': stats['success'] + stats['failed'],
        'successful': stats['success'],
        'failed': stats['failed'],
        'estimated_cost_usd': round(total_cost, 2),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("\n" + "=" * 70)
    log.info("SCORING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Model:      {model_key} ({MODELS[model_key]})")
    log.info(f"  Processed:  {stats['success'] + stats['failed']}")
    log.info(f"  Success:    {stats['success']}")
    log.info(f"  Failed:     {stats['failed']}")
    log.info(f"  Time:       {elapsed/60:.1f} min")
    log.info(f"  Est. cost:  ${total_cost:.2f}")
    log.info(f"  Raw file:   {SCORES_RAW}")
    log.info("=" * 70)

    return report


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: Score policies with LLM ensemble')
    parser.add_argument('--model', type=str, default='A',
                        help='Model to use: A (Claude), B (GPT-4o), C (Gemini), or "all"')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N entries (for testing)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing progress (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignoring existing progress')
    parser.add_argument('--merge', action='store_true',
                        help='Only merge existing raw scores into ensemble')
    args = parser.parse_args()

    if args.merge:
        merge_scores()
        sys.exit(0)

    resume = not args.no_resume

    if args.model.lower() == 'all':
        for key in ['A', 'B', 'C']:
            run_pipeline(model_key=key, limit=args.limit, resume=resume)
        merge_scores()
    else:
        key = args.model.upper()
        if key not in MODELS:
            log.error(f"Unknown model key: {key}. Use A, B, C, or 'all'")
            sys.exit(1)
        run_pipeline(model_key=key, limit=args.limit, resume=resume)
        # Auto-merge if we have scores from multiple models
        if SCORES_RAW.exists():
            models_present = set()
            with open(SCORES_RAW, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        models_present.add(rec.get('model', ''))
                    except:
                        pass
            if len(models_present) > 0:
                merge_scores()

    sys.exit(0)
