"""
Phase 5: Ethics Content Inventory
====================================
Extracts an inventory of specific VALUES, PRINCIPLES, and POLICY MECHANISMS
from all policies with non-zero ethics scores using LLM-based structured extraction.

Two-phase approach:
  Phase A – Mine existing evidence/rationale text from scores_raw.jsonl (no API cost)
  Phase B – New LLM pass to extract structured inventories from policy texts (API calls)

Inputs:
  data/analysis/scores_ensemble.json   – to identify ethics > 0 policies
  data/analysis/scores_raw.jsonl       – existing evidence text (Phase A)
  data/corpus/corpus_enriched.json     – original policy texts (Phase B)

Outputs:
  data/analysis/ethics_inventory/phase_a_evidence.json       – mined evidence per policy
  data/analysis/ethics_inventory/phase_b_inventory.jsonl     – structured inventory per policy
  data/analysis/ethics_inventory/inventory_summary.json      – aggregated inventory
  data/analysis/ethics_inventory/inventory_report.json       – run statistics

Usage:
  python src/analysis/ethics_inventory.py --phase A         # mine existing evidence (free)
  python src/analysis/ethics_inventory.py --phase B         # LLM extraction (API calls)
  python src/analysis/ethics_inventory.py --phase B --limit 5   # test on 5 entries
  python src/analysis/ethics_inventory.py --phase B --resume    # resume from checkpoint
  python src/analysis/ethics_inventory.py --phase B --workers 5 # concurrent API calls
  python src/analysis/ethics_inventory.py --summarise       # aggregate results
"""

import json
import os
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
CORPUS_PATH = ROOT / 'data' / 'corpus' / 'corpus_enriched.json'
SCORES_ENSEMBLE = ROOT / 'data' / 'analysis' / 'scores_ensemble.json'
SCORES_RAW = ROOT / 'data' / 'analysis' / 'scores_raw.jsonl'
OUTPUT_DIR = ROOT / 'data' / 'analysis' / 'ethics_inventory'
PHASE_A_OUTPUT = OUTPUT_DIR / 'phase_a_evidence.json'
PHASE_B_OUTPUT = OUTPUT_DIR / 'phase_b_inventory.jsonl'
SUMMARY_OUTPUT = OUTPUT_DIR / 'inventory_summary.json'
REPORT_OUTPUT = OUTPUT_DIR / 'inventory_report.json'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Config ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use GPT-4o for structured extraction (reliable JSON mode via response_format)
EXTRACTION_MODEL = 'openai/gpt-4o'

MAX_TEXT_CHARS = 30_000  # ~7,500 tokens

# ─── Extraction Taxonomy ──────────────────────────────────────────────────────
# These are the categories we want to identify in each policy document.
# Based on: Jobin et al. (2019), Floridi et al. (2018), UNESCO (2021),
# OECD AI Principles (2019), EU AI Act (2024)

TAXONOMY = {
    "values": [
        "Human dignity",
        "Human rights",
        "Fairness / Justice / Equity",
        "Non-discrimination",
        "Privacy / Data protection",
        "Autonomy / Self-determination",
        "Well-being / Beneficence",
        "Non-maleficence / Do no harm",
        "Freedom / Liberty",
        "Solidarity",
        "Sustainability / Environment",
        "Peace",
        "Cultural diversity",
        "Democracy",
        "Rule of law",
        "Trust",
        "Safety / Security",
        "Public interest / Common good",
    ],
    "principles": [
        "Transparency",
        "Explainability / Interpretability",
        "Accountability",
        "Responsibility",
        "Robustness / Reliability",
        "Human oversight / Human-in-the-loop",
        "Proportionality",
        "Precaution / Risk-based approach",
        "Inclusiveness / Accessibility",
        "Interoperability",
        "Contestability / Right to appeal",
        "Data governance / Data quality",
        "Open source / Openness",
        "International cooperation",
        "Multi-stakeholder governance",
        "Informed consent",
        "Purpose limitation",
        "Due diligence",
    ],
    "mechanisms": [
        "Ethics board / Ethics committee",
        "Impact assessment (AIIA / HRIA / DPIA)",
        "Algorithmic auditing / Third-party audit",
        "Certification / Conformity assessment",
        "Regulatory sandbox",
        "Complaints / Redress mechanism",
        "Standards / Technical standards",
        "Training / Capacity building requirements",
        "Labelling / Disclosure requirements",
        "Registration / Inventory of AI systems",
        "Risk classification / Tiered regulation",
        "Sectoral regulation (health, finance, etc.)",
        "Procurement requirements / Public sector AI",
        "Monitoring & evaluation / Reporting",
        "Penalties / Sanctions / Enforcement",
        "Insurance / Liability framework",
        "Code of conduct / Voluntary commitments",
        "Moratorium / Ban on specific uses",
        "Data sharing / Open data requirements",
        "Whistleblower protection",
    ],
}

# ─── Import country metadata ──────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / 'src' / 'analysis'))
from country_metadata import (
    INCOME_GROUP, REGION, GDP_PER_CAPITA, INTERNATIONAL,
    get_income_binary, get_metadata
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A: Mine existing evidence from scores_raw.jsonl
# ══════════════════════════════════════════════════════════════════════════════

def get_ethics_policy_ids() -> set:
    """Return set of entry IDs with ethics_score > 0 from ensemble."""
    with open(SCORES_ENSEMBLE, encoding='utf-8') as f:
        data = json.load(f)
    return {
        e['entry_id'] for e in data['entries']
        if e.get('analysis_ready', False) and e.get('ethics_score', 0) > 0
    }


def run_phase_a():
    """Mine evidence and rationale text from existing scores_raw.jsonl."""
    log.info("═══ PHASE A: Mining existing evidence from scores_raw.jsonl ═══")

    ethics_ids = get_ethics_policy_ids()
    log.info(f"Target policies with ethics > 0: {len(ethics_ids)}")

    # Collect evidence per policy (aggregate across 3 models)
    evidence_db = {}  # id → {title, jurisdiction, year, ethics_score, dimensions: {e1: [{model, score, evidence, rationale}]}}

    # Also load ensemble for metadata
    with open(SCORES_ENSEMBLE, encoding='utf-8') as f:
        ensemble = {e['entry_id']: e for e in json.load(f)['entries']}

    count = 0
    with open(SCORES_RAW, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            eid = entry.get('entry_id')
            if eid not in ethics_ids:
                continue

            count += 1
            if eid not in evidence_db:
                ens = ensemble.get(eid, {})
                evidence_db[eid] = {
                    'entry_id': eid,
                    'title': entry.get('title', ''),
                    'jurisdiction': entry.get('jurisdiction', ''),
                    'year': entry.get('year', 0),
                    'ethics_score': ens.get('ethics_score', 0),
                    'capacity_score': ens.get('capacity_score', 0),
                    'policy_type': ens.get('policy_type', ''),
                    'binding_nature': ens.get('binding_nature', ''),
                    'ethics_summary': '',
                    'dimensions': {d: [] for d in ['e1_framework', 'e2_rights', 'e3_governance',
                                                    'e4_operationalisation', 'e5_inclusion']},
                }

            scores = entry.get('scores', {})
            model_name = entry.get('model_name', entry.get('model_id', '?'))

            for dim in ['e1_framework', 'e2_rights', 'e3_governance',
                        'e4_operationalisation', 'e5_inclusion']:
                d = scores.get(dim, {})
                evidence_db[eid]['dimensions'][dim].append({
                    'model': model_name,
                    'score': d.get('score', 0),
                    'evidence': d.get('evidence', ''),
                    'rationale': d.get('rationale', ''),
                })

            # Take the longest ethics_summary across models
            summary = scores.get('ethics_summary', '')
            if len(summary) > len(evidence_db[eid].get('ethics_summary', '')):
                evidence_db[eid]['ethics_summary'] = summary

    log.info(f"Processed {count} raw score entries for {len(evidence_db)} unique policies")

    # Save Phase A output
    output = {
        'created': datetime.now().isoformat(),
        'phase': 'A',
        'description': 'Ethics evidence mined from existing LLM scores (scores_raw.jsonl)',
        'n_policies': len(evidence_db),
        'entries': list(evidence_db.values()),
    }

    with open(PHASE_A_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"Phase A output saved: {PHASE_A_OUTPUT}")
    log.info(f"Policies with evidence: {len(evidence_db)}")

    # Quick stats
    dims = ['e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion']
    for dim in dims:
        n_with_evidence = sum(
            1 for e in evidence_db.values()
            if any((obs.get('evidence') or '').strip() for obs in e['dimensions'][dim])
        )
        log.info(f"  {dim}: {n_with_evidence} policies with evidence text")

    return evidence_db


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B: LLM-based structured extraction
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert policy analyst specialising in AI ethics and governance.
You will analyse an AI governance policy document and extract a structured inventory
of the specific ethical values, principles, and policy mechanisms it contains.

Be rigorous and evidence-based. Only list items that are EXPLICITLY mentioned or
clearly implied in the document text. Do NOT infer items that are not present."""

EXTRACTION_PROMPT = """Analyse this AI governance policy and extract a structured inventory of its ethical content.

## CATEGORIES TO EXTRACT

### VALUES (ethical values the policy endorses or is grounded in)
Look for: human dignity, human rights, fairness/justice/equity, non-discrimination,
privacy/data protection, autonomy/self-determination, well-being/beneficence,
non-maleficence/do no harm, freedom/liberty, solidarity, sustainability/environment,
peace, cultural diversity, democracy, rule of law, trust, safety/security,
public interest/common good, and any OTHER values you find.

### PRINCIPLES (operational principles the policy articulates)
Look for: transparency, explainability/interpretability, accountability, responsibility,
robustness/reliability, human oversight/human-in-the-loop, proportionality,
precaution/risk-based approach, inclusiveness/accessibility, interoperability,
contestability/right to appeal, data governance/data quality, open source/openness,
international cooperation, multi-stakeholder governance, informed consent,
purpose limitation, due diligence, and any OTHER principles you find.

### MECHANISMS (concrete policy instruments, governance structures, or implementation tools)
Look for: ethics board/committee, impact assessment (AIIA/HRIA/DPIA),
algorithmic auditing/third-party audit, certification/conformity assessment,
regulatory sandbox, complaints/redress mechanism, standards/technical standards,
training/capacity building, labelling/disclosure requirements, registration/inventory of AI systems,
risk classification/tiered regulation, sectoral regulation, procurement requirements,
monitoring & evaluation/reporting, penalties/sanctions/enforcement,
insurance/liability framework, code of conduct/voluntary commitments,
moratorium/ban on specific uses, data sharing/open data, whistleblower protection,
and any OTHER mechanisms you find.

## POLICY DOCUMENT

**Title**: {title}
**Jurisdiction**: {jurisdiction}
**Year**: {year}
**Text quality**: {text_quality} ({word_count} words)

---
{text}
---

## REQUIRED OUTPUT

Return ONLY valid JSON (no markdown, no code fences) with this structure:

{{
  "values": [
    {{"name": "Human dignity", "strength": "explicit|implicit", "evidence": "brief quote or paraphrase"}},
    ...
  ],
  "principles": [
    {{"name": "Transparency", "strength": "explicit|implicit", "evidence": "brief quote or paraphrase"}},
    ...
  ],
  "mechanisms": [
    {{"name": "Impact assessment", "strength": "proposed|established|required", "evidence": "brief quote or paraphrase"}},
    ...
  ],
  "referenced_frameworks": ["UNESCO AI Recommendation", "OECD AI Principles", ...],
  "summary": "2-3 sentence summary of the policy's ethical approach"
}}

Rules:
- Only include items that appear in the text (explicit or clearly implied)
- Use the standardised names from the lists above where possible
- Add "Other: [description]" for items not in the standard lists
- For mechanisms, indicate strength: "proposed" (suggested), "established" (created), "required" (mandatory)
- referenced_frameworks: list any named ethical/governance frameworks the policy cites
- Keep evidence quotes brief (under 100 words each)"""


def _parse_json_robust(content: str) -> dict:
    """Parse JSON from LLM output, handling common formatting issues."""
    import re

    # Strip markdown code fences
    content = content.strip()
    if content.startswith('```'):
        # Remove opening fence (```json or ```)
        content = re.sub(r'^```\w*\n?', '', content)
    if content.endswith('```'):
        content = content[:-3]
    content = content.strip()

    # First try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    brace_start = content.find('{')
    brace_end = content.rfind('}')
    if brace_start >= 0 and brace_end > brace_start:
        extracted = content[brace_start:brace_end + 1]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        # Try fixing common issues: trailing commas, unescaped newlines in strings
        fixed = re.sub(r',\s*([}\]])', r'\1', extracted)  # remove trailing commas
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Try replacing unescaped control characters inside strings
        fixed2 = re.sub(r'(?<=": ")(.*?)(?="[,\s]*[}\]])', lambda m: m.group(0).replace('\n', ' ').replace('\r', ' '), fixed, flags=re.DOTALL)
        try:
            return json.loads(fixed2)
        except json.JSONDecodeError:
            pass

    # Last resort: raise the error
    raise json.JSONDecodeError("Could not parse JSON from LLM output", content[:200], 0)


def call_openrouter(text: str, title: str, jurisdiction: str, year: int,
                    text_quality: str, word_count: int,
                    max_retries: int = 3) -> dict | None:
    """Call OpenRouter API for structured ethics extraction."""
    import requests

    prompt = EXTRACTION_PROMPT.format(
        title=title,
        jurisdiction=jurisdiction,
        year=year,
        text_quality=text_quality,
        word_count=word_count,
        text=text[:MAX_TEXT_CHARS],
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lsempe77/ai-governance-capacity",
    }

    payload = {
        "model": EXTRACTION_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()

            content = result['choices'][0]['message']['content'].strip()
            parsed = _parse_json_robust(content)

            # Extract usage stats
            usage = result.get('usage', {})
            return {
                'inventory': parsed,
                'usage': {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                },
            }

        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error (attempt {attempt+1}): {e}")
            # Save first failed response for debugging
            debug_path = OUTPUT_DIR / 'debug_failed_response.txt'
            if not debug_path.exists():
                try:
                    with open(debug_path, 'w', encoding='utf-8') as df:
                        df.write(f"Title: {title}\nJurisdiction: {jurisdiction}\n")
                        df.write(f"Error: {e}\n\n=== RAW CONTENT ===\n")
                        df.write(content)
                        df.write(f"\n=== END ===\n\nrepr:\n{repr(content[:1000])}")
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            log.warning(f"  API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    return None


# Thread-safe counter
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
            elapsed = time.time() - self.start_time
            rate = self.done / elapsed if elapsed > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            log.info(f"  [{self.done}/{self.total}] "
                     f"({'OK' if success else 'FAIL'}) "
                     f"{rate:.1f}/min, ETA {eta/60:.0f}m "
                     f"(failed: {self.failed})")


def process_one_policy(entry: dict, corpus_texts: dict) -> dict | None:
    """Process a single policy through the LLM extraction."""
    eid = entry['entry_id']
    text_data = corpus_texts.get(eid, {})
    text = text_data.get('text', text_data.get('full_text', ''))

    if not text or len(text.strip()) < 50:
        return None

    result = call_openrouter(
        text=text,
        title=entry.get('title', ''),
        jurisdiction=entry.get('jurisdiction', ''),
        year=entry.get('year', 0),
        text_quality=entry.get('text_quality', 'unknown'),
        word_count=entry.get('word_count', 0),
    )

    if result is None:
        return None

    return {
        'entry_id': eid,
        'title': entry.get('title', ''),
        'jurisdiction': entry.get('jurisdiction', ''),
        'year': entry.get('year', 0),
        'ethics_score': entry.get('ethics_score', 0),
        'capacity_score': entry.get('capacity_score', 0),
        'policy_type': entry.get('policy_type', ''),
        'binding_nature': entry.get('binding_nature', ''),
        'inventory': result['inventory'],
        'usage': result['usage'],
        'timestamp': datetime.now().isoformat(),
    }


def run_phase_b(limit: int = 0, resume: bool = False, workers: int = 3):
    """Run LLM-based structured extraction on ethics policies."""
    log.info("═══ PHASE B: LLM-based structured ethics extraction ═══")

    # Load ensemble to get target policies
    with open(SCORES_ENSEMBLE, encoding='utf-8') as f:
        ensemble_data = json.load(f)

    ethics_entries = [
        e for e in ensemble_data['entries']
        if e.get('analysis_ready', False) and e.get('ethics_score', 0) > 0
    ]
    log.info(f"Target policies with ethics > 0: {len(ethics_entries)}")

    # Load corpus texts
    log.info("Loading corpus texts...")
    with open(CORPUS_PATH, encoding='utf-8') as f:
        corpus_data = json.load(f)

    # Build text lookup by matching title+jurisdiction (corpus doesn't have hash IDs)
    # First try to match by title + jurisdiction
    corpus_by_key = {}
    for ce in corpus_data.get('entries', []):
        key = (ce.get('title', ''), ce.get('jurisdiction', ''))
        text = ce.get('text', ce.get('full_text', ''))
        corpus_by_key[key] = {
            'text': text,
            'full_text': ce.get('full_text', text),
        }
    log.info(f"Corpus entries loaded: {len(corpus_by_key)}")

    # Build corpus_texts dict keyed by ensemble ID
    corpus_texts = {}
    for e in ethics_entries:
        key = (e.get('title', ''), e.get('jurisdiction', ''))
        if key in corpus_by_key:
            corpus_texts[e['entry_id']] = corpus_by_key[key]

    log.info(f"Matched corpus texts: {len(corpus_texts)} / {len(ethics_entries)}")

    # Check for existing results (resume support)
    done_ids = set()
    if resume and PHASE_B_OUTPUT.exists():
        with open(PHASE_B_OUTPUT, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r['entry_id'])
                except:
                    pass
        log.info(f"Resuming: {len(done_ids)} already completed")

    # Filter to remaining
    remaining = [e for e in ethics_entries if e['entry_id'] not in done_ids and e['entry_id'] in corpus_texts]

    if limit > 0:
        remaining = remaining[:limit]

    log.info(f"Policies to process: {len(remaining)}")

    if not remaining:
        log.info("Nothing to process!")
        return

    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    # Process with thread pool
    counter = ProgressCounter(len(remaining))
    results = []
    total_tokens = 0

    mode = 'a' if resume else 'w'
    with open(PHASE_B_OUTPUT, mode, encoding='utf-8') as out_f:
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

    elapsed = time.time() - counter.start_time

    # Save report
    report = {
        'created': datetime.now().isoformat(),
        'phase': 'B',
        'model': EXTRACTION_MODEL,
        'total_target': len(ethics_entries),
        'processed': len(results),
        'failed': counter.failed,
        'skipped_no_text': len(ethics_entries) - len(corpus_texts),
        'skipped_already_done': len(done_ids),
        'total_tokens': total_tokens,
        'estimated_cost_usd': total_tokens * 0.000005,  # approximate
        'elapsed_seconds': elapsed,
        'rate_per_minute': len(results) / (elapsed / 60) if elapsed > 0 else 0,
    }

    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Phase B complete!")
    log.info(f"  Processed: {len(results)}")
    log.info(f"  Failed: {counter.failed}")
    log.info(f"  Total tokens: {total_tokens:,}")
    log.info(f"  Est. cost: ${report['estimated_cost_usd']:.2f}")
    log.info(f"  Time: {elapsed/60:.1f} minutes")
    log.info(f"  Output: {PHASE_B_OUTPUT}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARISE: Aggregate inventory results
# ══════════════════════════════════════════════════════════════════════════════

def run_summarise():
    """Aggregate Phase B results into summary tables."""
    log.info("═══ SUMMARISE: Aggregating inventory results ═══")

    if not PHASE_B_OUTPUT.exists():
        log.error(f"Phase B output not found: {PHASE_B_OUTPUT}")
        log.error("Run --phase B first!")
        sys.exit(1)

    # Load all Phase B results
    entries = []
    with open(PHASE_B_OUTPUT, encoding='utf-8') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass

    log.info(f"Loaded {len(entries)} inventory entries")

    # ── Global frequency counts ────────────────────────────────────────────
    value_counts = Counter()
    principle_counts = Counter()
    mechanism_counts = Counter()
    framework_counts = Counter()

    # ── By jurisdiction ────────────────────────────────────────────────────
    jurisdiction_values = defaultdict(Counter)
    jurisdiction_principles = defaultdict(Counter)
    jurisdiction_mechanisms = defaultdict(Counter)

    # ── By income group ────────────────────────────────────────────────────
    income_values = defaultdict(Counter)
    income_principles = defaultdict(Counter)
    income_mechanisms = defaultdict(Counter)

    # ── By policy type ─────────────────────────────────────────────────────
    ptype_values = defaultdict(Counter)
    ptype_principles = defaultdict(Counter)
    ptype_mechanisms = defaultdict(Counter)

    # ── By binding nature ──────────────────────────────────────────────────
    binding_values = defaultdict(Counter)
    binding_principles = defaultdict(Counter)
    binding_mechanisms = defaultdict(Counter)

    # ── Strength distributions ─────────────────────────────────────────────
    value_strength = defaultdict(Counter)    # value_name → {explicit: n, implicit: n}
    mechanism_strength = defaultdict(Counter)  # mechanism_name → {proposed: n, established: n, required: n}

    # ── Per-entry stats ────────────────────────────────────────────────────
    entry_stats = []

    for entry in entries:
        inv = entry.get('inventory', {})
        jurisdiction = entry.get('jurisdiction', '')
        income = get_income_binary(jurisdiction) or 'Unknown'
        ptype = entry.get('policy_type', 'Unknown')
        binding = entry.get('binding_nature', 'Unknown')

        values = inv.get('values', [])
        principles = inv.get('principles', [])
        mechanisms = inv.get('mechanisms', [])
        frameworks = inv.get('referenced_frameworks', [])

        # Global counts
        for v in values:
            name = v.get('name', '').strip()
            if name:
                value_counts[name] += 1
                jurisdiction_values[jurisdiction][name] += 1
                income_values[income][name] += 1
                ptype_values[ptype][name] += 1
                binding_values[binding][name] += 1
                strength = v.get('strength', 'unknown')
                value_strength[name][strength] += 1

        for p in principles:
            name = p.get('name', '').strip()
            if name:
                principle_counts[name] += 1
                jurisdiction_principles[jurisdiction][name] += 1
                income_principles[income][name] += 1
                ptype_principles[ptype][name] += 1
                binding_principles[binding][name] += 1

        for m in mechanisms:
            name = m.get('name', '').strip()
            if name:
                mechanism_counts[name] += 1
                jurisdiction_mechanisms[jurisdiction][name] += 1
                income_mechanisms[income][name] += 1
                ptype_mechanisms[ptype][name] += 1
                binding_mechanisms[binding][name] += 1
                strength = m.get('strength', 'unknown')
                mechanism_strength[name][strength] += 1

        for fw in frameworks:
            if fw:
                framework_counts[fw.strip()] += 1

        entry_stats.append({
            'entry_id': entry['entry_id'],
            'title': entry['title'],
            'jurisdiction': jurisdiction,
            'income': income,
            'ethics_score': entry.get('ethics_score', 0),
            'n_values': len(values),
            'n_principles': len(principles),
            'n_mechanisms': len(mechanisms),
            'n_frameworks': len(frameworks),
        })

    # ── Build summary ──────────────────────────────────────────────────────
    import statistics

    n_values_list = [s['n_values'] for s in entry_stats]
    n_principles_list = [s['n_principles'] for s in entry_stats]
    n_mechanisms_list = [s['n_mechanisms'] for s in entry_stats]

    summary = {
        'created': datetime.now().isoformat(),
        'n_policies_inventoried': len(entries),

        'overview': {
            'total_unique_values': len(value_counts),
            'total_unique_principles': len(principle_counts),
            'total_unique_mechanisms': len(mechanism_counts),
            'total_unique_frameworks': len(framework_counts),
            'avg_values_per_policy': round(statistics.mean(n_values_list), 2) if n_values_list else 0,
            'avg_principles_per_policy': round(statistics.mean(n_principles_list), 2) if n_principles_list else 0,
            'avg_mechanisms_per_policy': round(statistics.mean(n_mechanisms_list), 2) if n_mechanisms_list else 0,
            'median_values_per_policy': round(statistics.median(n_values_list), 1) if n_values_list else 0,
            'median_principles_per_policy': round(statistics.median(n_principles_list), 1) if n_principles_list else 0,
            'median_mechanisms_per_policy': round(statistics.median(n_mechanisms_list), 1) if n_mechanisms_list else 0,
        },

        'values_ranking': [
            {'name': name, 'count': count, 'pct': round(count / len(entries) * 100, 1),
             'strength': dict(value_strength[name])}
            for name, count in value_counts.most_common(50)
        ],

        'principles_ranking': [
            {'name': name, 'count': count, 'pct': round(count / len(entries) * 100, 1)}
            for name, count in principle_counts.most_common(50)
        ],

        'mechanisms_ranking': [
            {'name': name, 'count': count, 'pct': round(count / len(entries) * 100, 1),
             'strength': dict(mechanism_strength[name])}
            for name, count in mechanism_counts.most_common(50)
        ],

        'frameworks_ranking': [
            {'name': name, 'count': count, 'pct': round(count / len(entries) * 100, 1)}
            for name, count in framework_counts.most_common(30)
        ],

        'by_income_group': {
            income: {
                'n_policies': sum(1 for s in entry_stats if s['income'] == income),
                'top_values': [{'name': n, 'count': c} for n, c in income_values[income].most_common(10)],
                'top_principles': [{'name': n, 'count': c} for n, c in income_principles[income].most_common(10)],
                'top_mechanisms': [{'name': n, 'count': c} for n, c in income_mechanisms[income].most_common(10)],
            }
            for income in sorted(income_values.keys())
        },

        'by_policy_type': {
            ptype: {
                'n_policies': sum(1 for s in entry_stats if s.get('policy_type', s.get('jurisdiction', '')) and entry_stats),
                'top_values': [{'name': n, 'count': c} for n, c in ptype_values[ptype].most_common(5)],
                'top_mechanisms': [{'name': n, 'count': c} for n, c in ptype_mechanisms[ptype].most_common(5)],
            }
            for ptype in ['Legislation', 'Regulation', 'Strategy', 'Framework', 'Guideline', 'Action Plan', 'Program']
            if ptype in ptype_values
        },

        'by_binding_nature': {
            binding: {
                'n_policies': sum(1 for s in entry_stats if True),  # will fix below
                'top_values': [{'name': n, 'count': c} for n, c in binding_values[binding].most_common(5)],
                'top_mechanisms': [{'name': n, 'count': c} for n, c in binding_mechanisms[binding].most_common(5)],
            }
            for binding in sorted(binding_values.keys())
        },

        'entry_level_stats': sorted(entry_stats, key=lambda x: x['n_values'] + x['n_principles'] + x['n_mechanisms'], reverse=True)[:50],
    }

    # Fix by_policy_type and by_binding_nature counts
    ptype_counter = Counter(s.get('policy_type', 'Unknown') for e in entries for s in [next(
        (es for es in entry_stats if es['entry_id'] == e['entry_id']), {})])
    # Simpler approach: count from entry_stats directly
    for ptype in summary.get('by_policy_type', {}):
        summary['by_policy_type'][ptype]['n_policies'] = sum(
            1 for e in entries if e.get('policy_type', '') == ptype)

    for binding in summary.get('by_binding_nature', {}):
        summary['by_binding_nature'][binding]['n_policies'] = sum(
            1 for e in entries if e.get('binding_nature', '') == binding)

    with open(SUMMARY_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"\nSummary saved: {SUMMARY_OUTPUT}")
    log.info(f"\n{'='*60}")
    log.info(f"INVENTORY OVERVIEW")
    log.info(f"{'='*60}")
    log.info(f"Policies inventoried: {len(entries)}")
    log.info(f"Unique values found: {len(value_counts)}")
    log.info(f"Unique principles found: {len(principle_counts)}")
    log.info(f"Unique mechanisms found: {len(mechanism_counts)}")
    log.info(f"Referenced frameworks: {len(framework_counts)}")
    log.info(f"\nAvg per policy: {summary['overview']['avg_values_per_policy']} values, "
             f"{summary['overview']['avg_principles_per_policy']} principles, "
             f"{summary['overview']['avg_mechanisms_per_policy']} mechanisms")

    log.info(f"\nTop 10 VALUES:")
    for item in summary['values_ranking'][:10]:
        log.info(f"  {item['pct']:5.1f}%  {item['name']} (n={item['count']})")

    log.info(f"\nTop 10 PRINCIPLES:")
    for item in summary['principles_ranking'][:10]:
        log.info(f"  {item['pct']:5.1f}%  {item['name']} (n={item['count']})")

    log.info(f"\nTop 10 MECHANISMS:")
    for item in summary['mechanisms_ranking'][:10]:
        log.info(f"  {item['pct']:5.1f}%  {item['name']} (n={item['count']})")

    log.info(f"\nTop 10 REFERENCED FRAMEWORKS:")
    for item in summary['frameworks_ranking'][:10]:
        log.info(f"  {item['pct']:5.1f}%  {item['name']} (n={item['count']})")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Ethics Content Inventory')
    parser.add_argument('--phase', choices=['A', 'B'], help='Phase to run (A=mine evidence, B=LLM extraction)')
    parser.add_argument('--summarise', action='store_true', help='Aggregate Phase B results')
    parser.add_argument('--limit', type=int, default=0, help='Limit entries to process (Phase B)')
    parser.add_argument('--resume', action='store_true', help='Resume Phase B from checkpoint')
    parser.add_argument('--workers', type=int, default=3, help='Concurrent API workers (Phase B)')

    args = parser.parse_args()

    if not args.phase and not args.summarise:
        parser.print_help()
        print("\nExample workflow:")
        print("  1. python src/analysis/ethics_inventory.py --phase A")
        print("  2. python src/analysis/ethics_inventory.py --phase B --limit 5   # test")
        print("  3. python src/analysis/ethics_inventory.py --phase B --resume     # full run")
        print("  4. python src/analysis/ethics_inventory.py --summarise")
        sys.exit(0)

    if args.phase == 'A':
        run_phase_a()
    elif args.phase == 'B':
        run_phase_b(limit=args.limit, resume=args.resume, workers=args.workers)

    if args.summarise:
        run_summarise()


if __name__ == '__main__':
    main()
