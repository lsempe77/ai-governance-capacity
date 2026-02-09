"""
Normalize Phase C depth labels back to the canonical 56-item taxonomy.

Strategy:
1. Case-insensitive exact match (after stripping slashes/spaces)
2. Fuzzy substring matching for near-duplicates
3. Remaining long-tail items → "Other" bucket per category

Reads:  phase_c_depth.jsonl
Writes: phase_c_depth_normalised.jsonl  (same structure, normalised names)
        normalisation_map.json          (label → canonical mapping for audit)
"""
import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "analysis" / "ethics_inventory"
INPUT  = DATA / "phase_c_depth.jsonl"
OUTPUT = DATA / "phase_c_depth_normalised.jsonl"
MAP_FILE = DATA / "normalisation_map.json"

# ── Canonical taxonomy (from ethics_inventory.py) ─────────────────────────────
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
        "Impact assessment",
        "Algorithmic auditing / Third-party audit",
        "Certification / Conformity assessment",
        "Regulatory sandbox",
        "Complaints / Redress mechanism",
        "Standards / Technical standards",
        "Training / Capacity building",
        "Labelling / Disclosure requirements",
        "Registration / Inventory of AI systems",
        "Risk classification / Tiered regulation",
        "Sectoral regulation",
        "Procurement requirements",
        "Monitoring & evaluation / Reporting",
        "Penalties / Sanctions / Enforcement",
        "Insurance / Liability framework",
        "Code of conduct / Voluntary commitments",
        "Moratorium / Ban on specific uses",
        "Data sharing / Open data",
        "Whistleblower protection",
    ],
}


def _normalise_key(s: str) -> str:
    """Lowercase, strip punctuation/spaces for matching."""
    s = s.lower().strip()
    s = re.sub(r'[/\-–—,()&]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── Build matching index ──────────────────────────────────────────────────────
# For each canonical name, generate multiple lookup keys

def _build_lookup(canonical_list: list[str]) -> dict[str, str]:
    """Build key→canonical mapping with multiple variant keys per canonical name."""
    lookup = {}
    for canon in canonical_list:
        nk = _normalise_key(canon)
        lookup[nk] = canon

        # Also index each sub-part split by /
        parts = [p.strip() for p in re.split(r'[/]', canon)]
        for p in parts:
            pk = _normalise_key(p)
            if pk and len(pk) > 3:
                lookup[pk] = canon
    return lookup


# ── Keyword-based fallback rules ──────────────────────────────────────────────
# Maps keyword patterns to canonical names for items that won't match exactly

KEYWORD_RULES = {
    "values": [
        (r'dignity', "Human dignity"),
        (r'human\s*rights?|fundamental\s*rights?|civil\s*rights?', "Human rights"),
        (r'fairness|justice|equity|equality|non.?discrimination|equal', "Fairness / Justice / Equity"),
        (r'non.?discriminat', "Non-discrimination"),
        (r'privacy|data\s*protect|gdpr|personal\s*data', "Privacy / Data protection"),
        (r'autonomy|self.?determin|individual\s*autonomy', "Autonomy / Self-determination"),
        (r'well.?being|beneficen|human\s*flourish|welfare', "Well-being / Beneficence"),
        (r'non.?malefi|do\s*no\s*harm|harm\s*prevent', "Non-maleficence / Do no harm"),
        (r'freedom|liberty|libert', "Freedom / Liberty"),
        (r'solidar', "Solidarity"),
        (r'sustainab|environment|climate|green|ecological', "Sustainability / Environment"),
        (r'^peace$', "Peace"),
        (r'cultural\s*divers|divers.*cultur', "Cultural diversity"),
        (r'democra', "Democracy"),
        (r'rule\s*of\s*law', "Rule of law"),
        (r'^trust$|public\s*trust', "Trust"),
        (r'safety|security|safe\b|cyber.?secur|national\s*secur', "Safety / Security"),
        (r'public\s*interest|common\s*good|public\s*good|gemeinwohl', "Public interest / Common good"),
        # Common extras → map to nearest
        (r'innovat|competit|econom|growth|prosper|productiv', "Public interest / Common good"),
        (r'inclusi|access', "Fairness / Justice / Equity"),
        (r'transparen', "Trust"),
        (r'accountab|responsib', "Trust"),
        (r'ethic', "Human dignity"),
        (r'sovereign', "Autonomy / Self-determination"),
        (r'collaborat|cooperat', "Solidarity"),
        (r'openness|open\s*source', "Freedom / Liberty"),
        (r'education|skill|training|learn|talent|workforce', "Well-being / Beneficence"),
        (r'resilien', "Safety / Security"),
        (r'human.?centr', "Human dignity"),
        (r'integrit|honest', "Trust"),
        (r'divers', "Cultural diversity"),
        (r'efficien|quality|cost', "Public interest / Common good"),
        (r'interoperab', "Trust"),
        (r'social\s*cohes|territorial|community', "Solidarity"),
        (r'respect|involvement|empower', "Human dignity"),
        (r'legal\s*certain|good\s*govern', "Rule of law"),
        (r'worker|job|labour|labor', "Well-being / Beneficence"),
        (r'availab|adapt|standard|pragmat|competen|communicat|civic|leadership|militar', "Public interest / Common good"),
    ],
    "principles": [
        (r'transparen', "Transparency"),
        (r'explainab|interpretab', "Explainability / Interpretability"),
        (r'accountab', "Accountability"),
        (r'responsib', "Responsibility"),
        (r'robust|reliab', "Robustness / Reliability"),
        (r'human\s*oversight|human.?in.?the.?loop|human\s*control', "Human oversight / Human-in-the-loop"),
        (r'proportional', "Proportionality"),
        (r'precaution|risk.?based|risk\s*manage|risk\s*assess', "Precaution / Risk-based approach"),
        (r'inclusiv|accessib|inclusion|equity|divers', "Inclusiveness / Accessibility"),
        (r'interoperab', "Interoperability"),
        (r'contestab|right\s*to\s*appeal|grievance|redress', "Contestability / Right to appeal"),
        (r'data\s*govern|data\s*qual|data\s*protect|data\s*privacy|data\s*minimi', "Data governance / Data quality"),
        (r'open\s*source|openness', "Open source / Openness"),
        (r'international\s*cooperat|international\s*collaborat|international\s*engag', "International cooperation"),
        (r'multi.?stakeholder|stakeholder\s*govern', "Multi-stakeholder governance"),
        (r'informed\s*consent|consent\s*manage|consent', "Informed consent"),
        (r'purpose\s*limitat', "Purpose limitation"),
        (r'due\s*diligen', "Due diligence"),
        # Extras
        (r'safety|security|safe\b|cyber', "Robustness / Reliability"),
        (r'sustainab|environment|climate|energy\s*effic', "Responsibility"),
        (r'ethic', "Responsibility"),
        (r'collaborat|cooperat|partnership|public.?private', "Multi-stakeholder governance"),
        (r'innovat|flexib|adapt|agil', "Precaution / Risk-based approach"),
        (r'privacy|gdpr', "Data governance / Data quality"),
        (r'fair|bias|equit', "Inclusiveness / Accessibility"),
        (r'standard|certif', "Interoperability"),
        (r'training|capacity|education|learn', "Inclusiveness / Accessibility"),
        (r'monitor|evaluat|audit|report', "Accountability"),
        (r'human.?cent', "Human oversight / Human-in-the-loop"),
        (r'efficien|cost.?effect|accura', "Accountability"),
        (r'technolog.*neutral|neutral', "Proportionality"),
        (r'evidence.?based|evidence', "Accountability"),
        (r'trust', "Transparency"),
        (r'rule\s*of\s*law|legal|law', "Accountability"),
        (r'need\s*to\s*know|least\s*privilege|authoriz|legitim|access', "Data governance / Data quality"),
        (r'cohere|consist|balanc|coordinat', "Proportionality"),
        (r'user.?centr|user\s*protect|voluntar', "Human oversight / Human-in-the-loop"),
        (r'sovereign', "Responsibility"),
        (r'regulator|modernis|complian', "Accountability"),
        (r'knowledge|skill|learning|long.?term', "Inclusiveness / Accessibility"),
        (r'humanism', "Responsibility"),
        (r'reasonable', "Proportionality"),
        (r'sharing', "Open source / Openness"),
        (r'continu', "Accountability"),
    ],
    "mechanisms": [
        (r'ethics\s*board|ethics\s*committee|advisory\s*board|advisory\s*committee|expert\s*group|council|steering|task\s*force|working\s*group|think\s*tank|commission', "Ethics board / Ethics committee"),
        (r'impact\s*assess|dpia|hria|aiia', "Impact assessment"),
        (r'algorithm.*audit|third.?party\s*audit|independent\s*audit|compliance\s*audit|bias\s*audit|audit', "Algorithmic auditing / Third-party audit"),
        (r'certif|conformity\s*assess|quality\s*management', "Certification / Conformity assessment"),
        (r'sandbox|experiment|pilot|test\s*bed|testbed|proof\s*of\s*concept|demonstrat', "Regulatory sandbox"),
        (r'complaint|redress|grievan|appeal|opt.?out|right\s*to\s*be\s*forgot|right\s*of\s*access|right\s*of\s*correct', "Complaints / Redress mechanism"),
        (r'standard|interoperab|technical\s*committee', "Standards / Technical standards"),
        (r'training|capacity\s*build|education|skill|learn|workshop|hackathon|fellowship|apprentice|mentor|webinar|career|curriculum|scholarship|intern|placement|talent|upskill|course', "Training / Capacity building"),
        (r'label|disclos|transparency\s*report|transparency\s*requir|transparency\s*provision|content\s*provenance', "Labelling / Disclosure requirements"),
        (r'registr|inventor|catalogu|repository|ai\s*register', "Registration / Inventory of AI systems"),
        (r'risk\s*classif|tiered\s*regulat|risk\s*categor', "Risk classification / Tiered regulation"),
        (r'sector.*regulat|regulat.*sector|legislation|legislative|regulat.*framework|regulatory\s*moderniz|regulatory\s*reform|regulat.*guidance|systemic\s*regulat', "Sectoral regulation"),
        (r'procurement|public\s*tender|public\s*bidding|vendor.?neutral', "Procurement requirements"),
        (r'monitor.*evaluat|report.*requir|reporting\b|mandatory\s*report|performance\s*indicator', "Monitoring & evaluation / Reporting"),
        (r'penalt|sanction|enforcement|fine|liable|liability', "Penalties / Sanctions / Enforcement"),
        (r'insurance|liability\s*framework|duty\s*of\s*care|compensation', "Insurance / Liability framework"),
        (r'code\s*of\s*conduct|voluntary\s*commit|voluntary\s*guidance|voluntary\s*standard|self.?assess|guidance|guideline', "Code of conduct / Voluntary commitments"),
        (r'moratori|ban\b|prohibit', "Moratorium / Ban on specific uses"),
        (r'data\s*shar|open\s*data|data\s*exchange|data\s*access|data\s*portab|data\s*trust|open\s*access|data\s*governance\b|data\s*integrat|data\s*locali|data\s*sovereign', "Data sharing / Open data"),
        (r'whistleblow', "Whistleblower protection"),
        # Extras → nearest
        (r'fund|grant|subsid|financ|invest|loan|equity|voucher|co.?fund|tax\s*(credit|incent)|budget|award|recognition|incentive|state\s*aid', "Training / Capacity building"),
        (r'collaborat|partner|cooperat|network|consort|hub|platform|forum|dialogue|roundtable|consultation|engagement|co.?creat|conference|event|public\s*participat|public\s*comment|crowdsourc', "Multi-stakeholder governance"),
        (r'research|r\s*&\s*d|innovat|accelerat|incubat|ecosystem|startup', "Regulatory sandbox"),
        (r'governance\s*struct|governance\s*board|coordination|oversight|national\s*ai\s*office|interagency|cross.?sector', "Ethics board / Ethics committee"),
        (r'cyber|encrypt|anonymi|privacy\s*setting|data\s*breach|data\s*protect.*officer|data\s*retention|data\s*classif|data\s*manage|pseudonym', "Data sharing / Open data"),
        (r'digital\s*infra|infrastructure|data\s*center|computing', "Standards / Technical standards"),
        (r'open\s*source|open\s*call', "Data sharing / Open data"),
        (r'procurement\s*process|public\s*procurement', "Procurement requirements"),
        (r'multi.?stakeholder|stakeholder', "Ethics board / Ethics committee"),
        (r'community\s*of\s*expert', "Ethics board / Ethics committee"),
        (r'ai\s*incident|hazard\s*monitor', "Monitoring & evaluation / Reporting"),
        (r'consent\s*manage|consent\s*mechanism|user\s*consent', "Complaints / Redress mechanism"),
        (r'privacy|data\s*protect', "Data sharing / Open data"),
        (r'transparen', "Labelling / Disclosure requirements"),
        (r'peer\s*review', "Algorithmic auditing / Third-party audit"),
        (r'roadmap|action\s*plan|strategic\s*plan|strateg', "Code of conduct / Voluntary commitments"),
        (r'document|record|publication|report', "Monitoring & evaluation / Reporting"),
        (r'recommend|guidance|guideline|practical\s*guide|code.*practice|support\s*material', "Code of conduct / Voluntary commitments"),
        (r'public\s*aware|awareness\s*campaign|public\s*engag|public\s*discuss|public\s*debat|public\s*listen', "Training / Capacity building"),
        (r'national\s*board|national\s*forum|select\s*commit|national\s*ai\s*market|marketplace|national\s*center|centre|center\s*of\s*excell', "Ethics board / Ethics committee"),
        (r'feedback|complaint|redress|opt.?out|right\s*to|right\s*of', "Complaints / Redress mechanism"),
        (r'expert\s*hear|round\s*table|roundtable|jury|board\s*of\s*direct|committee|commission|scientific\s*council', "Ethics board / Ethics committee"),
        (r'safe\s*harbor|no.?action|exemption|exception|experimentation\s*clause', "Regulatory sandbox"),
        (r'statutory\s*review|legislat|regulatory|regulation|executive\s*order', "Sectoral regulation"),
        (r'due\s*diligen', "Algorithmic auditing / Third-party audit"),
        (r'risk\s*manage|risk\s*assess|risk\s*reduc|risk\s*matrix|safety\s*case', "Impact assessment"),
        (r'recover|loan|guarantee|equity|subsid|financial\s*assist|financial\s*support|budget\s*alloc', "Training / Capacity building"),
        (r'model\s*of\s*governance|governance$', "Ethics board / Ethics committee"),
    ],
}


def _match_keyword(label: str, category: str) -> str | None:
    """Try keyword regex rules to find a canonical match."""
    nk = _normalise_key(label)
    # Strip "Other: " prefix
    nk = re.sub(r'^other\s*:\s*', '', nk)
    for pattern, canonical in KEYWORD_RULES.get(category, []):
        if re.search(pattern, nk):
            return canonical
    return None


def build_mapping(entries: list[dict]) -> dict[str, dict[str, str]]:
    """Build category→{raw_label: canonical_name} mapping."""
    mapping = {}  # category → {raw → canonical}
    stats = defaultdict(Counter)  # category → {method → count}

    for cat in ("values", "principles", "mechanisms"):
        canon_list = TAXONOMY[cat]
        lookup = _build_lookup(canon_list)
        cat_map = {}

        # Collect all unique labels for this category
        all_labels = set()
        for e in entries:
            dd = e.get("depth_data", {})
            for item in dd.get(cat, []):
                name = item.get("name", "").strip()
                if name:
                    all_labels.add(name)

        for label in sorted(all_labels):
            nk = _normalise_key(label)
            # Strip "Other: " prefix for matching
            nk_clean = re.sub(r'^other\s*:\s*', '', nk)

            # 1. Exact key match
            if nk in lookup:
                cat_map[label] = lookup[nk]
                stats[cat]["exact"] += 1
                continue
            if nk_clean in lookup:
                cat_map[label] = lookup[nk_clean]
                stats[cat]["exact_clean"] += 1
                continue

            # 2. Substring match: does any canonical key appear inside the label?
            matched = False
            for key, canon in sorted(lookup.items(), key=lambda x: -len(x[0])):
                if len(key) > 4 and key in nk_clean:
                    cat_map[label] = canon
                    stats[cat]["substring"] += 1
                    matched = True
                    break
            if matched:
                continue

            # 3. Fuzzy match (SequenceMatcher)
            best_ratio = 0
            best_canon = None
            for canon in canon_list:
                ratio = SequenceMatcher(None, nk_clean, _normalise_key(canon)).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_canon = canon
            if best_ratio >= 0.55:
                cat_map[label] = best_canon
                stats[cat]["fuzzy"] += 1
                continue

            # 4. Keyword rules
            kw_match = _match_keyword(label, cat)
            if kw_match:
                cat_map[label] = kw_match
                stats[cat]["keyword"] += 1
                continue

            # 5. Unmapped → "Other"
            cat_map[label] = "Other"
            stats[cat]["other"] += 1

        mapping[cat] = cat_map

    # Print stats
    for cat in ("values", "principles", "mechanisms"):
        total = sum(stats[cat].values())
        print(f"\n{cat.upper()} mapping ({total} unique labels):")
        for method, count in sorted(stats[cat].items(), key=lambda x: -x[1]):
            print(f"  {method:15s}: {count:4d} ({count/total*100:5.1f}%)")

    return mapping


def normalise_entries(entries: list[dict], mapping: dict) -> list[dict]:
    """Apply normalisation mapping to all entries."""
    for e in entries:
        dd = e.get("depth_data", {})
        for cat in ("values", "principles", "mechanisms"):
            cat_map = mapping[cat]
            for item in dd.get(cat, []):
                raw = item.get("name", "").strip()
                if raw and raw in cat_map:
                    item["name_original"] = raw
                    item["name"] = cat_map[raw]
    return entries


def main():
    print("Loading Phase C depth data...")
    entries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"  Loaded {len(entries)} entries")

    # Build mapping
    print("\nBuilding normalisation mapping...")
    mapping = build_mapping(entries)

    # Show "Other" items
    for cat in ("values", "principles", "mechanisms"):
        others = [k for k, v in mapping[cat].items() if v == "Other"]
        if others:
            print(f"\n  {cat} UNMAPPED → 'Other' ({len(others)}):")
            for o in others:
                print(f"    - {o}")

    # Save mapping for audit
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nMapping saved: {MAP_FILE}")

    # Apply normalisation
    print("\nApplying normalisation...")
    entries = normalise_entries(entries, mapping)

    # Write normalised output
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Normalised output: {OUTPUT}")

    # Summary stats after normalisation
    print("\n" + "=" * 70)
    print("POST-NORMALISATION SUMMARY")
    print("=" * 70)
    for cat in ("values", "principles", "mechanisms"):
        counts = Counter()
        for e in entries:
            dd = e.get("depth_data", {})
            for item in dd.get(cat, []):
                counts[item["name"]] += 1
        n_canonical = len(TAXONOMY[cat])
        n_actual = len(counts)
        n_other = counts.get("Other", 0)
        print(f"\n{cat.upper()}: {n_actual} unique (canonical: {n_canonical}, Other: {n_other})")
        for name, count in counts.most_common():
            print(f"  {name:<45s} {count:>5,}")


if __name__ == "__main__":
    main()
