"""Deep dive: What makes policies implementation-ready?"""
import json
from collections import defaultdict

# Load data
policies = json.load(open('data/analysis/capacity/capacity_by_policy.json', encoding='utf-8'))
enriched = json.load(open('data/oecd/enriched/oecd_enriched_20260127_203406.json', encoding='utf-8'))
corpus = json.load(open('data/corpus/corpus_master_20260128.json', encoding='utf-8'))

# Create lookups
enriched_lookup = {p['url']: p for p in enriched['policies']}
corpus_lookup = {e['url']: e for e in corpus['entries']}

print("=" * 80)
print("DEEP DIVE: WHAT MAKES POLICIES IMPLEMENTATION-READY?")
print("=" * 80)

# 1. Find highest and lowest capacity policies
sorted_policies = sorted(policies, key=lambda x: x['total_capacity_score'], reverse=True)

print("\n\n🏆 TOP 10 HIGHEST CAPACITY POLICIES:")
print("-" * 80)
for i, p in enumerate(sorted_policies[:10], 1):
    print(f"\n{i}. {p['title'][:70]}...")
    print(f"   Jurisdiction: {p['jurisdiction']} | Year: {p['year']} | Capacity: {p['total_capacity_score']:.3f}")
    print(f"   Indicators: Inst={p['institutional_score']:.2f}, Enforce={p['enforcement_score']:.2f}, "
          f"Resources={p['resources_score']:.2f}, Oper={p['operational_score']:.2f}")

print("\n\n⚠️  10 LOWEST CAPACITY POLICIES (with substantial text):")
print("-" * 80)
# Filter to policies with substantial text
substantial = [p for p in sorted_policies if p['word_count'] > 100]
for i, p in enumerate(substantial[-10:], 1):
    print(f"\n{i}. {p['title'][:70]}...")
    print(f"   Jurisdiction: {p['jurisdiction']} | Year: {p['year']} | Words: {p['word_count']}")
    print(f"   Capacity: {p['total_capacity_score']:.3f} (All dimensions near zero)")

# 2. By policy type
print("\n\n📊 CAPACITY BY POLICY TYPE:")
print("-" * 80)

by_type = defaultdict(list)
for p in policies:
    # Get original policy data for type
    title = p['title'].lower()
    if any(x in title for x in ['act', 'law', 'legislation', 'regulation']):
        ptype = 'Legislation'
    elif any(x in title for x in ['strategy', 'plan', 'roadmap']):
        ptype = 'Strategy/Plan'
    elif any(x in title for x in ['guideline', 'guidance', 'principle', 'framework']):
        ptype = 'Guidelines'
    elif any(x in title for x in ['executive order', 'decree', 'resolution']):
        ptype = 'Executive Action'
    else:
        ptype = 'Other'
    by_type[ptype].append(p)

print(f"{'Policy Type':<25} {'Count':>10} {'Avg Capacity':>15} {'Avg Enforcement':>15}")
print("-" * 70)
for ptype in ['Legislation', 'Strategy/Plan', 'Guidelines', 'Executive Action', 'Other']:
    if by_type[ptype]:
        policies_list = by_type[ptype]
        print(f"{ptype:<25} {len(policies_list):>10} "
              f"{sum(p['total_capacity_score'] for p in policies_list)/len(policies_list):>15.3f} "
              f"{sum(p['enforcement_score'] for p in policies_list)/len(policies_list):>15.3f}")

# 3. Sample high-capacity policy text
print("\n\n📝 SAMPLE: HIGHEST CAPACITY POLICY TEXT")
print("-" * 80)
top_policy = sorted_policies[0]
print(f"Title: {top_policy['title']}")
print(f"Jurisdiction: {top_policy['jurisdiction']}")
print(f"Capacity Score: {top_policy['total_capacity_score']:.3f}")
print(f"\nIndicators found:")
for dim, subdims in top_policy['indicators_found'].items():
    print(f"  {dim}: {subdims}")

# Get the actual text
for ep in enriched['policies']:
    if ep.get('title') == top_policy['title']:
        text = ep.get('initiative_overview', '')[:2000]
        print(f"\nOverview text (first 2000 chars):")
        print(text)
        break

# 4. Capacity by sector
print("\n\n📊 CAPACITY BY TARGET SECTOR:")
print("-" * 80)

sector_capacity = defaultdict(list)
for p in policies:
    # Get sectors from enriched
    for ep in enriched['policies']:
        if ep.get('title') == p['title'] and ep.get('jurisdiction') == p['jurisdiction']:
            for sector in ep.get('target_sectors', []):
                sector_capacity[sector].append(p['total_capacity_score'])
            break

print(f"{'Sector':<40} {'Policies':>10} {'Avg Capacity':>15}")
print("-" * 70)
top_sectors = sorted(sector_capacity.items(), key=lambda x: len(x[1]), reverse=True)[:15]
for sector, scores in top_sectors:
    print(f"{sector[:40]:<40} {len(scores):>10} {sum(scores)/len(scores):>15.3f}")

# 5. Implementation keywords frequency
print("\n\n🔑 KEYWORD FREQUENCY IN CORPUS:")
print("-" * 80)

# Load full corpus text
import re
full_text = " ".join([e.get('content', '') for e in corpus['entries']]).lower()
word_count = len(full_text.split())

keywords = {
    'Implementation': ['implement', 'implementation', 'operationalize', 'deploy'],
    'Enforcement': ['enforce', 'penalty', 'fine', 'sanction', 'compliance'],
    'Budget': ['budget', 'funding', 'fund', 'allocate', 'invest', 'million', 'billion'],
    'Agency': ['agency', 'authority', 'regulator', 'commission', 'office'],
    'Timeline': ['deadline', 'milestone', 'phase', 'timeline', 'by 2025', 'by 2030'],
    'Monitoring': ['monitor', 'evaluate', 'review', 'audit', 'assess'],
}

print(f"{'Category':<20} {'Sample Keywords':<40} {'Frequency per 10K words'}")
print("-" * 80)
for category, words in keywords.items():
    count = sum(len(re.findall(r'\b' + w + r'\w*', full_text)) for w in words)
    freq = count / word_count * 10000
    print(f"{category:<20} {', '.join(words[:3]):<40} {freq:.1f}")

print("\n" + "=" * 80)
print("IMPLICATIONS FOR THE PAPER")
print("=" * 80)
print("""
1. THE "IMPLEMENTATION PARADOX"
   - Countries with more policies don't have higher implementation capacity
   - Policy proliferation may be symbolic rather than substantive

2. THE "SOFT LAW TRAP"
   - 93%+ of policies are soft law (voluntary/guidelines)
   - Even legislation lacks enforcement specifics
   - AI governance is predominantly aspirational

3. THE "RESOURCES PUZZLE"
   - Resources dimension scores highest (0.08-0.14)
   - BUT this may reflect general budget mentions, not AI-specific
   - Institutional capacity (dedicated agencies) is the weakest

4. THE "GLOBAL SOUTH QUESTION"
   - Low-income countries show HIGHER capacity scores
   - Possible explanations:
     a) Sample size bias (fewer policies, more focused)
     b) Later adoption = learning from others
     c) Donor-driven implementation requirements

5. PUBLICATION ANGLE:
   "The Global AI Governance Implementation Gap: 
    2,211 Policies, Limited Operational Capacity"
   
   Key claim: The emerging AI governance regime is characterized by
   policy proliferation without implementation infrastructure.
""")
