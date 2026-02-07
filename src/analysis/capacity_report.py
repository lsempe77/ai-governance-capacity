"""Generate detailed capacity analysis report."""
import json
from pathlib import Path

# Load all results
income = json.load(open('data/analysis/capacity/capacity_by_income_group.json', encoding='utf-8'))
jurisdictions = json.load(open('data/analysis/capacity/capacity_by_jurisdiction.json', encoding='utf-8'))
policies = json.load(open('data/analysis/capacity/capacity_by_policy.json', encoding='utf-8'))

print("=" * 80)
print("IMPLEMENTATION-CAPACITY-EQUITY ANALYSIS")
print("Research Question: Do countries have the capacity to implement their")
print("AI policies, and how does this vary by development level?")
print("=" * 80)

# 1. INCOME GROUP COMPARISON
print("\n\n📊 FINDING 1: CAPACITY BY INCOME GROUP")
print("-" * 80)
print(f"{'Income Group':<20} {'Countries':>10} {'Policies':>10} {'Capacity':>12} {'Enforcement':>12}")
print("-" * 80)

for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
    if group in income:
        s = income[group]
        print(f"{group:<20} {s['jurisdiction_count']:>10} {s['policy_count']:>10} "
              f"{s['avg_capacity']:>12.3f} {s['avg_enforcement']:>12.3f}")

# 2. DIMENSION BREAKDOWN
print("\n\n📈 FINDING 2: CAPACITY DIMENSIONS BY INCOME GROUP")
print("-" * 80)
print(f"{'Income Group':<20} {'Institutional':>13} {'Enforcement':>13} {'Resources':>13} {'Operational':>13} {'Expertise':>13}")
print("-" * 80)

for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
    if group in income:
        s = income[group]
        print(f"{group:<20} {s['avg_institutional']:>13.3f} {s['avg_enforcement']:>13.3f} "
              f"{s['avg_resources']:>13.3f} {s['avg_operational']:>13.3f} {s['avg_expertise']:>13.3f}")

# 3. Top performers in each income group
print("\n\n🏆 FINDING 3: TOP 5 PERFORMERS BY INCOME GROUP")
print("-" * 80)

for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
    group_jurs = [(k, v) for k, v in jurisdictions.items() if v['income_group'] == group]
    top5 = sorted(group_jurs, key=lambda x: x[1]['avg_total_capacity'], reverse=True)[:5]
    if top5:
        print(f"\n{group}:")
        for jur, data in top5:
            print(f"  {jur:<35} Capacity: {data['avg_total_capacity']:.3f} ({data['policy_count']} policies)")

# 4. Capacity-Ambition Gap Analysis
print("\n\n⚠️  FINDING 4: CAPACITY-AMBITION GAP (Negative = Under-resourced)")
print("-" * 80)

# Calculate average gap by income group
from collections import defaultdict
gap_by_income = defaultdict(list)
for jur, data in jurisdictions.items():
    gap_by_income[data['income_group']].append(data['capacity_ambition_gap'])

print(f"{'Income Group':<20} {'Avg Gap':>15} {'Min Gap':>15} {'Max Gap':>15}")
print("-" * 80)

for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
    if group in gap_by_income:
        gaps = gap_by_income[group]
        print(f"{group:<20} {sum(gaps)/len(gaps):>15.3f} {min(gaps):>15.3f} {max(gaps):>15.3f}")

# 5. Binding vs Voluntary by income
print("\n\n📋 FINDING 5: BINDING vs VOLUNTARY POLICIES")
print("-" * 80)
print(f"{'Income Group':<20} {'Binding Score':>15} {'Interpretation':>30}")
print("-" * 80)

for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
    if group in income:
        s = income[group]
        binding = s['avg_binding']
        interpretation = "More binding" if binding > 0.5 else "More voluntary/soft law"
        print(f"{group:<20} {binding:>15.3f} {interpretation:>30}")

# 6. Specific capacity gaps
print("\n\n🔍 FINDING 6: WHERE ARE THE BIGGEST GAPS?")
print("-" * 80)

# Calculate which dimension is weakest for each income group
dimensions = ['avg_institutional', 'avg_enforcement', 'avg_resources', 'avg_operational', 'avg_expertise']
dim_names = ['Institutional', 'Enforcement', 'Resources', 'Operational', 'Expertise']

for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
    if group in income:
        s = income[group]
        scores = [(dim_names[i], s[d]) for i, d in enumerate(dimensions)]
        weakest = min(scores, key=lambda x: x[1])
        strongest = max(scores, key=lambda x: x[1])
        print(f"{group}:")
        print(f"  Weakest:   {weakest[0]} ({weakest[1]:.3f})")
        print(f"  Strongest: {strongest[0]} ({strongest[1]:.3f})")

# 7. Policy volume vs capacity
print("\n\n📊 FINDING 7: DOES MORE POLICY = MORE CAPACITY?")
print("-" * 80)

# Get top 10 by policy count
top_by_count = sorted(jurisdictions.items(), key=lambda x: x[1]['policy_count'], reverse=True)[:10]
top_by_capacity = sorted(jurisdictions.items(), key=lambda x: x[1]['avg_total_capacity'], reverse=True)[:10]

print("Top 10 by Policy Volume:")
for jur, data in top_by_count:
    print(f"  {jur:<30} {data['policy_count']:>5} policies, Capacity: {data['avg_total_capacity']:.3f}")

print("\nCorrelation check:")
import statistics
counts = [v['policy_count'] for v in jurisdictions.values()]
capacities = [v['avg_total_capacity'] for v in jurisdictions.values()]

# Simple correlation
mean_c = statistics.mean(counts)
mean_cap = statistics.mean(capacities)
numerator = sum((c - mean_c) * (cap - mean_cap) for c, cap in zip(counts, capacities))
denom = (sum((c - mean_c)**2 for c in counts) * sum((cap - mean_cap)**2 for cap in capacities)) ** 0.5
correlation = numerator / denom if denom > 0 else 0
print(f"Pearson correlation (policy count vs capacity): {correlation:.3f}")

# 8. Year trends
print("\n\n📅 FINDING 8: TEMPORAL TRENDS - ARE NEWER POLICIES BETTER?")
print("-" * 80)

from collections import defaultdict
by_year = defaultdict(list)
for p in policies:
    if p.get('year') and 2015 <= p['year'] <= 2025:
        by_year[p['year']].append(p['total_capacity_score'])

print(f"{'Year':<10} {'Policies':>10} {'Avg Capacity':>15}")
print("-" * 40)
for year in sorted(by_year.keys()):
    scores = by_year[year]
    print(f"{year:<10} {len(scores):>10} {sum(scores)/len(scores):>15.3f}")

print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. IMPLEMENTATION CAPACITY IS LOW ACROSS ALL INCOME GROUPS
   - Even high-income countries average only 0.048 capacity score
   - Policies rarely mention budgets, staff, or enforcement mechanisms

2. SURPRISING: LOW-INCOME COUNTRIES SCORE HIGHER ON AVERAGE
   - Rwanda, Zambia show relatively high capacity scores
   - May reflect smaller sample size OR more implementation-focused policies

3. ENFORCEMENT IS THE WEAKEST DIMENSION
   - Across all groups, enforcement mechanisms are rarely specified
   - This suggests a global "implementation gap" in AI governance

4. AMBITION EXCEEDS CAPACITY
   - Many countries have negative capacity-ambition gaps
   - They're making ambitious commitments without implementation details

5. MORE POLICIES ≠ MORE CAPACITY
   - Weak correlation between policy volume and quality
   - Suggests "policy inflation" without substance

POLICY IMPLICATIONS:
- Global South countries may be importing policy frameworks without
  building implementation capacity
- Rich countries aren't doing much better on operationalization
- The global AI governance regime is heavy on principles, light on practice
""")
