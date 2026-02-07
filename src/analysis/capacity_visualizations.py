"""
Generate visualizations for Implementation-Capacity-Equity Analysis
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Load data
income = json.load(open('data/analysis/capacity/capacity_by_income_group.json', encoding='utf-8'))
jurisdictions = json.load(open('data/analysis/capacity/capacity_by_jurisdiction.json', encoding='utf-8'))
policies = json.load(open('data/analysis/capacity/capacity_by_policy.json', encoding='utf-8'))

output_dir = Path('data/analysis/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'High Income': '#2E86AB',
    'Upper Middle': '#A23B72',
    'Lower Middle': '#F18F01',
    'Low Income': '#C73E1D',
    'Unclassified': '#808080'
}

# =============================================================================
# FIGURE 1: Capacity by Income Group - Radar Chart
# =============================================================================
def fig1_radar():
    categories = ['Institutional', 'Enforcement', 'Resources', 'Operational', 'Expertise']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    groups = ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']
    for group in groups:
        if group in income:
            s = income[group]
            values = [s['avg_institutional'], s['avg_enforcement'], s['avg_resources'],
                     s['avg_operational'], s['avg_expertise']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=group, color=COLORS[group])
            ax.fill(angles, values, alpha=0.15, color=COLORS[group])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 0.15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.set_title('AI Governance Implementation Capacity by Income Group\n', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_capacity_radar.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_capacity_radar.pdf', bbox_inches='tight')
    print("Saved Figure 1: Capacity Radar")
    plt.close()

# =============================================================================
# FIGURE 2: Capacity-Ambition Gap by Jurisdiction
# =============================================================================
def fig2_gap_scatter():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for jur, data in jurisdictions.items():
        x = data['avg_total_capacity']
        y = data['avg_binding'] + data['avg_sector_breadth'] / 10  # Ambition proxy
        size = data['policy_count'] * 3
        color = COLORS.get(data['income_group'], COLORS['Unclassified'])
        
        ax.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='white')
        
        # Label top jurisdictions
        if x > 0.08 or y > 0.6 or data['policy_count'] > 70:
            ax.annotate(jur, (x, y), fontsize=8, alpha=0.8)
    
    # Add diagonal line (capacity = ambition)
    ax.plot([0, 0.2], [0, 0.2], 'k--', alpha=0.3, label='Capacity = Ambition')
    
    # Highlight "gap zone"
    ax.fill_between([0, 0.2], [0, 0.2], [0.8, 0.8], alpha=0.1, color='red', label='Ambition > Capacity')
    
    ax.set_xlabel('Implementation Capacity Score', fontsize=12)
    ax.set_ylabel('Policy Ambition (Binding + Sector Breadth)', fontsize=12)
    ax.set_title('The Implementation Gap: Capacity vs Ambition in AI Governance', fontsize=14, fontweight='bold')
    
    # Legend for income groups
    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']]
    ax.legend(handles=handles, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_capacity_ambition_gap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_capacity_ambition_gap.pdf', bbox_inches='tight')
    print("Saved Figure 2: Capacity-Ambition Gap")
    plt.close()

# =============================================================================
# FIGURE 3: Capacity Over Time
# =============================================================================
def fig3_temporal():
    by_year_income = defaultdict(lambda: defaultdict(list))
    for p in policies:
        if p.get('year') and 2015 <= p['year'] <= 2025:
            by_year_income[p['year']][p['income_group']].append(p['total_capacity_score'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = sorted([y for y in by_year_income.keys() if y >= 2015])
    
    for group in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']:
        avg_by_year = []
        for year in years:
            scores = by_year_income[year].get(group, [])
            avg_by_year.append(np.mean(scores) if scores else 0)
        ax.plot(years, avg_by_year, 'o-', label=group, color=COLORS[group], linewidth=2, markersize=6)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Capacity Score', fontsize=12)
    ax.set_title('Evolution of AI Governance Implementation Capacity (2015-2025)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(2015, 2025)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_temporal_capacity.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_temporal_capacity.pdf', bbox_inches='tight')
    print("Saved Figure 3: Temporal Trends")
    plt.close()

# =============================================================================
# FIGURE 4: Bar Chart - Top 15 Countries
# =============================================================================
def fig4_top_countries():
    top15 = sorted(jurisdictions.items(), key=lambda x: x[1]['avg_total_capacity'], reverse=True)[:15]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    countries = [j[0] for j in top15]
    capacities = [j[1]['avg_total_capacity'] for j in top15]
    colors = [COLORS.get(j[1]['income_group'], COLORS['Unclassified']) for j in top15]
    
    bars = ax.barh(countries, capacities, color=colors, edgecolor='white')
    
    ax.set_xlabel('Implementation Capacity Score', fontsize=12)
    ax.set_title('Top 15 Jurisdictions by AI Governance Implementation Capacity', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add policy count annotations
    for i, (jur, data) in enumerate(top15):
        ax.annotate(f"n={data['policy_count']}", (data['avg_total_capacity'] + 0.005, i), va='center', fontsize=9)
    
    # Legend
    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']]
    ax.legend(handles=handles, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_top_countries.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_top_countries.pdf', bbox_inches='tight')
    print("Saved Figure 4: Top Countries")
    plt.close()

# =============================================================================
# FIGURE 5: Policy Count vs Capacity
# =============================================================================
def fig5_count_vs_capacity():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for jur, data in jurisdictions.items():
        x = data['policy_count']
        y = data['avg_total_capacity']
        color = COLORS.get(data['income_group'], COLORS['Unclassified'])
        ax.scatter(x, y, c=color, s=80, alpha=0.6, edgecolors='white')
        
        if x > 60 or y > 0.1:
            ax.annotate(jur, (x, y), fontsize=8)
    
    ax.set_xlabel('Number of AI Policies', fontsize=12)
    ax.set_ylabel('Average Implementation Capacity', fontsize=12)
    ax.set_title('Policy Volume vs Implementation Quality: More ≠ Better', fontsize=14, fontweight='bold')
    
    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']]
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_count_vs_capacity.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_count_vs_capacity.pdf', bbox_inches='tight')
    print("Saved Figure 5: Count vs Capacity")
    plt.close()

# =============================================================================
# FIGURE 6: Stacked Bar - Capacity Dimensions by Income Group
# =============================================================================
def fig6_stacked_dimensions():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']
    dimensions = ['avg_institutional', 'avg_enforcement', 'avg_resources', 'avg_operational', 'avg_expertise']
    dim_labels = ['Institutional', 'Enforcement', 'Resources', 'Operational', 'Expertise']
    dim_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    
    x = np.arange(len(groups))
    width = 0.6
    
    bottom = np.zeros(len(groups))
    for dim, label, color in zip(dimensions, dim_labels, dim_colors):
        values = [income.get(g, {}).get(dim, 0) for g in groups]
        ax.bar(x, values, width, label=label, bottom=bottom, color=color)
        bottom += values
    
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel('Cumulative Capacity Score', fontsize=12)
    ax.set_title('Capacity Dimensions by Income Group', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_stacked_dimensions.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_stacked_dimensions.pdf', bbox_inches='tight')
    print("Saved Figure 6: Stacked Dimensions")
    plt.close()

# Run all
print("\n" + "=" * 60)
print("GENERATING FIGURES")
print("=" * 60)

fig1_radar()
fig2_gap_scatter()
fig3_temporal()
fig4_top_countries()
fig5_count_vs_capacity()
fig6_stacked_dimensions()

print("\n✓ All figures saved to data/analysis/figures/")
