"""
Phase 3: SOTA Analysis Pipeline
=================================
Comprehensive analysis for TWO interconnected papers:

  Paper 1 — AI Governance CAPACITY
    "Do countries have the capacity to implement their AI policies?"
    Dimensions: C1-C5 (Clarity, Resources, Authority, Accountability, Coherence)

  Paper 2 — AI ETHICS Governance
    "How substantively do AI policies address ethical governance?"
    Dimensions: E1-E5 (Framework, Rights, Governance, Operationalisation, Inclusion)

Both papers share:
  - The same 2,216-policy corpus from OECD.AI
  - The same 3-model LLM ensemble scoring methodology
  - The capacity–ethics nexus analysis (linking the two papers)

Analyses:
  1. Descriptive statistics & distributions
  2. Income-group comparisons (t-tests, Cohen's d)
  3. Regional analysis
  4. Temporal trends
  5. Policy-type analysis
  6. Country-level rankings & profiles
  7. OLS regression (predictors of scores)
  8. K-means clustering (country typologies)
  9. Capacity–ethics nexus
  10. Publication-ready figures

Output:
  data/analysis/paper1_capacity/    – tables, figures, stats for Paper 1
  data/analysis/paper2_ethics/      – tables, figures, stats for Paper 2
  data/analysis/shared/             – cross-paper outputs

Usage:
  python src/analysis/sota_analysis.py
"""

import json
import math
import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import our country metadata
sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import (
    INCOME_GROUP, INCOME_LABELS, REGION, REGION_LABELS,
    GDP_PER_CAPITA, INTERNATIONAL, get_income_binary, get_metadata,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
ENSEMBLE_PATH = ROOT / 'data' / 'analysis' / 'scores_ensemble.json'
OUT_P1 = ROOT / 'data' / 'analysis' / 'paper1_capacity'
OUT_P2 = ROOT / 'data' / 'analysis' / 'paper2_ethics'
OUT_SHARED = ROOT / 'data' / 'analysis' / 'shared'

# Dimensions
CAP_DIMS = ['c1_clarity', 'c2_resources', 'c3_authority', 'c4_accountability', 'c5_coherence']
ETH_DIMS = ['e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion']
ALL_DIMS = CAP_DIMS + ETH_DIMS

CAP_LABELS = {
    'c1_clarity': 'C1 Clarity',
    'c2_resources': 'C2 Resources',
    'c3_authority': 'C3 Authority',
    'c4_accountability': 'C4 Accountability',
    'c5_coherence': 'C5 Coherence',
}
ETH_LABELS = {
    'e1_framework': 'E1 Framework',
    'e2_rights': 'E2 Rights',
    'e3_governance': 'E3 Governance',
    'e4_operationalisation': 'E4 Operationalisation',
    'e5_inclusion': 'E5 Inclusion',
}

# ─── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.figsize': (10, 6),
}
plt.rcParams.update(STYLE)
PAL = sns.color_palette('Set2', 8)
INCOME_COLORS = {'High income': PAL[0], 'Upper middle income': PAL[1],
                  'Lower middle income': PAL[2], 'Low income': PAL[3]}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load ensemble scores and enrich with country metadata."""
    print("Loading data...")
    with open(ENSEMBLE_PATH, 'r', encoding='utf-8') as f:
        ens = json.load(f)

    rows = []
    for e in ens['entries']:
        row = {
            'entry_id': e['entry_id'],
            'title': e['title'],
            'jurisdiction': e['jurisdiction'],
            'year': e.get('year'),
            'text_quality': e.get('text_quality', 'unknown'),
            'n_models': e['n_models'],
            'policy_type': e.get('policy_type', 'Unknown'),
            'binding_nature': e.get('binding_nature', 'Unknown'),
        }

        # Dimension scores (median)
        for d in ALL_DIMS:
            row[d] = e.get(d, {}).get('median', 0) if isinstance(e.get(d), dict) else 0

        # Composites
        row['capacity_score'] = e.get('capacity_score', 0)
        row['ethics_score'] = e.get('ethics_score', 0)
        row['overall_score'] = e.get('overall_score', 0)
        row['mean_spread'] = e.get('mean_spread', 0)

        # Country metadata
        meta = get_metadata(e['jurisdiction'])
        row['income_group'] = meta['income_label']
        row['income_binary'] = meta['income_binary']
        row['region'] = meta['region_label']
        row['region_code'] = meta['region']
        row['gdp_pc'] = meta['gdp_per_capita']
        row['is_international'] = meta['is_international']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Clean year
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Log GDP
    df['log_gdp_pc'] = np.log(df['gdp_pc'].replace(0, np.nan))

    print(f"  Loaded {len(df)} entries, {df['jurisdiction'].nunique()} jurisdictions")
    print(f"  Country-level (excl. international): {(~df['is_international']).sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(df: pd.DataFrame, paper: str, dims: list,
                      composite: str, out_dir: Path):
    """Compute and save descriptive statistics."""
    print(f"\n{'='*70}")
    print(f"1. DESCRIPTIVE STATISTICS — {paper}")
    print(f"{'='*70}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall distribution
    cols = dims + [composite]
    desc = df[cols].describe().T
    desc['median'] = df[cols].median()
    desc['skew'] = df[cols].skew()
    desc['kurtosis'] = df[cols].kurtosis()
    desc = desc[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skew', 'kurtosis']]
    desc = desc.round(3)

    desc.to_csv(out_dir / 'descriptive_stats.csv')
    print(desc.to_string())

    # Score distribution histogram
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(dims):
        ax = axes[i]
        label = CAP_LABELS.get(col) or ETH_LABELS.get(col, col)
        ax.hist(df[col], bins=np.arange(-0.25, 4.75, 0.5), color=PAL[i], edgecolor='white', alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel('Score (0–4)')
        ax.set_ylabel('Count')
        ax.set_xlim(-0.5, 4.5)

    # Composite in last subplot
    ax = axes[5]
    ax.hist(df[composite], bins=30, color=PAL[5], edgecolor='white', alpha=0.8)
    ax.set_title(f'{composite.replace("_", " ").title()} (Composite)')
    ax.set_xlabel('Score (0–4)')
    ax.set_ylabel('Count')
    fig.suptitle(f'{paper}: Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_distributions.png')
    plt.close(fig)

    # Correlation matrix
    corr = df[dims].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    labels = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in dims]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                xticklabels=labels, yticklabels=labels, mask=mask,
                vmin=-0.2, vmax=1.0, ax=ax, square=True)
    ax.set_title(f'{paper}: Dimension Correlations')
    fig.savefig(out_dir / 'fig_correlations.png')
    plt.close(fig)

    corr.to_csv(out_dir / 'correlation_matrix.csv')

    return desc


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INCOME-GROUP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def income_analysis(df: pd.DataFrame, paper: str, dims: list,
                    composite: str, out_dir: Path):
    """Compare scores by World Bank income group."""
    print(f"\n{'='*70}")
    print(f"2. INCOME-GROUP ANALYSIS — {paper}")
    print(f"{'='*70}")

    dfc = df[~df['is_international']].copy()

    # 4-group comparison
    groups = dfc.groupby('income_group')[dims + [composite]].agg(['mean', 'median', 'std', 'count'])
    groups.to_csv(out_dir / 'income_group_stats.csv')

    print("\n  Mean scores by income group:")
    for ig in ['High income', 'Upper middle income', 'Lower middle income', 'Low income']:
        sub = dfc[dfc['income_group'] == ig]
        if len(sub) > 0:
            m = sub[composite].mean()
            n = len(sub)
            print(f"    {ig:<25s}  n={n:>4d}  mean={m:.3f}")

    # Binary comparison: High income vs Developing
    hi = dfc[dfc['income_binary'] == 'High income'][composite]
    dev = dfc[dfc['income_binary'] == 'Developing'][composite]

    results = {}
    if len(hi) > 10 and len(dev) > 10:
        t_stat, p_val = sp_stats.ttest_ind(hi, dev, equal_var=False)
        cohens_d = (hi.mean() - dev.mean()) / math.sqrt((hi.std()**2 + dev.std()**2) / 2)
        u_stat, u_p = sp_stats.mannwhitneyu(hi, dev, alternative='two-sided')

        results['binary'] = {
            'high_income': {'n': len(hi), 'mean': round(hi.mean(), 4), 'std': round(hi.std(), 4)},
            'developing': {'n': len(dev), 'mean': round(dev.mean(), 4), 'std': round(dev.std(), 4)},
            'welch_t': round(t_stat, 4),
            'welch_p': round(p_val, 6),
            'cohens_d': round(cohens_d, 4),
            'mann_whitney_u': round(u_stat, 1),
            'mann_whitney_p': round(u_p, 6),
        }

        print(f"\n  Binary comparison ({composite}):")
        print(f"    High income:  mean={hi.mean():.3f} (n={len(hi)})")
        print(f"    Developing:   mean={dev.mean():.3f} (n={len(dev)})")
        print(f"    Welch's t:    {t_stat:.3f}  (p={p_val:.6f})")
        print(f"    Cohen's d:    {cohens_d:.3f}")
        print(f"    Mann-Whitney: U={u_stat:.0f}  (p={u_p:.6f})")

    # Per-dimension t-tests
    dim_tests = {}
    for d in dims:
        h = dfc[dfc['income_binary'] == 'High income'][d]
        v = dfc[dfc['income_binary'] == 'Developing'][d]
        if len(h) > 10 and len(v) > 10:
            t, p = sp_stats.ttest_ind(h, v, equal_var=False)
            cd = (h.mean() - v.mean()) / math.sqrt((h.std()**2 + v.std()**2) / 2)
            dim_tests[d] = {
                'hi_mean': round(h.mean(), 3), 'dev_mean': round(v.mean(), 3),
                'diff': round(h.mean() - v.mean(), 3),
                't': round(t, 3), 'p': round(p, 6), 'd': round(cd, 3),
            }
    results['dimension_tests'] = dim_tests

    print(f"\n  Per-dimension (High income − Developing):")
    print(f"    {'Dimension':<25s} {'HI mean':>8s} {'Dev mean':>9s} {'Diff':>6s} {'t':>7s} {'p':>9s} {'d':>6s}")
    for d, v in dim_tests.items():
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        sig = '***' if v['p'] < 0.001 else '**' if v['p'] < 0.01 else '*' if v['p'] < 0.05 else ''
        print(f"    {label:<25s} {v['hi_mean']:>8.3f} {v['dev_mean']:>9.3f} {v['diff']:>+6.3f} "
              f"{v['t']:>7.3f} {v['p']:>8.5f}{sig} {v['d']:>+6.3f}")

    with open(out_dir / 'income_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Box plot by income group
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    order = [o for o in order if o in dfc['income_group'].values]
    colors = [INCOME_COLORS.get(o, PAL[0]) for o in order]
    bp = sns.boxplot(data=dfc, x='income_group', y=composite, order=order,
                     palette=colors, showfliers=False, ax=ax)
    sns.stripplot(data=dfc, x='income_group', y=composite, order=order,
                  color='black', alpha=0.08, size=2, ax=ax)
    ax.set_xlabel('World Bank Income Group')
    ax.set_ylabel(f'{composite.replace("_", " ").title()} (0–4)')
    ax.set_title(f'{paper}: Scores by Income Group')
    fig.savefig(out_dir / 'fig_income_boxplot.png')
    plt.close(fig)

    # Violin plot by binary
    fig, axes = plt.subplots(1, len(dims), figsize=(3 * len(dims), 5))
    for i, d in enumerate(dims):
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        sns.violinplot(data=dfc, x='income_binary', y=d,
                       palette=[PAL[0], PAL[2]], cut=0, ax=axes[i])
        axes[i].set_title(label, fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score' if i == 0 else '')
        axes[i].set_ylim(-0.3, 4.3)
    fig.suptitle(f'{paper}: Dimension Scores by Income Level', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_income_violins.png')
    plt.close(fig)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGIONAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def regional_analysis(df: pd.DataFrame, paper: str, dims: list,
                      composite: str, out_dir: Path):
    """Compare scores by World Bank region."""
    print(f"\n{'='*70}")
    print(f"3. REGIONAL ANALYSIS — {paper}")
    print(f"{'='*70}")

    dfc = df[~df['is_international'] & df['region'].notna()].copy()

    # Summary by region
    reg_stats = dfc.groupby('region').agg(
        n_policies=(composite, 'count'),
        n_countries=('jurisdiction', 'nunique'),
        mean=(composite, 'mean'),
        median=(composite, 'median'),
        std=(composite, 'std'),
    ).round(3)
    reg_stats = reg_stats.sort_values('mean', ascending=False)
    reg_stats.to_csv(out_dir / 'regional_stats.csv')

    print(f"\n  {'Region':<30s} {'N':>5s} {'Countries':>10s} {'Mean':>6s} {'Median':>7s}")
    for r, row in reg_stats.iterrows():
        print(f"  {r:<30s} {int(row['n_policies']):>5d} {int(row['n_countries']):>10d} "
              f"{row['mean']:>6.3f} {row['median']:>7.3f}")

    # ANOVA
    groups = [g[composite].values for _, g in dfc.groupby('region') if len(g) >= 5]
    if len(groups) >= 3:
        f_stat, f_p = sp_stats.f_oneway(*groups)
        # Kruskal-Wallis (non-parametric)
        h_stat, h_p = sp_stats.kruskal(*groups)
        print(f"\n  ANOVA: F={f_stat:.3f}, p={f_p:.6f}")
        print(f"  Kruskal-Wallis: H={h_stat:.3f}, p={h_p:.6f}")

    # Heatmap: region × dimension
    heat_data = dfc.groupby('region')[dims].mean()
    heat_data.columns = [CAP_LABELS.get(c) or ETH_LABELS.get(c, c) for c in heat_data.columns]
    heat_data = heat_data.sort_values(heat_data.columns[0], ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=2.5, ax=ax, linewidths=0.5)
    ax.set_title(f'{paper}: Mean Scores by Region × Dimension')
    fig.savefig(out_dir / 'fig_region_heatmap.png')
    plt.close(fig)

    return reg_stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_analysis(df: pd.DataFrame, paper: str, dims: list,
                      composite: str, out_dir: Path):
    """Analyse score trends over time."""
    print(f"\n{'='*70}")
    print(f"4. TEMPORAL TRENDS — {paper}")
    print(f"{'='*70}")

    dfc = df[df['year'].between(2017, 2025)].copy()

    # Mean by year
    yearly = dfc.groupby('year').agg(
        n=(composite, 'count'),
        mean=(composite, 'mean'),
        median=(composite, 'median'),
        std=(composite, 'std'),
    ).round(3)
    yearly.to_csv(out_dir / 'temporal_stats.csv')

    print(f"\n  {'Year':>6s} {'N':>5s} {'Mean':>7s} {'Median':>8s}")
    for y, row in yearly.iterrows():
        print(f"  {int(y):>6d} {int(row['n']):>5d} {row['mean']:>7.3f} {row['median']:>8.3f}")

    # Linear trend
    x = dfc['year'].values.astype(float)
    y = dfc[composite].values
    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    print(f"\n  Linear trend: slope={slope:.4f}/year (r²={r**2:.4f}, p={p:.6f})")

    # Trend plot
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_mean = dfc.groupby('year')[composite].mean()
    yearly_ci = dfc.groupby('year')[composite].agg(['mean', 'std', 'count'])
    yearly_ci['se'] = yearly_ci['std'] / np.sqrt(yearly_ci['count'])
    yearly_ci['ci95'] = 1.96 * yearly_ci['se']

    ax.fill_between(yearly_ci.index,
                     yearly_ci['mean'] - yearly_ci['ci95'],
                     yearly_ci['mean'] + yearly_ci['ci95'],
                     alpha=0.2, color=PAL[0])
    ax.plot(yearly_ci.index, yearly_ci['mean'], 'o-', color=PAL[0],
            markersize=8, linewidth=2, label='Mean ± 95% CI')

    # Trend line
    x_line = np.linspace(2017, 2025, 100)
    ax.plot(x_line, slope * x_line + intercept, '--', color='gray',
            label=f'Trend: {slope:+.3f}/yr (p={p:.3f})')

    ax.set_xlabel('Year')
    ax.set_ylabel(f'{composite.replace("_", " ").title()} (0–4)')
    ax.set_title(f'{paper}: Temporal Trend')
    ax.legend()
    ax.set_xlim(2016.5, 2025.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(out_dir / 'fig_temporal_trend.png')
    plt.close(fig)

    # Dimension trends
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, d in enumerate(dims):
        ym = dfc.groupby('year')[d].mean()
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        ax.plot(ym.index, ym.values, 'o-', color=PAL[i], label=label, markersize=5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Score (0–4)')
    ax.set_title(f'{paper}: Dimension Trends Over Time')
    ax.legend(loc='upper left', fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(out_dir / 'fig_temporal_dimensions.png')
    plt.close(fig)

    # By income and time
    dfc2 = dfc[~dfc['is_international'] & dfc['income_binary'].notna()]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [('High income', PAL[0]), ('Developing', PAL[2])]:
        sub = dfc2[dfc2['income_binary'] == label]
        ym = sub.groupby('year')[composite].mean()
        ax.plot(ym.index, ym.values, 'o-', color=color, label=label, markersize=6, linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{composite.replace("_", " ").title()} (0–4)')
    ax.set_title(f'{paper}: Temporal Trend by Income Level')
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(out_dir / 'fig_temporal_income.png')
    plt.close(fig)

    return {
        'slope': round(slope, 5),
        'intercept': round(intercept, 4),
        'r_squared': round(r**2, 4),
        'p_value': round(p, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POLICY TYPE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def policy_type_analysis(df: pd.DataFrame, paper: str, dims: list,
                         composite: str, out_dir: Path):
    """Analyse scores by policy type and binding nature."""
    print(f"\n{'='*70}")
    print(f"5. POLICY TYPE ANALYSIS — {paper}")
    print(f"{'='*70}")

    # By policy type
    pt = df.groupby('policy_type').agg(
        n=(composite, 'count'),
        mean=(composite, 'mean'),
        median=(composite, 'median'),
    ).round(3).sort_values('mean', ascending=False)
    pt = pt[pt['n'] >= 10]
    pt.to_csv(out_dir / 'policy_type_stats.csv')

    print(f"\n  Policy Type (n≥10):")
    print(f"  {'Type':<25s} {'N':>5s} {'Mean':>7s} {'Median':>8s}")
    for t, row in pt.iterrows():
        print(f"  {t:<25s} {int(row['n']):>5d} {row['mean']:>7.3f} {row['median']:>8.3f}")

    # By binding nature
    bn = df.groupby('binding_nature').agg(
        n=(composite, 'count'),
        mean=(composite, 'mean'),
        median=(composite, 'median'),
    ).round(3).sort_values('mean', ascending=False)
    bn.to_csv(out_dir / 'binding_nature_stats.csv')

    print(f"\n  Binding Nature:")
    print(f"  {'Nature':<25s} {'N':>5s} {'Mean':>7s} {'Median':>8s}")
    for t, row in bn.iterrows():
        print(f"  {t:<25s} {int(row['n']):>5d} {row['mean']:>7.3f} {row['median']:>8.3f}")

    # Bar chart — policy type
    fig, ax = plt.subplots(figsize=(10, 6))
    pt_plot = pt.head(8)
    bars = ax.barh(range(len(pt_plot)), pt_plot['mean'], color=PAL[0], edgecolor='white')
    ax.set_yticks(range(len(pt_plot)))
    ax.set_yticklabels(pt_plot.index)
    ax.set_xlabel(f'Mean {composite.replace("_", " ").title()} (0–4)')
    ax.set_title(f'{paper}: Scores by Policy Type')
    for i, (_, row) in enumerate(pt_plot.iterrows()):
        ax.text(row['mean'] + 0.02, i, f"n={int(row['n'])}", va='center', fontsize=9)
    ax.invert_yaxis()
    fig.savefig(out_dir / 'fig_policy_type.png')
    plt.close(fig)

    return pt


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COUNTRY RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════

def country_rankings(df: pd.DataFrame, paper: str, dims: list,
                     composite: str, out_dir: Path):
    """Country-level aggregation and rankings."""
    print(f"\n{'='*70}")
    print(f"6. COUNTRY RANKINGS — {paper}")
    print(f"{'='*70}")

    dfc = df[~df['is_international']].copy()

    # Country-level means
    agg = {composite: 'mean'}
    for d in dims:
        agg[d] = 'mean'
    agg['entry_id'] = 'count'

    country = dfc.groupby('jurisdiction').agg(agg).round(3)
    country = country.rename(columns={'entry_id': 'n_policies'})
    country = country.sort_values(composite, ascending=False)

    # Add metadata
    country['income_group'] = country.index.map(lambda j: get_metadata(j)['income_label'])
    country['region'] = country.index.map(lambda j: get_metadata(j)['region_label'])
    country['gdp_pc'] = country.index.map(lambda j: get_metadata(j)['gdp_per_capita'])

    country.to_csv(out_dir / 'country_rankings.csv')

    # Top 20
    print(f"\n  Top 20 countries (mean {composite}, min 5 policies):")
    top = country[country['n_policies'] >= 5].head(20)
    print(f"  {'#':>3s}  {'Country':<30s} {'Score':>6s} {'N':>4s}  {'Income':<20s}")
    for i, (j, row) in enumerate(top.iterrows()):
        print(f"  {i+1:>3d}  {j:<30s} {row[composite]:>6.3f} {int(row['n_policies']):>4d}  {row['income_group']:<20s}")

    # Bottom 10
    print(f"\n  Bottom 10 countries (min 5 policies):")
    bot = country[country['n_policies'] >= 5].tail(10)
    for i, (j, row) in enumerate(bot.iterrows()):
        print(f"       {j:<30s} {row[composite]:>6.3f} {int(row['n_policies']):>4d}  {row['income_group']:<20s}")

    return country


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def regression_analysis(df: pd.DataFrame, paper: str, dims: list,
                        composite: str, out_dir: Path):
    """OLS regression: what predicts scores?"""
    print(f"\n{'='*70}")
    print(f"7. REGRESSION ANALYSIS — {paper}")
    print(f"{'='*70}")

    dfc = df[~df['is_international'] & df['gdp_pc'].notna() & df['year'].between(2017, 2025)].copy()

    # Create dummy variables
    dfc['year_centered'] = dfc['year'] - 2021  # center on 2021
    dfc['is_high_income'] = (dfc['income_binary'] == 'High income').astype(int)
    dfc['is_binding'] = dfc['binding_nature'].isin(['Binding regulation', 'Hard law']).astype(int)
    dfc['is_good_text'] = (dfc['text_quality'] == 'good').astype(int)

    results = {}

    # Model 1: Bivariate — GDP
    mask1 = dfc['log_gdp_pc'].notna()
    x = dfc.loc[mask1, 'log_gdp_pc'].values
    y = dfc.loc[mask1, composite].values
    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    results['model1_gdp'] = {
        'slope': round(slope, 4), 'intercept': round(intercept, 4),
        'r_squared': round(r**2, 4), 'p': round(p, 6), 'n': len(x),
    }
    print(f"\n  Model 1 (bivariate): log(GDP_pc) → {composite}")
    print(f"    β={slope:.4f}, R²={r**2:.4f}, p={p:.6f}, n={len(x)}")

    # GDP scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    dfc_plot = dfc[mask1]
    for ig, color in INCOME_COLORS.items():
        sub = dfc_plot[dfc_plot['income_group'] == ig]
        if len(sub) > 0:
            ax.scatter(sub['log_gdp_pc'], sub[composite], c=[color], label=ig,
                       alpha=0.3, s=20, edgecolors='none')

    x_line = np.linspace(dfc_plot['log_gdp_pc'].min(), dfc_plot['log_gdp_pc'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, '--', color='black',
            label=f'β={slope:.3f} (p={p:.4f})')
    ax.set_xlabel('log(GDP per capita, PPP)')
    ax.set_ylabel(f'{composite.replace("_", " ").title()} (0–4)')
    ax.set_title(f'{paper}: Score vs Economic Development')
    ax.legend(fontsize=9)
    fig.savefig(out_dir / 'fig_gdp_scatter.png')
    plt.close(fig)

    # Model 2: Multiple regression (manual OLS since no statsmodels dependency)
    X_cols = ['log_gdp_pc', 'year_centered', 'is_binding', 'is_good_text']
    mask2 = dfc[X_cols + [composite]].notna().all(axis=1)
    dfc2 = dfc[mask2].copy()

    if len(dfc2) >= 50:
        X = dfc2[X_cols].values
        y = dfc2[composite].values
        n, k = X.shape

        # Add intercept
        X_full = np.column_stack([np.ones(n), X])

        try:
            # OLS: β = (X'X)^(-1) X'y
            XtX_inv = np.linalg.inv(X_full.T @ X_full)
            beta = XtX_inv @ (X_full.T @ y)
            y_hat = X_full @ beta
            residuals = y - y_hat
            sse = np.sum(residuals**2)
            sst = np.sum((y - y.mean())**2)
            r_sq = 1 - sse / sst
            adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k - 1)

            # Standard errors
            mse = sse / (n - k - 1)
            se_beta = np.sqrt(np.diag(XtX_inv) * mse)
            t_stats = beta / se_beta
            p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), df=n - k - 1))

            var_names = ['(Intercept)'] + X_cols
            print(f"\n  Model 2 (OLS multiple): {composite} ~ " + ' + '.join(X_cols))
            print(f"    R²={r_sq:.4f}, Adj R²={adj_r_sq:.4f}, n={n}")
            print(f"    {'Variable':<20s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
            print(f"    {'─'*58}")

            model2 = {'r_squared': round(r_sq, 4), 'adj_r_squared': round(adj_r_sq, 4), 'n': n}
            model2['coefficients'] = {}
            for i, v in enumerate(var_names):
                sig = '***' if p_vals[i] < 0.001 else '**' if p_vals[i] < 0.01 else '*' if p_vals[i] < 0.05 else ''
                print(f"    {v:<20s} {beta[i]:>+8.4f} {se_beta[i]:>8.4f} {t_stats[i]:>8.3f} {p_vals[i]:>9.5f} {sig}")
                model2['coefficients'][v] = {
                    'beta': round(float(beta[i]), 5),
                    'se': round(float(se_beta[i]), 5),
                    't': round(float(t_stats[i]), 4),
                    'p': round(float(p_vals[i]), 6),
                }

            results['model2_ols'] = model2

        except np.linalg.LinAlgError:
            print("    OLS failed (singular matrix)")

    with open(out_dir / 'regression_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLUSTERING / TYPOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def clustering_analysis(df: pd.DataFrame, paper: str, dims: list,
                        composite: str, out_dir: Path):
    """K-means clustering on country-level mean scores to identify typologies."""
    print(f"\n{'='*70}")
    print(f"8. CLUSTERING / TYPOLOGY — {paper}")
    print(f"{'='*70}")

    dfc = df[~df['is_international']].copy()

    # Country-level means (min 3 policies)
    country = dfc.groupby('jurisdiction')[dims].mean()
    country = country[dfc.groupby('jurisdiction').size() >= 3]

    if len(country) < 10:
        print("  Not enough countries for clustering")
        return {}

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(country.values)

    # Elbow method
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    # Use 4 clusters (interpretable typology)
    n_clusters = 4
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    country['cluster'] = km.fit_predict(X)

    # Cluster profiles
    print(f"\n  Cluster profiles ({n_clusters} clusters, {len(country)} countries):")
    profiles = {}
    for c in range(n_clusters):
        members = country[country['cluster'] == c]
        profile = members[dims].mean()
        countries = sorted(members.index.tolist())

        # Income distribution
        incomes = [get_metadata(j)['income_label'] for j in countries]
        income_dist = Counter(incomes)

        profiles[f'cluster_{c}'] = {
            'n_countries': len(members),
            'mean_scores': {d: round(v, 3) for d, v in profile.items()},
            'composite_mean': round(members[dims].mean(axis=1).mean(), 3),
            'income_distribution': dict(income_dist),
            'countries': countries,
        }

        labels = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in dims]
        print(f"\n  Cluster {c} ({len(members)} countries): "
              f"composite={members[dims].mean(axis=1).mean():.2f}")
        for d, v in profile.items():
            label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
            print(f"    {label:<20s} {v:.2f}")
        print(f"    Income: {dict(income_dist)}")
        print(f"    Members: {', '.join(countries[:8])}{'...' if len(countries) > 8 else ''}")

    # Radar chart for cluster profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    dim_labels = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in dims]

    for c in range(min(n_clusters, 4)):
        ax = axes[c // 2][c % 2]
        members = country[country['cluster'] == c]
        values = members[dims].mean().tolist()
        values += values[:1]

        ax.fill(angles, values, alpha=0.25, color=PAL[c])
        ax.plot(angles, values, 'o-', color=PAL[c], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, fontsize=8)
        ax.set_ylim(0, 4)
        ax.set_title(f'Cluster {c} (n={len(members)})', fontsize=12, pad=20)

    fig.suptitle(f'{paper}: Country Typologies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_cluster_radar.png')
    plt.close(fig)

    # Save
    country.to_csv(out_dir / 'country_clusters.csv')
    with open(out_dir / 'cluster_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CAPACITY–ETHICS NEXUS (shared across both papers)
# ═══════════════════════════════════════════════════════════════════════════════

def capacity_ethics_nexus(df: pd.DataFrame, out_dir: Path):
    """Analyse the relationship between capacity and ethics scores."""
    print(f"\n{'='*70}")
    print(f"9. CAPACITY–ETHICS NEXUS")
    print(f"{'='*70}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dfc = df[~df['is_international']].copy()

    cap = dfc['capacity_score'].values
    eth = dfc['ethics_score'].values

    # Correlation
    r, p = sp_stats.pearsonr(cap, eth)
    rho, rho_p = sp_stats.spearmanr(cap, eth)
    print(f"\n  Pearson r:   {r:.4f} (p={p:.2e})")
    print(f"  Spearman ρ:  {rho:.4f} (p={rho_p:.2e})")

    # Gap analysis
    dfc['cap_eth_gap'] = dfc['capacity_score'] - dfc['ethics_score']
    gap_stats = {
        'mean_gap': round(dfc['cap_eth_gap'].mean(), 4),
        'median_gap': round(dfc['cap_eth_gap'].median(), 4),
        'pct_capacity_higher': round((dfc['cap_eth_gap'] > 0).mean() * 100, 1),
        'pct_ethics_higher': round((dfc['cap_eth_gap'] < 0).mean() * 100, 1),
        'pct_equal': round((dfc['cap_eth_gap'] == 0).mean() * 100, 1),
    }
    print(f"\n  Capacity–Ethics gap:")
    print(f"    Mean gap:             {gap_stats['mean_gap']:+.4f}")
    print(f"    Capacity > Ethics:    {gap_stats['pct_capacity_higher']:.1f}%")
    print(f"    Ethics > Capacity:    {gap_stats['pct_ethics_higher']:.1f}%")
    print(f"    Equal:                {gap_stats['pct_equal']:.1f}%")

    # By income
    for ig in ['High income', 'Developing']:
        sub = dfc[dfc['income_binary'] == ig]
        if len(sub) > 0:
            g = sub['cap_eth_gap'].mean()
            print(f"    Gap ({ig}): {g:+.3f}")

    # T-test: does gap differ by income?
    hi_gap = dfc[dfc['income_binary'] == 'High income']['cap_eth_gap']
    dev_gap = dfc[dfc['income_binary'] == 'Developing']['cap_eth_gap']
    if len(hi_gap) > 10 and len(dev_gap) > 10:
        t, p_gap = sp_stats.ttest_ind(hi_gap, dev_gap, equal_var=False)
        print(f"    Gap diff t-test: t={t:.3f}, p={p_gap:.6f}")

    results = {
        'pearson_r': round(r, 4),
        'pearson_p': round(p, 8),
        'spearman_rho': round(rho, 4),
        'spearman_p': round(rho_p, 8),
        'gap_analysis': gap_stats,
    }

    # Scatter: capacity vs ethics
    fig, ax = plt.subplots(figsize=(10, 9))
    for ig, color in INCOME_COLORS.items():
        sub = dfc[dfc['income_group'] == ig]
        if len(sub) > 0:
            ax.scatter(sub['capacity_score'], sub['ethics_score'],
                       c=[color], label=ig, alpha=0.3, s=20)

    # Diagonal
    ax.plot([0, 4], [0, 4], '--', color='gray', alpha=0.5, label='Capacity = Ethics')

    # Regression line
    slope, intercept, _, _, _ = sp_stats.linregress(cap, eth)
    x_line = np.linspace(0, 3.5, 100)
    ax.plot(x_line, slope * x_line + intercept, '-', color='black', linewidth=2,
            label=f'r={r:.3f}')

    ax.set_xlabel('Capacity Score (0–4)')
    ax.set_ylabel('Ethics Score (0–4)')
    ax.set_title('Capacity–Ethics Nexus: AI Policy Scores')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.1, 3.5)
    ax.set_ylim(-0.1, 3.5)
    ax.set_aspect('equal')
    fig.savefig(out_dir / 'fig_capacity_ethics_scatter.png')
    plt.close(fig)

    # Gap by country (country-level)
    country = dfc.groupby('jurisdiction').agg(
        cap=('capacity_score', 'mean'),
        eth=('ethics_score', 'mean'),
        n=('entry_id', 'count'),
    )
    country['gap'] = country['cap'] - country['eth']
    country = country[country['n'] >= 5].sort_values('gap', ascending=False)
    country['income'] = country.index.map(lambda j: get_metadata(j)['income_label'])
    country.to_csv(out_dir / 'capacity_ethics_gap.csv')

    # Gap bar chart (top 15 biggest gaps)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_gap = pd.concat([country.head(10), country.tail(10)]).drop_duplicates()
    top_gap = top_gap.sort_values('gap')
    colors = [PAL[0] if g > 0 else PAL[2] for g in top_gap['gap']]
    ax.barh(range(len(top_gap)), top_gap['gap'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(top_gap)))
    ax.set_yticklabels(top_gap.index)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Capacity − Ethics Gap')
    ax.set_title('Capacity–Ethics Gap by Country (min 5 policies)')
    fig.savefig(out_dir / 'fig_capacity_ethics_gap.png')
    plt.close(fig)

    # Quadrant analysis
    cap_med = dfc['capacity_score'].median()
    eth_med = dfc['ethics_score'].median()
    dfc['quadrant'] = 'Low-Low'
    dfc.loc[(dfc['capacity_score'] >= cap_med) & (dfc['ethics_score'] >= eth_med), 'quadrant'] = 'High-High'
    dfc.loc[(dfc['capacity_score'] >= cap_med) & (dfc['ethics_score'] < eth_med), 'quadrant'] = 'High Cap / Low Eth'
    dfc.loc[(dfc['capacity_score'] < cap_med) & (dfc['ethics_score'] >= eth_med), 'quadrant'] = 'Low Cap / High Eth'

    quad_dist = dfc['quadrant'].value_counts()
    results['quadrant_distribution'] = quad_dist.to_dict()
    print(f"\n  Quadrant distribution:")
    for q, n in quad_dist.items():
        print(f"    {q:<25s} {n:>5d} ({n/len(dfc)*100:.1f}%)")

    with open(out_dir / 'nexus_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY DASHBOARD FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def summary_figure(df: pd.DataFrame, out_dir: Path):
    """Create a 2×2 summary figure for each paper."""
    print(f"\n{'='*70}")
    print("10. SUMMARY FIGURES")
    print(f"{'='*70}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dfc = df[~df['is_international']].copy()

    for paper, dims, composite, colors in [
        ('Paper 1: AI Governance Capacity', CAP_DIMS, 'capacity_score', PAL[:5]),
        ('Paper 2: AI Ethics Governance', ETH_DIMS, 'ethics_score', PAL[3:8]),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # (a) Overall distribution
        ax = axes[0, 0]
        ax.hist(dfc[composite], bins=30, color=colors[0], edgecolor='white', alpha=0.8)
        ax.axvline(dfc[composite].mean(), color='red', linestyle='--', label=f'Mean={dfc[composite].mean():.2f}')
        ax.set_xlabel(f'{composite.replace("_", " ").title()} (0–4)')
        ax.set_ylabel('Count')
        ax.set_title('(a) Score Distribution')
        ax.legend()

        # (b) By income group
        ax = axes[0, 1]
        order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
        order = [o for o in order if o in dfc['income_group'].values]
        ic = [INCOME_COLORS.get(o, PAL[0]) for o in order]
        sns.boxplot(data=dfc, x='income_group', y=composite, order=order,
                    palette=ic, showfliers=False, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(f'Score (0–4)')
        ax.set_title('(b) By Income Group')
        ax.tick_params(axis='x', rotation=15)

        # (c) Temporal trend
        ax = axes[1, 0]
        dft = dfc[dfc['year'].between(2017, 2025)]
        for ig, color in [('High income', PAL[0]), ('Developing', PAL[2])]:
            sub = dft[dft['income_binary'] == ig]
            if len(sub) > 0:
                ym = sub.groupby('year')[composite].mean()
                ax.plot(ym.index, ym.values, 'o-', color=color, label=ig, markersize=5)
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Mean Score (0–4)')
        ax.set_title('(c) Temporal Trend by Income')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # (d) Dimension means
        ax = axes[1, 1]
        dim_labels = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in dims]
        for ig, color in [('High income', PAL[0]), ('Developing', PAL[2])]:
            sub = dfc[dfc['income_binary'] == ig]
            if len(sub) > 0:
                means = [sub[d].mean() for d in dims]
                x_pos = np.arange(len(dims))
                offset = -0.15 if ig == 'High income' else 0.15
                ax.bar(x_pos + offset, means, width=0.3, color=color, label=ig, alpha=0.8)
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dim_labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Mean Score (0–4)')
        ax.set_title('(d) Dimensions by Income')
        ax.legend(fontsize=9)

        fig.suptitle(paper, fontsize=15, fontweight='bold')
        plt.tight_layout()
        fname = 'fig_summary_capacity.png' if 'Capacity' in paper else 'fig_summary_ethics.png'
        fig.savefig(out_dir / fname)
        plt.close(fig)

    print("  Saved summary figures.")


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run the complete SOTA analysis pipeline."""
    start = datetime.now()
    print("=" * 70)
    print("PHASE 3: SOTA ANALYSIS PIPELINE")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create output directories
    for d in [OUT_P1, OUT_P2, OUT_SHARED]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # Export master CSV
    df.to_csv(OUT_SHARED / 'master_dataset.csv', index=False)
    print(f"  Master CSV saved: {OUT_SHARED / 'master_dataset.csv'}")

    # ── Run analyses for Paper 1 (Capacity) ──
    print(f"\n{'#'*70}")
    print("PAPER 1: AI GOVERNANCE CAPACITY")
    print(f"{'#'*70}")
    descriptive_stats(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    income_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    regional_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    temporal_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    policy_type_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    country_rankings(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    regression_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    clustering_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)

    # ── Run analyses for Paper 2 (Ethics) ──
    print(f"\n{'#'*70}")
    print("PAPER 2: AI ETHICS GOVERNANCE")
    print(f"{'#'*70}")
    descriptive_stats(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    income_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    regional_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    temporal_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    policy_type_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    country_rankings(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    regression_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    clustering_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)

    # ── Shared analyses ──
    print(f"\n{'#'*70}")
    print("SHARED: CAPACITY–ETHICS NEXUS")
    print(f"{'#'*70}")
    capacity_ethics_nexus(df, OUT_SHARED)
    summary_figure(df, OUT_SHARED)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*70}")
    print(f"PHASE 3 COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Paper 1 outputs: {OUT_P1}")
    print(f"  Paper 2 outputs: {OUT_P2}")
    print(f"  Shared outputs:  {OUT_SHARED}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_all()
