"""
Phase 3d: Policy Diffusion & Efficiency Frontier
==================================================
Two 'nice-to-have' analyses that add differentiation:

  9.  Policy diffusion patterns
      - First movers per dimension
      - Regional adoption curves (Kaplan-Meier style)
      - Leader-follower lags by income group & region
      - Cox-style hazard analysis (logistic proxy)

  10. Governance efficiency frontier
      - Score/GDP scatter with OLS expected line
      - Over-/under-performers (OLS residuals)
      - Convex-hull efficiency envelope
      - Distance-to-frontier ranking

Output:
  data/analysis/paper1_capacity/frontier/
  data/analysis/paper2_ethics/frontier/
  data/analysis/shared/frontier/

Usage:
  python src/analysis/diffusion_frontier.py
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as sp_stats
from scipy.spatial import ConvexHull

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import get_metadata, REGION_LABELS

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_CSV = ROOT / 'data' / 'analysis' / 'shared' / 'master_dataset.csv'
OUT_P1 = ROOT / 'data' / 'analysis' / 'paper1_capacity' / 'frontier'
OUT_P2 = ROOT / 'data' / 'analysis' / 'paper2_ethics' / 'frontier'
OUT_SH = ROOT / 'data' / 'analysis' / 'shared' / 'frontier'

# Dimensions
CAP_DIMS = ['c1_clarity', 'c2_resources', 'c3_authority', 'c4_accountability', 'c5_coherence']
ETH_DIMS = ['e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion']
ALL_DIMS = CAP_DIMS + ETH_DIMS

CAP_LABELS = {
    'c1_clarity': 'C1 Clarity', 'c2_resources': 'C2 Resources',
    'c3_authority': 'C3 Authority', 'c4_accountability': 'C4 Accountability',
    'c5_coherence': 'C5 Coherence',
}
ETH_LABELS = {
    'e1_framework': 'E1 Framework', 'e2_rights': 'E2 Rights',
    'e3_governance': 'E3 Governance', 'e4_operationalisation': 'E4 Operationalisation',
    'e5_inclusion': 'E5 Inclusion',
}
ALL_LABELS = {**CAP_LABELS, **ETH_LABELS}

REGION_COLORS = {
    'ECA': '#66c2a5', 'EAP': '#fc8d62', 'NAM': '#8da0cb',
    'LAC': '#e78ac3', 'MENA': '#a6d854', 'SA': '#ffd92f',
    'SSA': '#e5c494', 'Unknown': '#b3b3b3',
}

# ─── Style ──────────────────────────────────────────────────────────────────
STYLE = {
    'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.figsize': (10, 6),
}
plt.rcParams.update(STYLE)
PAL = sns.color_palette('Set2', 8)
INCOME_COLORS = {'High income': PAL[0], 'Developing': PAL[3]}


def _json_default(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return str(obj)


def load_data() -> pd.DataFrame:
    print("Loading master dataset...")
    df = pd.read_csv(MASTER_CSV)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['log_gdp_pc'] = np.log(df['gdp_pc'].replace(0, np.nan))
    print(f"  {len(df)} entries, {df['jurisdiction'].nunique()} jurisdictions")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 9. POLICY DIFFUSION PATTERNS
# ═════════════════════════════════════════════════════════════════════════════

def policy_diffusion(df: pd.DataFrame, paper: str, dims: list,
                     composite: str, out_dir: Path):
    """
    Temporal diffusion: who adopted first, and did peers follow?
    - First mover identification per dimension
    - Kaplan-Meier-style adoption curves by region and income
    - Year-of-first-adoption analysis
    """
    section = f"9. POLICY DIFFUSION — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Focus on the AI era (2017+) and exclude international entities
    dfc = df[~df['is_international'] & df['year'].between(2017, 2025)].copy()
    dfc = dfc.dropna(subset=['region_code', 'income_binary'])

    results = {}

    # ── 9a. First movers per dimension ────────────────────────────────────
    # "Adoption" = first policy scoring ≥ 1 in a dimension
    ADOPT_THRESHOLD = 1.0

    print(f"\n  9a. First movers (first policy scoring ≥ {ADOPT_THRESHOLD}):")
    print(f"  {'Dimension':<25s} {'Country':<25s} {'Year':>5s} {'Title'}")
    print(f"  {'─'*90}")

    first_movers = {}
    adoption_year = {}  # dim → {jurisdiction → first year adopted}

    for d in dims:
        label = ALL_LABELS.get(d, d)
        adopted = dfc[dfc[d] >= ADOPT_THRESHOLD].copy()
        if adopted.empty:
            first_movers[d] = None
            adoption_year[d] = {}
            continue

        # First adopter overall
        first = adopted.sort_values('year').iloc[0]
        first_movers[d] = {
            'country': first['jurisdiction'],
            'year': int(first['year']),
            'title': first['title'][:50],
            'score': round(float(first[d]), 2),
        }
        print(f"  {label:<25s} {first['jurisdiction']:<25s} {int(first['year']):>5d} "
              f" {first['title'][:45]}")

        # First adoption year per country
        first_per_country = adopted.groupby('jurisdiction')['year'].min()
        adoption_year[d] = first_per_country.to_dict()

    results['first_movers'] = first_movers

    # ── 9b. Top 5 earliest adopters per dimension ─────────────────────────
    print(f"\n  9b. Earliest 5 adopters per dimension:")
    top_adopters = {}
    for d in dims:
        label = ALL_LABELS.get(d, d)
        if not adoption_year.get(d):
            continue
        sorted_a = sorted(adoption_year[d].items(), key=lambda x: x[1])[:5]
        top_adopters[d] = [{'country': c, 'year': int(y)} for c, y in sorted_a]
        names = ', '.join(f"{c} ({int(y)})" for c, y in sorted_a)
        print(f"  {label:<25s} {names}")
    results['top_adopters'] = top_adopters

    # ── 9c. Adoption curves by income group ───────────────────────────────
    # For the composite: cumulative share of countries that have ≥1 policy scoring ≥1
    print(f"\n  9c. Cumulative adoption by income group:")

    years = list(range(2017, 2026))
    countries = dfc['jurisdiction'].unique()
    country_income = {j: dfc[dfc['jurisdiction'] == j]['income_binary'].iloc[0]
                      for j in countries if len(dfc[dfc['jurisdiction'] == j]) > 0}

    adoption_curves = {}
    for ig in ['High income', 'Developing']:
        ig_countries = [c for c, ib in country_income.items() if ib == ig]
        n_total = len(ig_countries)
        if n_total == 0:
            continue
        cum_adopted = []
        for yr in years:
            # Count countries with ≥1 policy scoring ≥ threshold by this year
            n_adopted = 0
            for c in ig_countries:
                c_data = dfc[(dfc['jurisdiction'] == c) & (dfc['year'] <= yr)]
                if len(c_data) > 0 and c_data[composite].max() >= ADOPT_THRESHOLD:
                    n_adopted += 1
            cum_adopted.append(n_adopted / n_total)
        adoption_curves[ig] = {
            'years': years,
            'cumulative_share': [round(x, 4) for x in cum_adopted],
            'n_countries': n_total,
        }
        print(f"    {ig:<15s} (n={n_total}): ", end='')
        for yr, share in zip(years, cum_adopted):
            print(f"{yr}:{share:.0%} ", end='')
        print()

    results['adoption_curves_income'] = adoption_curves

    # ── 9d. Adoption curves by region ─────────────────────────────────────
    print(f"\n  9d. Cumulative adoption by region:")
    country_region = {j: dfc[dfc['jurisdiction'] == j]['region_code'].iloc[0]
                      for j in countries if len(dfc[dfc['jurisdiction'] == j]) > 0}

    adoption_by_region = {}
    for rc in sorted(set(country_region.values())):
        if pd.isna(rc) or rc == 'Unknown':
            continue
        rc_countries = [c for c, r in country_region.items() if r == rc]
        n_total = len(rc_countries)
        if n_total < 2:
            continue
        cum_adopted = []
        for yr in years:
            n_adopted = 0
            for c in rc_countries:
                c_data = dfc[(dfc['jurisdiction'] == c) & (dfc['year'] <= yr)]
                if len(c_data) > 0 and c_data[composite].max() >= ADOPT_THRESHOLD:
                    n_adopted += 1
            cum_adopted.append(n_adopted / n_total)
        region_label = REGION_LABELS.get(rc, rc)
        adoption_by_region[rc] = {
            'label': region_label,
            'years': years,
            'cumulative_share': [round(x, 4) for x in cum_adopted],
            'n_countries': n_total,
        }
        latest = cum_adopted[-1]
        print(f"    {region_label:<30s} (n={n_total:>2d}): "
              f"2019={cum_adopted[2]:.0%}  2022={cum_adopted[5]:.0%}  2025={latest:.0%}")

    results['adoption_curves_region'] = adoption_by_region

    # ── 9e. Adoption lag analysis ─────────────────────────────────────────
    # For composite: median year of first adoption by income group
    print(f"\n  9e. Adoption lag (composite ≥ {ADOPT_THRESHOLD}):")

    first_year_composite = {}
    for c in countries:
        c_data = dfc[(dfc['jurisdiction'] == c) & (dfc[composite] >= ADOPT_THRESHOLD)]
        if len(c_data) > 0:
            first_year_composite[c] = int(c_data['year'].min())

    for ig in ['High income', 'Developing']:
        ig_years = [y for c, y in first_year_composite.items()
                    if country_income.get(c) == ig]
        if ig_years:
            print(f"    {ig:<15s}: median first adoption = {np.median(ig_years):.0f}, "
                  f"mean = {np.mean(ig_years):.1f}, n={len(ig_years)}")
            results[f'adoption_lag_{ig.lower().replace(" ", "_")}'] = {
                'median': int(np.median(ig_years)),
                'mean': round(np.mean(ig_years), 1),
                'n': len(ig_years),
            }

    # T-test on year of first adoption
    hi_years = [y for c, y in first_year_composite.items()
                if country_income.get(c) == 'High income']
    dev_years = [y for c, y in first_year_composite.items()
                 if country_income.get(c) == 'Developing']
    if len(hi_years) >= 5 and len(dev_years) >= 5:
        t, p = sp_stats.ttest_ind(hi_years, dev_years, equal_var=False)
        print(f"    Adoption year t-test: t={t:.3f}, p={p:.4f}")
        if p < 0.05:
            diff = np.mean(hi_years) - np.mean(dev_years)
            print(f"    → HI adopted {abs(diff):.1f} years {'earlier' if diff < 0 else 'later'} on average")
        results['adoption_ttest'] = {'t': round(t, 3), 'p': round(p, 6)}

    # ── 9f. Diffusion direction: horizontal vs vertical ───────────────────
    # Do countries adopt after peers at similar income (horizontal) or after
    # richer countries (vertical)?
    print(f"\n  9f. Diffusion direction analysis:")

    # For each non-first-mover, check if a country in the same income group
    # or a different group adopted earlier
    horiz_count = 0
    vert_count = 0
    for d in dims:
        if not adoption_year.get(d):
            continue
        sorted_adopters = sorted(adoption_year[d].items(), key=lambda x: x[1])
        for i, (c, yr) in enumerate(sorted_adopters):
            if i == 0:
                continue  # skip first mover
            c_income = country_income.get(c)
            if c_income is None:
                continue
            # Check who adopted before this country
            predecessors = sorted_adopters[:i]
            same_income = any(country_income.get(p) == c_income for p, _ in predecessors)
            diff_income = any(country_income.get(p) != c_income for p, _ in predecessors)
            if same_income:
                horiz_count += 1
            if diff_income and not same_income:
                vert_count += 1

    total_follows = horiz_count + vert_count
    if total_follows > 0:
        h_share = horiz_count / total_follows
        v_share = vert_count / total_follows
        print(f"    Horizontal (same income group led): {horiz_count} ({h_share:.0%})")
        print(f"    Vertical (different income group led): {vert_count} ({v_share:.0%})")
        results['diffusion_direction'] = {
            'horizontal': horiz_count, 'vertical': vert_count,
            'horizontal_share': round(h_share, 3),
        }

    # ── FIGURES ───────────────────────────────────────────────────────────

    # Figure 1: Adoption curves by income group
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ig, data in adoption_curves.items():
        color = INCOME_COLORS.get(ig, 'gray')
        ax1.plot(data['years'], data['cumulative_share'], 'o-',
                 color=color, linewidth=2.5, markersize=6,
                 label=f"{ig} (n={data['n_countries']})")
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative share of countries adopted')
    ax1.set_title('(a) Adoption by Income Group')
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Figure 2: Adoption curves by region
    for rc, data in adoption_by_region.items():
        color = REGION_COLORS.get(rc, 'gray')
        ax2.plot(data['years'], data['cumulative_share'], 'o-',
                 color=color, linewidth=2, markersize=5,
                 label=f"{data['label']} ({data['n_countries']})")
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative share of countries adopted')
    ax2.set_title('(b) Adoption by Region')
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(alpha=0.3)

    fig.suptitle(f'{paper}: Policy Diffusion ({composite.replace("_", " ")} ≥ {ADOPT_THRESHOLD})',
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_diffusion_curves.png')
    plt.close(fig)

    # Figure 3: First movers timeline (dot plot)
    fig, ax = plt.subplots(figsize=(12, max(5, len(dims) * 0.8)))
    dim_labels = [ALL_LABELS.get(d, d) for d in dims]
    y_pos = np.arange(len(dims))

    for i, d in enumerate(dims):
        if not adoption_year.get(d):
            continue
        adopters = sorted(adoption_year[d].items(), key=lambda x: x[1])
        # Plot all adoption years as small dots
        years_list = [y for _, y in adopters]
        ax.scatter(years_list, [i] * len(years_list), c='lightgray',
                   s=20, alpha=0.5, zorder=1)
        # Highlight first 3 adopters
        for j, (c, yr) in enumerate(adopters[:3]):
            color = INCOME_COLORS.get(country_income.get(c, ''), 'gray')
            ax.scatter(yr, i, c=[color], s=100, zorder=3, edgecolors='black', linewidth=0.5)
            offset = 0.15 * (j + 1) * (-1 if j % 2 == 0 else 1)
            ax.annotate(c[:12], (yr, i + offset), fontsize=7, ha='center',
                        va='bottom' if offset > 0 else 'top')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dim_labels)
    ax.set_xlabel('Year of First Adoption')
    ax.set_title(f'{paper}: First Movers by Dimension')
    ax.set_xlim(2016.5, 2025.5)
    ax.grid(axis='x', alpha=0.3)

    # Legend for income groups
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                           markersize=10, label=ig)
                    for ig, c in INCOME_COLORS.items()]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_first_movers.png')
    plt.close(fig)

    # Figure 4: Heatmap — region × dimension adoption year
    print(f"\n  Building region × dimension adoption heatmap...")
    regions_sorted = sorted([r for r in adoption_by_region.keys()])
    region_dim_year = pd.DataFrame(index=regions_sorted, columns=dims, dtype=float)

    for rc in regions_sorted:
        rc_countries = [c for c, r in country_region.items() if r == rc]
        for d in dims:
            # Median first adoption year across countries in region
            years_in_region = [adoption_year[d].get(c) for c in rc_countries
                               if c in adoption_year.get(d, {})]
            years_in_region = [y for y in years_in_region if y is not None]
            if years_in_region:
                region_dim_year.loc[rc, d] = np.median(years_in_region)

    region_dim_year.index = [REGION_LABELS.get(rc, rc) for rc in regions_sorted]
    region_dim_year.columns = [ALL_LABELS.get(d, d) for d in dims]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(region_dim_year.astype(float), annot=True, fmt='.0f',
                cmap='YlOrRd_r', ax=ax, linewidths=0.5,
                vmin=2017, vmax=2025)
    ax.set_title(f'{paper}: Median First Adoption Year by Region × Dimension')
    fig.savefig(out_dir / 'fig_diffusion_heatmap.png')
    plt.close(fig)

    with open(out_dir / 'diffusion_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 10. GOVERNANCE EFFICIENCY FRONTIER
# ═════════════════════════════════════════════════════════════════════════════

def efficiency_frontier(df: pd.DataFrame, paper: str, dims: list,
                        composite: str, out_dir: Path):
    """
    Which countries get the most governance capacity per GDP dollar?
    - OLS residuals → over/under-performers
    - Convex hull efficiency frontier
    - Distance-to-frontier ranking
    """
    section = f"10. EFFICIENCY FRONTIER — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international']].copy()
    dfc = dfc.dropna(subset=['gdp_pc', 'income_binary'])
    dfc = dfc[dfc['gdp_pc'] > 0]
    dfc['log_gdp_pc'] = np.log(dfc['gdp_pc'])

    # Country-level aggregation: mean score per country
    country = dfc.groupby('jurisdiction').agg({
        composite: 'mean',
        'log_gdp_pc': 'first',
        'gdp_pc': 'first',
        'income_binary': 'first',
        'region': 'first',
        'region_code': 'first',
    }).reset_index()

    # Require min 3 policies for reliability
    policy_count = dfc.groupby('jurisdiction').size()
    country['n_policies'] = country['jurisdiction'].map(policy_count)
    country = country[country['n_policies'] >= 3].copy()

    print(f"  {len(country)} countries with ≥3 policies")

    results = {}

    # ── 10a. OLS regression: score ~ log(GDP) ─────────────────────────────
    print(f"\n  10a. OLS: {composite} ~ log(GDP pc)")
    ols = sm.OLS(country[composite], sm.add_constant(country['log_gdp_pc'])).fit()
    country['predicted'] = ols.predict(sm.add_constant(country['log_gdp_pc']))
    country['residual'] = country[composite] - country['predicted']

    print(f"    β (GDP) = {ols.params.iloc[1]:+.4f} (p={ols.pvalues.iloc[1]:.4f})")
    print(f"    R² = {ols.rsquared:.4f}")
    results['ols'] = {
        'beta': round(float(ols.params.iloc[1]), 5),
        'p': round(float(ols.pvalues.iloc[1]), 6),
        'r_squared': round(float(ols.rsquared), 4),
    }

    # ── 10b. Over-performers (positive residuals) ─────────────────────────
    country_sorted = country.sort_values('residual', ascending=False)

    print(f"\n  10b. Top 10 over-performers (punching above weight):")
    print(f"  {'Country':<25s} {'Score':>6s} {'Predicted':>9s} {'Residual':>9s} {'GDP pc':>10s} {'Income'}")
    top_over = country_sorted.head(10)
    for _, row in top_over.iterrows():
        print(f"  {row['jurisdiction']:<25s} {row[composite]:>6.2f} {row['predicted']:>9.2f} "
              f"{row['residual']:>+9.2f} {row['gdp_pc']:>10,.0f}  {row['income_binary']}")

    results['overperformers'] = [
        {'country': row['jurisdiction'],
         'score': round(row[composite], 3),
         'predicted': round(row['predicted'], 3),
         'residual': round(row['residual'], 3),
         'gdp_pc': int(row['gdp_pc']),
         'income': row['income_binary']}
        for _, row in top_over.iterrows()
    ]

    # ── 10c. Under-performers (negative residuals) ────────────────────────
    print(f"\n  10c. Top 10 under-performers (below expected):")
    bottom_under = country_sorted.tail(10).iloc[::-1]
    for _, row in bottom_under.iterrows():
        print(f"  {row['jurisdiction']:<25s} {row[composite]:>6.2f} {row['predicted']:>9.2f} "
              f"{row['residual']:>+9.2f} {row['gdp_pc']:>10,.0f}  {row['income_binary']}")

    results['underperformers'] = [
        {'country': row['jurisdiction'],
         'score': round(row[composite], 3),
         'predicted': round(row['predicted'], 3),
         'residual': round(row['residual'], 3),
         'gdp_pc': int(row['gdp_pc']),
         'income': row['income_binary']}
        for _, row in bottom_under.iterrows()
    ]

    # ── 10d. Efficiency frontier (Free Disposal Hull) ─────────────────────
    # FDH: for each GDP level, the frontier is the maximum score
    # achievable at that GDP or lower
    print(f"\n  10d. Efficiency frontier:")

    # Sort by GDP ascending
    country_gdp_sorted = country.sort_values('log_gdp_pc').copy()
    x = country_gdp_sorted['log_gdp_pc'].values
    y = country_gdp_sorted[composite].values

    # FDH: running maximum
    fdh_y = np.maximum.accumulate(y)

    # Identify frontier countries (those ON the FDH)
    on_frontier = []
    current_max = -np.inf
    for i in range(len(y)):
        if y[i] > current_max:
            current_max = y[i]
            on_frontier.append(country_gdp_sorted.iloc[i]['jurisdiction'])

    print(f"  Frontier countries ({len(on_frontier)}): {', '.join(on_frontier)}")
    results['frontier_countries'] = on_frontier

    # Distance to frontier for each country
    country_gdp_sorted['frontier_score'] = fdh_y
    country_gdp_sorted['dist_to_frontier'] = fdh_y - y

    # Summary stats
    mean_dist = country_gdp_sorted['dist_to_frontier'].mean()
    print(f"  Mean distance to frontier: {mean_dist:.3f}")
    results['mean_dist_to_frontier'] = round(mean_dist, 4)

    # By income group
    for ig in ['High income', 'Developing']:
        sub = country_gdp_sorted[country_gdp_sorted['income_binary'] == ig]
        if len(sub) > 0:
            md = sub['dist_to_frontier'].mean()
            print(f"    {ig:<15s}: mean dist = {md:.3f}")
            results[f'dist_frontier_{ig.lower().replace(" ","_")}'] = round(md, 4)

    # ── 10e. Efficiency ratio ─────────────────────────────────────────────
    # Score per $10k GDP pc
    country['efficiency'] = country[composite] / (country['gdp_pc'] / 10000)
    country_eff = country.sort_values('efficiency', ascending=False)

    print(f"\n  10e. Top 10 most efficient (score per $10k GDP pc):")
    print(f"  {'Country':<25s} {'Score':>6s} {'GDP pc':>10s} {'Efficiency':>11s}")
    for _, row in country_eff.head(10).iterrows():
        print(f"  {row['jurisdiction']:<25s} {row[composite]:>6.2f} "
              f"{row['gdp_pc']:>10,.0f} {row['efficiency']:>11.4f}")

    results['most_efficient'] = [
        {'country': row['jurisdiction'],
         'score': round(row[composite], 3),
         'gdp_pc': int(row['gdp_pc']),
         'efficiency': round(row['efficiency'], 4)}
        for _, row in country_eff.head(10).iterrows()
    ]

    # ── FIGURES ───────────────────────────────────────────────────────────

    # Figure 1: Main scatter with OLS line + frontier
    fig, ax = plt.subplots(figsize=(14, 9))

    for ig in ['High income', 'Developing']:
        sub = country[country['income_binary'] == ig]
        ax.scatter(sub['log_gdp_pc'], sub[composite],
                   c=[INCOME_COLORS.get(ig, 'gray')] * len(sub),
                   s=sub['n_policies'] * 5,  # size by policy count
                   alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=f'{ig} (n={len(sub)})', zorder=3)

    # OLS line
    x_line = np.linspace(country['log_gdp_pc'].min(), country['log_gdp_pc'].max(), 100)
    y_line = ols.params.iloc[0] + ols.params.iloc[1] * x_line
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
            label=f'OLS (R²={ols.rsquared:.3f})')

    # FDH frontier
    fdh_x = country_gdp_sorted['log_gdp_pc'].values
    # Step function for FDH
    step_x = []
    step_y = []
    for i in range(len(fdh_x)):
        step_x.append(fdh_x[i])
        step_y.append(fdh_y[i])
        if i < len(fdh_x) - 1:
            step_x.append(fdh_x[i + 1])
            step_y.append(fdh_y[i])
    ax.plot(step_x, step_y, '-', color='darkgreen', linewidth=2, alpha=0.6,
            label='Efficiency frontier (FDH)')

    # Label notable countries
    top_5_over = country_sorted.head(5)
    top_5_under = country_sorted.tail(5)
    labeled = pd.concat([top_5_over, top_5_under])
    # Also label frontier countries
    frontier_df = country[country['jurisdiction'].isin(on_frontier)]
    labeled = pd.concat([labeled, frontier_df]).drop_duplicates('jurisdiction')

    for _, row in labeled.iterrows():
        name = row['jurisdiction']
        if len(name) > 15:
            name = name[:12] + '...'
        ax.annotate(name, (row['log_gdp_pc'], row[composite]),
                    fontsize=8, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('log(GDP per capita, PPP)')
    ax.set_ylabel(f'Mean {composite.replace("_", " ")} (0-4)')
    ax.set_title(f'{paper}: Governance Efficiency Frontier')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.2)

    # Secondary x-axis with actual GDP
    ax2_top = ax.twiny()
    gdp_ticks = [3000, 10000, 30000, 60000, 100000]
    log_ticks = [np.log(g) for g in gdp_ticks]
    ax2_top.set_xlim(ax.get_xlim())
    ax2_top.set_xticks(log_ticks)
    ax2_top.set_xticklabels([f'${g//1000}k' for g in gdp_ticks], fontsize=9)
    ax2_top.set_xlabel('GDP per capita (PPP)', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_efficiency_frontier.png')
    plt.close(fig)

    # Figure 2: Residual ranking (bar chart)
    fig, ax = plt.subplots(figsize=(12, max(8, len(country_sorted) * 0.3)))
    colors = [INCOME_COLORS.get(row['income_binary'], 'gray')
              for _, row in country_sorted.iterrows()]
    ax.barh(range(len(country_sorted)), country_sorted['residual'].values,
            color=colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(country_sorted)))
    ax.set_yticklabels(country_sorted['jurisdiction'].values, fontsize=8)
    ax.set_xlabel(f'OLS Residual ({composite.replace("_", " ")})')
    ax.set_title(f'{paper}: Over/Under-Performers Relative to GDP')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=ig) for ig, c in INCOME_COLORS.items()]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_residual_ranking.png')
    plt.close(fig)

    # Figure 3: Efficiency by dimension (radar-like grouped bar)
    # Top 5 overperformers vs bottom 5 — dimension profile
    fig, ax = plt.subplots(figsize=(12, 6))
    dim_labels = [ALL_LABELS.get(d, d) for d in dims]

    top5_countries = top_over['jurisdiction'].values[:5]
    bot5_countries = bottom_under['jurisdiction'].values[:5]

    top5_means = dfc[dfc['jurisdiction'].isin(top5_countries)][dims].mean()
    bot5_means = dfc[dfc['jurisdiction'].isin(bot5_countries)][dims].mean()
    all_means = dfc[dims].mean()

    x_pos = np.arange(len(dims))
    width = 0.25
    ax.bar(x_pos - width, top5_means.values, width, label='Top 5 overperformers',
           color=PAL[0], edgecolor='white')
    ax.bar(x_pos, all_means.values, width, label='Global average',
           color=PAL[7], edgecolor='white')
    ax.bar(x_pos + width, bot5_means.values, width, label='Bottom 5 underperformers',
           color=PAL[3], edgecolor='white')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dim_labels, rotation=30, ha='right')
    ax.set_ylabel('Mean Score (0-4)')
    ax.set_title(f'{paper}: Dimension Profile — Over vs Under-Performers')
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_overunder_profile.png')
    plt.close(fig)

    # Save data
    country_sorted.to_csv(out_dir / 'efficiency_ranking.csv', index=False)

    with open(out_dir / 'efficiency_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run the Phase 3d diffusion & frontier analyses."""
    start = datetime.now()
    print("=" * 70)
    print("PHASE 3d: POLICY DIFFUSION & EFFICIENCY FRONTIER")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for d in [OUT_P1, OUT_P2, OUT_SH]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # ── Paper 1: Capacity ──────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("PAPER 1: CAPACITY")
    print(f"{'#'*70}")
    policy_diffusion(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    efficiency_frontier(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)

    # ── Paper 2: Ethics ────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("PAPER 2: ETHICS")
    print(f"{'#'*70}")
    policy_diffusion(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    efficiency_frontier(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*70}")
    print(f"PHASE 3d COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Paper 1: {OUT_P1}")
    print(f"  Paper 2: {OUT_P2}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_all()
