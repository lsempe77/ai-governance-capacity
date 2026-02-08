"""
Phase 3c: Extended Analysis Pipeline
======================================
Four medium-priority analyses strengthening the contribution:

  5. Inequality decomposition
     - Gini coefficient for capacity & ethics scores
     - Theil index decomposition: between vs within income groups
     - Lorenz curves

  6. Policy portfolio breadth
     - Per-country coverage index (dimensions scoring ≥1)
     - Portfolio gap identification
     - Country × dimension heatmaps

  7. Quantile regression
     - Does GDP matter more at the bottom of the distribution?
     - Quantile regression at τ = 0.10, 0.25, 0.50, 0.75, 0.90
     - Compare coefficients across quantiles

  8. Tobit regression
     - Handle floor effects (64% score 0–0.9)
     - Left-censored at 0 Tobit model
     - Compare with OLS & mixed models

Output:
  data/analysis/paper1_capacity/extended/
  data/analysis/paper2_ethics/extended/
  data/analysis/shared/extended/

Usage:
  python src/analysis/extended_analysis.py
"""

import json
import math
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

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import get_metadata

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_CSV = ROOT / 'data' / 'analysis' / 'shared' / 'master_dataset.csv'
OUT_P1 = ROOT / 'data' / 'analysis' / 'paper1_capacity' / 'extended'
OUT_P2 = ROOT / 'data' / 'analysis' / 'paper2_ethics' / 'extended'
OUT_EXT = ROOT / 'data' / 'analysis' / 'shared' / 'extended'

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

# ─── Style ──────────────────────────────────────────────────────────────────
STYLE = {
    'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.figsize': (10, 6),
}
plt.rcParams.update(STYLE)
PAL = sns.color_palette('Set2', 8)
INCOME_COLORS = {
    'High income': PAL[0], 'Upper middle income': PAL[1],
    'Lower middle income': PAL[2], 'Low income': PAL[3],
}


def _json_default(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_data() -> pd.DataFrame:
    print("Loading master dataset...")
    df = pd.read_csv(MASTER_CSV)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['log_gdp_pc'] = np.log(df['gdp_pc'].replace(0, np.nan))
    print(f"  {len(df)} entries, {df['jurisdiction'].nunique()} jurisdictions")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 5. INEQUALITY DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════

def _gini(x: np.ndarray) -> float:
    """Compute the Gini coefficient for a 1-D array of non-negative values."""
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def _theil_t(x: np.ndarray) -> float:
    """Theil T (GE(1)) index."""
    x = x[~np.isnan(x)]
    x = x[x > 0]  # Theil T requires strictly positive values
    n = len(x)
    if n == 0:
        return 0.0
    mu = x.mean()
    return np.mean((x / mu) * np.log(x / mu))


def _theil_decompose(values: np.ndarray, groups: np.ndarray):
    """
    Decompose Theil T index into between-group and within-group components.
    Returns (total, between, within, between_share).
    """
    # Drop entries where group is NaN
    mask = pd.notna(groups) & ~np.isnan(values)
    values = values[mask]
    groups = groups[mask]

    total_t = _theil_t(values)
    mu = values.mean()
    n = len(values)

    between = 0.0
    within = 0.0

    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        x_g = values[mask]
        x_g = x_g[~np.isnan(x_g)]
        x_g = x_g[x_g > 0]
        if len(x_g) == 0:
            continue
        n_g = len(x_g)
        mu_g = x_g.mean()
        s_g = (n_g * mu_g) / (n * mu)  # income share of group g

        # Between component
        between += s_g * np.log(mu_g / mu)

        # Within component
        t_g = _theil_t(x_g) if len(x_g) > 1 else 0.0
        within += s_g * t_g

    between_share = between / total_t if total_t > 0 else 0.0
    return total_t, between, within, between_share


def inequality_decomposition(df: pd.DataFrame, paper: str, dims: list,
                             composite: str, out_dir: Path):
    """Gini coefficients and Theil index decomposition."""
    section = f"5. INEQUALITY DECOMPOSITION — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international']].copy()
    results = {}

    # ── 5a. Gini coefficients ──────────────────────────────────────────────
    print(f"\n  5a. Gini coefficients:")
    print(f"  {'Scope':<30s} {'Composite':>10s}  {'Dims':>40s}")

    gini_results = {}
    for scope_label, sub in [('All countries', dfc),
                              ('High income', dfc[dfc['income_binary'] == 'High income']),
                              ('Developing', dfc[dfc['income_binary'] == 'Developing'])]:
        g_comp = _gini(sub[composite].values)
        g_dims = {d: _gini(sub[d].values) for d in dims}
        gini_results[scope_label] = {
            'composite': round(g_comp, 4),
            'dimensions': {d: round(v, 4) for d, v in g_dims.items()},
            'n': len(sub),
        }
        dim_str = ', '.join(f"{_gini(sub[d].values):.3f}" for d in dims[:3]) + '...'
        print(f"  {scope_label:<30s} {g_comp:>10.4f}  {dim_str:>40s}")

    # Country-level Gini
    country_means = dfc.groupby('jurisdiction')[composite].mean()
    g_country = _gini(country_means.values)
    print(f"  {'Country-level means':<30s} {g_country:>10.4f}")
    gini_results['country_level'] = {'composite': round(g_country, 4), 'n': len(country_means)}

    results['gini'] = gini_results

    # ── 5b. Theil decomposition ────────────────────────────────────────────
    print(f"\n  5b. Theil T index decomposition (by income group):")

    # Need positive values for Theil — shift scores by +0.01 to avoid zeros
    dfc_ib = dfc.dropna(subset=['income_binary'])
    vals = dfc_ib[composite].values + 0.01
    groups = dfc_ib['income_binary'].values

    total, between, within, between_share = _theil_decompose(vals, groups)
    print(f"    Total Theil T:    {total:.4f}")
    print(f"    Between groups:   {between:.4f} ({between_share*100:.1f}%)")
    print(f"    Within groups:    {within:.4f} ({(1-between_share)*100:.1f}%)")
    print(f"    → {'Between' if between_share > 0.5 else 'Within'}-group inequality dominates")

    results['theil_binary'] = {
        'total': round(total, 5), 'between': round(between, 5),
        'within': round(within, 5), 'between_share': round(between_share, 4),
    }

    # 4-group decomposition
    dfc_ig = dfc.dropna(subset=['income_group'])
    groups4 = dfc_ig['income_group'].values
    vals4 = dfc_ig[composite].values + 0.01
    total4, between4, within4, bs4 = _theil_decompose(vals4, groups4)
    print(f"\n    Decomposition by 4 income groups:")
    print(f"    Total Theil T:    {total4:.4f}")
    print(f"    Between groups:   {between4:.4f} ({bs4*100:.1f}%)")
    print(f"    Within groups:    {within4:.4f} ({(1-bs4)*100:.1f}%)")

    results['theil_4group'] = {
        'total': round(total4, 5), 'between': round(between4, 5),
        'within': round(within4, 5), 'between_share': round(bs4, 4),
    }

    # Per-dimension Theil
    print(f"\n    Per-dimension Theil decomposition (binary):")
    dim_theil = {}
    dfc_ib = dfc.dropna(subset=['income_binary'])
    for d in dims:
        v = dfc_ib[d].values + 0.01
        t, b, w, bs = _theil_decompose(v, dfc_ib['income_binary'].values)
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        print(f"    {label:<25s} Total={t:.4f}  Between={bs*100:.1f}%")
        dim_theil[d] = {'total': round(t, 5), 'between_share': round(bs, 4)}
    results['theil_dimensions'] = dim_theil

    # ── 5c. Lorenz curve ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, scope, sub in [(ax1, 'Policy-level', dfc),
                            (ax2, 'Country-level', None)]:
        if sub is None:
            vals_sorted = np.sort(country_means.values)
        else:
            vals_sorted = np.sort(sub[composite].values)

        n = len(vals_sorted)
        cum_pop = np.arange(1, n + 1) / n
        cum_income = np.cumsum(vals_sorted) / vals_sorted.sum()

        ax.plot(cum_pop, cum_income, color=PAL[0], linewidth=2, label='Lorenz curve')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect equality')
        ax.fill_between(cum_pop, cum_income, cum_pop, alpha=0.15, color=PAL[0])

        gini_val = _gini(vals_sorted)
        ax.set_xlabel('Cumulative share of policies')
        ax.set_ylabel(f'Cumulative share of {composite.replace("_", " ")}')
        ax.set_title(f'{scope} (Gini = {gini_val:.3f})')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    fig.suptitle(f'{paper}: Inequality in Governance Scores', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_lorenz.png')
    plt.close(fig)

    # Theil decomposition bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    dim_labels = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in dims]
    bs_vals = [dim_theil[d]['between_share'] * 100 for d in dims]
    ws_vals = [100 - b for b in bs_vals]
    x = np.arange(len(dims))
    ax.bar(x, bs_vals, color=PAL[2], label='Between income groups', edgecolor='white')
    ax.bar(x, ws_vals, bottom=bs_vals, color=PAL[0], label='Within income groups', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=30, ha='right')
    ax.set_ylabel('Share of Total Inequality (%)')
    ax.set_title(f'{paper}: Theil Decomposition by Dimension')
    ax.legend()
    ax.set_ylim(0, 100)
    fig.savefig(out_dir / 'fig_theil_decomposition.png')
    plt.close(fig)

    with open(out_dir / 'inequality_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 6. POLICY PORTFOLIO BREADTH
# ═════════════════════════════════════════════════════════════════════════════

def policy_portfolio(df: pd.DataFrame, paper: str, dims: list,
                     composite: str, out_dir: Path):
    """
    Per-country analysis: how many dimensions does the national portfolio cover?
    Coverage = dimension has at least one policy scoring ≥ 1.
    """
    section = f"6. POLICY PORTFOLIO BREADTH — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international']].copy()

    # Country-level: max score per dimension (does the portfolio cover it?)
    country_max = dfc.groupby('jurisdiction')[dims].max()
    country_count = dfc.groupby('jurisdiction').size()
    country_max = country_max[country_count >= 3]  # min 3 policies

    # Coverage: dimension ≥ 1
    coverage = (country_max >= 1).astype(int)
    coverage['breadth'] = coverage.sum(axis=1)
    coverage['n_policies'] = country_count[coverage.index]
    coverage['income'] = coverage.index.map(lambda j: get_metadata(j)['income_label'])
    coverage['income_binary'] = coverage.index.map(lambda j: get_metadata(j)['income_binary'])

    n_dims = len(dims)
    print(f"\n  Coverage breadth distribution ({n_dims} dimensions):")
    for b in range(n_dims + 1):
        count = (coverage['breadth'] == b).sum()
        if count > 0:
            countries = sorted(coverage[coverage['breadth'] == b].index.tolist())
            print(f"    {b}/{n_dims} dimensions: {count} countries  — {', '.join(countries[:5])}{'...' if len(countries)>5 else ''}")

    results = {
        'n_countries': len(coverage),
        'mean_breadth': round(coverage['breadth'].mean(), 2),
        'median_breadth': int(coverage['breadth'].median()),
    }

    # By income group
    print(f"\n  Mean breadth by income level:")
    for ig in ['High income', 'Developing']:
        sub = coverage[coverage['income_binary'] == ig]
        if len(sub) > 0:
            m = sub['breadth'].mean()
            print(f"    {ig:<15s}  mean={m:.2f}/{n_dims}  (n={len(sub)})")
            results[f'breadth_{ig.lower().replace(" ", "_")}'] = round(m, 2)

    # T-test on breadth
    hi_b = coverage[coverage['income_binary'] == 'High income']['breadth'].values
    dev_b = coverage[coverage['income_binary'] == 'Developing']['breadth'].values
    if len(hi_b) >= 5 and len(dev_b) >= 5:
        t, p = sp_stats.ttest_ind(hi_b, dev_b, equal_var=False)
        d = (hi_b.mean() - dev_b.mean()) / math.sqrt((hi_b.std()**2 + dev_b.std()**2) / 2)
        print(f"    t={t:.3f}, p={p:.4f}, d={d:.3f}")
        results['breadth_ttest'] = {'t': round(t, 3), 'p': round(p, 6), 'd': round(d, 3)}

    # Biggest gaps: dimensions least covered
    dim_coverage = coverage[dims].mean()
    print(f"\n  Dimension coverage rate (% countries with at least one policy ≥ 1):")
    for d in dims:
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        pct = dim_coverage[d] * 100
        print(f"    {label:<25s} {pct:.1f}%")
    results['dimension_coverage_rate'] = {d: round(dim_coverage[d], 4) for d in dims}

    # Country × dimension heatmap
    dim_labels = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in dims]

    # Use mean scores for heatmap (more informative than binary coverage)
    country_mean = dfc.groupby('jurisdiction')[dims].mean()
    country_mean = country_mean[country_count >= 5]  # min 5 for heatmap
    country_mean = country_mean.sort_values(dims[0], ascending=False)

    # Sort by composite mean
    country_mean['_composite'] = country_mean[dims].mean(axis=1)
    country_mean = country_mean.sort_values('_composite', ascending=False)
    country_mean = country_mean.drop('_composite', axis=1)

    fig, ax = plt.subplots(figsize=(10, max(8, len(country_mean) * 0.35)))
    display = country_mean.copy()
    display.columns = dim_labels
    sns.heatmap(display, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=2.5, ax=ax, linewidths=0.3)
    ax.set_title(f'{paper}: Country × Dimension Mean Scores (min 5 policies)')
    ax.set_ylabel('')
    fig.savefig(out_dir / 'fig_portfolio_heatmap.png')
    plt.close(fig)

    # Gap identification: which dimensions are most "missing"?
    # For each country, identify dimensions where max score < 1
    gap_counts = (country_max[dims] < 1).sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dim_labels, gap_counts.values, color=PAL[3], edgecolor='white')
    ax.set_ylabel('Number of Countries with Gap')
    ax.set_title(f'{paper}: Portfolio Gaps (Dimension max < 1, min 3 policies)')
    ax.set_xticklabels(dim_labels, rotation=30, ha='right')
    for bar, val in zip(bars, gap_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, str(val),
                ha='center', va='bottom', fontsize=10)
    fig.savefig(out_dir / 'fig_portfolio_gaps.png')
    plt.close(fig)

    # Save data
    coverage.to_csv(out_dir / 'portfolio_coverage.csv')
    country_mean.to_csv(out_dir / 'portfolio_country_means.csv')

    with open(out_dir / 'portfolio_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 7. QUANTILE REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

def quantile_regression(df: pd.DataFrame, paper: str, dims: list,
                        composite: str, out_dir: Path):
    """
    Does GDP matter more at the bottom of the score distribution?
    Quantile regression at τ = 0.10, 0.25, 0.50, 0.75, 0.90.
    """
    section = f"7. QUANTILE REGRESSION — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international'] & df['year'].between(2017, 2025)].copy()
    dfc = dfc[dfc['gdp_pc'] > 0].copy()
    dfc['log_gdp_pc'] = np.log(dfc['gdp_pc'])
    dfc['year_c'] = dfc['year'] - 2021
    dfc['is_binding'] = dfc['binding_nature'].isin(['Binding regulation', 'Hard law']).astype(int)
    dfc['is_good_text'] = (dfc['text_quality'] == 'good').astype(int)
    dfc = dfc.dropna(subset=['log_gdp_pc', composite])

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    formula = f'{composite} ~ log_gdp_pc + year_c + is_binding + is_good_text'

    # Because 30-50% of the dependent variable is exactly 0, quantile
    # regression at low τ hits the mass-point and produces degenerate
    # β=0 coefficients. We run QR on ALL data (showing the mass-point
    # issue) and also on the POSITIVE subset (where QR is well-behaved).

    results = {}

    # ── OLS baseline ──────────────────────────────────────────────────────
    print(f"\n  OLS baseline: {formula}")
    ols = smf.ols(formula, data=dfc).fit()
    ols_gdp = float(ols.params['log_gdp_pc'])
    ols_gdp_p = float(ols.pvalues['log_gdp_pc'])
    print(f"    GDP β = {ols_gdp:+.4f} (p={ols_gdp_p:.4f}), R²={ols.rsquared:.4f}")

    results['ols'] = {
        'gdp_beta': round(ols_gdp, 5),
        'gdp_p': round(ols_gdp_p, 6),
        'r_squared': round(float(ols.rsquared), 4),
    }

    # ── Quantile regressions ──────────────────────────────────────────────
    print(f"\n  Quantile regression results:")
    print(f"  {'τ':>5s} {'GDP β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s} {'Text β':>8s} {'Bind β':>8s}")

    qr_results = {}
    gdp_betas = []
    gdp_cis_low = []
    gdp_cis_high = []

    for q in quantiles:
        try:
            qr = smf.quantreg(formula, data=dfc).fit(q=q, max_iter=1000)
            gdp_b = float(qr.params['log_gdp_pc'])
            gdp_se = float(qr.bse['log_gdp_pc'])
            gdp_t = float(qr.tvalues['log_gdp_pc'])
            gdp_p = float(qr.pvalues['log_gdp_pc'])
            text_b = float(qr.params['is_good_text'])
            bind_b = float(qr.params['is_binding'])

            sig = '***' if gdp_p < 0.001 else '**' if gdp_p < 0.01 else '*' if gdp_p < 0.05 else ''
            print(f"  {q:>5.2f} {gdp_b:>+8.4f} {gdp_se:>8.4f} {gdp_t:>8.3f} "
                  f"{gdp_p:>9.5f}{sig} {text_b:>+8.4f} {bind_b:>+8.4f}")

            gdp_betas.append(gdp_b)
            gdp_cis_low.append(gdp_b - 1.96 * gdp_se)
            gdp_cis_high.append(gdp_b + 1.96 * gdp_se)

            qr_results[str(q)] = {
                'gdp_beta': round(gdp_b, 5), 'gdp_se': round(gdp_se, 5),
                'gdp_p': round(gdp_p, 6),
                'text_beta': round(text_b, 5), 'binding_beta': round(bind_b, 5),
                'intercept': round(float(qr.params['Intercept']), 5),
                'year_beta': round(float(qr.params['year_c']), 5),
                'pseudo_r2': round(float(qr.prsquared), 4),
            }
        except Exception as e:
            print(f"  {q:>5.2f}  FAILED: {e}")
            gdp_betas.append(np.nan)
            gdp_cis_low.append(np.nan)
            gdp_cis_high.append(np.nan)

    results['quantile_regressions'] = qr_results

    # ── Positive-subset quantile regression ───────────────────────────────
    # With many exact zeros, QR at low τ is degenerate.
    # Run on y > 0 subset to get well-behaved estimates.
    dfc_pos = dfc[dfc[composite] > 0].copy()
    print(f"\n  Positive-subset QR (n={len(dfc_pos)}, excluding {len(dfc)-len(dfc_pos)} zeros):")
    print(f"  {'τ':>5s} {'GDP β':>8s} {'SE':>8s} {'p':>10s}")

    pos_results = {}
    pos_gdp_betas = []
    pos_gdp_cis_low = []
    pos_gdp_cis_high = []

    for q in quantiles:
        try:
            qr = smf.quantreg(formula, data=dfc_pos).fit(q=q, max_iter=1000)
            gdp_b = float(qr.params['log_gdp_pc'])
            gdp_se = float(qr.bse['log_gdp_pc'])
            gdp_p = float(qr.pvalues['log_gdp_pc'])
            sig = '***' if gdp_p < 0.001 else '**' if gdp_p < 0.01 else '*' if gdp_p < 0.05 else ''
            print(f"  {q:>5.2f} {gdp_b:>+8.4f} {gdp_se:>8.4f} {gdp_p:>9.5f}{sig}")
            pos_gdp_betas.append(gdp_b)
            pos_gdp_cis_low.append(gdp_b - 1.96 * gdp_se)
            pos_gdp_cis_high.append(gdp_b + 1.96 * gdp_se)
            pos_results[str(q)] = {
                'gdp_beta': round(gdp_b, 5), 'gdp_se': round(gdp_se, 5),
                'gdp_p': round(gdp_p, 6),
                'pseudo_r2': round(float(qr.prsquared), 4),
            }
        except Exception as e:
            print(f"  {q:>5.2f}  FAILED: {e}")
            pos_gdp_betas.append(np.nan)
            pos_gdp_cis_low.append(np.nan)
            pos_gdp_cis_high.append(np.nan)

    results['quantile_regressions_positive'] = pos_results

    # Use positive-subset results for interpretation
    use_betas = pos_gdp_betas if any(b != 0 and not np.isnan(b) for b in pos_gdp_betas) else gdp_betas

    # Interpretation
    if len(use_betas) >= 3 and not np.isnan(use_betas[0]):
        low_q = use_betas[0]  # τ=0.10
        high_q = use_betas[-1]  # τ=0.90
        ratio = abs(low_q / high_q) if high_q != 0 else float('inf')
        print(f"\n  GDP effect at bottom (τ=0.10): β = {low_q:+.4f}")
        print(f"  GDP effect at top (τ=0.90):    β = {high_q:+.4f}")
        print(f"  Ratio: {ratio:.2f}x")
        if abs(low_q) > abs(high_q) * 1.5:
            print(f"  → GDP matters MORE at the bottom of the distribution")
        elif abs(high_q) > abs(low_q) * 1.5:
            print(f"  → GDP matters MORE at the top of the distribution")
        else:
            print(f"  → GDP effect is relatively uniform across quantiles")

        results['interpretation'] = {
            'bottom_beta': round(low_q, 5), 'top_beta': round(high_q, 5),
            'ratio': round(ratio, 2),
            'source': 'positive_subset' if use_betas is pos_gdp_betas else 'full_sample',
        }

    # ── Quantile coefficient plot ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Use positive-subset for plot (better behaved)
    plot_betas = pos_gdp_betas if pos_results else gdp_betas
    plot_cl = pos_gdp_cis_low if pos_results else gdp_cis_low
    plot_ch = pos_gdp_cis_high if pos_results else gdp_cis_high

    valid = [not np.isnan(b) for b in plot_betas]
    if any(valid):
        q_valid = [q for q, v in zip(quantiles, valid) if v]
        b_valid = [b for b, v in zip(plot_betas, valid) if v]
        cl = [c for c, v in zip(plot_cl, valid) if v]
        ch = [c for c, v in zip(plot_ch, valid) if v]

        ax1.fill_between(q_valid, cl, ch, alpha=0.2, color=PAL[0])
        ax1.plot(q_valid, b_valid, 'o-', color=PAL[0], linewidth=2, markersize=8,
                 label='QR (y > 0 subset)')
        ax1.axhline(ols_gdp, color='red', linestyle='--', alpha=0.7,
                    label=f'OLS β = {ols_gdp:.4f}')
        ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Quantile (τ)')
        ax1.set_ylabel('GDP coefficient (β)')
        ax1.set_title(f'(a) log(GDP per capita) Effect by Quantile')
        ax1.legend(fontsize=9)

    # Text quality: use positive-subset results
    pos_text_betas = [pos_results.get(str(q), qr_results.get(str(q), {}))
                      .get('text_beta', np.nan) if str(q) in pos_results else
                      qr_results.get(str(q), {}).get('text_beta', np.nan)
                      for q in quantiles]
    # Fallback to full-sample text betas
    text_betas = [qr_results.get(str(q), {}).get('text_beta', np.nan) for q in quantiles]
    valid_t = [not np.isnan(b) for b in text_betas]
    if any(valid_t):
        q_valid_t = [q for q, v in zip(quantiles, valid_t) if v]
        t_valid = [b for b, v in zip(text_betas, valid_t) if v]
        ax2.plot(q_valid_t, t_valid, 'o-', color=PAL[2], linewidth=2, markersize=8,
                 label='Quantile regression')
        ols_text = float(ols.params['is_good_text'])
        ax2.axhline(ols_text, color='red', linestyle='--', alpha=0.7,
                    label=f'OLS β = {ols_text:.4f}')
        ax2.set_xlabel('Quantile (τ)')
        ax2.set_ylabel('Text quality coefficient (β)')
        ax2.set_title(f'(b) Good Text Effect by Quantile')
        ax2.legend(fontsize=9)

    fig.suptitle(f'{paper}: Quantile Regression Coefficients', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_quantile_regression.png')
    plt.close(fig)

    with open(out_dir / 'quantile_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 8. TOBIT REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

def tobit_regression(df: pd.DataFrame, paper: str, dims: list,
                     composite: str, out_dir: Path):
    """
    Tobit model: left-censored at 0 (floor effects).
    64% of policies score 0–0.9 — OLS is technically misspecified.
    Uses statsmodels Tobit via maximum likelihood.
    """
    section = f"8. TOBIT REGRESSION — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international'] & df['year'].between(2017, 2025)].copy()
    dfc = dfc[dfc['gdp_pc'] > 0].copy()
    dfc['log_gdp_pc'] = np.log(dfc['gdp_pc'])
    dfc['year_c'] = dfc['year'] - 2021
    dfc['is_binding'] = dfc['binding_nature'].isin(['Binding regulation', 'Hard law']).astype(int)
    dfc['is_good_text'] = (dfc['text_quality'] == 'good').astype(int)
    dfc = dfc.dropna(subset=['log_gdp_pc', composite])

    y = dfc[composite].values
    X_vars = ['log_gdp_pc', 'year_c', 'is_binding', 'is_good_text']
    X = dfc[X_vars].values
    X = sm.add_constant(X)

    results = {}

    # ── 8a. Floor effect description ──────────────────────────────────────
    n_zero = (y == 0).sum()
    n_near_zero = (y < 0.5).sum()
    print(f"\n  8a. Floor effects:")
    print(f"    Exact zeros:     {n_zero:>5d} ({n_zero/len(y)*100:.1f}%)")
    print(f"    Score < 0.5:     {n_near_zero:>5d} ({n_near_zero/len(y)*100:.1f}%)")
    print(f"    Score < 1.0:     {(y < 1).sum():>5d} ({(y < 1).sum()/len(y)*100:.1f}%)")

    results['floor_effects'] = {
        'n_zero': int(n_zero), 'pct_zero': round(n_zero/len(y)*100, 1),
        'n_below_0.5': int(n_near_zero), 'pct_below_0.5': round(n_near_zero/len(y)*100, 1),
        'n_below_1.0': int((y < 1).sum()), 'pct_below_1.0': round((y < 1).sum()/len(y)*100, 1),
    }

    # ── 8b. Tobit model (left-censored at 0) ─────────────────────────────
    print(f"\n  8b. Tobit model (left-censored at 0):")

    try:
        # statsmodels doesn't have a built-in Tobit, so we implement via
        # censored regression using the Tobit log-likelihood
        from scipy.optimize import minimize
        from scipy.stats import norm

        def tobit_loglik(params, y, X, censor_val=0):
            """Negative log-likelihood for left-censored Tobit."""
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)

            Xb = X @ beta
            censored = y <= censor_val
            uncensored = ~censored

            ll = 0
            # Uncensored observations
            if uncensored.any():
                z = (y[uncensored] - Xb[uncensored]) / sigma
                ll += np.sum(-0.5 * np.log(2 * np.pi) - log_sigma - 0.5 * z**2)

            # Censored observations
            if censored.any():
                z_c = (censor_val - Xb[censored]) / sigma
                ll += np.sum(np.log(norm.cdf(z_c) + 1e-10))

            return -ll  # minimize negative LL

        # Initial values from OLS
        ols = sm.OLS(y, X).fit()
        init_params = np.append(ols.params, np.log(np.std(ols.resid)))

        # Optimize — try multiple methods for robustness
        best_result = None
        for method in ['L-BFGS-B', 'Nelder-Mead', 'Powell', 'BFGS']:
            try:
                res = minimize(tobit_loglik, init_params, args=(y, X),
                               method=method,
                               options={'maxiter': 10000, 'gtol': 1e-5})
                if res.success or (best_result is None):
                    if best_result is None or res.fun < best_result.fun:
                        best_result = res
                if res.success:
                    break
            except Exception:
                continue

        result = best_result

        if result is not None and np.isfinite(result.fun):
            if not result.success:
                print(f"    (Warning: optimizer reports '{result.message}', using best result)")
            params = result.x
            beta = params[:-1]
            sigma = np.exp(params[-1])

            # Standard errors from inverse Hessian (or numerical approximation)
            try:
                if hasattr(result, 'hess_inv') and result.hess_inv is not None:
                    hess_inv = result.hess_inv
                    if hasattr(hess_inv, 'todense'):
                        hess_inv = hess_inv.todense()
                    se = np.sqrt(np.diag(np.abs(hess_inv)))
                else:
                    # Numerical Hessian via finite differences
                    from scipy.optimize import approx_fprime
                    eps = 1e-5
                    n_p = len(params)
                    hess = np.zeros((n_p, n_p))
                    for i in range(n_p):
                        def grad_i(p):
                            return approx_fprime(p, tobit_loglik, eps, y, X)[i]
                        hess[i, :] = approx_fprime(params, grad_i, eps)
                    try:
                        hess_inv = np.linalg.inv(hess)
                        se = np.sqrt(np.diag(np.abs(hess_inv)))
                    except np.linalg.LinAlgError:
                        se = np.full(len(params), np.nan)
            except Exception:
                se = np.full(len(params), np.nan)

            var_names = ['(Intercept)'] + X_vars + ['log(σ)']
            print(f"    {'Variable':<20s} {'β':>8s} {'SE':>8s} {'z':>8s} {'p':>10s}")
            print(f"    {'─'*56}")

            tobit_coefs = {}
            for i, v in enumerate(var_names):
                b = params[i]
                s = se[i] if not np.isnan(se[i]) else 0
                z = b / s if s > 0 else 0
                p = 2 * (1 - norm.cdf(abs(z))) if s > 0 else 1
                sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"    {v:<20s} {b:>+8.4f} {s:>8.4f} {z:>8.3f} {p:>9.5f} {sig_str}")
                tobit_coefs[v] = {
                    'beta': round(float(b), 5), 'se': round(float(s), 5),
                    'z': round(float(z), 4), 'p': round(float(p), 6),
                }

            print(f"\n    σ = {sigma:.4f}")
            print(f"    Log-likelihood: {-result.fun:.2f}")

            results['tobit'] = {
                'coefficients': tobit_coefs,
                'sigma': round(sigma, 4),
                'log_likelihood': round(-result.fun, 2),
                'n_obs': len(y),
                'n_censored': int(n_zero),
                'converged': result.success,
            }

            # ── 8c. OLS for comparison ────────────────────────────────────
            print(f"\n  8c. OLS comparison:")
            print(f"    {'Variable':<20s} {'OLS β':>8s} {'Tobit β':>9s} {'Diff':>8s}")
            print(f"    {'─'*48}")

            ols_coefs = {}
            ols_names = ['(Intercept)'] + X_vars
            for i, v in enumerate(ols_names):
                ols_b = ols.params[i]
                tobit_b = beta[i]
                diff = tobit_b - ols_b
                print(f"    {v:<20s} {ols_b:>+8.4f} {tobit_b:>+9.4f} {diff:>+8.4f}")
                ols_coefs[v] = round(float(ols_b), 5)

            results['ols_comparison'] = {
                'coefficients': ols_coefs,
                'r_squared': round(float(ols.rsquared), 4),
                'log_likelihood': round(float(ols.llf), 2),
            }

            # ── Marginal effects ──────────────────────────────────────────
            # E[y*|y>0] marginal effects: β * Φ(Xβ/σ)
            Xb_mean = X.mean(axis=0) @ beta
            prob_uncensored = norm.cdf(Xb_mean / sigma)
            print(f"\n  8d. Average marginal effects (Tobit):")
            print(f"    P(uncensored at mean): {prob_uncensored:.3f}")
            me_results = {}
            for i, v in enumerate(ols_names):
                me = beta[i] * prob_uncensored
                print(f"    {v:<20s}  ME = {me:+.4f}")
                me_results[v] = round(float(me), 5)
            results['marginal_effects'] = me_results

        else:
            print(f"    Tobit optimization failed completely")
            results['tobit'] = {'error': 'Optimization failed', 'converged': False}

    except Exception as e:
        print(f"    Tobit model failed: {e}")
        import traceback
        traceback.print_exc()
        results['tobit'] = {'error': str(e)}

    # ── Figure: OLS vs Tobit comparison ───────────────────────────────────
    if 'tobit' in results and 'coefficients' in results.get('tobit', {}):
        fig, ax = plt.subplots(figsize=(10, 6))
        var_plot = X_vars  # exclude intercept and log(σ)
        ols_vals = [float(ols.params[i+1]) for i in range(len(var_plot))]
        tobit_vals = [results['tobit']['coefficients'].get(v, {}).get('beta', 0)
                      for v in var_plot]

        x_pos = np.arange(len(var_plot))
        width = 0.35
        ax.bar(x_pos - width/2, ols_vals, width, label='OLS', color=PAL[0], edgecolor='white')
        ax.bar(x_pos + width/2, tobit_vals, width, label='Tobit', color=PAL[2], edgecolor='white')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['log(GDP pc)', 'Year', 'Binding', 'Good text'], rotation=15)
        ax.set_ylabel('Coefficient (β)')
        ax.set_title(f'{paper}: OLS vs Tobit Coefficients')
        ax.legend()
        ax.axhline(0, color='gray', linewidth=0.5)
        fig.savefig(out_dir / 'fig_tobit_comparison.png')
        plt.close(fig)

    with open(out_dir / 'tobit_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run the complete Phase 3c extended analysis pipeline."""
    start = datetime.now()
    print("=" * 70)
    print("PHASE 3c: EXTENDED ANALYSIS PIPELINE")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for d in [OUT_P1, OUT_P2, OUT_EXT]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # ── Paper 1: Capacity ──────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("PAPER 1: CAPACITY — EXTENDED ANALYSES")
    print(f"{'#'*70}")
    inequality_decomposition(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    policy_portfolio(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    quantile_regression(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    tobit_regression(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)

    # ── Paper 2: Ethics ────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("PAPER 2: ETHICS — EXTENDED ANALYSES")
    print(f"{'#'*70}")
    inequality_decomposition(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    policy_portfolio(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    quantile_regression(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    tobit_regression(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*70}")
    print(f"PHASE 3c COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Paper 1 extended: {OUT_P1}")
    print(f"  Paper 2 extended: {OUT_P2}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_all()
