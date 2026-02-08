"""
Phase 3b: Advanced Analysis Pipeline
======================================
Four reviewer-essential analyses building on Phase 3a:

  1. Robustness checks
     - Re-estimate income-group gaps excluding stubs/thin texts
     - Bootstrap 95% CIs for key effect sizes (1,000 replications)
     - Cluster stability: k = 2–7 with silhouette scores

  2. Multilevel / hierarchical models
     - Random-intercepts model (country grouping) for capacity & ethics
     - Compare with pooled OLS (likelihood-ratio test)
     - Country-level ICC: how much variance is between- vs within-country?

  3. PCA / Factor analysis
     - PCA on all 10 dimensions (policy-level)
     - Do C1–C5 and E1–E5 form two distinct constructs?
     - Scree plot, loadings matrix, explained variance

  4. Convergence / divergence analysis
     - Income × year interaction in OLS
     - Separate temporal slopes for HI vs developing
     - Gap trajectory 2017–2025

Output:
  data/analysis/paper1_capacity/robustness/
  data/analysis/paper2_ethics/robustness/
  data/analysis/shared/advanced/

Usage:
  python src/analysis/advanced_analysis.py
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import shared config from sota_analysis
sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import (
    INCOME_GROUP, INCOME_LABELS, REGION, REGION_LABELS,
    GDP_PER_CAPITA, INTERNATIONAL, get_income_binary, get_metadata,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
ENSEMBLE_PATH = ROOT / 'data' / 'analysis' / 'scores_ensemble.json'
MASTER_CSV = ROOT / 'data' / 'analysis' / 'shared' / 'master_dataset.csv'
OUT_P1 = ROOT / 'data' / 'analysis' / 'paper1_capacity' / 'robustness'
OUT_P2 = ROOT / 'data' / 'analysis' / 'paper2_ethics' / 'robustness'
OUT_ADV = ROOT / 'data' / 'analysis' / 'shared' / 'advanced'

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
    """Handle numpy / pandas types for JSON serialisation."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (reuse the master CSV from Phase 3a)
# ═════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load the master dataset produced by sota_analysis.py."""
    print("Loading master dataset...")
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    else:
        # Fall back to rebuilding from ensemble JSON
        from sota_analysis import load_data as _load
        df = _load()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['log_gdp_pc'] = np.log(df['gdp_pc'].replace(0, np.nan))
    print(f"  {len(df)} entries, {df['jurisdiction'].nunique()} jurisdictions")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 1. ROBUSTNESS CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def robustness_checks(df: pd.DataFrame, paper: str, dims: list,
                      composite: str, out_dir: Path):
    """
    1a. Text-quality sensitivity: re-run income-gap t-tests on good-text only
    1b. Bootstrap CIs (1,000 reps) for income-gap Cohen's d
    1c. Cluster stability: silhouette scores for k = 2–7
    """
    section = f"1. ROBUSTNESS CHECKS — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international']].copy()
    results = {}

    # ── 1a. Text-quality sensitivity ────────────────────────────────────────
    print("\n  1a. Text-quality sensitivity analysis")
    print(f"  {'Sample':<25s} {'N':>5s} {'HI mean':>8s} {'Dev mean':>9s} {'t':>8s} {'p':>10s} {'d':>7s}")

    samples = {
        'All texts':        dfc,
        'Good only':        dfc[dfc['text_quality'] == 'good'],
        'Good + thin':      dfc[dfc['text_quality'].isin(['good', 'thin'])],
        'Excl. stubs':      dfc[dfc['text_quality'] != 'stub'],
    }

    sensitivity = {}
    for label, sub in samples.items():
        hi = sub[sub['income_binary'] == 'High income'][composite]
        dev = sub[sub['income_binary'] == 'Developing'][composite]
        if len(hi) >= 10 and len(dev) >= 10:
            t, p = sp_stats.ttest_ind(hi, dev, equal_var=False)
            d = (hi.mean() - dev.mean()) / math.sqrt((hi.std()**2 + dev.std()**2) / 2)
            n = len(hi) + len(dev)
            sensitivity[label] = {
                'n': n, 'n_hi': len(hi), 'n_dev': len(dev),
                'hi_mean': round(hi.mean(), 4), 'dev_mean': round(dev.mean(), 4),
                't': round(t, 4), 'p': round(p, 6), 'd': round(d, 4),
            }
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {label:<25s} {n:>5d} {hi.mean():>8.3f} {dev.mean():>9.3f} "
                  f"{t:>8.3f} {p:>9.6f}{sig} {d:>+7.3f}")

    results['text_quality_sensitivity'] = sensitivity

    # Per-dimension sensitivity (good-text only)
    print(f"\n  Per-dimension (good-text only, HI − Dev):")
    good = dfc[dfc['text_quality'] == 'good']
    dim_sens = {}
    for d in dims:
        hi = good[good['income_binary'] == 'High income'][d]
        dev = good[good['income_binary'] == 'Developing'][d]
        if len(hi) >= 10 and len(dev) >= 10:
            t, p = sp_stats.ttest_ind(hi, dev, equal_var=False)
            cd = (hi.mean() - dev.mean()) / math.sqrt((hi.std()**2 + dev.std()**2) / 2)
            dim_sens[d] = {'hi': round(hi.mean(), 3), 'dev': round(dev.mean(), 3),
                           'diff': round(hi.mean() - dev.mean(), 3),
                           't': round(t, 3), 'p': round(p, 6), 'd': round(cd, 3)}
            label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {label:<25s} {hi.mean():>7.3f} {dev.mean():>8.3f} {cd:>+7.3f} (p={p:.4f}{sig})")
    results['dimension_sensitivity_good_text'] = dim_sens

    # ── 1b. Bootstrap CIs ──────────────────────────────────────────────────
    print(f"\n  1b. Bootstrap 95% CIs (1,000 reps) for income-group gap")
    rng = np.random.default_rng(42)
    n_boot = 1000

    boot_results = {}
    for comp_label, comp_var in [(composite, composite)]:
        hi = dfc[dfc['income_binary'] == 'High income'][comp_var].values
        dev = dfc[dfc['income_binary'] == 'Developing'][comp_var].values

        boot_diffs = np.empty(n_boot)
        boot_cohens = np.empty(n_boot)

        for b in range(n_boot):
            hi_b = rng.choice(hi, size=len(hi), replace=True)
            dev_b = rng.choice(dev, size=len(dev), replace=True)
            boot_diffs[b] = hi_b.mean() - dev_b.mean()
            pooled_sd = math.sqrt((hi_b.std()**2 + dev_b.std()**2) / 2)
            boot_cohens[b] = boot_diffs[b] / pooled_sd if pooled_sd > 0 else 0

        obs_diff = hi.mean() - dev.mean()
        obs_d = obs_diff / math.sqrt((hi.std()**2 + dev.std()**2) / 2)

        boot_results[comp_label] = {
            'observed_diff': round(obs_diff, 4),
            'observed_cohens_d': round(obs_d, 4),
            'diff_ci95': [round(np.percentile(boot_diffs, 2.5), 4),
                          round(np.percentile(boot_diffs, 97.5), 4)],
            'cohens_d_ci95': [round(np.percentile(boot_cohens, 2.5), 4),
                              round(np.percentile(boot_cohens, 97.5), 4)],
            'boot_diff_mean': round(boot_diffs.mean(), 4),
            'boot_diff_se': round(boot_diffs.std(), 4),
        }

        ci_d = boot_results[comp_label]['diff_ci95']
        ci_c = boot_results[comp_label]['cohens_d_ci95']
        print(f"    {comp_label}:")
        print(f"      Mean diff: {obs_diff:.4f}  95% CI [{ci_d[0]:.4f}, {ci_d[1]:.4f}]")
        print(f"      Cohen's d: {obs_d:.4f}  95% CI [{ci_c[0]:.4f}, {ci_c[1]:.4f}]")

    # Per-dimension bootstrap
    print(f"\n  Per-dimension bootstrap CIs:")
    for d in dims:
        hi = dfc[dfc['income_binary'] == 'High income'][d].values
        dev = dfc[dfc['income_binary'] == 'Developing'][d].values
        boot_d = np.empty(n_boot)
        for b in range(n_boot):
            hi_b = rng.choice(hi, size=len(hi), replace=True)
            dev_b = rng.choice(dev, size=len(dev), replace=True)
            diff = hi_b.mean() - dev_b.mean()
            ps = math.sqrt((hi_b.std()**2 + dev_b.std()**2) / 2)
            boot_d[b] = diff / ps if ps > 0 else 0
        obs = (hi.mean() - dev.mean()) / math.sqrt((hi.std()**2 + dev.std()**2) / 2)
        ci = [round(np.percentile(boot_d, 2.5), 4), round(np.percentile(boot_d, 97.5), 4)]
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        print(f"    {label:<25s} d={obs:>+.3f}  95% CI [{ci[0]:>+.3f}, {ci[1]:>+.3f}]")
        boot_results[d] = {'observed_d': round(obs, 4), 'ci95': ci}

    results['bootstrap'] = boot_results

    # Bootstrap CI figure
    fig, ax = plt.subplots(figsize=(10, 6))
    dim_list = dims + [composite]
    y_pos = np.arange(len(dim_list))
    obs_vals = []
    ci_lows = []
    ci_highs = []
    labels_list = []
    for d in dim_list:
        br = boot_results[d]
        if 'observed_cohens_d' in br:
            obs_vals.append(br['observed_cohens_d'])
            ci_lows.append(br['cohens_d_ci95'][0])
            ci_highs.append(br['cohens_d_ci95'][1])
        else:
            obs_vals.append(br['observed_d'])
            ci_lows.append(br['ci95'][0])
            ci_highs.append(br['ci95'][1])
        labels_list.append(CAP_LABELS.get(d) or ETH_LABELS.get(d, d) or composite.replace('_', ' ').title())

    errors_low = [o - l for o, l in zip(obs_vals, ci_lows)]
    errors_high = [h - o for o, h in zip(obs_vals, ci_highs)]
    ax.errorbar(obs_vals, y_pos, xerr=[errors_low, errors_high],
                fmt='o', color=PAL[0], capsize=5, markersize=8, linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_list)
    ax.set_xlabel("Cohen's d (High Income − Developing)")
    ax.set_title(f"{paper}: Income-Group Effect Sizes with Bootstrap 95% CIs")
    ax.invert_yaxis()
    fig.savefig(out_dir / 'fig_bootstrap_ci.png')
    plt.close(fig)

    # ── 1c. Cluster stability ──────────────────────────────────────────────
    print(f"\n  1c. Cluster stability (silhouette scores, k=2–7)")
    country = dfc.groupby('jurisdiction')[dims].mean()
    country = country[dfc.groupby('jurisdiction').size() >= 3]

    if len(country) >= 10:
        scaler = StandardScaler()
        X = scaler.fit_transform(country.values)

        sil_scores = {}
        inertias = {}
        for k in range(2, 8):
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            sil_scores[k] = round(sil, 4)
            inertias[k] = round(km.inertia_, 2)
            print(f"    k={k}: silhouette={sil:.4f}, inertia={km.inertia_:.1f}")

        best_k = max(sil_scores, key=sil_scores.get)
        print(f"    Best k by silhouette: {best_k} (score={sil_scores[best_k]:.4f})")

        results['cluster_stability'] = {
            'silhouette_scores': sil_scores,
            'inertias': inertias,
            'best_k': best_k,
        }

        # Elbow + silhouette plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ks = list(sil_scores.keys())
        ax1.plot(ks, [inertias[k] for k in ks], 'o-', color=PAL[0], linewidth=2)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (within-cluster SS)')
        ax1.set_title('Elbow Plot')
        ax1.set_xticks(ks)

        colors_bar = [PAL[2] if k == best_k else PAL[0] for k in ks]
        ax2.bar(ks, [sil_scores[k] for k in ks], color=colors_bar, edgecolor='white')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title(f'Silhouette Analysis (best k={best_k})')
        ax2.set_xticks(ks)

        fig.suptitle(f'{paper}: Cluster Stability', fontweight='bold')
        plt.tight_layout()
        fig.savefig(out_dir / 'fig_cluster_stability.png')
        plt.close(fig)

    # Save
    with open(out_dir / 'robustness_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    # Summary table CSV
    sens_df = pd.DataFrame(sensitivity).T
    sens_df.to_csv(out_dir / 'sensitivity_table.csv')

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. MULTILEVEL / HIERARCHICAL MODELS
# ═════════════════════════════════════════════════════════════════════════════

def multilevel_models(df: pd.DataFrame, paper: str, dims: list,
                      composite: str, out_dir: Path):
    """
    Random-intercepts model: score ~ fixed_effects + (1 | country)
    Compare with pooled OLS using log-likelihood.
    """
    section = f"2. MULTILEVEL MODELS — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international'] & df['year'].between(2017, 2025)].copy()
    dfc = dfc[dfc['gdp_pc'] > 0].copy()
    dfc['log_gdp_pc'] = np.log(dfc['gdp_pc'])
    dfc['year_c'] = dfc['year'] - 2021
    dfc['is_binding'] = dfc['binding_nature'].isin(['Binding regulation', 'Hard law']).astype(int)
    dfc['is_good_text'] = (dfc['text_quality'] == 'good').astype(int)

    # Need at least 2 observations per group for mixed models
    counts = dfc['jurisdiction'].value_counts()
    valid_j = counts[counts >= 2].index
    dfc = dfc[dfc['jurisdiction'].isin(valid_j)].copy()

    results = {}

    # ── 2a. Country-level ICC (null model) ─────────────────────────────────
    print(f"\n  2a. Null model: {composite} ~ 1 + (1 | jurisdiction)")
    try:
        null_model = smf.mixedlm(
            f'{composite} ~ 1',
            data=dfc,
            groups=dfc['jurisdiction'],
        ).fit(reml=True)

        var_country = float(null_model.cov_re.iloc[0, 0])
        var_residual = float(null_model.scale)
        icc = var_country / (var_country + var_residual)

        print(f"    Country variance:  {var_country:.4f}")
        print(f"    Residual variance: {var_residual:.4f}")
        print(f"    ICC (country):     {icc:.4f}")
        print(f"    → {icc*100:.1f}% of variance is between countries")

        results['null_model'] = {
            'var_country': round(var_country, 5),
            'var_residual': round(var_residual, 5),
            'icc': round(icc, 4),
            'n_obs': len(dfc),
            'n_groups': dfc['jurisdiction'].nunique(),
            'aic': round(null_model.aic, 2) if hasattr(null_model, 'aic') else None,
            'bic': round(null_model.bic, 2) if hasattr(null_model, 'bic') else None,
        }
    except Exception as e:
        print(f"    Null model failed: {e}")
        results['null_model'] = {'error': str(e)}

    # ── 2b. Full mixed model ───────────────────────────────────────────────
    formula = f'{composite} ~ log_gdp_pc + year_c + is_binding + is_good_text'
    print(f"\n  2b. Full model: {formula} + (1 | jurisdiction)")

    try:
        # Mixed model (random intercepts)
        mixed = smf.mixedlm(
            formula, data=dfc, groups=dfc['jurisdiction'],
        ).fit(reml=True)

        print(f"\n    Random-intercepts model:")
        print(f"    {'Variable':<20s} {'β':>8s} {'SE':>8s} {'z':>8s} {'p':>10s}")
        print(f"    {'─'*56}")

        mixed_coefs = {}
        for var in mixed.fe_params.index:
            b = mixed.fe_params[var]
            se = mixed.bse_fe[var]
            z = mixed.tvalues[var]
            p = mixed.pvalues[var]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {var:<20s} {b:>+8.4f} {se:>8.4f} {z:>8.3f} {p:>9.5f} {sig}")
            mixed_coefs[var] = {
                'beta': round(float(b), 5), 'se': round(float(se), 5),
                'z': round(float(z), 4), 'p': round(float(p), 6),
            }

        var_re = float(mixed.cov_re.iloc[0, 0])
        var_resid = float(mixed.scale)
        print(f"\n    Random effects:")
        print(f"      Country variance: {var_re:.4f}")
        print(f"      Residual:         {var_resid:.4f}")

        results['mixed_model'] = {
            'coefficients': mixed_coefs,
            'var_country': round(var_re, 5),
            'var_residual': round(var_resid, 5),
            'n_obs': int(mixed.nobs),
            'n_groups': len(mixed.random_effects),
            'converged': mixed.converged,
            'llf': round(float(mixed.llf), 2),
        }

    except Exception as e:
        print(f"    Mixed model failed: {e}")
        results['mixed_model'] = {'error': str(e)}

    # ── 2c. Comparison with pooled OLS ─────────────────────────────────────
    print(f"\n  2c. Pooled OLS (for comparison)")
    try:
        ols = smf.ols(formula, data=dfc).fit()

        print(f"    {'Variable':<20s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
        print(f"    {'─'*56}")
        ols_coefs = {}
        for var in ols.params.index:
            b = ols.params[var]
            se = ols.bse[var]
            t = ols.tvalues[var]
            p = ols.pvalues[var]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {var:<20s} {b:>+8.4f} {se:>8.4f} {t:>8.3f} {p:>9.5f} {sig}")
            ols_coefs[var] = {
                'beta': round(float(b), 5), 'se': round(float(se), 5),
                't': round(float(t), 4), 'p': round(float(p), 6),
            }

        print(f"\n    OLS R²: {ols.rsquared:.4f}, Adj R²: {ols.rsquared_adj:.4f}")
        print(f"    OLS AIC: {ols.aic:.1f}, BIC: {ols.bic:.1f}")

        results['ols_model'] = {
            'coefficients': ols_coefs,
            'r_squared': round(float(ols.rsquared), 4),
            'adj_r_squared': round(float(ols.rsquared_adj), 4),
            'aic': round(float(ols.aic), 2),
            'bic': round(float(ols.bic), 2),
            'llf': round(float(ols.llf), 2),
            'n_obs': int(ols.nobs),
        }

        # Compare: LR test (mixed vs OLS)
        if 'llf' in results.get('mixed_model', {}):
            llf_mixed = results['mixed_model']['llf']
            llf_ols = results['ols_model']['llf']
            lr_stat = 2 * (llf_mixed - llf_ols)
            # Chi-squared with df=1 (one additional parameter: country variance)
            lr_p = sp_stats.chi2.sf(max(lr_stat, 0), df=1)
            print(f"\n  2d. Likelihood-ratio test (Mixed vs OLS):")
            print(f"    LR statistic: {lr_stat:.2f}")
            print(f"    p-value:      {lr_p:.6f}")
            sig = '***' if lr_p < 0.001 else '**' if lr_p < 0.01 else '*' if lr_p < 0.05 else ''
            print(f"    → {'Mixed model significantly better' if lr_p < 0.05 else 'Models not significantly different'} {sig}")

            results['lr_test'] = {
                'lr_statistic': round(lr_stat, 4),
                'p_value': round(lr_p, 6),
                'df': 1,
                'mixed_better': lr_p < 0.05,
            }

    except Exception as e:
        print(f"    OLS failed: {e}")
        results['ols_model'] = {'error': str(e)}

    # ── 2e. Country random effects (BLUP) ──────────────────────────────────
    if 'mixed_model' in results and 'error' not in results['mixed_model']:
        try:
            re = mixed.random_effects
            re_df = pd.DataFrame({
                'jurisdiction': list(re.keys()),
                'random_intercept': [float(v.iloc[0]) for v in re.values()],
            })
            re_df['income'] = re_df['jurisdiction'].map(lambda j: get_metadata(j)['income_label'])
            re_df = re_df.sort_values('random_intercept', ascending=False)
            re_df.to_csv(out_dir / 'country_random_effects.csv', index=False)

            # Plot top/bottom random effects
            fig, ax = plt.subplots(figsize=(10, 8))
            top_bot = pd.concat([re_df.head(10), re_df.tail(10)]).drop_duplicates()
            top_bot = top_bot.sort_values('random_intercept')
            colors_re = [INCOME_COLORS.get(inc, PAL[0]) for inc in top_bot['income']]
            ax.barh(range(len(top_bot)), top_bot['random_intercept'], color=colors_re, edgecolor='white')
            ax.set_yticks(range(len(top_bot)))
            ax.set_yticklabels(top_bot['jurisdiction'])
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Country Random Intercept (BLUP)')
            ax.set_title(f'{paper}: Country-Level Deviations from Fixed Effects')
            # Legend
            from matplotlib.patches import Patch
            handles = [Patch(color=INCOME_COLORS[ig], label=ig)
                       for ig in ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
                       if ig in top_bot['income'].values]
            ax.legend(handles=handles, fontsize=8, loc='lower right')
            fig.savefig(out_dir / 'fig_country_random_effects.png')
            plt.close(fig)

            print(f"\n  Top 5 country random intercepts:")
            for _, row in re_df.head(5).iterrows():
                print(f"    {row['jurisdiction']:<25s} {row['random_intercept']:>+.4f}")
            print(f"  Bottom 5:")
            for _, row in re_df.tail(5).iterrows():
                print(f"    {row['jurisdiction']:<25s} {row['random_intercept']:>+.4f}")

        except Exception as e:
            print(f"    Random effects extraction failed: {e}")

    # Save
    with open(out_dir / 'multilevel_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. PCA / FACTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def pca_analysis(df: pd.DataFrame, out_dir: Path):
    """
    PCA on all 10 dimensions.
    Key question: Do C1–C5 and E1–E5 form two distinct constructs?
    """
    section = "3. PCA / FACTOR ANALYSIS"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use all policies (not just country-level)
    X = df[ALL_DIMS].values
    labels = [ALL_LABELS[d] for d in ALL_DIMS]

    # Standardise
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Full PCA
    pca = PCA()
    pca.fit(X_std)

    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)

    print(f"\n  Explained variance by component:")
    print(f"  {'PC':>4s} {'Eigenvalue':>12s} {'Variance %':>12s} {'Cumulative %':>14s}")
    results = {'components': []}
    for i, (ev, ve, cv) in enumerate(zip(pca.explained_variance_, var_explained, cum_var)):
        print(f"  PC{i+1:>2d} {ev:>12.4f} {ve*100:>11.2f}% {cv*100:>13.2f}%")
        results['components'].append({
            'pc': i + 1,
            'eigenvalue': round(float(ev), 4),
            'variance_pct': round(float(ve) * 100, 2),
            'cumulative_pct': round(float(cv) * 100, 2),
        })

    # Kaiser criterion: eigenvalue > 1
    n_kaiser = sum(1 for ev in pca.explained_variance_ if ev > 1)
    print(f"\n  Kaiser criterion (eigenvalue > 1): {n_kaiser} components")
    print(f"  → {'Supports' if n_kaiser == 2 else 'Does NOT support'} two-factor structure")
    results['n_kaiser'] = n_kaiser

    # ── Scree plot ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scree
    ax1.plot(range(1, 11), pca.explained_variance_, 'o-', color=PAL[0], linewidth=2, markersize=8)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Kaiser criterion (λ=1)')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')
    ax1.set_xticks(range(1, 11))
    ax1.legend()

    # Cumulative variance
    ax2.bar(range(1, 11), var_explained * 100, color=PAL[1], alpha=0.7, label='Individual')
    ax2.plot(range(1, 11), cum_var * 100, 'o-', color=PAL[3], linewidth=2, label='Cumulative')
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance Explained (%)')
    ax2.set_title('Explained Variance')
    ax2.set_xticks(range(1, 11))
    ax2.legend()

    fig.suptitle('PCA: Dimensionality of AI Governance Scores', fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_pca_scree.png')
    plt.close(fig)

    # ── Loadings matrix (first 3 PCs) ──────────────────────────────────────
    n_show = min(3, n_kaiser + 1)
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        index=labels,
        columns=[f'PC{i+1}' for i in range(n_show)],
    ).round(3)

    print(f"\n  Factor loadings (first {n_show} PCs):")
    print(loadings.to_string())
    loadings.to_csv(out_dir / 'pca_loadings.csv')
    results['loadings'] = loadings.to_dict()

    # ── Loadings heatmap ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-0.8, vmax=0.8, ax=ax, linewidths=0.5)
    ax.set_title(f'PCA Loadings (first {n_show} components)')
    ax.set_ylabel('')

    # Add a box around C1-C5 and E1-E5
    ax.axhline(y=5, color='black', linewidth=2)

    fig.savefig(out_dir / 'fig_pca_loadings.png')
    plt.close(fig)

    # ── Two-factor structure test ──────────────────────────────────────────
    # Check if PC1 loads on all dims, PC2 separates capacity from ethics
    pc1_cap = np.mean(np.abs(pca.components_[0][:5]))
    pc1_eth = np.mean(np.abs(pca.components_[0][5:]))
    if n_show >= 2:
        pc2_cap = np.mean(pca.components_[1][:5])
        pc2_eth = np.mean(pca.components_[1][5:])
        separation = abs(pc2_cap - pc2_eth)
    else:
        pc2_cap = pc2_eth = separation = 0

    print(f"\n  Two-factor structure analysis:")
    print(f"    PC1 mean |loading| — Capacity dims: {pc1_cap:.3f}")
    print(f"    PC1 mean |loading| — Ethics dims:   {pc1_eth:.3f}")
    print(f"    → PC1 is a {'general governance' if abs(pc1_cap - pc1_eth) < 0.1 else 'mixed'} factor")
    if n_show >= 2:
        print(f"    PC2 mean loading — Capacity dims:  {pc2_cap:+.3f}")
        print(f"    PC2 mean loading — Ethics dims:     {pc2_eth:+.3f}")
        print(f"    PC2 separation (|cap - eth|):       {separation:.3f}")
        print(f"    → PC2 {'DOES' if separation > 0.3 else 'does NOT'} separate capacity from ethics")

    results['two_factor_test'] = {
        'pc1_cap_mean_abs_loading': round(pc1_cap, 4),
        'pc1_eth_mean_abs_loading': round(pc1_eth, 4),
        'pc2_cap_mean_loading': round(pc2_cap, 4),
        'pc2_eth_mean_loading': round(pc2_eth, 4),
        'pc2_separation': round(separation, 4),
        'supports_two_factors': separation > 0.3,
    }

    # ── Biplot (PC1 vs PC2) ────────────────────────────────────────────────
    if n_show >= 2:
        scores = pca.transform(X_std)
        fig, ax = plt.subplots(figsize=(10, 9))

        # Policy points coloured by income
        for ig, color in INCOME_COLORS.items():
            mask = df['income_group'] == ig
            if mask.any():
                ax.scatter(scores[mask, 0], scores[mask, 1], c=[color],
                           alpha=0.15, s=10, label=ig)

        # Loading vectors
        scale = 3  # scale arrows for visibility
        for i, label in enumerate(labels):
            ax.annotate('', xy=(pca.components_[0][i] * scale,
                                pca.components_[1][i] * scale),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            # Colour C dims blue, E dims red
            txt_color = 'navy' if i < 5 else 'darkred'
            ax.annotate(label,
                        xy=(pca.components_[0][i] * scale * 1.12,
                            pca.components_[1][i] * scale * 1.12),
                        fontsize=9, fontweight='bold', color=txt_color,
                        ha='center', va='center')

        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)')
        ax.set_title('PCA Biplot: Capacity (blue) vs Ethics (red) Dimensions')
        ax.axhline(0, color='gray', alpha=0.3)
        ax.axvline(0, color='gray', alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        fig.savefig(out_dir / 'fig_pca_biplot.png')
        plt.close(fig)

    # ── Cronbach's alpha for each construct ────────────────────────────────
    print(f"\n  Internal consistency (Cronbach's α):")
    for construct_name, construct_dims in [('Capacity (C1–C5)', CAP_DIMS),
                                            ('Ethics (E1–E5)', ETH_DIMS),
                                            ('All 10 dimensions', ALL_DIMS)]:
        items = df[construct_dims].values
        k = items.shape[1]
        item_vars = items.var(axis=0, ddof=1)
        total_var = items.sum(axis=1).var(ddof=1)
        alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
        print(f"    {construct_name:<25s} α = {alpha:.3f}")
        results[f'cronbach_alpha_{construct_name.split("(")[0].strip().lower()}'] = round(alpha, 4)

    # Save
    with open(out_dir / 'pca_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 4. CONVERGENCE / DIVERGENCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def convergence_analysis(df: pd.DataFrame, paper: str, dims: list,
                         composite: str, out_dir: Path):
    """
    Are developing countries catching up or falling further behind?
    - Income × year interaction in OLS
    - Separate slopes by income group
    - Gap trajectory over time
    """
    section = f"4. CONVERGENCE / DIVERGENCE — {paper}"
    print(f"\n{'='*70}\n{section}\n{'='*70}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfc = df[~df['is_international'] & df['year'].between(2017, 2025)].copy()
    dfc['year_c'] = dfc['year'] - 2021
    dfc['is_hi'] = (dfc['income_binary'] == 'High income').astype(int)
    dfc['is_good_text'] = (dfc['text_quality'] == 'good').astype(int)
    dfc['is_binding'] = dfc['binding_nature'].isin(['Binding regulation', 'Hard law']).astype(int)
    dfc['log_gdp_pc'] = np.log(dfc['gdp_pc'].replace(0, np.nan))
    dfc = dfc.dropna(subset=['log_gdp_pc'])

    results = {}

    # ── 4a. OLS with interaction ───────────────────────────────────────────
    formula = f'{composite} ~ year_c * is_hi + is_good_text + is_binding'
    print(f"\n  4a. Interaction model: {formula}")

    try:
        model = smf.ols(formula, data=dfc).fit()

        print(f"\n    {'Variable':<25s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
        print(f"    {'─'*62}")
        coefs = {}
        for var in model.params.index:
            b = model.params[var]
            se = model.bse[var]
            t = model.tvalues[var]
            p = model.pvalues[var]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {var:<25s} {b:>+8.4f} {se:>8.4f} {t:>8.3f} {p:>9.5f} {sig}")
            coefs[var] = {
                'beta': round(float(b), 5), 'se': round(float(se), 5),
                't': round(float(t), 4), 'p': round(float(p), 6),
            }
        print(f"\n    R²={model.rsquared:.4f}, Adj R²={model.rsquared_adj:.4f}")

        # Interpret the interaction term
        interaction_key = 'year_c:is_hi'
        if interaction_key in coefs:
            inter = coefs[interaction_key]
            if inter['p'] < 0.05:
                direction = 'widening' if inter['beta'] > 0 else 'narrowing'
                print(f"\n    ⚡ Income × Year interaction is SIGNIFICANT (p={inter['p']:.4f})")
                print(f"       → The gap is {direction} over time (β={inter['beta']:+.4f}/yr)")
            else:
                print(f"\n    → Income × Year interaction NOT significant (p={inter['p']:.4f})")
                print(f"       → No evidence of convergence or divergence")

        results['interaction_model'] = {
            'coefficients': coefs,
            'r_squared': round(float(model.rsquared), 4),
            'adj_r_squared': round(float(model.rsquared_adj), 4),
            'n_obs': int(model.nobs),
        }

    except Exception as e:
        print(f"    Interaction model failed: {e}")
        results['interaction_model'] = {'error': str(e)}

    # ── 4b. Separate temporal slopes ───────────────────────────────────────
    print(f"\n  4b. Separate temporal slopes by income group:")
    slope_results = {}
    for ig in ['High income', 'Developing']:
        sub = dfc[dfc['income_binary'] == ig]
        if len(sub) >= 20:
            x = sub['year'].values.astype(float)
            y = sub[composite].values
            slope, intercept, r, p, se = sp_stats.linregress(x, y)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"    {ig:<15s}  slope={slope:+.4f}/yr  (r²={r**2:.4f}, p={p:.4f}{sig}, n={len(sub)})")
            slope_results[ig] = {
                'slope': round(slope, 5), 'intercept': round(intercept, 4),
                'r_squared': round(r**2, 4), 'p': round(p, 6), 'n': len(sub),
            }
    results['separate_slopes'] = slope_results

    # Per-dimension slopes
    print(f"\n  Per-dimension temporal slopes:")
    print(f"    {'Dimension':<25s} {'HI slope':>10s} {'Dev slope':>10s} {'Converging?':>12s}")
    dim_slopes = {}
    for d in dims:
        label = CAP_LABELS.get(d) or ETH_LABELS.get(d, d)
        slopes = {}
        for ig in ['High income', 'Developing']:
            sub = dfc[dfc['income_binary'] == ig]
            if len(sub) >= 20:
                s, _, _, p, _ = sp_stats.linregress(sub['year'].values.astype(float), sub[d].values)
                slopes[ig] = round(s, 5)
        if len(slopes) == 2:
            converging = slopes['Developing'] > slopes['High income']
            flag = '↑ Catching up' if converging else '↓ Falling behind'
            print(f"    {label:<25s} {slopes['High income']:>+10.4f} {slopes['Developing']:>+10.4f} {flag:>12s}")
            dim_slopes[d] = {**slopes, 'converging': converging}
    results['dimension_slopes'] = dim_slopes

    # ── 4c. Gap trajectory ─────────────────────────────────────────────────
    print(f"\n  4c. Income-group gap trajectory (HI mean − Dev mean by year):")
    yearly_gap = []
    for yr in range(2017, 2026):
        hi = dfc[(dfc['income_binary'] == 'High income') & (dfc['year'] == yr)][composite]
        dev = dfc[(dfc['income_binary'] == 'Developing') & (dfc['year'] == yr)][composite]
        if len(hi) >= 5 and len(dev) >= 3:
            gap = hi.mean() - dev.mean()
            yearly_gap.append({
                'year': yr, 'hi_mean': round(hi.mean(), 4),
                'dev_mean': round(dev.mean(), 4), 'gap': round(gap, 4),
                'n_hi': len(hi), 'n_dev': len(dev),
            })
            print(f"    {yr}  HI={hi.mean():.3f}  Dev={dev.mean():.3f}  Gap={gap:+.3f}  "
                  f"(n_hi={len(hi)}, n_dev={len(dev)})")

    results['gap_trajectory'] = yearly_gap

    # Linear trend on the gap
    if len(yearly_gap) >= 4:
        gap_years = [g['year'] for g in yearly_gap]
        gap_vals = [g['gap'] for g in yearly_gap]
        g_slope, g_int, g_r, g_p, g_se = sp_stats.linregress(gap_years, gap_vals)
        print(f"\n    Gap trend: slope={g_slope:+.4f}/yr (r²={g_r**2:.4f}, p={g_p:.4f})")
        direction = 'widening' if g_slope > 0 else 'narrowing'
        sig = '***' if g_p < 0.001 else '**' if g_p < 0.01 else '*' if g_p < 0.05 else ''
        sig_text = 'significantly' if g_p < 0.05 else 'not significantly'
        print(f"    → Gap is {sig_text} {direction} {sig}")

        results['gap_trend'] = {
            'slope': round(g_slope, 5), 'r_squared': round(g_r**2, 4),
            'p': round(g_p, 6), 'direction': direction,
            'significant': g_p < 0.05,
        }

    # ── Convergence figure ─────────────────────────────────────────────────
    if yearly_gap:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: separate trends
        for ig, color, marker in [('High income', PAL[0], 'o'), ('Developing', PAL[2], 's')]:
            sub = dfc[dfc['income_binary'] == ig]
            ym = sub.groupby('year')[composite].agg(['mean', 'std', 'count'])
            ym['ci95'] = 1.96 * ym['std'] / np.sqrt(ym['count'])
            ax1.fill_between(ym.index, ym['mean'] - ym['ci95'], ym['mean'] + ym['ci95'],
                             alpha=0.15, color=color)
            ax1.plot(ym.index, ym['mean'], f'{marker}-', color=color, label=ig,
                     markersize=6, linewidth=2)

            # Trend line
            if ig in slope_results:
                sr = slope_results[ig]
                x_line = np.linspace(2017, 2025, 100)
                ax1.plot(x_line, sr['slope'] * (x_line - 2021) + sr['intercept'] + sr['slope'] * 2021,
                         '--', color=color, alpha=0.5)

        ax1.set_xlabel('Year')
        ax1.set_ylabel(f'{composite.replace("_", " ").title()} (0–4)')
        ax1.set_title('(a) Score Trends by Income Level')
        ax1.legend(fontsize=9)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Right: gap over time
        gap_years = [g['year'] for g in yearly_gap]
        gap_vals = [g['gap'] for g in yearly_gap]
        ax2.bar(gap_years, gap_vals, color=PAL[4], edgecolor='white', alpha=0.8)
        if 'gap_trend' in results:
            gt = results['gap_trend']
            x_line = np.linspace(min(gap_years), max(gap_years), 100)
            y_line = gt['slope'] * x_line + (np.mean(gap_vals) - gt['slope'] * np.mean(gap_years))
            ax2.plot(x_line, y_line, '--', color='black', linewidth=2,
                     label=f"Trend: {gt['slope']:+.3f}/yr (p={gt['p']:.3f})")
            ax2.legend(fontsize=9)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('HI − Developing Gap')
        ax2.set_title('(b) Income-Group Gap Trajectory')
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        fig.suptitle(f'{paper}: Convergence / Divergence Analysis', fontweight='bold')
        plt.tight_layout()
        fig.savefig(out_dir / 'fig_convergence.png')
        plt.close(fig)

    # ── Dimension convergence heatmap ──────────────────────────────────────
    if dim_slopes:
        fig, ax = plt.subplots(figsize=(10, 5))
        heat_data = pd.DataFrame(dim_slopes).T
        heat_data.index = [CAP_LABELS.get(d) or ETH_LABELS.get(d, d) for d in heat_data.index]
        # Keep only numeric columns for heatmap
        heat_numeric = heat_data[['High income', 'Developing']].astype(float)
        heat_numeric['Gap change'] = heat_numeric['High income'] - heat_numeric['Developing']
        sns.heatmap(heat_numeric, annot=True, fmt='+.4f', cmap='RdYlGn_r', center=0,
                    ax=ax, linewidths=0.5)
        ax.set_title(f'{paper}: Dimension-Level Temporal Slopes by Income Group')
        ax.set_ylabel('')
        fig.savefig(out_dir / 'fig_dimension_convergence.png')
        plt.close(fig)

    # Save
    with open(out_dir / 'convergence_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run the complete Phase 3b advanced analysis pipeline."""
    start = datetime.now()
    print("=" * 70)
    print("PHASE 3b: ADVANCED ANALYSIS PIPELINE")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for d in [OUT_P1, OUT_P2, OUT_ADV]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # ── 1. Robustness checks (per paper) ───────────────────────────────────
    print(f"\n{'#'*70}")
    print("PAPER 1: CAPACITY — ROBUSTNESS & ADVANCED")
    print(f"{'#'*70}")
    robustness_checks(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    multilevel_models(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)
    convergence_analysis(df, 'Paper 1: Capacity', CAP_DIMS, 'capacity_score', OUT_P1)

    print(f"\n{'#'*70}")
    print("PAPER 2: ETHICS — ROBUSTNESS & ADVANCED")
    print(f"{'#'*70}")
    robustness_checks(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    multilevel_models(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)
    convergence_analysis(df, 'Paper 2: Ethics', ETH_DIMS, 'ethics_score', OUT_P2)

    # ── 3. PCA (shared — uses all 10 dimensions) ──────────────────────────
    print(f"\n{'#'*70}")
    print("SHARED: PCA / FACTOR ANALYSIS")
    print(f"{'#'*70}")
    pca_analysis(df, OUT_ADV)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*70}")
    print(f"PHASE 3b COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Paper 1 robustness: {OUT_P1}")
    print(f"  Paper 2 robustness: {OUT_P2}")
    print(f"  Shared advanced:    {OUT_ADV}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_all()
