"""
Phase 2.3: Inter-Rater Reliability Analysis
=============================================
Computes agreement statistics across the 3-model LLM ensemble:
  - Intraclass Correlation Coefficient (ICC 2,1 — two-way random, single measures)
  - Pearson & Spearman correlations (pairwise)
  - Weighted Cohen's kappa (pairwise, linear weights)
  - Fleiss' kappa (all 3 raters)
  - Mean absolute deviation, spread distributions
  - Dimension-level and composite-level agreement
  - Bland-Altman style bias analysis

Output:
  data/analysis/inter_rater_report.json   – full statistics
  prints summary to console

Usage:
  python src/analysis/inter_rater.py
"""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SCORES_RAW = ROOT / 'data' / 'analysis' / 'scores_raw.jsonl'
REPORT_PATH = ROOT / 'data' / 'analysis' / 'inter_rater_report.json'

DIMS = [
    'c1_clarity', 'c2_resources', 'c3_authority', 'c4_accountability', 'c5_coherence',
    'e1_framework', 'e2_rights', 'e3_governance', 'e4_operationalisation', 'e5_inclusion',
]
CAP_DIMS = DIMS[:5]
ETH_DIMS = DIMS[5:]
MODEL_PAIRS = [('A', 'B'), ('A', 'C'), ('B', 'C')]


# ─── Load Data ─────────────────────────────────────────────────────────────────

def load_scores():
    """Load raw scores into {entry_id: {model: {dim: score}}} structure."""
    data = defaultdict(dict)
    meta = defaultdict(dict)

    with open(SCORES_RAW, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            eid = rec['entry_id']
            model = rec['model']

            # Extract dimension scores
            scores = {}
            for d in DIMS:
                s = rec['scores'].get(d, {})
                if isinstance(s, dict) and 'score' in s:
                    scores[d] = s['score']

            # Composite scores
            cap = [scores[d] for d in CAP_DIMS if d in scores]
            eth = [scores[d] for d in ETH_DIMS if d in scores]
            if cap:
                scores['capacity_score'] = round(sum(cap) / len(cap), 2)
            if eth:
                scores['ethics_score'] = round(sum(eth) / len(eth), 2)
            if cap and eth:
                scores['overall_score'] = round(
                    (scores['capacity_score'] + scores['ethics_score']) / 2, 2
                )

            data[eid][model] = scores
            meta[eid] = {
                'title': rec.get('title', ''),
                'jurisdiction': rec.get('jurisdiction', ''),
                'text_quality': rec.get('text_quality', ''),
            }

    return data, meta


# ─── Statistical Functions ─────────────────────────────────────────────────────

def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def variance(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs) if xs else 0

def std(xs):
    return math.sqrt(variance(xs)) if xs else 0

def pearson(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

def spearman(xs, ys):
    """Spearman rank correlation."""
    def rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2  # 1-based average
            for k in range(i, j):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j
        return ranks

    n = len(xs)
    if n < 3:
        return None
    rx = rank(xs)
    ry = rank(ys)
    return pearson(rx, ry)

def weighted_kappa_linear(xs, ys, max_score=4):
    """Cohen's weighted kappa with linear weights."""
    n = len(xs)
    if n == 0:
        return None

    # Build confusion matrix
    cats = list(range(max_score + 1))
    k = len(cats)
    observed = [[0] * k for _ in range(k)]
    for x, y in zip(xs, ys):
        xi = min(int(round(x)), max_score)
        yi = min(int(round(y)), max_score)
        observed[xi][yi] += 1

    # Weight matrix (linear)
    w = [[abs(i - j) / max_score for j in range(k)] for i in range(k)]

    # Marginals
    row_sum = [sum(observed[i]) for i in range(k)]
    col_sum = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    # Expected
    expected = [[row_sum[i] * col_sum[j] / n for j in range(k)] for i in range(k)]

    # Weighted observed and expected
    po = sum(w[i][j] * observed[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w[i][j] * expected[i][j] for i in range(k) for j in range(k)) / n

    if pe == 0:
        return 1.0 if po == 0 else None
    return 1 - po / pe

def fleiss_kappa(ratings_matrix, n_categories=5):
    """
    Fleiss' kappa for 3+ raters.
    ratings_matrix: list of dicts {category: count} per subject.
    """
    N = len(ratings_matrix)  # number of subjects
    if N == 0:
        return None

    n = sum(ratings_matrix[0].values())  # number of raters per subject
    if n <= 1:
        return None

    cats = list(range(n_categories))

    # Proportion of all assignments to each category
    p_j = {}
    for c in cats:
        total = sum(row.get(c, 0) for row in ratings_matrix)
        p_j[c] = total / (N * n)

    # P_i for each subject
    P_i = []
    for row in ratings_matrix:
        s = sum(row.get(c, 0) ** 2 for c in cats)
        P_i.append((s - n) / (n * (n - 1)) if n > 1 else 0)

    P_bar = mean(P_i)
    Pe = sum(p ** 2 for p in p_j.values())

    if Pe >= 1.0:
        return 1.0 if P_bar >= 1.0 else None
    return (P_bar - Pe) / (1 - Pe)

def icc_2_1(data_matrix):
    """
    ICC(2,1) — Two-way random, single measures, absolute agreement.
    data_matrix: list of lists, shape [n_subjects x n_raters].
    Each row is a subject, each column is a rater.
    Missing values should be excluded before calling.
    """
    n = len(data_matrix)
    if n < 3:
        return None

    k = len(data_matrix[0])  # number of raters
    if k < 2:
        return None

    # Grand mean
    grand = mean([x for row in data_matrix for x in row])

    # Subject means
    subj_means = [mean(row) for row in data_matrix]

    # Rater means
    rater_means = [mean([data_matrix[i][j] for i in range(n)]) for j in range(k)]

    # Mean squares
    # Between subjects
    BMS = k * sum((sm - grand) ** 2 for sm in subj_means) / (n - 1)

    # Between raters (judges)
    JMS = n * sum((rm - grand) ** 2 for rm in rater_means) / (k - 1) if k > 1 else 0

    # Residual (error)
    SSE = sum(
        (data_matrix[i][j] - subj_means[i] - rater_means[j] + grand) ** 2
        for i in range(n) for j in range(k)
    )
    EMS = SSE / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0

    # ICC(2,1) formula
    denom = BMS + (k - 1) * EMS + k * (JMS - EMS) / n
    if denom == 0:
        return None
    return (BMS - EMS) / denom


# ─── Main Analysis ─────────────────────────────────────────────────────────────

def run_analysis():
    print("=" * 70)
    print("INTER-RATER RELIABILITY ANALYSIS")
    print("=" * 70)

    data, meta = load_scores()

    # Only use entries scored by all 3 models
    three_model = {eid: scores for eid, scores in data.items()
                   if all(m in scores for m in ['A', 'B', 'C'])}
    two_plus = {eid: scores for eid, scores in data.items() if len(scores) >= 2}

    print(f"\n  Total entries with scores: {len(data)}")
    print(f"  Entries with all 3 models: {len(three_model)}")
    print(f"  Entries with 2+ models:    {len(two_plus)}")

    report = {
        'n_total': len(data),
        'n_three_models': len(three_model),
        'n_two_plus': len(two_plus),
    }

    # ── 1. ICC(2,1) per dimension ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("1. INTRACLASS CORRELATION — ICC(2,1)")
    print(f"   Two-way random, single measures, absolute agreement")
    print(f"{'─' * 70}")

    all_measures = DIMS + ['capacity_score', 'ethics_score', 'overall_score']
    icc_results = {}

    for measure in all_measures:
        matrix = []
        for eid in three_model:
            row = []
            for m in ['A', 'B', 'C']:
                score = three_model[eid][m].get(measure)
                if score is not None:
                    row.append(score)
            if len(row) == 3:
                matrix.append(row)

        if len(matrix) >= 10:
            icc_val = icc_2_1(matrix)
            icc_results[measure] = round(icc_val, 3) if icc_val is not None else None
        else:
            icc_results[measure] = None

    # Print ICC table
    print(f"\n  {'Dimension':<25s} {'ICC(2,1)':>8s}  Interpretation")
    print(f"  {'─' * 55}")
    for m in all_measures:
        val = icc_results[m]
        if val is not None:
            if val >= 0.75:
                interp = "Excellent"
            elif val >= 0.60:
                interp = "Good"
            elif val >= 0.40:
                interp = "Fair"
            else:
                interp = "Poor"
            print(f"  {m:<25s} {val:>8.3f}  {interp}")
        else:
            print(f"  {m:<25s}     N/A")

    report['icc'] = icc_results

    # ── 2. Pairwise Correlations ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("2. PAIRWISE CORRELATIONS (Pearson / Spearman)")
    print(f"{'─' * 70}")

    corr_results = {}
    for m1, m2 in MODEL_PAIRS:
        corr_results[f"{m1}v{m2}"] = {}
        for measure in all_measures:
            xs, ys = [], []
            for eid in two_plus:
                if m1 in two_plus[eid] and m2 in two_plus[eid]:
                    s1 = two_plus[eid][m1].get(measure)
                    s2 = two_plus[eid][m2].get(measure)
                    if s1 is not None and s2 is not None:
                        xs.append(s1)
                        ys.append(s2)

            if len(xs) >= 10:
                p = pearson(xs, ys)
                s = spearman(xs, ys)
                corr_results[f"{m1}v{m2}"][measure] = {
                    'pearson': round(p, 3) if p else None,
                    'spearman': round(s, 3) if s else None,
                    'n': len(xs),
                }

    # Print summary for composites
    print(f"\n  {'Pair':<8s} {'Measure':<20s} {'Pearson':>8s} {'Spearman':>9s} {'N':>6s}")
    print(f"  {'─' * 55}")
    for pair_key in ['AvB', 'AvC', 'BvC']:
        for measure in ['capacity_score', 'ethics_score', 'overall_score']:
            c = corr_results.get(pair_key, {}).get(measure, {})
            if c:
                print(f"  {pair_key:<8s} {measure:<20s} {c['pearson']:>8.3f} {c['spearman']:>9.3f} {c['n']:>6d}")
        print()

    report['pairwise_correlations'] = corr_results

    # ── 3. Weighted Kappa (pairwise) ───────────────────────────────────────────
    print(f"{'─' * 70}")
    print("3. WEIGHTED COHEN'S KAPPA (linear weights, pairwise)")
    print(f"{'─' * 70}")

    kappa_results = {}
    for m1, m2 in MODEL_PAIRS:
        kappa_results[f"{m1}v{m2}"] = {}
        for measure in DIMS:
            xs, ys = [], []
            for eid in two_plus:
                if m1 in two_plus[eid] and m2 in two_plus[eid]:
                    s1 = two_plus[eid][m1].get(measure)
                    s2 = two_plus[eid][m2].get(measure)
                    if s1 is not None and s2 is not None:
                        xs.append(s1)
                        ys.append(s2)

            if len(xs) >= 10:
                k = weighted_kappa_linear(xs, ys)
                kappa_results[f"{m1}v{m2}"][measure] = round(k, 3) if k else None

    print(f"\n  {'Dimension':<25s} {'A vs B':>8s} {'A vs C':>8s} {'B vs C':>8s}")
    print(f"  {'─' * 55}")
    for d in DIMS:
        vals = []
        for pair_key in ['AvB', 'AvC', 'BvC']:
            v = kappa_results.get(pair_key, {}).get(d)
            vals.append(f"{v:.3f}" if v is not None else "  N/A")
        print(f"  {d:<25s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s}")

    report['weighted_kappa'] = kappa_results

    # ── 4. Fleiss' Kappa ───────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("4. FLEISS' KAPPA (3 raters)")
    print(f"{'─' * 70}")

    fleiss_results = {}
    for measure in DIMS:
        ratings = []
        for eid in three_model:
            row = {}
            for m in ['A', 'B', 'C']:
                score = three_model[eid][m].get(measure)
                if score is not None:
                    cat = min(int(round(score)), 4)
                    row[cat] = row.get(cat, 0) + 1
            if sum(row.values()) == 3:
                ratings.append(row)

        if len(ratings) >= 10:
            fk = fleiss_kappa(ratings, n_categories=5)
            fleiss_results[measure] = round(fk, 3) if fk is not None else None

    print(f"\n  {'Dimension':<25s} {'Fleiss κ':>8s}  Interpretation")
    print(f"  {'─' * 55}")
    for d in DIMS:
        val = fleiss_results.get(d)
        if val is not None:
            if val >= 0.61:
                interp = "Substantial"
            elif val >= 0.41:
                interp = "Moderate"
            elif val >= 0.21:
                interp = "Fair"
            elif val >= 0:
                interp = "Slight"
            else:
                interp = "Poor"
            print(f"  {d:<25s} {val:>8.3f}  {interp}")
        else:
            print(f"  {d:<25s}     N/A")

    report['fleiss_kappa'] = fleiss_results

    # ── 5. Mean Absolute Deviation & Spread ────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("5. SCORE SPREAD & MEAN ABSOLUTE DEVIATION")
    print(f"{'─' * 70}")

    spread_results = {}
    for measure in all_measures:
        spreads = []
        mads = []
        for eid in three_model:
            scores = []
            for m in ['A', 'B', 'C']:
                s = three_model[eid][m].get(measure)
                if s is not None:
                    scores.append(s)
            if len(scores) >= 2:
                spreads.append(max(scores) - min(scores))
                avg = mean(scores)
                mads.append(mean([abs(s - avg) for s in scores]))

        if spreads:
            spread_results[measure] = {
                'mean_spread': round(mean(spreads), 3),
                'median_spread': round(sorted(spreads)[len(spreads) // 2], 3),
                'mean_mad': round(mean(mads), 3),
                'pct_exact_agree': round(sum(1 for s in spreads if s == 0) / len(spreads) * 100, 1),
                'pct_within_1': round(sum(1 for s in spreads if s <= 1) / len(spreads) * 100, 1),
            }

    print(f"\n  {'Measure':<25s} {'Spread':>7s} {'MAD':>6s} {'Exact%':>7s} {'≤1pt%':>7s}")
    print(f"  {'─' * 60}")
    for m in all_measures:
        sr = spread_results.get(m, {})
        if sr:
            print(f"  {m:<25s} {sr['mean_spread']:>7.3f} {sr['mean_mad']:>6.3f} "
                  f"{sr['pct_exact_agree']:>6.1f}% {sr['pct_within_1']:>6.1f}%")

    report['spread'] = spread_results

    # ── 6. Model Bias (systematic over/under-scoring) ──────────────────────────
    print(f"\n{'─' * 70}")
    print("6. MODEL BIAS (mean score per model)")
    print(f"{'─' * 70}")

    bias_results = {}
    for m_key in ['A', 'B', 'C']:
        model_scores = defaultdict(list)
        for eid in three_model:
            if m_key in three_model[eid]:
                for measure in all_measures:
                    s = three_model[eid][m_key].get(measure)
                    if s is not None:
                        model_scores[measure].append(s)
        bias_results[m_key] = {m: round(mean(v), 3) for m, v in model_scores.items()}

    print(f"\n  {'Measure':<25s} {'Model A':>8s} {'Model B':>8s} {'Model C':>8s} {'Δmax':>6s}")
    print(f"  {'─' * 60}")
    for measure in all_measures:
        vals = [bias_results[m].get(measure, 0) for m in ['A', 'B', 'C']]
        delta = max(vals) - min(vals)
        print(f"  {measure:<25s} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {delta:>6.3f}")

    report['model_bias'] = bias_results

    # ── 7. Agreement by Text Quality ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("7. AGREEMENT BY TEXT QUALITY")
    print(f"{'─' * 70}")

    quality_groups = defaultdict(list)
    for eid in three_model:
        q = meta[eid].get('text_quality', 'unknown')
        scores = []
        for m in ['A', 'B', 'C']:
            s = three_model[eid][m].get('overall_score')
            if s is not None:
                scores.append(s)
        if len(scores) == 3:
            spread = max(scores) - min(scores)
            quality_groups[q].append(spread)

    quality_agreement = {}
    print(f"\n  {'Quality':<12s} {'N':>6s} {'Mean Spread':>12s} {'Median':>8s} {'≤1pt%':>7s}")
    print(f"  {'─' * 50}")
    for q in ['good', 'thin', 'stub', 'empty']:
        vals = quality_groups.get(q, [])
        if vals:
            qa = {
                'n': len(vals),
                'mean_spread': round(mean(vals), 3),
                'median_spread': round(sorted(vals)[len(vals) // 2], 3),
                'pct_within_1': round(sum(1 for v in vals if v <= 1) / len(vals) * 100, 1),
            }
            quality_agreement[q] = qa
            print(f"  {q:<12s} {qa['n']:>6d} {qa['mean_spread']:>12.3f} "
                  f"{qa['median_spread']:>8.3f} {qa['pct_within_1']:>6.1f}%")

    report['agreement_by_quality'] = quality_agreement

    # ── Save Report ────────────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    icc_overall = icc_results.get('overall_score')
    icc_cap = icc_results.get('capacity_score')
    icc_eth = icc_results.get('ethics_score')
    mean_spread_overall = spread_results.get('overall_score', {}).get('mean_spread', 0)

    print(f"\n  ICC(2,1) overall:    {icc_overall:.3f}" if icc_overall else "  ICC(2,1) overall:    N/A")
    print(f"  ICC(2,1) capacity:   {icc_cap:.3f}" if icc_cap else "  ICC(2,1) capacity:   N/A")
    print(f"  ICC(2,1) ethics:     {icc_eth:.3f}" if icc_eth else "  ICC(2,1) ethics:     N/A")
    print(f"  Mean spread overall: {mean_spread_overall:.3f}")

    dim_iccs = [icc_results[d] for d in DIMS if icc_results.get(d) is not None]
    if dim_iccs:
        print(f"  Mean dim ICC:        {mean(dim_iccs):.3f} (range {min(dim_iccs):.3f}–{max(dim_iccs):.3f})")

    print(f"\n  Report saved: {REPORT_PATH}")
    print(f"{'=' * 70}")

    return report


if __name__ == '__main__':
    run_analysis()
