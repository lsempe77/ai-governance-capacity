"""
Phase C – Depth Analysis & Visualisation
=========================================
Reads the normalised Phase C depth output and produces:

Figures:
  1. fig_depth_distribution.png        – overall depth distribution (stacked bar)
  2. fig_depth_by_category.png         – values vs principles vs mechanisms
  3. fig_depth_heatmap_values.png      – items × depth heatmap (values)
  4. fig_depth_heatmap_principles.png  – items × depth heatmap (principles)
  5. fig_depth_heatmap_mechanisms.png  – items × depth heatmap (mechanisms)
  6. fig_depth_income_gap.png          – high-income vs developing depth scores
  7. fig_depth_income_items.png        – income gap per canonical item
  8. fig_depth_vs_breadth.png          – scatter: #items vs avg depth per policy
  9. fig_depth_top_bottom.png          – deepest vs shallowest items
 10. fig_depth_by_region.png           – regional depth comparison
 11. fig_verbatim_length.png           – verbatim length distribution by depth

Tables (CSV):
  depth_summary_table.csv             – canonical item × depth level counts
  depth_by_income.csv                 – income-group depth comparison
  depth_by_region.csv                 – regional depth comparison
  depth_by_country.csv                – country-level depth scores
  policy_depth_scores.csv             – per-policy depth score

Stats (JSON):
  depth_stats.json                    – t-tests, effect sizes, descriptive stats
"""

import json, sys, warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data' / 'analysis' / 'ethics_inventory'
INPUT = DATA / 'phase_c_depth_normalised.jsonl'
OUT = ROOT / 'data' / 'analysis' / 'paper2_ethics' / 'depth'
OUT.mkdir(parents=True, exist_ok=True)

# ── Country metadata ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import (
    INCOME_GROUP, REGION, REGION_LABELS,
    INTERNATIONAL, get_income_binary, get_metadata,
)

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    'font.family':    'serif',
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  12,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':   'tight',
    'figure.figsize': (10, 6),
}
plt.rcParams.update(STYLE)
PAL = sns.color_palette('Set2', 8)

DEPTH_ORDER = ['word', 'phrase', 'sentence', 'paragraph', 'section']
DEPTH_WEIGHTS = {'word': 1, 'phrase': 2, 'sentence': 3, 'paragraph': 4, 'section': 5, 'not_found': 0}
DEPTH_COLORS = {
    'word':      '#fee5d9',
    'phrase':    '#fcae91',
    'sentence':  '#fb6a4a',
    'paragraph': '#de2d26',
    'section':   '#a50f15',
    'not_found': '#cccccc',
}
INCOME_COLORS = {'High income': PAL[0], 'Developing': PAL[2]}
CATEGORIES = ['values', 'principles', 'mechanisms']
CAT_LABELS = {'values': 'Values', 'principles': 'Principles', 'mechanisms': 'Mechanisms'}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    entries = []
    with open(INPUT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} normalised depth entries")
    return entries


def enrich_entries(entries):
    """Add income_binary, region, depth_score to each entry."""
    for e in entries:
        jur = e.get('jurisdiction', '')
        e['income_binary'] = get_income_binary(jur) or 'Unknown'
        meta = get_metadata(jur)
        e['region'] = REGION_LABELS.get(meta.get('region', ''), meta.get('region_label', 'Unknown'))
        e['is_international'] = jur in INTERNATIONAL

        # Compute per-entry depth score
        all_depths = []
        dd = e.get('depth_data', {})
        for cat in CATEGORIES:
            for item in dd.get(cat, []):
                d = item.get('depth', 'not_found').lower().strip()
                w = DEPTH_WEIGHTS.get(d, 0)
                if w > 0:
                    all_depths.append(w)
        e['depth_score'] = np.mean(all_depths) if all_depths else 0
        e['n_items'] = len(all_depths)
        e['pct_deep'] = sum(1 for d in all_depths if d >= 4) / len(all_depths) * 100 if all_depths else 0

        # Per-category depth scores
        for cat in CATEGORIES:
            cat_depths = []
            for item in dd.get(cat, []):
                d = item.get('depth', 'not_found').lower().strip()
                w = DEPTH_WEIGHTS.get(d, 0)
                if w > 0:
                    cat_depths.append(w)
            e[f'{cat}_depth_score'] = np.mean(cat_depths) if cat_depths else 0
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def collect_items(entries):
    """Collect all items grouped by category and canonical name."""
    items = defaultdict(lambda: defaultdict(list))  # cat → name → [depth_levels]
    for e in entries:
        dd = e.get('depth_data', {})
        for cat in CATEGORIES:
            for item in dd.get(cat, []):
                name = item.get('name', 'Other')
                depth = item.get('depth', 'not_found').lower().strip()
                items[cat][name].append(depth)
    return items


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Overall Depth Distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_distribution(entries):
    all_depths = Counter()
    for e in entries:
        dd = e.get('depth_data', {})
        for cat in CATEGORIES:
            for item in dd.get(cat, []):
                d = item.get('depth', 'not_found').lower().strip()
                all_depths[d] += 1

    total = sum(all_depths.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    depths = DEPTH_ORDER + (['not_found'] if all_depths.get('not_found', 0) > 0 else [])
    counts = [all_depths.get(d, 0) for d in depths]
    pcts = [c / total * 100 for c in counts]
    colors = [DEPTH_COLORS[d] for d in depths]

    bars = ax.bar(depths, pcts, color=colors, edgecolor='white', linewidth=0.5)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%\n({cnt:,})', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Share of items (%)')
    ax.set_xlabel('Depth level')
    ax.set_title('How Deeply Do AI Policies Engage with Ethics Content?')
    ax.set_ylim(0, max(pcts) + 8)
    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_distribution.png')
    plt.close(fig)
    print("  ✓ fig_depth_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Depth by Category (values vs principles vs mechanisms)
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_by_category(entries):
    cat_depths = defaultdict(Counter)
    for e in entries:
        dd = e.get('depth_data', {})
        for cat in CATEGORIES:
            for item in dd.get(cat, []):
                d = item.get('depth', 'not_found').lower().strip()
                if d in DEPTH_ORDER:
                    cat_depths[cat][d] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CATEGORIES))
    width = 0.15
    for i, depth in enumerate(DEPTH_ORDER):
        vals = []
        for cat in CATEGORIES:
            total = sum(cat_depths[cat].values())
            vals.append(cat_depths[cat].get(depth, 0) / total * 100 if total else 0)
        ax.bar(x + i * width - 2 * width, vals, width, label=depth.capitalize(),
               color=DEPTH_COLORS[depth], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES])
    ax.set_ylabel('Share of items (%)')
    ax.set_title('Depth Distribution by Ethics Category')
    ax.legend(title='Depth', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_by_category.png')
    plt.close(fig)
    print("  ✓ fig_depth_by_category.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 3-5: Heatmaps per category
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_heatmaps(items_data):
    for cat in CATEGORIES:
        cat_items = items_data[cat]
        # Filter to items with ≥5 mentions, sort by count
        filtered = {k: v for k, v in cat_items.items() if len(v) >= 5 and k != 'Other'}
        if not filtered:
            continue
        names = sorted(filtered.keys(), key=lambda n: -len(filtered[n]))

        # Build matrix: rows=items, cols=depth levels, values=percentage
        matrix = []
        for name in names:
            depths = Counter(filtered[name])
            total = sum(depths.values())
            row = [depths.get(d, 0) / total * 100 for d in DEPTH_ORDER]
            matrix.append(row)

        matrix = np.array(matrix)
        n_items = len(names)
        fig_height = max(4, n_items * 0.4 + 1.5)

        fig, ax = plt.subplots(figsize=(8, fig_height))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)

        ax.set_xticks(range(len(DEPTH_ORDER)))
        ax.set_xticklabels([d.capitalize() for d in DEPTH_ORDER])
        ax.set_yticks(range(n_items))
        ax.set_yticklabels(names, fontsize=9)

        # Annotate cells
        for i in range(n_items):
            for j in range(len(DEPTH_ORDER)):
                val = matrix[i, j]
                color = 'white' if val > 35 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=7, color=color)

        ax.set_title(f'Depth Profile: {CAT_LABELS[cat]}')
        fig.colorbar(im, ax=ax, label='% of mentions', shrink=0.6)
        plt.tight_layout()
        fig.savefig(OUT / f'fig_depth_heatmap_{cat}.png')
        plt.close(fig)
        print(f"  ✓ fig_depth_heatmap_{cat}.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Income Group Depth Gap
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_income_gap(entries, stats_out):
    # Filter to country-level (not international)
    country = [e for e in entries if not e['is_international'] and e['income_binary'] != 'Unknown']

    hi = [e['depth_score'] for e in country if e['income_binary'] == 'High income']
    dev = [e['depth_score'] for e in country if e['income_binary'] == 'Developing']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Overall depth score
    ax = axes[0]
    bp = ax.boxplot([hi, dev], labels=['High income', 'Developing'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(INCOME_COLORS['High income'])
    bp['boxes'][1].set_facecolor(INCOME_COLORS['Developing'])
    ax.set_ylabel('Average depth score (1-5)')
    ax.set_title('A. Overall Depth Score')
    t, p = sp_stats.ttest_ind(hi, dev)
    cohens_d = (np.mean(hi) - np.mean(dev)) / np.sqrt((np.std(hi)**2 + np.std(dev)**2) / 2)
    ax.text(0.05, 0.95, f'Δ = {np.mean(hi)-np.mean(dev):.2f}\nt = {t:.2f}, p = {p:.3f}\nd = {cohens_d:.2f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    stats_out['income_gap'] = {
        'hi_mean': round(np.mean(hi), 3), 'hi_std': round(np.std(hi), 3), 'hi_n': len(hi),
        'dev_mean': round(np.mean(dev), 3), 'dev_std': round(np.std(dev), 3), 'dev_n': len(dev),
        't_stat': round(t, 3), 'p_value': round(p, 4), 'cohens_d': round(cohens_d, 3),
    }

    # Panel B: % deep (paragraph+section)
    ax = axes[1]
    hi_deep = [e['pct_deep'] for e in country if e['income_binary'] == 'High income']
    dev_deep = [e['pct_deep'] for e in country if e['income_binary'] == 'Developing']
    bp = ax.boxplot([hi_deep, dev_deep], labels=['High income', 'Developing'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(INCOME_COLORS['High income'])
    bp['boxes'][1].set_facecolor(INCOME_COLORS['Developing'])
    ax.set_ylabel('% of items at paragraph/section depth')
    ax.set_title('B. Share of Deep Engagement')
    t2, p2 = sp_stats.ttest_ind(hi_deep, dev_deep)
    ax.text(0.05, 0.95, f'Δ = {np.mean(hi_deep)-np.mean(dev_deep):.1f}pp\np = {p2:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel C: Per-category
    ax = axes[2]
    x = np.arange(len(CATEGORIES))
    w = 0.35
    hi_means = [np.mean([e[f'{c}_depth_score'] for e in country if e['income_binary'] == 'High income' and e[f'{c}_depth_score'] > 0]) for c in CATEGORIES]
    dev_means = [np.mean([e[f'{c}_depth_score'] for e in country if e['income_binary'] == 'Developing' and e[f'{c}_depth_score'] > 0]) for c in CATEGORIES]
    ax.bar(x - w/2, hi_means, w, label='High income', color=INCOME_COLORS['High income'], edgecolor='white')
    ax.bar(x + w/2, dev_means, w, label='Developing', color=INCOME_COLORS['Developing'], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES])
    ax.set_ylabel('Average depth score (1-5)')
    ax.set_title('C. Depth by Category')
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_income_gap.png')
    plt.close(fig)
    print("  ✓ fig_depth_income_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Income Gap per Canonical Item (dot plot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_income_items(entries):
    country = [e for e in entries if not e['is_international'] and e['income_binary'] != 'Unknown']

    # For each canonical item (across all categories), compute mean depth by income
    item_scores = defaultdict(lambda: {'High income': [], 'Developing': []})
    for e in country:
        inc = e['income_binary']
        dd = e.get('depth_data', {})
        for cat in CATEGORIES:
            for item in dd.get(cat, []):
                name = item.get('name', 'Other')
                d = item.get('depth', 'not_found').lower().strip()
                w = DEPTH_WEIGHTS.get(d, 0)
                if w > 0 and name != 'Other':
                    item_scores[name][inc].append(w)

    # Filter to items with ≥ 20 mentions in each group
    rows = []
    for name, groups in item_scores.items():
        if len(groups['High income']) >= 20 and len(groups['Developing']) >= 10:
            hi_m = np.mean(groups['High income'])
            dev_m = np.mean(groups['Developing'])
            rows.append({'name': name, 'hi': hi_m, 'dev': dev_m, 'gap': hi_m - dev_m})

    rows.sort(key=lambda r: r['gap'])
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.35 + 1)))
    y = np.arange(len(rows))
    names = [r['name'] for r in rows]

    ax.barh(y, [r['gap'] for r in rows],
            color=[INCOME_COLORS['High income'] if r['gap'] > 0 else INCOME_COLORS['Developing'] for r in rows],
            edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Depth gap (High income − Developing)')
    ax.set_title('Income Gap in Treatment Depth by Ethics Item')
    ax.text(0.98, 0.02, '← Developing deeper    High income deeper →',
            transform=ax.transAxes, ha='right', fontsize=8, style='italic', color='gray')
    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_income_items.png')
    plt.close(fig)
    print("  ✓ fig_depth_income_items.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Depth vs Breadth scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_vs_breadth(entries):
    country = [e for e in entries if not e['is_international'] and e['income_binary'] != 'Unknown']

    fig, ax = plt.subplots(figsize=(8, 6))
    for inc in ['High income', 'Developing']:
        subset = [e for e in country if e['income_binary'] == inc]
        ax.scatter([e['n_items'] for e in subset],
                   [e['depth_score'] for e in subset],
                   alpha=0.4, s=30, label=inc, color=INCOME_COLORS[inc],
                   edgecolor='white', linewidth=0.3)

    # Regression line
    all_n = [e['n_items'] for e in country]
    all_d = [e['depth_score'] for e in country]
    if len(all_n) > 10:
        z = np.polyfit(all_n, all_d, 1)
        x_line = np.linspace(min(all_n), max(all_n), 100)
        ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', linewidth=1)
        r, p = sp_stats.pearsonr(all_n, all_d)
        ax.text(0.05, 0.05, f'r = {r:.3f}, p = {p:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Number of ethics items mentioned (breadth)')
    ax.set_ylabel('Average depth score (1-5)')
    ax.set_title('Breadth vs Depth: Do More Comprehensive Policies Sacrifice Depth?')
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_vs_breadth.png')
    plt.close(fig)
    print("  ✓ fig_depth_vs_breadth.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Deepest vs Shallowest Items
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_top_bottom(items_data):
    all_items = {}
    for cat in CATEGORIES:
        for name, depths in items_data[cat].items():
            if name == 'Other' or len(depths) < 10:
                continue
            weighted = [DEPTH_WEIGHTS.get(d, 0) for d in depths if DEPTH_WEIGHTS.get(d, 0) > 0]
            if weighted:
                all_items[f"{name}"] = {
                    'avg': np.mean(weighted),
                    'cat': cat,
                    'n': len(weighted),
                }

    sorted_items = sorted(all_items.items(), key=lambda x: x[1]['avg'])
    n_show = min(10, len(sorted_items) // 2)
    bottom = sorted_items[:n_show]
    top = sorted_items[-n_show:]
    combined = bottom + top

    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.4)))
    y = np.arange(len(combined))
    names = [c[0] for c in combined]
    scores = [c[1]['avg'] for c in combined]
    cats = [c[1]['cat'] for c in combined]
    cat_colors = {'values': PAL[0], 'principles': PAL[1], 'mechanisms': PAL[2]}

    ax.barh(y, scores, color=[cat_colors[c] for c in cats], edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({c[1]['n']})" for n, c in zip(names, combined)], fontsize=8)
    ax.set_xlabel('Average depth score (1=word, 5=section)')
    ax.set_title('Deepest vs Shallowest Ethics Items')
    ax.axvline(3, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    # Separator line
    ax.axhline(n_show - 0.5, color='black', linewidth=1, linestyle='-')
    ax.text(0.5, n_show - 0.5, '— shallowest ↓   deepest ↑ —', ha='center', va='center',
            fontsize=8, style='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[c], label=CAT_LABELS[c]) for c in CATEGORIES]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_top_bottom.png')
    plt.close(fig)
    print("  ✓ fig_depth_top_bottom.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Regional Depth Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_by_region(entries):
    country = [e for e in entries if not e['is_international'] and e['region'] != 'Unknown']

    region_scores = defaultdict(list)
    for e in country:
        region_scores[e['region']].append(e['depth_score'])

    # Sort by median
    regions = sorted(region_scores.keys(), key=lambda r: np.median(region_scores[r]), reverse=True)
    regions = [r for r in regions if len(region_scores[r]) >= 3]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot([region_scores[r] for r in regions],
                    labels=[f"{r}\n(n={len(region_scores[r])})" for r in regions],
                    patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor(PAL[1])
        patch.set_alpha(0.7)
    ax.set_ylabel('Average depth score (1-5)')
    ax.set_title('Depth of Ethics Engagement by Region')
    plt.xticks(fontsize=9, rotation=15, ha='right')
    plt.tight_layout()
    fig.savefig(OUT / 'fig_depth_by_region.png')
    plt.close(fig)
    print("  ✓ fig_depth_by_region.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Verbatim Length by Depth Level
# ══════════════════════════════════════════════════════════════════════════════

def fig_verbatim_length(entries):
    depth_lengths = defaultdict(list)
    for e in entries:
        dd = e.get('depth_data', {})
        for cat in CATEGORIES:
            for item in dd.get(cat, []):
                d = item.get('depth', 'not_found').lower().strip()
                v = item.get('verbatim', '')
                if d in DEPTH_ORDER and v:
                    depth_lengths[d].append(len(v))

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [depth_lengths.get(d, []) for d in DEPTH_ORDER]
    bp = ax.boxplot(data, labels=[d.capitalize() for d in DEPTH_ORDER],
                    patch_artist=True, widths=0.6, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(DEPTH_COLORS[DEPTH_ORDER[i]])
        patch.set_alpha(0.8)

    ax.set_ylabel('Verbatim text length (characters)')
    ax.set_xlabel('Depth classification')
    ax.set_title('Verbatim Quote Length Validates Depth Classification')

    # Add median labels
    for i, d in enumerate(DEPTH_ORDER):
        vals = depth_lengths.get(d, [])
        if vals:
            med = np.median(vals)
            ax.text(i + 1, med + 15, f'{med:.0f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT / 'fig_verbatim_length.png')
    plt.close(fig)
    print("  ✓ fig_verbatim_length.png")


# ══════════════════════════════════════════════════════════════════════════════
# TABLES (CSV)
# ══════════════════════════════════════════════════════════════════════════════

def write_tables(entries, items_data):
    import csv

    # 1. Depth summary table: canonical item × depth level
    with open(OUT / 'depth_summary_table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['category', 'item', 'total', 'word', 'phrase', 'sentence', 'paragraph',
                     'section', 'not_found', 'avg_depth', 'pct_deep'])
        for cat in CATEGORIES:
            for name in sorted(items_data[cat].keys()):
                if name == 'Other':
                    continue
                depths = items_data[cat][name]
                c = Counter(depths)
                total = len(depths)
                weighted = [DEPTH_WEIGHTS.get(d, 0) for d in depths if DEPTH_WEIGHTS.get(d, 0) > 0]
                avg = np.mean(weighted) if weighted else 0
                deep = (c.get('paragraph', 0) + c.get('section', 0)) / total * 100 if total else 0
                w.writerow([cat, name, total,
                            c.get('word', 0), c.get('phrase', 0), c.get('sentence', 0),
                            c.get('paragraph', 0), c.get('section', 0), c.get('not_found', 0),
                            f'{avg:.2f}', f'{deep:.1f}'])
    print("  ✓ depth_summary_table.csv")

    # 2. Income group comparison
    country = [e for e in entries if not e['is_international'] and e['income_binary'] != 'Unknown']
    with open(OUT / 'depth_by_income.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['income_group', 'n_policies', 'mean_depth', 'median_depth', 'std_depth',
                     'mean_pct_deep', 'mean_n_items',
                     'values_depth', 'principles_depth', 'mechanisms_depth'])
        for inc in ['High income', 'Developing']:
            sub = [e for e in country if e['income_binary'] == inc]
            scores = [e['depth_score'] for e in sub]
            w.writerow([inc, len(sub),
                        f'{np.mean(scores):.3f}', f'{np.median(scores):.3f}', f'{np.std(scores):.3f}',
                        f'{np.mean([e["pct_deep"] for e in sub]):.1f}',
                        f'{np.mean([e["n_items"] for e in sub]):.1f}',
                        f'{np.mean([e["values_depth_score"] for e in sub if e["values_depth_score"] > 0]):.3f}',
                        f'{np.mean([e["principles_depth_score"] for e in sub if e["principles_depth_score"] > 0]):.3f}',
                        f'{np.mean([e["mechanisms_depth_score"] for e in sub if e["mechanisms_depth_score"] > 0]):.3f}'])
    print("  ✓ depth_by_income.csv")

    # 3. Regional comparison
    region_data = defaultdict(list)
    for e in country:
        if e['region'] != 'Unknown':
            region_data[e['region']].append(e)
    with open(OUT / 'depth_by_region.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['region', 'n_policies', 'mean_depth', 'median_depth', 'mean_pct_deep'])
        for reg in sorted(region_data.keys(), key=lambda r: -np.mean([e['depth_score'] for e in region_data[r]])):
            sub = region_data[reg]
            scores = [e['depth_score'] for e in sub]
            w.writerow([reg, len(sub), f'{np.mean(scores):.3f}', f'{np.median(scores):.3f}',
                        f'{np.mean([e["pct_deep"] for e in sub]):.1f}'])
    print("  ✓ depth_by_region.csv")

    # 4. Country-level depth scores
    country_data = defaultdict(list)
    for e in country:
        country_data[e['jurisdiction']].append(e)
    with open(OUT / 'depth_by_country.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['jurisdiction', 'income_group', 'region', 'n_policies',
                     'mean_depth', 'mean_pct_deep', 'mean_n_items'])
        for jur in sorted(country_data.keys(), key=lambda j: -np.mean([e['depth_score'] for e in country_data[j]])):
            sub = country_data[jur]
            scores = [e['depth_score'] for e in sub]
            w.writerow([jur, sub[0]['income_binary'], sub[0]['region'], len(sub),
                        f'{np.mean(scores):.3f}', f'{np.mean([e["pct_deep"] for e in sub]):.1f}',
                        f'{np.mean([e["n_items"] for e in sub]):.1f}'])
    print("  ✓ depth_by_country.csv")

    # 5. Per-policy depth scores
    with open(OUT / 'policy_depth_scores.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['entry_id', 'title', 'jurisdiction', 'income_group', 'region',
                     'depth_score', 'pct_deep', 'n_items',
                     'values_depth', 'principles_depth', 'mechanisms_depth'])
        for e in sorted(entries, key=lambda x: -x['depth_score']):
            w.writerow([e['entry_id'], e.get('title', ''), e.get('jurisdiction', ''),
                        e['income_binary'], e['region'],
                        f'{e["depth_score"]:.3f}', f'{e["pct_deep"]:.1f}', e['n_items'],
                        f'{e["values_depth_score"]:.3f}', f'{e["principles_depth_score"]:.3f}',
                        f'{e["mechanisms_depth_score"]:.3f}'])
    print("  ✓ policy_depth_scores.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STATS (JSON)
# ══════════════════════════════════════════════════════════════════════════════

def write_stats(entries, items_data, stats_out):
    country = [e for e in entries if not e['is_international'] and e['income_binary'] != 'Unknown']

    # Descriptive
    all_scores = [e['depth_score'] for e in entries if e['depth_score'] > 0]
    stats_out['descriptive'] = {
        'n_policies': len(entries),
        'n_country': len(country),
        'mean_depth': round(np.mean(all_scores), 3),
        'median_depth': round(np.median(all_scores), 3),
        'std_depth': round(np.std(all_scores), 3),
        'mean_pct_deep': round(np.mean([e['pct_deep'] for e in entries]), 1),
        'mean_n_items': round(np.mean([e['n_items'] for e in entries]), 1),
    }

    # Per-category t-tests
    stats_out['category_income_tests'] = {}
    for cat in CATEGORIES:
        hi = [e[f'{cat}_depth_score'] for e in country if e['income_binary'] == 'High income' and e[f'{cat}_depth_score'] > 0]
        dev = [e[f'{cat}_depth_score'] for e in country if e['income_binary'] == 'Developing' and e[f'{cat}_depth_score'] > 0]
        if len(hi) > 5 and len(dev) > 5:
            t, p = sp_stats.ttest_ind(hi, dev)
            d = (np.mean(hi) - np.mean(dev)) / np.sqrt((np.std(hi)**2 + np.std(dev)**2) / 2)
            stats_out['category_income_tests'][cat] = {
                'hi_mean': round(np.mean(hi), 3), 'dev_mean': round(np.mean(dev), 3),
                'gap': round(np.mean(hi) - np.mean(dev), 3),
                't': round(t, 3), 'p': round(p, 4), 'd': round(d, 3),
            }

    # Breadth-depth correlation
    n_items = [e['n_items'] for e in country]
    d_scores = [e['depth_score'] for e in country]
    r, p = sp_stats.pearsonr(n_items, d_scores)
    stats_out['breadth_depth_correlation'] = {
        'pearson_r': round(r, 3), 'p_value': round(p, 4),
    }

    stats_out['created'] = datetime.now().isoformat()

    with open(OUT / 'depth_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print("  ✓ depth_stats.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DEPTH ANALYSIS & VISUALISATION")
    print("=" * 60)

    entries = load_data()
    entries = enrich_entries(entries)
    items_data = collect_items(entries)
    stats_out = {}

    print("\nGenerating figures...")
    fig_depth_distribution(entries)
    fig_depth_by_category(entries)
    fig_depth_heatmaps(items_data)
    fig_depth_income_gap(entries, stats_out)
    fig_depth_income_items(entries)
    fig_depth_vs_breadth(entries)
    fig_depth_top_bottom(items_data)
    fig_depth_by_region(entries)
    fig_verbatim_length(entries)

    print("\nGenerating tables...")
    write_tables(entries, items_data)

    print("\nGenerating stats...")
    write_stats(entries, items_data, stats_out)

    print(f"\nAll outputs saved to: {OUT}")
    print("Done!")


if __name__ == '__main__':
    main()
