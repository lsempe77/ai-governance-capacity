"""
Paper 3 — Extended UNESCO Alignment Analysis
=============================================
Builds on unesco_alignment.py (11 figures already done) and adds:

Phase A – Data join & enrichment:
  Join phase_c_depth_normalised.jsonl with master_dataset.csv to obtain
  policy_type, binding_nature, capacity_score, ethics_score, all C1-C5/E1-E5.

Phase B – New figures (12–22):
 12. fig_unesco_coverage_by_layer.png       – grouped bar: values / principles / policy areas
 13. fig_unesco_coverage_vs_depth.png       – scatter: coverage % vs mean depth per UNESCO item
 14. fig_unesco_income_alignment_dist.png   – violin: alignment score by income
 15. fig_unesco_region_scores.png           – bar: mean alignment score by region
 16. fig_unesco_pre_post.png               – paired bar: coverage before/after UNESCO adoption
 17. fig_unesco_pre_post_income.png         – pre/post × income interaction
 18. fig_unesco_binding_nature.png          – boxplot: alignment by binding nature
 19. fig_unesco_policy_type.png             – boxplot: alignment by policy type
 20. fig_unesco_cluster_radar.png           – radar per cluster
 21. fig_unesco_cluster_map.png             – choropleth map by cluster
 22. fig_unesco_cluster_income.png          – cluster × income stacked bar

Phase C – New tables:
  unesco_pre_post.csv                      – before/after UNESCO comparison
  unesco_binding.csv                       – binding nature breakdown
  unesco_regression.csv                    – OLS regression table
  unesco_clusters.csv                      – cluster profiles

Phase D – Stats extension:
  unesco_paper3_stats.json                 – regression, pre/post tests, cluster info
"""

import json, csv, sys, warnings, math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "analysis" / "ethics_inventory"
MASTER = ROOT / "data" / "analysis" / "shared" / "master_dataset.csv"
INPUT = DATA / "phase_c_depth_normalised.jsonl"
OUT = ROOT / "data" / "analysis" / "paper2_ethics" / "unesco"
OUT.mkdir(parents=True, exist_ok=True)

# ── Country metadata ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import (
    INCOME_GROUP, REGION, REGION_LABELS,
    INTERNATIONAL, get_income_binary, get_metadata, GDP_PER_CAPITA,
)

# ── Import shared constants from base script ─────────────────────────────────
from unesco_alignment import (
    UNESCO_VALUES, UNESCO_PRINCIPLES, UNESCO_POLICY_AREAS,
    ALL_UNESCO, UNESCO_ITEMS_FLAT, CANONICAL_TO_UNESCO,
    DEPTH_ORDER, DEPTH_WEIGHTS, DEPTH_COLORS,
)

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "font.family":    "serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":   "tight",
    "figure.figsize": (10, 6),
}
plt.rcParams.update(STYLE)
PAL = sns.color_palette("Set2", 8)
INCOME_COLORS = {"High income": PAL[0], "Developing": PAL[2]}
TYPE_MAP = {name: typ for name, typ in ALL_UNESCO}
LAYER_COLORS = {"value": PAL[0], "principle": PAL[1], "policy_area": PAL[2]}

# UNESCO adoption date split
UNESCO_CUTOFF = 2021  # Adopted Nov 2021; policies ≤2021 = "pre", ≥2022 = "post"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_master():
    """Load master_dataset.csv into {entry_id: row} dict."""
    master = {}
    with open(MASTER, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            master[row["entry_id"]] = row
    print(f"  Master dataset: {len(master)} rows")
    return master


def load_and_enrich():
    """Load depth data, join with master, compute UNESCO scores."""
    # Load depth data
    entries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"  Depth data: {len(entries)} entries")

    # Load master for joining
    master = load_master()

    # Enrich each entry
    joined = 0
    for e in entries:
        eid = e.get("entry_id", "")
        jur = e.get("jurisdiction", "")

        # --- Join master fields ---
        if eid in master:
            m = master[eid]
            e["policy_type"] = m.get("policy_type", "Other")
            e["binding_nature"] = m.get("binding_nature", "Unknown")
            e["capacity_score"] = float(m["capacity_score"]) if m.get("capacity_score") else None
            e["ethics_score_master"] = float(m["ethics_score"]) if m.get("ethics_score") else None
            e["overall_score"] = float(m["overall_score"]) if m.get("overall_score") else None
            for dim in ("c1_clarity", "c2_resources", "c3_authority",
                        "c4_accountability", "c5_coherence",
                        "e1_framework", "e2_rights", "e3_governance",
                        "e4_operationalisation", "e5_inclusion"):
                e[dim] = float(m[dim]) if m.get(dim) else None
            e["gdp_pc"] = float(m["gdp_pc"]) if m.get("gdp_pc") else None
            e["log_gdp_pc"] = float(m["log_gdp_pc"]) if m.get("log_gdp_pc") else None
            joined += 1
        else:
            e["policy_type"] = "Other"
            e["binding_nature"] = "Unknown"
            e["capacity_score"] = None
            e["ethics_score_master"] = None
            e["overall_score"] = None
            e["gdp_pc"] = GDP_PER_CAPITA.get(jur)
            e["log_gdp_pc"] = np.log(GDP_PER_CAPITA[jur]) if jur in GDP_PER_CAPITA else None

        # --- Metadata ---
        e["income_binary"] = get_income_binary(jur) or "Unknown"
        meta = get_metadata(jur)
        e["region"] = REGION_LABELS.get(
            meta.get("region", ""), meta.get("region_label", "Unknown")
        )
        e["is_international"] = jur in INTERNATIONAL

        # --- UNESCO scores ---
        unesco_hits = defaultdict(list)
        dd = e.get("depth_data", {})
        for cat in ("values", "principles", "mechanisms"):
            for item in dd.get(cat, []):
                canon_name = item.get("name", "Other")
                depth_str = item.get("depth", "not_found").lower().strip()
                dw = DEPTH_WEIGHTS.get(depth_str, 0)
                for ut in CANONICAL_TO_UNESCO.get(canon_name, []):
                    unesco_hits[ut].append(dw)

        e["unesco_scores"] = {}
        for u_name in UNESCO_ITEMS_FLAT:
            weights = unesco_hits.get(u_name, [])
            positive = [w for w in weights if w > 0]
            e["unesco_scores"][u_name] = np.mean(positive) if positive else 0

        e["unesco_coverage"] = {
            u_name: (1 if e["unesco_scores"][u_name] > 0 else 0)
            for u_name in UNESCO_ITEMS_FLAT
        }

        coverage_pct = sum(e["unesco_coverage"].values()) / len(UNESCO_ITEMS_FLAT) * 100
        depths = [e["unesco_scores"][u] for u in UNESCO_ITEMS_FLAT if e["unesco_scores"][u] > 0]
        avg_depth = np.mean(depths) if depths else 0
        e["unesco_alignment_score"] = 0.6 * coverage_pct + 0.4 * (avg_depth / 5 * 100)
        e["unesco_coverage_pct"] = coverage_pct
        e["unesco_avg_depth"] = avg_depth

        # Sub-framework coverage
        for label, items in [("values", UNESCO_VALUES),
                              ("principles", UNESCO_PRINCIPLES),
                              ("policy_areas", UNESCO_POLICY_AREAS)]:
            e[f"unesco_{label}_cov"] = (
                sum(e["unesco_coverage"][u] for u in items) / len(items) * 100
            )

        # Pre/post UNESCO
        year = e.get("year")
        if year and year > 0:
            e["unesco_era"] = "Pre-UNESCO" if year <= UNESCO_CUTOFF else "Post-UNESCO"
        else:
            e["unesco_era"] = "Unknown"

    print(f"  Joined with master: {joined}/{len(entries)}")
    # Filter to entries with depth data
    entries = [e for e in entries if e.get("depth_data")]
    print(f"  Entries with depth data: {len(entries)}")
    return entries


def compute_coverage(entries):
    """Compute overall coverage dict {item: pct}."""
    n = len(entries)
    return {
        u_name: sum(1 for e in entries if e["unesco_coverage"][u_name] == 1) / n * 100
        for u_name in UNESCO_ITEMS_FLAT
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Coverage by Layer (Values / Principles / Policy Areas)
# ══════════════════════════════════════════════════════════════════════════════

def fig_coverage_by_layer(entries):
    """Grouped bar: mean coverage for each UNESCO sub-framework."""
    layers = [
        ("Values\n(4 items)", UNESCO_VALUES, PAL[0]),
        ("Principles\n(10 items)", UNESCO_PRINCIPLES, PAL[1]),
        ("Policy Areas\n(11 items)", UNESCO_POLICY_AREAS, PAL[2]),
    ]

    means = []
    ses = []
    for label, items, _ in layers:
        # Per-policy coverage within this layer
        policy_covs = [
            sum(e["unesco_coverage"][u] for u in items) / len(items) * 100
            for e in entries
        ]
        means.append(np.mean(policy_covs))
        ses.append(sp_stats.sem(policy_covs))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layers))
    colors = [c for _, _, c in layers]
    bars = ax.bar(x, means, yerr=ses, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in layers], fontsize=11)
    ax.set_ylabel("Mean per-policy coverage (%)")
    ax.set_title("UNESCO Coverage by Framework Layer")
    ax.set_ylim(0, max(means) + 15)

    for bar, m, se in zip(bars, means, ses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_coverage_by_layer.png")
    plt.close(fig)
    print("  ✓ fig_unesco_coverage_by_layer.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: Coverage vs Depth Scatter
# ══════════════════════════════════════════════════════════════════════════════

def fig_coverage_vs_depth(entries, coverage):
    """Scatter: per-UNESCO-item coverage % vs mean depth score."""
    # Compute mean depth per UNESCO item (among policies that mention it)
    item_depth = {}
    for u_name in UNESCO_ITEMS_FLAT:
        depths = [e["unesco_scores"][u_name] for e in entries
                  if e["unesco_scores"][u_name] > 0]
        item_depth[u_name] = np.mean(depths) if depths else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    for u_name in UNESCO_ITEMS_FLAT:
        x = coverage[u_name]
        y = item_depth[u_name]
        color = LAYER_COLORS[TYPE_MAP[u_name]]
        ax.scatter(x, y, color=color, s=80, zorder=3, edgecolor="white", linewidth=0.5)
        # Label
        offset_x = 1.5
        offset_y = 0.05
        ha = "left"
        # Shorten label for plot
        short = u_name.split(" & ")[0] if len(u_name) > 28 else u_name
        ax.annotate(short, (x, y), fontsize=7.5,
                    xytext=(offset_x, offset_y),
                    textcoords="offset points", ha=ha, va="bottom")

    ax.set_xlabel("Coverage: % of policies mentioning this item")
    ax.set_ylabel("Depth: mean depth score (1–5) among mentioning policies")
    ax.set_title("The Coverage–Depth Trade-off in UNESCO Framework Items")

    # Quadrant lines at medians
    med_x = np.median(list(coverage.values()))
    med_y = np.median(list(item_depth.values()))
    ax.axvline(med_x, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(med_y, color="gray", linestyle=":", alpha=0.5)

    # Quadrant labels
    ax.text(5, ax.get_ylim()[1] - 0.1, "Low coverage\nHigh depth",
            fontsize=8, color="gray", ha="left", va="top")
    ax.text(max(coverage.values()) - 5, ax.get_ylim()[1] - 0.1,
            "High coverage\nHigh depth", fontsize=8, color="gray", ha="right", va="top")
    ax.text(5, ax.get_ylim()[0] + 0.1, "Low coverage\nLow depth",
            fontsize=8, color="gray", ha="left", va="bottom")
    ax.text(max(coverage.values()) - 5, ax.get_ylim()[0] + 0.1,
            "High coverage\nLow depth", fontsize=8, color="gray", ha="right", va="bottom")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PAL[0], label="Values (4)"),
        Patch(facecolor=PAL[1], label="Principles (10)"),
        Patch(facecolor=PAL[2], label="Policy Areas (11)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_coverage_vs_depth.png")
    plt.close(fig)
    print("  ✓ fig_unesco_coverage_vs_depth.png")

    return item_depth


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Income Alignment Distribution (Violin)
# ══════════════════════════════════════════════════════════════════════════════

def fig_income_alignment_dist(entries):
    """Violin plot: UNESCO alignment score by income group."""
    data = [(e["income_binary"], e["unesco_alignment_score"])
            for e in entries if e["income_binary"] != "Unknown"]

    fig, ax = plt.subplots(figsize=(8, 6))
    groups = ["High income", "Developing"]
    positions = [0, 1]
    for i, grp in enumerate(groups):
        vals = [v for g, v in data if g == grp]
        parts = ax.violinplot([vals], positions=[i], showmeans=True,
                               showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(INCOME_COLORS[grp])
            pc.set_alpha(0.6)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("black")

        n = len(vals)
        mu = np.mean(vals)
        ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > -5 else -2,
                f"n={n}\nμ={mu:.1f}", ha="center", va="top", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("UNESCO Alignment Score (0–100)")
    ax.set_title("UNESCO Alignment by Income Group")
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_income_alignment_dist.png")
    plt.close(fig)
    print("  ✓ fig_unesco_income_alignment_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Regional Scores Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_region_scores(entries):
    """Horizontal bar: mean UNESCO alignment score by region."""
    filtered = [e for e in entries
                if e["region"] != "Unknown" and not e["is_international"]]
    regions = sorted(set(e["region"] for e in filtered))

    region_data = []
    for reg in regions:
        sub = [e for e in filtered if e["region"] == reg]
        if len(sub) >= 3:
            scores = [e["unesco_alignment_score"] for e in sub]
            region_data.append((reg, np.mean(scores), sp_stats.sem(scores), len(sub)))

    region_data.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(region_data))
    means = [d[1] for d in region_data]
    ses = [d[2] for d in region_data]
    names = [d[0] for d in region_data]
    ns = [d[3] for d in region_data]

    bars = ax.barh(y_pos, means, xerr=ses, capsize=4,
                   color=PAL[1], edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean UNESCO Alignment Score (0–100)")
    ax.set_title("UNESCO Alignment by Region")

    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(m + 1, i, f"{m:.1f} (n={n})", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_region_scores.png")
    plt.close(fig)
    print("  ✓ fig_unesco_region_scores.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 16: Pre/Post UNESCO Coverage Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_pre_post(entries):
    """Paired bar: coverage % per UNESCO item, before vs after adoption."""
    pre = [e for e in entries if e["unesco_era"] == "Pre-UNESCO"]
    post = [e for e in entries if e["unesco_era"] == "Post-UNESCO"]

    # Compute coverage per item for each era
    pre_cov = {}
    post_cov = {}
    for u_name in UNESCO_ITEMS_FLAT:
        pre_cov[u_name] = (np.mean([e["unesco_coverage"][u_name] for e in pre]) * 100
                           if pre else 0)
        post_cov[u_name] = (np.mean([e["unesco_coverage"][u_name] for e in post]) * 100
                            if post else 0)

    # Sort by difference (post - pre)
    diffs = {u: post_cov[u] - pre_cov[u] for u in UNESCO_ITEMS_FLAT}
    sorted_items = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(names))
    w = 0.35

    bars1 = ax.barh(y + w / 2, [pre_cov[n] for n in names], w,
                    color=PAL[3], label=f"Pre-UNESCO ≤{UNESCO_CUTOFF} (n={len(pre)})",
                    edgecolor="white", linewidth=0.3)
    bars2 = ax.barh(y - w / 2, [post_cov[n] for n in names], w,
                    color=PAL[4], label=f"Post-UNESCO ≥{UNESCO_CUTOFF+1} (n={len(post)})",
                    edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("UNESCO Framework Coverage: Before vs After Adoption (Nov 2021)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)

    # Annotate differences
    for i, name in enumerate(names):
        diff = diffs[name]
        x_pos = max(pre_cov[name], post_cov[name]) + 1
        color = "green" if diff > 0 else ("red" if diff < 0 else "gray")
        ax.text(x_pos, i, f"{diff:+.1f}pp", va="center", fontsize=7.5,
                color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_pre_post.png")
    plt.close(fig)
    print("  ✓ fig_unesco_pre_post.png")

    return pre, post, pre_cov, post_cov, diffs


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 17: Pre/Post × Income Interaction
# ══════════════════════════════════════════════════════════════════════════════

def fig_pre_post_income(entries):
    """Grouped bar: pre/post × income for overall alignment score."""
    groups = {
        ("Pre-UNESCO", "High income"): [],
        ("Pre-UNESCO", "Developing"): [],
        ("Post-UNESCO", "High income"): [],
        ("Post-UNESCO", "Developing"): [],
    }
    for e in entries:
        era = e.get("unesco_era", "Unknown")
        inc = e.get("income_binary", "Unknown")
        key = (era, inc)
        if key in groups:
            groups[key].append(e["unesco_alignment_score"])

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(2)
    w = 0.3

    for j, inc in enumerate(["High income", "Developing"]):
        pre_vals = groups[("Pre-UNESCO", inc)]
        post_vals = groups[("Post-UNESCO", inc)]
        means = [np.mean(pre_vals) if pre_vals else 0,
                 np.mean(post_vals) if post_vals else 0]
        ses = [sp_stats.sem(pre_vals) if len(pre_vals) > 1 else 0,
               sp_stats.sem(post_vals) if len(post_vals) > 1 else 0]
        offset = (j - 0.5) * w
        bars = ax.bar(x + offset, means, w, yerr=ses, capsize=4,
                      color=INCOME_COLORS[inc], label=inc,
                      edgecolor="white", linewidth=0.5)
        for bar, m, n in zip(bars, means,
                              [len(pre_vals), len(post_vals)]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{m:.1f}\nn={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Pre-UNESCO\n(≤{UNESCO_CUTOFF})",
                        f"Post-UNESCO\n(≥{UNESCO_CUTOFF+1})"], fontsize=11)
    ax.set_ylabel("Mean UNESCO Alignment Score")
    ax.set_title("UNESCO Alignment: Temporal × Income Interaction")
    ax.legend(fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] + 10)
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_pre_post_income.png")
    plt.close(fig)
    print("  ✓ fig_unesco_pre_post_income.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 18: Binding Nature
# ══════════════════════════════════════════════════════════════════════════════

def fig_binding_nature(entries):
    """Boxplot: UNESCO alignment score by binding nature."""
    bn_order = ["Hard law", "Binding regulation", "Soft law", "Non-binding"]
    bn_data = defaultdict(list)
    for e in entries:
        bn = e.get("binding_nature", "Unknown")
        if bn in bn_order:
            bn_data[bn].append(e["unesco_alignment_score"])

    # Only keep groups with enough data
    valid = [(bn, bn_data[bn]) for bn in bn_order if len(bn_data[bn]) >= 5]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [v[0] for v in valid]
    data = [v[1] for v in valid]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))

    colors = [PAL[i] for i in range(len(valid))]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (label, vals) in enumerate(valid):
        ax.text(i + 1, ax.get_ylim()[0] - 2,
                f"n={len(vals)}\nμ={np.mean(vals):.1f}",
                ha="center", va="top", fontsize=8)

    ax.set_ylabel("UNESCO Alignment Score (0–100)")
    ax.set_title("UNESCO Alignment by Binding Nature of Policy")
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_binding_nature.png")
    plt.close(fig)
    print("  ✓ fig_unesco_binding_nature.png")

    return {l: v for l, v in valid}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 19: Policy Type
# ══════════════════════════════════════════════════════════════════════════════

def fig_policy_type(entries):
    """Boxplot: UNESCO alignment score by policy type."""
    pt_data = defaultdict(list)
    for e in entries:
        pt = e.get("policy_type", "Other")
        pt_data[pt].append(e["unesco_alignment_score"])

    # Sort by mean, keep top groups
    sorted_types = sorted(pt_data.items(), key=lambda x: -np.mean(x[1]))
    valid = [(pt, vals) for pt, vals in sorted_types if len(vals) >= 10]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [v[0] for v in valid]
    data = [v[1] for v in valid]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PAL[i % len(PAL)])
        patch.set_alpha(0.7)

    for i, (label, vals) in enumerate(valid):
        ax.text(i + 1, ax.get_ylim()[0] - 2,
                f"n={len(vals)}\nμ={np.mean(vals):.1f}",
                ha="center", va="top", fontsize=8)

    ax.set_ylabel("UNESCO Alignment Score (0–100)")
    ax.set_title("UNESCO Alignment by Policy Type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_policy_type.png")
    plt.close(fig)
    print("  ✓ fig_unesco_policy_type.png")


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER ANALYSIS (Figures 20–22)
# ══════════════════════════════════════════════════════════════════════════════

def run_cluster_analysis(entries):
    """K-means clustering on 25-item coverage vectors."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Build feature matrix (one row per policy, 25 columns = UNESCO coverage)
    X = np.array([
        [e["unesco_coverage"][u] for u in UNESCO_ITEMS_FLAT]
        for e in entries
    ])

    # Find optimal k via silhouette score
    from sklearn.metrics import silhouette_score
    best_k = 4
    best_sil = -1
    sil_scores = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=300)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_scores[k] = round(sil, 4)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"  Silhouette scores: {sil_scores} → best k={best_k}")

    # Final clustering
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42, max_iter=300)
    labels = km.fit_predict(X)

    # Assign labels to entries
    for e, lab in zip(entries, labels):
        e["cluster"] = int(lab)

    # Build cluster profiles
    cluster_profiles = {}
    for c in range(best_k):
        members = [e for e in entries if e["cluster"] == c]
        profile = {
            "n": len(members),
            "mean_alignment": round(np.mean([e["unesco_alignment_score"] for e in members]), 1),
            "mean_coverage_pct": round(np.mean([e["unesco_coverage_pct"] for e in members]), 1),
        }
        # Per-item coverage rate
        for u_name in UNESCO_ITEMS_FLAT:
            profile[u_name] = round(
                np.mean([e["unesco_coverage"][u_name] for e in members]) * 100, 1
            )
        # Income distribution
        inc_counts = Counter(e["income_binary"] for e in members if e["income_binary"] != "Unknown")
        profile["income_dist"] = dict(inc_counts)
        # Top policy types
        pt_counts = Counter(e.get("policy_type", "Other") for e in members)
        profile["top_types"] = dict(pt_counts.most_common(3))
        cluster_profiles[c] = profile

    # Name clusters based on coverage level
    cluster_names = {}
    sorted_clusters = sorted(cluster_profiles.items(),
                              key=lambda x: x[1]["mean_coverage_pct"], reverse=True)
    archetype_labels = [
        "Comprehensive aligners",
        "Moderate aligners",
        "Selective aligners",
        "Minimal engagement",
        "Very limited",
        "Marginal",
    ]
    for i, (cid, _) in enumerate(sorted_clusters):
        cluster_names[cid] = archetype_labels[i] if i < len(archetype_labels) else f"Cluster {cid}"

    for e in entries:
        e["cluster_name"] = cluster_names[e["cluster"]]

    print(f"  Clusters: {best_k}")
    for cid, name in sorted(cluster_names.items()):
        p = cluster_profiles[cid]
        print(f"    {name}: n={p['n']}, cov={p['mean_coverage_pct']:.0f}%, "
              f"score={p['mean_alignment']:.0f}")

    return cluster_profiles, cluster_names, best_k


def fig_cluster_radar(entries, cluster_profiles, cluster_names, n_clusters):
    """Radar chart: one polygon per cluster across UNESCO items."""
    # Use a subset of UNESCO items for readability (short labels)
    # Use all 25 items
    labels = UNESCO_ITEMS_FLAT
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    cluster_colors = sns.color_palette("husl", n_clusters)

    sorted_clusters = sorted(cluster_profiles.items(),
                              key=lambda x: x[1]["mean_coverage_pct"], reverse=True)

    for cid, profile in sorted_clusters:
        vals = [profile.get(u, 0) for u in labels]
        vals += vals[:1]
        name = cluster_names[cid]
        ax.plot(angles, vals, "o-", linewidth=1.5, markersize=3,
                color=cluster_colors[cid],
                label=f"{name} (n={profile['n']})")
        ax.fill(angles, vals, alpha=0.08, color=cluster_colors[cid])

    ax.set_xticks(angles[:-1])
    short_labels = [l.split(" & ")[0] if len(l) > 22 else l for l in labels]
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title("UNESCO Coverage Profiles by Cluster", fontsize=14, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_cluster_radar.png")
    plt.close(fig)
    print("  ✓ fig_unesco_cluster_radar.png")


def fig_cluster_map(entries, cluster_names):
    """Choropleth world map coloured by most-common cluster per jurisdiction."""
    try:
        import geopandas as gpd
        # Try to get Natural Earth data (works for geopandas < 1.0 and via URL)
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except (AttributeError, Exception):
            url = ("https://naciscdn.org/naturalearth/110m/cultural/"
                   "ne_110m_admin_0_countries.zip")
            try:
                world = gpd.read_file(url)
                # Normalise column name
                if "NAME" in world.columns and "name" not in world.columns:
                    world = world.rename(columns={"NAME": "name"})
            except Exception:
                print("  ⚠ Cannot download Natural Earth data — using bar fallback")
                _fig_cluster_jurisdiction_bar(entries, cluster_names)
                return
    except ImportError:
        print("  ⚠ geopandas not installed — using bar fallback")
        _fig_cluster_jurisdiction_bar(entries, cluster_names)
        return

    # Aggregate: for each jurisdiction, assign the modal cluster
    jur_cluster = defaultdict(list)
    for e in entries:
        jur = e.get("jurisdiction", "")
        if jur and not e["is_international"]:
            jur_cluster[jur].append(e.get("cluster_name", "Unknown"))

    jur_mode = {jur: Counter(clusters).most_common(1)[0][0]
                for jur, clusters in jur_cluster.items()}

    # Name mapping for geopandas natural earth
    name_map = {
        "Korea": "South Korea",
        "Türkiye": "Turkey",
        "China (People's Republic of)": "China",
        "Czech Republic": "Czechia",
        "Slovak Republic": "Slovakia",
        "Viet Nam": "Vietnam",
    }

    cluster_col = []
    for _, row in world.iterrows():
        country = row["name"]
        matched = jur_mode.get(country)
        if not matched:
            for our_name, geo_name in name_map.items():
                if geo_name == country:
                    matched = jur_mode.get(our_name)
                    break
        cluster_col.append(matched if matched else "No data")

    world["cluster"] = cluster_col

    all_clusters = sorted(set(cluster_names.values()))
    color_map = dict(zip(all_clusters, sns.color_palette("husl", len(all_clusters))))
    color_map["No data"] = "#e0e0e0"

    fig, ax = plt.subplots(figsize=(16, 9))
    for cluster_label in all_clusters + ["No data"]:
        subset = world[world["cluster"] == cluster_label]
        if not subset.empty:
            subset.plot(ax=ax, color=color_map[cluster_label],
                       edgecolor="white", linewidth=0.3,
                       label=cluster_label)

    ax.set_title("UNESCO Alignment Clusters: Global Map", fontsize=14)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_cluster_map.png")
    plt.close(fig)
    print("  ✓ fig_unesco_cluster_map.png")


def _fig_cluster_jurisdiction_bar(entries, cluster_names):
    """Fallback: bar chart of cluster distribution by jurisdiction (if no geopandas)."""
    jur_cluster = defaultdict(list)
    for e in entries:
        jur = e.get("jurisdiction", "")
        if jur and not e["is_international"]:
            jur_cluster[jur].append(e.get("cluster_name", "Unknown"))

    # For each jurisdiction, pick the mode cluster
    jur_mode = {jur: Counter(clusters).most_common(1)[0][0]
                for jur, clusters in jur_cluster.items()}

    # Count jurisdictions per cluster
    cluster_counts = Counter(jur_mode.values())
    all_clusters = sorted(set(cluster_names.values()))
    counts = [cluster_counts.get(c, 0) for c in all_clusters]

    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_colors = sns.color_palette("husl", len(all_clusters))
    ax.bar(all_clusters, counts, color=cluster_colors, edgecolor="white")
    for i, (c, cnt) in enumerate(zip(all_clusters, counts)):
        ax.text(i, cnt + 0.5, str(cnt), ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Number of jurisdictions")
    ax.set_title("Jurisdictions per UNESCO Alignment Cluster")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_cluster_map.png")
    plt.close(fig)
    print("  ✓ fig_unesco_cluster_map.png (bar fallback)")


def fig_cluster_income(entries, cluster_names, n_clusters):
    """Stacked bar: income distribution within each cluster."""
    all_clusters = sorted(set(cluster_names.values()),
                           key=lambda x: -np.mean([
                               e["unesco_alignment_score"] for e in entries
                               if e.get("cluster_name") == x
                           ]))

    inc_labels = ["High income", "Developing", "Unknown"]
    inc_colors = [INCOME_COLORS.get(l, "#cccccc") for l in inc_labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_clusters))
    bottoms = np.zeros(len(all_clusters))

    for i_inc, inc_label in enumerate(inc_labels):
        heights = []
        for cluster in all_clusters:
            members = [e for e in entries if e.get("cluster_name") == cluster]
            total = len(members)
            count = sum(1 for e in members if e["income_binary"] == inc_label)
            heights.append(count / total * 100 if total > 0 else 0)
        ax.bar(x, heights, bottom=bottoms, color=inc_colors[i_inc],
               label=inc_label, edgecolor="white", linewidth=0.3)
        bottoms += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(all_clusters, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Share of policies (%)")
    ax.set_title("Income Composition of UNESCO Alignment Clusters")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)

    # Add N labels on top
    for i, cluster in enumerate(all_clusters):
        n = sum(1 for e in entries if e.get("cluster_name") == cluster)
        ax.text(i, 102, f"n={n}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_cluster_income.png")
    plt.close(fig)
    print("  ✓ fig_unesco_cluster_income.png")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(entries):
    """OLS regression: UNESCO alignment score ~ predictors."""
    import statsmodels.api as sm

    # Build regression dataframe
    rows = []
    for e in entries:
        if e["income_binary"] == "Unknown":
            continue
        row = {
            "alignment": e["unesco_alignment_score"],
            "coverage_pct": e["unesco_coverage_pct"],
            "is_developing": 1 if e["income_binary"] == "Developing" else 0,
            "year": e.get("year", 0),
            "post_unesco": 1 if e.get("unesco_era") == "Post-UNESCO" else 0,
            "log_gdp": e.get("log_gdp_pc"),
            "capacity": e.get("capacity_score"),
            "ethics_master": e.get("ethics_score_master"),
            "is_hard_law": 1 if e.get("binding_nature") in ("Hard law", "Binding regulation") else 0,
            "is_soft_law": 1 if e.get("binding_nature") == "Soft law" else 0,
        }
        # Skip if missing key vars
        if row["year"] and row["year"] > 0 and row["log_gdp"] is not None:
            rows.append(row)

    print(f"  Regression sample: {len(rows)} observations")

    if len(rows) < 50:
        print("  ⚠ Too few observations for regression")
        return None

    Y = np.array([r["alignment"] for r in rows])
    X_cols = ["is_developing", "post_unesco", "log_gdp", "is_hard_law", "is_soft_law", "year"]
    X = np.column_stack([[r[c] for r in rows] for c in X_cols])
    X = sm.add_constant(X)

    model1 = sm.OLS(Y, X).fit(cov_type="HC1")
    print(f"  Model 1 (base): R²={model1.rsquared:.3f}, adj-R²={model1.rsquared_adj:.3f}")

    # Model 2: add capacity & ethics scores
    rows2 = [r for r in rows if r["capacity"] is not None and r["ethics_master"] is not None]
    if len(rows2) >= 50:
        Y2 = np.array([r["alignment"] for r in rows2])
        X_cols2 = X_cols + ["capacity", "ethics_master"]
        X2 = np.column_stack([[r[c] for r in rows2] for c in X_cols2])
        X2 = sm.add_constant(X2)
        model2 = sm.OLS(Y2, X2).fit(cov_type="HC1")
        print(f"  Model 2 (+ scores): R²={model2.rsquared:.3f}, adj-R²={model2.rsquared_adj:.3f}")
    else:
        model2 = None

    # Model 3: interaction term (developing × post_unesco)
    interaction = np.array([r["is_developing"] * r["post_unesco"] for r in rows])
    X_cols3 = X_cols + ["dev_x_post"]
    X3_data = np.column_stack([[r[c] for r in rows] for c in X_cols[:-0]] + [interaction])
    # Rebuild properly
    X3 = np.column_stack([
        [r["is_developing"] for r in rows],
        [r["post_unesco"] for r in rows],
        [r["log_gdp"] for r in rows],
        [r["is_hard_law"] for r in rows],
        [r["is_soft_law"] for r in rows],
        [r["year"] for r in rows],
        interaction,
    ])
    X3 = sm.add_constant(X3)
    X_cols3 = ["is_developing", "post_unesco", "log_gdp", "is_hard_law",
               "is_soft_law", "year", "dev_x_post"]
    model3 = sm.OLS(Y, X3).fit(cov_type="HC1")
    print(f"  Model 3 (interaction): R²={model3.rsquared:.3f}")

    # --- Write regression table ---
    with open(OUT / "unesco_regression.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variable", "Model 1 β", "Model 1 SE", "Model 1 p",
                     "Model 3 β", "Model 3 SE", "Model 3 p"])

        col_names_m1 = ["const"] + X_cols
        col_names_m3 = ["const"] + X_cols3

        for i, var in enumerate(col_names_m1):
            m1_b = f"{model1.params[i]:.3f}"
            m1_se = f"{model1.bse[i]:.3f}"
            m1_p = f"{model1.pvalues[i]:.4f}"

            if var in col_names_m3:
                j = col_names_m3.index(var)
                m3_b = f"{model3.params[j]:.3f}"
                m3_se = f"{model3.bse[j]:.3f}"
                m3_p = f"{model3.pvalues[j]:.4f}"
            else:
                m3_b = m3_se = m3_p = ""

            w.writerow([var, m1_b, m1_se, m1_p, m3_b, m3_se, m3_p])

        # Add interaction row
        j_int = col_names_m3.index("dev_x_post")
        w.writerow(["dev_x_post", "", "", "",
                     f"{model3.params[j_int]:.3f}",
                     f"{model3.bse[j_int]:.3f}",
                     f"{model3.pvalues[j_int]:.4f}"])

        w.writerow([])
        w.writerow(["N", len(rows), "", "", len(rows), "", ""])
        w.writerow(["R²", f"{model1.rsquared:.3f}", "", "",
                     f"{model3.rsquared:.3f}", "", ""])
        w.writerow(["Adj R²", f"{model1.rsquared_adj:.3f}", "", "",
                     f"{model3.rsquared_adj:.3f}", "", ""])

    print("  ✓ unesco_regression.csv")

    reg_stats = {
        "model1": {
            "n": int(model1.nobs),
            "r_squared": round(model1.rsquared, 4),
            "adj_r_squared": round(model1.rsquared_adj, 4),
            "f_stat": round(float(model1.fvalue), 3),
            "f_pvalue": round(float(model1.f_pvalue), 4),
            "coefficients": {
                name: {
                    "beta": round(float(model1.params[i]), 4),
                    "se": round(float(model1.bse[i]), 4),
                    "t": round(float(model1.tvalues[i]), 3),
                    "p": round(float(model1.pvalues[i]), 4),
                }
                for i, name in enumerate(col_names_m1)
            },
        },
        "model3_interaction": {
            "n": int(model3.nobs),
            "r_squared": round(model3.rsquared, 4),
            "dev_x_post_beta": round(float(model3.params[j_int]), 4),
            "dev_x_post_p": round(float(model3.pvalues[j_int]), 4),
        },
    }

    if model2:
        col_names_m2 = ["const"] + X_cols + ["capacity", "ethics_master"]
        reg_stats["model2"] = {
            "n": int(model2.nobs),
            "r_squared": round(model2.rsquared, 4),
            "adj_r_squared": round(model2.rsquared_adj, 4),
            "coefficients": {
                name: {
                    "beta": round(float(model2.params[i]), 4),
                    "se": round(float(model2.bse[i]), 4),
                    "p": round(float(model2.pvalues[i]), 4),
                }
                for i, name in enumerate(col_names_m2)
            },
        }

    return reg_stats


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def write_pre_post_table(pre, post, pre_cov, post_cov, diffs):
    """Write pre/post comparison CSV."""
    with open(OUT / "unesco_pre_post.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["UNESCO Item", "Type", "Pre Coverage %", "Post Coverage %",
                     "Δ (pp)", "Chi²", "p-value"])
        for u_name in UNESCO_ITEMS_FLAT:
            typ = TYPE_MAP[u_name]
            pre_yes = sum(1 for e in pre if e["unesco_coverage"][u_name] == 1)
            pre_no = len(pre) - pre_yes
            post_yes = sum(1 for e in post if e["unesco_coverage"][u_name] == 1)
            post_no = len(post) - post_yes
            if (pre_yes + post_yes) > 0:
                chi2, p = sp_stats.chi2_contingency(
                    [[pre_yes, pre_no], [post_yes, post_no]]
                )[:2]
            else:
                chi2, p = 0, 1
            w.writerow([u_name, typ,
                        f"{pre_cov[u_name]:.1f}", f"{post_cov[u_name]:.1f}",
                        f"{diffs[u_name]:+.1f}",
                        f"{chi2:.3f}", f"{p:.4f}"])

        # Overall row
        pre_scores = [e["unesco_alignment_score"] for e in pre]
        post_scores = [e["unesco_alignment_score"] for e in post]
        t, p = sp_stats.ttest_ind(pre_scores, post_scores, equal_var=False)
        w.writerow([])
        w.writerow(["OVERALL (alignment score)", "",
                     f"{np.mean(pre_scores):.1f}", f"{np.mean(post_scores):.1f}",
                     f"{np.mean(post_scores)-np.mean(pre_scores):+.1f}",
                     f"t={t:.3f}", f"{p:.4f}"])
    print("  ✓ unesco_pre_post.csv")

    return {
        "pre_n": len(pre), "post_n": len(post),
        "pre_mean_score": round(np.mean(pre_scores), 2),
        "post_mean_score": round(np.mean(post_scores), 2),
        "t_stat": round(float(t), 3),
        "p_value": round(float(p), 4),
    }


def write_binding_table(entries, bn_data):
    """Write binding nature comparison CSV."""
    with open(OUT / "unesco_binding.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Binding Nature", "N", "Mean Alignment", "SD", "Mean Coverage %"])
        for bn, vals in sorted(bn_data.items(), key=lambda x: -np.mean(x[1])):
            members = [e for e in entries if e.get("binding_nature") == bn]
            covs = [e["unesco_coverage_pct"] for e in members]
            w.writerow([bn, len(vals), f"{np.mean(vals):.1f}",
                        f"{np.std(vals):.1f}", f"{np.mean(covs):.1f}"])

    # ANOVA
    groups = list(bn_data.values())
    if len(groups) >= 2:
        f_stat, p_anova = sp_stats.f_oneway(*groups)
    else:
        f_stat, p_anova = 0, 1

    print("  ✓ unesco_binding.csv")
    return {"f_stat": round(float(f_stat), 3), "p_anova": round(float(p_anova), 4)}


def write_cluster_table(entries, cluster_profiles, cluster_names):
    """Write cluster profiles CSV."""
    with open(OUT / "unesco_clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Cluster", "N", "Mean Alignment", "Mean Coverage %"]
        header += UNESCO_ITEMS_FLAT
        header += ["HI %", "Dev %"]
        w.writerow(header)

        sorted_clusters = sorted(cluster_profiles.items(),
                                  key=lambda x: -x[1]["mean_coverage_pct"])
        for cid, profile in sorted_clusters:
            name = cluster_names[cid]
            n = profile["n"]
            inc_dist = profile.get("income_dist", {})
            hi_pct = inc_dist.get("High income", 0) / n * 100 if n > 0 else 0
            dev_pct = inc_dist.get("Developing", 0) / n * 100 if n > 0 else 0
            row = [name, n, profile["mean_alignment"], profile["mean_coverage_pct"]]
            row += [profile.get(u, 0) for u in UNESCO_ITEMS_FLAT]
            row += [f"{hi_pct:.0f}", f"{dev_pct:.0f}"]
            w.writerow(row)

    print("  ✓ unesco_clusters.csv")


# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE STATS
# ══════════════════════════════════════════════════════════════════════════════

def write_stats(entries, coverage, pre_post_stats, binding_stats, reg_stats,
                cluster_profiles, cluster_names, item_depth):
    """Write comprehensive paper 3 stats JSON."""
    stats = {"generated": datetime.now().isoformat()}

    # Descriptives
    all_scores = [e["unesco_alignment_score"] for e in entries]
    stats["descriptives"] = {
        "n_policies": len(entries),
        "n_jurisdictions": len(set(e.get("jurisdiction", "") for e in entries)),
        "alignment_mean": round(np.mean(all_scores), 2),
        "alignment_median": round(float(np.median(all_scores)), 2),
        "alignment_sd": round(np.std(all_scores), 2),
        "coverage_mean": round(np.mean([e["unesco_coverage_pct"] for e in entries]), 2),
        "values_coverage_mean": round(np.mean([e["unesco_values_cov"] for e in entries]), 2),
        "principles_coverage_mean": round(np.mean([e["unesco_principles_cov"] for e in entries]), 2),
        "policy_areas_coverage_mean": round(np.mean([e["unesco_policy_areas_cov"] for e in entries]), 2),
    }

    # Coverage vs depth
    stats["coverage_depth_items"] = {
        u: {"coverage_pct": round(coverage[u], 1),
            "mean_depth": round(item_depth.get(u, 0), 2)}
        for u in UNESCO_ITEMS_FLAT
    }

    # Coverage-depth correlation
    cov_vals = [coverage[u] for u in UNESCO_ITEMS_FLAT]
    dep_vals = [item_depth.get(u, 0) for u in UNESCO_ITEMS_FLAT]
    r, p = sp_stats.pearsonr(cov_vals, dep_vals)
    stats["coverage_depth_correlation"] = {"r": round(r, 3), "p": round(p, 4)}

    # Pre/post
    stats["pre_post"] = pre_post_stats

    # Binding nature
    stats["binding_nature"] = binding_stats

    # Regression
    if reg_stats:
        stats["regression"] = reg_stats

    # Clusters
    stats["clusters"] = {
        "n_clusters": len(cluster_profiles),
        "profiles": {
            cluster_names[cid]: {
                "n": p["n"],
                "mean_alignment": p["mean_alignment"],
                "mean_coverage_pct": p["mean_coverage_pct"],
                "income_dist": p.get("income_dist", {}),
            }
            for cid, p in cluster_profiles.items()
        },
    }

    with open(OUT / "unesco_paper3_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  ✓ unesco_paper3_stats.json")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PAPER 3 — EXTENDED UNESCO ALIGNMENT ANALYSIS")
    print("=" * 70)

    print("\n── Phase A: Loading & enriching data ──")
    entries = load_and_enrich()

    hi = [e for e in entries if e["income_binary"] == "High income"]
    dev = [e for e in entries if e["income_binary"] == "Developing"]
    print(f"\n  High income: {len(hi)}  |  Developing: {len(dev)}")
    print(f"  Pre-UNESCO (≤{UNESCO_CUTOFF}): "
          f"{sum(1 for e in entries if e['unesco_era'] == 'Pre-UNESCO')}")
    print(f"  Post-UNESCO (≥{UNESCO_CUTOFF+1}): "
          f"{sum(1 for e in entries if e['unesco_era'] == 'Post-UNESCO')}")

    coverage = compute_coverage(entries)

    # ── Phase B: Figures ──
    print("\n── Phase B: Generating new figures ──")
    fig_coverage_by_layer(entries)
    item_depth = fig_coverage_vs_depth(entries, coverage)
    fig_income_alignment_dist(entries)
    fig_region_scores(entries)
    pre, post, pre_cov, post_cov, diffs = fig_pre_post(entries)
    fig_pre_post_income(entries)
    bn_data = fig_binding_nature(entries)
    fig_policy_type(entries)

    print("\n── Phase B.2: Cluster analysis ──")
    cluster_profiles, cluster_names, n_clusters = run_cluster_analysis(entries)
    fig_cluster_radar(entries, cluster_profiles, cluster_names, n_clusters)
    fig_cluster_map(entries, cluster_names)
    fig_cluster_income(entries, cluster_names, n_clusters)

    # ── Phase C: Tables ──
    print("\n── Phase C: Writing tables ──")
    pre_post_stats = write_pre_post_table(pre, post, pre_cov, post_cov, diffs)
    binding_stats = write_binding_table(entries, bn_data)
    write_cluster_table(entries, cluster_profiles, cluster_names)

    # ── Phase C.2: Regression ──
    print("\n── Phase C.2: Regression analysis ──")
    reg_stats = run_regression(entries)

    # ── Phase D: Stats ──
    print("\n── Phase D: Writing stats ──")
    stats = write_stats(entries, coverage, pre_post_stats, binding_stats,
                        reg_stats, cluster_profiles, cluster_names, item_depth)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    d = stats["descriptives"]
    print(f"\n  Policies: {d['n_policies']} from {d['n_jurisdictions']} jurisdictions")
    print(f"  Alignment: μ={d['alignment_mean']}, σ={d['alignment_sd']}")
    print(f"  Coverage layers: values={d['values_coverage_mean']:.1f}%, "
          f"principles={d['principles_coverage_mean']:.1f}%, "
          f"policy areas={d['policy_areas_coverage_mean']:.1f}%")

    pp = stats["pre_post"]
    print(f"\n  Pre/Post UNESCO: {pp['pre_mean_score']} → {pp['post_mean_score']} "
          f"(p={pp['p_value']})")

    bn = stats["binding_nature"]
    print(f"  Binding nature ANOVA: F={bn['f_stat']}, p={bn['p_anova']}")

    if reg_stats and "model1" in reg_stats:
        m1 = reg_stats["model1"]
        print(f"  Regression R²={m1['r_squared']}")
        for var, coef in m1["coefficients"].items():
            if var != "const" and coef["p"] < 0.05:
                print(f"    {var}: β={coef['beta']}, p={coef['p']}")

    cd = stats.get("coverage_depth_correlation", {})
    print(f"\n  Coverage–depth correlation: r={cd.get('r')}, p={cd.get('p')}")

    print(f"\n✅ All Paper 3 outputs saved to: {OUT}")
    print(f"  New figures: 11")
    print(f"  New tables: 4 CSV")
    print(f"  New stats: 1 JSON")


if __name__ == "__main__":
    main()
