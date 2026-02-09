"""
UNESCO Alignment Analysis
=========================
Maps each policy's ethics content against the UNESCO Recommendation on the
Ethics of Artificial Intelligence (2021) framework:

  • 4 Values
  • 10 Principles
  • 11 Policy Action Areas

Reads:   phase_c_depth_normalised.jsonl
Outputs: data/analysis/paper2_ethics/unesco/

Figures:
  1. fig_unesco_coverage_overview.png       – % policies covering each UNESCO item
  2. fig_unesco_depth_heatmap.png           – UNESCO items × depth level heatmap
  3. fig_unesco_radar_income.png            – radar chart: HI vs Developing
  4. fig_unesco_compliance_score.png        – histogram of per-policy UNESCO scores
  5. fig_unesco_income_gap.png              – income gap per UNESCO item
  6. fig_unesco_region_heatmap.png          – regions × UNESCO items heatmap
  7. fig_unesco_gaps_top_bottom.png         – most/least covered UNESCO items
  8. fig_unesco_values_depth.png            – 4 UNESCO values depth breakdown
  9. fig_unesco_principles_depth.png        – 10 UNESCO principles depth breakdown
 10. fig_unesco_policy_areas_depth.png      – 11 policy action areas depth breakdown
 11. fig_unesco_score_vs_year.png           – UNESCO score trend over time

Tables:
  unesco_coverage_table.csv                – coverage & depth per UNESCO item
  unesco_by_income.csv                     – income-group comparison
  unesco_by_region.csv                     – regional comparison
  unesco_policy_scores.csv                 – per-policy UNESCO alignment scores
  unesco_mapping.csv                       – canonical → UNESCO mapping reference

Stats:
  unesco_stats.json                        – statistical tests & descriptive stats
"""

import json, sys, warnings, math
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "analysis" / "ethics_inventory"
INPUT = DATA / "phase_c_depth_normalised.jsonl"
OUT = ROOT / "data" / "analysis" / "paper2_ethics" / "unesco"
OUT.mkdir(parents=True, exist_ok=True)

# ── Country metadata ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from country_metadata import (
    INCOME_GROUP, REGION, REGION_LABELS,
    INTERNATIONAL, get_income_binary, get_metadata,
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

DEPTH_ORDER = ["word", "phrase", "sentence", "paragraph", "section"]
DEPTH_WEIGHTS = {
    "word": 1, "phrase": 2, "sentence": 3, "paragraph": 4, "section": 5,
    "not_found": 0,
}
DEPTH_COLORS = {
    "word":      "#fee5d9",
    "phrase":    "#fcae91",
    "sentence":  "#fb6a4a",
    "paragraph": "#de2d26",
    "section":   "#a50f15",
    "not_found": "#cccccc",
}
INCOME_COLORS = {"High income": PAL[0], "Developing": PAL[2]}
CAT_COLORS = {"values": PAL[0], "principles": PAL[1], "mechanisms": PAL[2]}

# ══════════════════════════════════════════════════════════════════════════════
# UNESCO RECOMMENDATION ON THE ETHICS OF AI (2021)
# ══════════════════════════════════════════════════════════════════════════════

UNESCO_VALUES = [
    "Human rights & human dignity",
    "Peaceful, just & interconnected societies",
    "Diversity & inclusiveness",
    "Environment & ecosystem flourishing",
]

UNESCO_PRINCIPLES = [
    "Proportionality & do no harm",
    "Safety & security",
    "Fairness & non-discrimination",
    "Sustainability",
    "Right to privacy & data protection",
    "Human oversight & determination",
    "Transparency & explainability",
    "Responsibility & accountability",
    "Awareness & literacy",
    "Multi-stakeholder & adaptive governance",
]

UNESCO_POLICY_AREAS = [
    "Ethical impact assessment",
    "Ethical governance & stewardship",
    "Data policy",
    "Development & international cooperation",
    "Environment & ecosystem",
    "Gender",
    "Culture",
    "Education & research",
    "Communication & information",
    "Economy & labour",
    "Health & social well-being",
]

ALL_UNESCO = (
    [(v, "value") for v in UNESCO_VALUES]
    + [(p, "principle") for p in UNESCO_PRINCIPLES]
    + [(a, "policy_area") for a in UNESCO_POLICY_AREAS]
)
UNESCO_ITEMS_FLAT = [name for name, _ in ALL_UNESCO]

# ══════════════════════════════════════════════════════════════════════════════
# MAPPING: Our 56 canonical items  →  UNESCO framework
# ══════════════════════════════════════════════════════════════════════════════
# Many-to-one: each of our items maps to one or more UNESCO items.

CANONICAL_TO_UNESCO = {
    # ── VALUES ──────────────────────────────────────────────────────────────
    # Our canonical values → UNESCO values
    "Human dignity":                    ["Human rights & human dignity"],
    "Human rights":                     ["Human rights & human dignity"],
    "Fairness / Justice / Equity":      ["Peaceful, just & interconnected societies",
                                         "Fairness & non-discrimination"],
    "Non-discrimination":               ["Fairness & non-discrimination",
                                         "Gender"],
    "Privacy / Data protection":        ["Right to privacy & data protection",
                                         "Data policy"],
    "Autonomy / Self-determination":    ["Human rights & human dignity",
                                         "Human oversight & determination"],
    "Well-being / Beneficence":         ["Health & social well-being"],
    "Non-maleficence / Do no harm":     ["Proportionality & do no harm"],
    "Freedom / Liberty":                ["Human rights & human dignity"],
    "Solidarity":                       ["Peaceful, just & interconnected societies",
                                         "Development & international cooperation"],
    "Sustainability / Environment":     ["Sustainability",
                                         "Environment & ecosystem flourishing",
                                         "Environment & ecosystem"],
    "Peace":                            ["Peaceful, just & interconnected societies"],
    "Cultural diversity":               ["Diversity & inclusiveness",
                                         "Culture"],
    "Democracy":                        ["Peaceful, just & interconnected societies"],
    "Rule of law":                      ["Ethical governance & stewardship"],
    "Trust":                            ["Transparency & explainability"],
    "Safety / Security":                ["Safety & security"],
    "Public interest / Common good":    ["Peaceful, just & interconnected societies",
                                         "Economy & labour"],

    # ── PRINCIPLES ──────────────────────────────────────────────────────────
    "Transparency":                     ["Transparency & explainability"],
    "Explainability / Interpretability":["Transparency & explainability"],
    "Accountability":                   ["Responsibility & accountability"],
    "Responsibility":                   ["Responsibility & accountability"],
    "Robustness / Reliability":         ["Safety & security"],
    "Human oversight / Human-in-the-loop": ["Human oversight & determination"],
    "Proportionality":                  ["Proportionality & do no harm"],
    "Precaution / Risk-based approach": ["Proportionality & do no harm",
                                         "Ethical impact assessment"],
    "Inclusiveness / Accessibility":    ["Diversity & inclusiveness",
                                         "Fairness & non-discrimination"],
    "Interoperability":                 ["Ethical governance & stewardship"],
    "Contestability / Right to appeal": ["Human rights & human dignity"],
    "Data governance / Data quality":   ["Data policy",
                                         "Right to privacy & data protection"],
    "Open source / Openness":           ["Transparency & explainability"],
    "International cooperation":        ["Development & international cooperation",
                                         "Multi-stakeholder & adaptive governance"],
    "Multi-stakeholder governance":     ["Multi-stakeholder & adaptive governance"],
    "Informed consent":                 ["Right to privacy & data protection"],
    "Purpose limitation":               ["Right to privacy & data protection",
                                         "Data policy"],
    "Due diligence":                    ["Responsibility & accountability",
                                         "Ethical impact assessment"],

    # ── MECHANISMS ──────────────────────────────────────────────────────────
    "Ethics board / Ethics committee":  ["Ethical governance & stewardship",
                                         "Multi-stakeholder & adaptive governance"],
    "Impact assessment":                ["Ethical impact assessment"],
    "Algorithmic auditing / Third-party audit":
                                        ["Responsibility & accountability",
                                         "Ethical impact assessment"],
    "Certification / Conformity assessment":
                                        ["Ethical governance & stewardship"],
    "Regulatory sandbox":               ["Ethical governance & stewardship"],
    "Complaints / Redress mechanism":   ["Human rights & human dignity"],
    "Standards / Technical standards":  ["Ethical governance & stewardship"],
    "Training / Capacity building":     ["Awareness & literacy",
                                         "Education & research"],
    "Labelling / Disclosure requirements":
                                        ["Transparency & explainability",
                                         "Communication & information"],
    "Registration / Inventory of AI systems":
                                        ["Ethical governance & stewardship"],
    "Risk classification / Tiered regulation":
                                        ["Proportionality & do no harm",
                                         "Ethical impact assessment"],
    "Sectoral regulation":              ["Ethical governance & stewardship",
                                         "Economy & labour"],
    "Procurement requirements":         ["Ethical governance & stewardship"],
    "Monitoring & evaluation / Reporting":
                                        ["Responsibility & accountability",
                                         "Ethical governance & stewardship"],
    "Penalties / Sanctions / Enforcement":
                                        ["Ethical governance & stewardship"],
    "Insurance / Liability framework":  ["Responsibility & accountability"],
    "Code of conduct / Voluntary commitments":
                                        ["Ethical governance & stewardship"],
    "Moratorium / Ban on specific uses":["Safety & security",
                                         "Proportionality & do no harm"],
    "Data sharing / Open data":         ["Data policy"],
    "Whistleblower protection":         ["Responsibility & accountability"],
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & ENRICH
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    entries = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} normalised depth entries")
    return entries


def enrich_entries(entries):
    """Add income_binary, region, year, and UNESCO-aligned scores."""
    for e in entries:
        jur = e.get("jurisdiction", "")
        e["income_binary"] = get_income_binary(jur) or "Unknown"
        meta = get_metadata(jur)
        e["region"] = REGION_LABELS.get(
            meta.get("region", ""), meta.get("region_label", "Unknown")
        )
        e["is_international"] = jur in INTERNATIONAL

        # Map each item to UNESCO categories and compute per-UNESCO scores
        unesco_hits = defaultdict(list)  # unesco_item → [depth_weights]
        dd = e.get("depth_data", {})
        for cat in ("values", "principles", "mechanisms"):
            for item in dd.get(cat, []):
                canon_name = item.get("name", "Other")
                depth_str = item.get("depth", "not_found").lower().strip()
                dw = DEPTH_WEIGHTS.get(depth_str, 0)
                # Look up UNESCO mapping
                unesco_targets = CANONICAL_TO_UNESCO.get(canon_name, [])
                for ut in unesco_targets:
                    unesco_hits[ut].append(dw)

        # Per-UNESCO-item depth score (mean of depth weights, 0 if absent)
        e["unesco_scores"] = {}
        for u_name in UNESCO_ITEMS_FLAT:
            weights = unesco_hits.get(u_name, [])
            if weights:
                positive = [w for w in weights if w > 0]
                e["unesco_scores"][u_name] = np.mean(positive) if positive else 0
            else:
                e["unesco_scores"][u_name] = 0

        # Coverage: binary (1 if mentioned, 0 if not)
        e["unesco_coverage"] = {
            u_name: (1 if e["unesco_scores"][u_name] > 0 else 0)
            for u_name in UNESCO_ITEMS_FLAT
        }

        # Aggregate UNESCO compliance score (0-100)
        coverage_pct = sum(e["unesco_coverage"].values()) / len(UNESCO_ITEMS_FLAT) * 100
        avg_depth = np.mean([
            e["unesco_scores"][u] for u in UNESCO_ITEMS_FLAT
            if e["unesco_scores"][u] > 0
        ]) if any(e["unesco_scores"][u] > 0 for u in UNESCO_ITEMS_FLAT) else 0
        # Combined score: 60% coverage breadth + 40% depth quality (normalised to 0-100)
        e["unesco_alignment_score"] = 0.6 * coverage_pct + 0.4 * (avg_depth / 5 * 100)
        e["unesco_coverage_pct"] = coverage_pct
        e["unesco_avg_depth"] = avg_depth

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: UNESCO Coverage Overview
# ══════════════════════════════════════════════════════════════════════════════

def fig_coverage_overview(entries):
    """Bar chart: % of policies covering each UNESCO item."""
    n = len(entries)
    coverage = {}
    for u_name in UNESCO_ITEMS_FLAT:
        count = sum(1 for e in entries if e["unesco_coverage"][u_name] == 1)
        coverage[u_name] = count / n * 100

    # Sort by coverage
    sorted_items = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    pcts = [x[1] for x in sorted_items]

    # Color by type
    type_map = {name: typ for name, typ in ALL_UNESCO}
    type_colors = {"value": PAL[0], "principle": PAL[1], "policy_area": PAL[2]}
    colors = [type_colors[type_map[n]] for n in names]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, pcts, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Policies covering this UNESCO item (%)")
    ax.set_title("UNESCO Framework Coverage Across AI Policies")
    ax.invert_yaxis()

    # Add percentage labels
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PAL[0], label="Values (4)"),
        Patch(facecolor=PAL[1], label="Principles (10)"),
        Patch(facecolor=PAL[2], label="Policy Areas (11)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_xlim(0, max(pcts) + 8)
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_coverage_overview.png")
    plt.close(fig)
    print("  ✓ fig_unesco_coverage_overview.png")
    return coverage


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: UNESCO Depth Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_depth_heatmap(entries):
    """Heatmap: UNESCO items × depth levels."""
    depth_counts = defaultdict(Counter)  # unesco_item → {depth_level: count}

    for e in entries:
        dd = e.get("depth_data", {})
        for cat in ("values", "principles", "mechanisms"):
            for item in dd.get(cat, []):
                canon_name = item.get("name", "Other")
                depth_str = item.get("depth", "not_found").lower().strip()
                unesco_targets = CANONICAL_TO_UNESCO.get(canon_name, [])
                for ut in unesco_targets:
                    depth_counts[ut][depth_str] += 1

    # Build matrix
    rows = UNESCO_ITEMS_FLAT
    cols = DEPTH_ORDER
    matrix = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        total = sum(depth_counts[r].values())
        for j, c in enumerate(cols):
            matrix[i, j] = depth_counts[r].get(c, 0) / total * 100 if total > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="YlOrRd",
        xticklabels=[d.capitalize() for d in cols],
        yticklabels=rows,
        ax=ax, cbar_kws={"label": "% of mentions"},
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Depth of Engagement with UNESCO Framework Items")
    ax.set_xlabel("Depth Level")

    # Add type separators
    ax.axhline(y=4, color="black", linewidth=2)
    ax.axhline(y=14, color="black", linewidth=2)

    # Type labels on right
    ax.text(len(cols) + 0.3, 2, "VALUES", va="center", ha="left",
            fontsize=9, fontweight="bold", color=PAL[0])
    ax.text(len(cols) + 0.3, 9, "PRINCIPLES", va="center", ha="left",
            fontsize=9, fontweight="bold", color=PAL[1])
    ax.text(len(cols) + 0.3, 19.5, "POLICY AREAS", va="center", ha="left",
            fontsize=9, fontweight="bold", color=PAL[2])

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_depth_heatmap.png")
    plt.close(fig)
    print("  ✓ fig_unesco_depth_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Radar Chart – HI vs Developing
# ══════════════════════════════════════════════════════════════════════════════

def fig_radar_income(entries):
    """Radar chart comparing High-income vs Developing on UNESCO items."""
    hi = [e for e in entries if e["income_binary"] == "High income"]
    dev = [e for e in entries if e["income_binary"] == "Developing"]

    # Use coverage rates
    hi_coverage = {}
    dev_coverage = {}
    for u_name in UNESCO_ITEMS_FLAT:
        hi_coverage[u_name] = np.mean([e["unesco_coverage"][u_name] for e in hi]) * 100 if hi else 0
        dev_coverage[u_name] = np.mean([e["unesco_coverage"][u_name] for e in dev]) * 100 if dev else 0

    # Radar
    labels = UNESCO_ITEMS_FLAT
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    hi_vals = [hi_coverage[l] for l in labels] + [hi_coverage[labels[0]]]
    dev_vals = [dev_coverage[l] for l in labels] + [dev_coverage[labels[0]]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, hi_vals, "o-", color=INCOME_COLORS["High income"],
            linewidth=1.5, markersize=3, label="High income")
    ax.fill(angles, hi_vals, alpha=0.15, color=INCOME_COLORS["High income"])
    ax.plot(angles, dev_vals, "o-", color=INCOME_COLORS["Developing"],
            linewidth=1.5, markersize=3, label="Developing")
    ax.fill(angles, dev_vals, alpha=0.15, color=INCOME_COLORS["Developing"])

    ax.set_xticks(angles[:-1])
    # Shorten labels for readability
    short_labels = [l.split(" & ")[0] if len(l) > 25 else l for l in labels]
    ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_title("UNESCO Framework Coverage:\nHigh Income vs Developing Countries",
                 fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_radar_income.png")
    plt.close(fig)
    print("  ✓ fig_unesco_radar_income.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: UNESCO Alignment Score Distribution
# ══════════════════════════════════════════════════════════════════════════════

def fig_compliance_score(entries):
    """Histogram of per-policy UNESCO alignment scores."""
    hi = [e["unesco_alignment_score"] for e in entries if e["income_binary"] == "High income"]
    dev = [e["unesco_alignment_score"] for e in entries if e["income_binary"] == "Developing"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 100, 26)
    ax.hist(hi, bins=bins, alpha=0.6, color=INCOME_COLORS["High income"],
            label=f"High income (n={len(hi)}, μ={np.mean(hi):.1f})", edgecolor="white")
    ax.hist(dev, bins=bins, alpha=0.6, color=INCOME_COLORS["Developing"],
            label=f"Developing (n={len(dev)}, μ={np.mean(dev):.1f})", edgecolor="white")

    ax.axvline(np.mean(hi), color=INCOME_COLORS["High income"], linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(dev), color=INCOME_COLORS["Developing"], linestyle="--", linewidth=1.5)

    ax.set_xlabel("UNESCO Alignment Score (0–100)")
    ax.set_ylabel("Number of policies")
    ax.set_title("Distribution of UNESCO Alignment Scores")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_compliance_score.png")
    plt.close(fig)
    print("  ✓ fig_unesco_compliance_score.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Income Gap per UNESCO Item
# ══════════════════════════════════════════════════════════════════════════════

def fig_income_gap(entries):
    """Horizontal bar: income gap (HI% − Dev%) for each UNESCO item."""
    hi = [e for e in entries if e["income_binary"] == "High income"]
    dev = [e for e in entries if e["income_binary"] == "Developing"]

    gaps = {}
    for u_name in UNESCO_ITEMS_FLAT:
        hi_pct = np.mean([e["unesco_coverage"][u_name] for e in hi]) * 100 if hi else 0
        dev_pct = np.mean([e["unesco_coverage"][u_name] for e in dev]) * 100 if dev else 0
        gaps[u_name] = hi_pct - dev_pct

    sorted_items = sorted(gaps.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PAL[0] if v >= 0 else PAL[2] for v in vals]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coverage gap (High income % − Developing %)")
    ax.set_title("Income Gap in UNESCO Framework Coverage")

    for i, (name, val) in enumerate(zip(names, vals)):
        ax.text(val + (0.5 if val >= 0 else -0.5), i,
                f"{val:+.1f}pp", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_income_gap.png")
    plt.close(fig)
    print("  ✓ fig_unesco_income_gap.png")
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Region × UNESCO Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_region_heatmap(entries):
    """Heatmap: regions × UNESCO items (coverage %)."""
    # Exclude international organisations and Unknown
    filtered = [e for e in entries
                if e["region"] not in ("Unknown",) and not e["is_international"]]

    regions = sorted(set(e["region"] for e in filtered))
    region_coverage = {}
    for reg in regions:
        reg_entries = [e for e in filtered if e["region"] == reg]
        if len(reg_entries) < 3:
            continue
        reg_cov = {}
        for u_name in UNESCO_ITEMS_FLAT:
            reg_cov[u_name] = np.mean([
                e["unesco_coverage"][u_name] for e in reg_entries
            ]) * 100
        region_coverage[reg] = reg_cov

    valid_regions = sorted(region_coverage.keys())
    if not valid_regions:
        print("  ⚠ Not enough regional data for heatmap")
        return

    matrix = np.zeros((len(valid_regions), len(UNESCO_ITEMS_FLAT)))
    for i, reg in enumerate(valid_regions):
        for j, u_name in enumerate(UNESCO_ITEMS_FLAT):
            matrix[i, j] = region_coverage[reg].get(u_name, 0)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="YlGnBu",
        xticklabels=UNESCO_ITEMS_FLAT,
        yticklabels=valid_regions,
        ax=ax, cbar_kws={"label": "Coverage (%)"},
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=100,
    )
    ax.set_title("UNESCO Framework Coverage by Region")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)

    # Vertical separators
    ax.axvline(x=4, color="black", linewidth=2)
    ax.axvline(x=14, color="black", linewidth=2)

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_region_heatmap.png")
    plt.close(fig)
    print("  ✓ fig_unesco_region_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Top/Bottom Coverage Gaps
# ══════════════════════════════════════════════════════════════════════════════

def fig_gaps_top_bottom(entries, coverage):
    """Side-by-side: top 10 most and least covered UNESCO items."""
    sorted_items = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_items[:10]
    bottom10 = sorted_items[-10:][::-1]

    type_map = {name: typ for name, typ in ALL_UNESCO}
    type_colors = {"value": PAL[0], "principle": PAL[1], "policy_area": PAL[2]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Top 10
    names = [x[0] for x in top10]
    pcts = [x[1] for x in top10]
    colors = [type_colors[type_map[n]] for n in names]
    y = np.arange(len(names))
    ax1.barh(y, pcts, color=colors, edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Coverage (%)")
    ax1.set_title("Most Covered UNESCO Items")
    for i, pct in enumerate(pcts):
        ax1.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=8)

    # Bottom 10
    names = [x[0] for x in bottom10]
    pcts = [x[1] for x in bottom10]
    colors = [type_colors[type_map[n]] for n in names]
    y = np.arange(len(names))
    ax2.barh(y, pcts, color=colors, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Coverage (%)")
    ax2.set_title("Least Covered UNESCO Items")
    for i, pct in enumerate(pcts):
        ax2.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PAL[0], label="Values"),
        Patch(facecolor=PAL[1], label="Principles"),
        Patch(facecolor=PAL[2], label="Policy Areas"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_gaps_top_bottom.png")
    plt.close(fig)
    print("  ✓ fig_unesco_gaps_top_bottom.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 8-10: Depth Breakdown per UNESCO Sub-Framework
# ══════════════════════════════════════════════════════════════════════════════

def _depth_breakdown_figure(entries, items_list, title, filename, color):
    """Stacked horizontal bar: depth-level breakdown for a set of UNESCO items."""
    depth_counts = defaultdict(Counter)

    for e in entries:
        dd = e.get("depth_data", {})
        for cat in ("values", "principles", "mechanisms"):
            for item in dd.get(cat, []):
                canon_name = item.get("name", "Other")
                depth_str = item.get("depth", "not_found").lower().strip()
                unesco_targets = CANONICAL_TO_UNESCO.get(canon_name, [])
                for ut in unesco_targets:
                    if ut in items_list:
                        depth_counts[ut][depth_str] += 1

    fig, ax = plt.subplots(figsize=(12, max(5, len(items_list) * 0.6)))
    y_pos = np.arange(len(items_list))
    lefts = np.zeros(len(items_list))

    for depth_level in DEPTH_ORDER:
        widths = []
        for u_name in items_list:
            total = sum(depth_counts[u_name].values())
            pct = depth_counts[u_name].get(depth_level, 0) / total * 100 if total > 0 else 0
            widths.append(pct)
        ax.barh(y_pos, widths, left=lefts, color=DEPTH_COLORS[depth_level],
                label=depth_level.capitalize(), edgecolor="white", linewidth=0.3)
        lefts += np.array(widths)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(items_list, fontsize=9)
    ax.set_xlabel("Share of mentions (%)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9)

    # Add total mention count on right
    for i, u_name in enumerate(items_list):
        total = sum(depth_counts[u_name].values())
        ax.text(101, i, f"n={total:,}", va="center", fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def fig_values_depth(entries):
    _depth_breakdown_figure(
        entries, UNESCO_VALUES,
        "Depth of Engagement: UNESCO Values (4)",
        "fig_unesco_values_depth.png", PAL[0],
    )


def fig_principles_depth(entries):
    _depth_breakdown_figure(
        entries, UNESCO_PRINCIPLES,
        "Depth of Engagement: UNESCO Principles (10)",
        "fig_unesco_principles_depth.png", PAL[1],
    )


def fig_policy_areas_depth(entries):
    _depth_breakdown_figure(
        entries, UNESCO_POLICY_AREAS,
        "Depth of Engagement: UNESCO Policy Action Areas (11)",
        "fig_unesco_policy_areas_depth.png", PAL[2],
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: UNESCO Score vs Year
# ══════════════════════════════════════════════════════════════════════════════

def fig_score_vs_year(entries):
    """Scatter + trend: UNESCO alignment score vs year."""
    years = [e.get("year", 0) for e in entries]
    scores = [e["unesco_alignment_score"] for e in entries]
    income = [e["income_binary"] for e in entries]

    # Filter valid years
    valid = [(y, s, inc) for y, s, inc in zip(years, scores, income)
             if y and y >= 2015]
    if not valid:
        print("  ⚠ No valid year data for score-vs-year")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for inc_label, color in INCOME_COLORS.items():
        sub = [(y, s) for y, s, inc in valid if inc == inc_label]
        if sub:
            yrs, scs = zip(*sub)
            ax.scatter(yrs, scs, alpha=0.3, s=20, color=color, label=inc_label)

    # Overall trend line
    all_yrs = np.array([v[0] for v in valid])
    all_scs = np.array([v[1] for v in valid])
    if len(all_yrs) > 10:
        z = np.polyfit(all_yrs, all_scs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_yrs.min(), all_yrs.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="gray", linewidth=1.5,
                label=f"Trend (slope={z[0]:.2f}/yr)")

    ax.set_xlabel("Year")
    ax.set_ylabel("UNESCO Alignment Score (0–100)")
    ax.set_title("UNESCO Alignment Over Time")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "fig_unesco_score_vs_year.png")
    plt.close(fig)
    print("  ✓ fig_unesco_score_vs_year.png")


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def write_tables(entries, coverage, gaps):
    import csv

    # 1. Coverage & depth per UNESCO item
    with open(OUT / "unesco_coverage_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["UNESCO Item", "Type", "Coverage %", "Mean Depth Score",
                     "HI Coverage %", "Dev Coverage %", "Gap (pp)"])

        type_map = {name: typ for name, typ in ALL_UNESCO}
        hi = [e for e in entries if e["income_binary"] == "High income"]
        dev = [e for e in entries if e["income_binary"] == "Developing"]

        for u_name in UNESCO_ITEMS_FLAT:
            typ = type_map[u_name]
            cov = coverage.get(u_name, 0)
            mean_depth = np.mean([
                e["unesco_scores"][u_name] for e in entries
                if e["unesco_scores"][u_name] > 0
            ]) if any(e["unesco_scores"][u_name] > 0 for e in entries) else 0
            hi_cov = np.mean([e["unesco_coverage"][u_name] for e in hi]) * 100 if hi else 0
            dev_cov = np.mean([e["unesco_coverage"][u_name] for e in dev]) * 100 if dev else 0
            gap = gaps.get(u_name, 0)
            w.writerow([u_name, typ, f"{cov:.1f}", f"{mean_depth:.2f}",
                        f"{hi_cov:.1f}", f"{dev_cov:.1f}", f"{gap:+.1f}"])
    print("  ✓ unesco_coverage_table.csv")

    # 2. Income-group comparison
    with open(OUT / "unesco_by_income.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Income Group", "N", "Mean Coverage %", "Mean Depth",
                     "Mean Alignment Score"])
        for inc_label in ("High income", "Developing"):
            sub = [e for e in entries if e["income_binary"] == inc_label]
            if sub:
                w.writerow([
                    inc_label, len(sub),
                    f"{np.mean([e['unesco_coverage_pct'] for e in sub]):.1f}",
                    f"{np.mean([e['unesco_avg_depth'] for e in sub]):.2f}",
                    f"{np.mean([e['unesco_alignment_score'] for e in sub]):.1f}",
                ])
    print("  ✓ unesco_by_income.csv")

    # 3. Regional comparison
    filtered = [e for e in entries
                if e["region"] not in ("Unknown",) and not e["is_international"]]
    regions = sorted(set(e["region"] for e in filtered))
    with open(OUT / "unesco_by_region.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Region", "N", "Mean Coverage %", "Mean Depth",
                     "Mean Alignment Score"])
        for reg in regions:
            sub = [e for e in filtered if e["region"] == reg]
            if len(sub) >= 3:
                w.writerow([
                    reg, len(sub),
                    f"{np.mean([e['unesco_coverage_pct'] for e in sub]):.1f}",
                    f"{np.mean([e['unesco_avg_depth'] for e in sub]):.2f}",
                    f"{np.mean([e['unesco_alignment_score'] for e in sub]):.1f}",
                ])
    print("  ✓ unesco_by_region.csv")

    # 4. Per-policy scores
    with open(OUT / "unesco_policy_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["entry_id", "title", "jurisdiction", "year", "income",
                   "coverage_pct", "avg_depth", "alignment_score"]
        header += UNESCO_ITEMS_FLAT
        w.writerow(header)
        for e in sorted(entries, key=lambda x: -x["unesco_alignment_score"]):
            row = [
                e["entry_id"], e["title"], e.get("jurisdiction", ""),
                e.get("year", ""), e["income_binary"],
                f"{e['unesco_coverage_pct']:.1f}",
                f"{e['unesco_avg_depth']:.2f}",
                f"{e['unesco_alignment_score']:.1f}",
            ]
            row += [f"{e['unesco_scores'][u]:.2f}" for u in UNESCO_ITEMS_FLAT]
            w.writerow(row)
    print("  ✓ unesco_policy_scores.csv")

    # 5. Mapping reference
    with open(OUT / "unesco_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Canonical Item", "Category", "UNESCO Target(s)"])
        for canon, targets in sorted(CANONICAL_TO_UNESCO.items()):
            # Determine category
            from normalise_depth import TAXONOMY
            cat = "unknown"
            for c, items in TAXONOMY.items():
                if canon in items:
                    cat = c
                    break
            w.writerow([canon, cat, "; ".join(targets)])
    print("  ✓ unesco_mapping.csv")


# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(entries, coverage, gaps):
    """Compute statistical tests and write to JSON."""
    stats_out = {"generated": datetime.now().isoformat()}

    hi = [e for e in entries if e["income_binary"] == "High income"]
    dev = [e for e in entries if e["income_binary"] == "Developing"]

    # Overall alignment score t-test
    hi_scores = [e["unesco_alignment_score"] for e in hi]
    dev_scores = [e["unesco_alignment_score"] for e in dev]
    if hi_scores and dev_scores:
        t, p = sp_stats.ttest_ind(hi_scores, dev_scores, equal_var=False)
        d = (np.mean(hi_scores) - np.mean(dev_scores)) / np.sqrt(
            (np.std(hi_scores)**2 + np.std(dev_scores)**2) / 2
        )
        stats_out["alignment_score_test"] = {
            "hi_mean": round(np.mean(hi_scores), 2),
            "hi_sd": round(np.std(hi_scores), 2),
            "hi_n": len(hi_scores),
            "dev_mean": round(np.mean(dev_scores), 2),
            "dev_sd": round(np.std(dev_scores), 2),
            "dev_n": len(dev_scores),
            "t_stat": round(float(t), 3),
            "p_value": round(float(p), 4),
            "cohens_d": round(float(d), 3),
        }

    # Coverage t-test
    hi_cov = [e["unesco_coverage_pct"] for e in hi]
    dev_cov = [e["unesco_coverage_pct"] for e in dev]
    if hi_cov and dev_cov:
        t, p = sp_stats.ttest_ind(hi_cov, dev_cov, equal_var=False)
        d = (np.mean(hi_cov) - np.mean(dev_cov)) / np.sqrt(
            (np.std(hi_cov)**2 + np.std(dev_cov)**2) / 2
        )
        stats_out["coverage_test"] = {
            "hi_mean": round(np.mean(hi_cov), 2),
            "dev_mean": round(np.mean(dev_cov), 2),
            "gap_pp": round(np.mean(hi_cov) - np.mean(dev_cov), 2),
            "t_stat": round(float(t), 3),
            "p_value": round(float(p), 4),
            "cohens_d": round(float(d), 3),
        }

    # Per-UNESCO-item tests
    item_tests = {}
    for u_name in UNESCO_ITEMS_FLAT:
        hi_vals = [e["unesco_coverage"][u_name] for e in hi]
        dev_vals = [e["unesco_coverage"][u_name] for e in dev]
        if hi_vals and dev_vals:
            hi_pct = np.mean(hi_vals) * 100
            dev_pct = np.mean(dev_vals) * 100
            # Chi-squared test for proportions
            hi_yes = sum(hi_vals)
            hi_no = len(hi_vals) - hi_yes
            dev_yes = sum(dev_vals)
            dev_no = len(dev_vals) - dev_yes
            if (hi_yes + dev_yes) > 0:
                chi2, p_chi = sp_stats.chi2_contingency(
                    [[hi_yes, hi_no], [dev_yes, dev_no]]
                )[:2]
                item_tests[u_name] = {
                    "hi_pct": round(hi_pct, 1),
                    "dev_pct": round(dev_pct, 1),
                    "gap_pp": round(hi_pct - dev_pct, 1),
                    "chi2": round(float(chi2), 3),
                    "p_value": round(float(p_chi), 4),
                }
    stats_out["item_tests"] = item_tests

    # Overall descriptives
    all_scores = [e["unesco_alignment_score"] for e in entries]
    all_cov = [e["unesco_coverage_pct"] for e in entries]
    stats_out["descriptives"] = {
        "n_policies": len(entries),
        "alignment_score": {
            "mean": round(np.mean(all_scores), 2),
            "median": round(float(np.median(all_scores)), 2),
            "sd": round(np.std(all_scores), 2),
            "min": round(min(all_scores), 2),
            "max": round(max(all_scores), 2),
        },
        "coverage_pct": {
            "mean": round(np.mean(all_cov), 2),
            "median": round(float(np.median(all_cov)), 2),
            "sd": round(np.std(all_cov), 2),
        },
    }

    # Sub-framework coverage
    for label, items in [("values", UNESCO_VALUES),
                          ("principles", UNESCO_PRINCIPLES),
                          ("policy_areas", UNESCO_POLICY_AREAS)]:
        cov_rates = []
        for u_name in items:
            cov_rates.append(coverage.get(u_name, 0))
        stats_out[f"{label}_coverage"] = {
            "mean_coverage_pct": round(np.mean(cov_rates), 2),
            "min_coverage_pct": round(min(cov_rates), 2),
            "max_coverage_pct": round(max(cov_rates), 2),
            "items": {u: round(coverage.get(u, 0), 1) for u in items},
        }

    # Top gaps
    sig_gaps = {k: v for k, v in gaps.items()
                if k in item_tests and item_tests[k]["p_value"] < 0.05}
    stats_out["significant_gaps"] = dict(
        sorted(sig_gaps.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    with open(OUT / "unesco_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print("  ✓ unesco_stats.json")
    return stats_out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("UNESCO ALIGNMENT ANALYSIS")
    print("=" * 70)

    entries = load_data()
    entries = enrich_entries(entries)
    print(f"Enriched {len(entries)} entries with UNESCO scores")

    # Filter to entries with depth data
    entries = [e for e in entries if e.get("depth_data")]
    print(f"Entries with depth data: {len(entries)}")

    # Quick summary
    hi = [e for e in entries if e["income_binary"] == "High income"]
    dev = [e for e in entries if e["income_binary"] == "Developing"]
    print(f"\n  High income: {len(hi)}  |  Developing: {len(dev)}")
    print(f"  Mean alignment score: HI={np.mean([e['unesco_alignment_score'] for e in hi]):.1f}"
          f"  Dev={np.mean([e['unesco_alignment_score'] for e in dev]):.1f}")
    print(f"  Mean coverage: HI={np.mean([e['unesco_coverage_pct'] for e in hi]):.1f}%"
          f"  Dev={np.mean([e['unesco_coverage_pct'] for e in dev]):.1f}%")

    print("\n── Generating figures ──")
    coverage = fig_coverage_overview(entries)
    fig_depth_heatmap(entries)
    fig_radar_income(entries)
    fig_compliance_score(entries)
    gaps = fig_income_gap(entries)
    fig_region_heatmap(entries)
    fig_gaps_top_bottom(entries, coverage)
    fig_values_depth(entries)
    fig_principles_depth(entries)
    fig_policy_areas_depth(entries)
    fig_score_vs_year(entries)

    print("\n── Generating tables ──")
    write_tables(entries, coverage, gaps)

    print("\n── Computing statistics ──")
    stats = compute_stats(entries, coverage, gaps)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    desc = stats["descriptives"]
    print(f"\nOverall: {desc['n_policies']} policies analysed")
    print(f"  Alignment score: μ={desc['alignment_score']['mean']:.1f} "
          f"(median={desc['alignment_score']['median']:.1f}, "
          f"σ={desc['alignment_score']['sd']:.1f})")
    print(f"  Coverage: μ={desc['coverage_pct']['mean']:.1f}%")

    if "alignment_score_test" in stats:
        ast = stats["alignment_score_test"]
        print(f"\nIncome gap (alignment): "
              f"HI={ast['hi_mean']} vs Dev={ast['dev_mean']}, "
              f"t={ast['t_stat']}, p={ast['p_value']}, d={ast['cohens_d']}")

    if "coverage_test" in stats:
        ct = stats["coverage_test"]
        print(f"Income gap (coverage):  "
              f"HI={ct['hi_mean']:.1f}% vs Dev={ct['dev_mean']:.1f}% "
              f"(Δ={ct['gap_pp']:+.1f}pp, p={ct['p_value']})")

    # Most/least covered
    sorted_cov = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    print(f"\nMost covered:  {sorted_cov[0][0]} ({sorted_cov[0][1]:.1f}%)")
    print(f"Least covered: {sorted_cov[-1][0]} ({sorted_cov[-1][1]:.1f}%)")

    # Significant gaps
    if stats.get("significant_gaps"):
        print(f"\nSignificant income gaps (p<0.05):")
        for item, gap in list(stats["significant_gaps"].items())[:5]:
            print(f"  {item}: {gap:+.1f}pp")

    print(f"\n✅ All outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
