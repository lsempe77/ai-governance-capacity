# Paper 3 — UNESCO Alignment of Global AI Policies

## Full Plan & Roadmap

---

## 1. Working Title

**"Do AI Policies Walk the Talk? Measuring Alignment with the UNESCO
Recommendation on the Ethics of Artificial Intelligence"**

*Alternative:* "Beyond Principles: A Global Benchmarking of AI Policy Alignment
with the UNESCO Ethics Framework"

---

## 2. Research Questions

1. **Coverage:** To what extent do national and regional AI policies address the
   UNESCO Recommendation's 4 values, 10 principles, and 11 policy action areas?
2. **Depth:** When policies do engage with UNESCO items, how substantively do
   they do so — at word-, phrase-, sentence-, paragraph-, or section-level?
3. **Income divide:** Do high-income and developing countries differ in their
   alignment with the UNESCO framework, and if so, where?
4. **Temporal evolution:** Has alignment improved since the Recommendation's
   adoption in November 2021?
5. **Binding force:** Does binding nature (hard law vs. soft law vs.
   non-binding) predict deeper UNESCO alignment?
6. **Gaps:** Which UNESCO items are systematically under-addressed, and what
   does this reveal about the global AI governance landscape?

---

## 3. Contribution & Motivation

- The UNESCO Recommendation (Nov 2021) is the **first global normative
  instrument** on AI ethics, adopted by all 194 member states
- Yet no systematic empirical study has benchmarked actual policy texts against
  its full framework (4 + 10 + 11 = 25 items)
- Our contribution: a **large-N, text-level analysis** of 1,326 AI policies
  from 79 jurisdictions, measuring not just *whether* but *how deeply* each
  policy engages with each UNESCO item
- Key innovation: a **coverage + depth scoring methodology** using LLM-based
  verbatim extraction, moving beyond manual content analysis

---

## 4. Data & Methods Summary

| Element | Detail |
|---------|--------|
| Corpus | 1,326 AI policies with extractable ethics content (from 2,216 in OECD.AI) |
| Jurisdictions | 79 countries/regions |
| Time span | 2017–2025 (727 pre-UNESCO, 594 post-UNESCO adoption) |
| UNESCO benchmark | 4 values + 10 principles + 11 policy areas = 25 items |
| Our taxonomy | 56 canonical items (18 values + 18 principles + 20 mechanisms) |
| Mapping | 56 → 25 UNESCO items (many-to-one, curated) |
| Depth extraction | LLM (Claude Opus 4.6) verbatim extraction with 5-level depth scale |
| Depth scale | word (1) → phrase (2) → sentence (3) → paragraph (4) → section (5) |
| Alignment score | 60% coverage breadth + 40% normalised depth quality (0–100) |
| Covariates | Income group, region, GDP/capita, binding nature, policy type, year |

---

## 5. Paper Outline & Chapter Plan

### 5.1  Introduction  [~1,500 words]
- The "principles-to-practice" gap in AI governance
- UNESCO Recommendation as the benchmark of choice
- Research questions
- Preview of findings

### 5.2  Background & Related Work  [~2,000 words]
- UNESCO Recommendation: genesis, adoption, structure
- Prior attempts to measure AI ethics in policy (Jobin et al. 2019,
  Floridi et al. 2018, Schiff et al. 2021, Hagendorff 2020)
- Gap: no large-N empirical benchmark against UNESCO specifically
- Theoretical frame: policy diffusion + norm localisation

### 5.3  Data & Methods  [~2,500 words]
- 5.3.1 Corpus construction (OECD.AI → text extraction → quality filter)
- 5.3.2 Ethics inventory: LLM extraction of values/principles/mechanisms
- 5.3.3 Depth scoring: verbatim extraction with 5-level depth scale
- 5.3.4 Normalisation: 1,019 raw labels → 56 canonical items
- 5.3.5 UNESCO mapping: 56 canonical → 25 UNESCO items
- 5.3.6 Alignment scoring methodology
- 5.3.7 Statistical approach (t-tests, chi-squared, OLS, ordered logit)

### 5.4  Results  [~4,000 words]

#### 5.4.1 — Descriptive: The UNESCO Alignment Landscape
- Overall alignment score distribution (μ=53.9, σ=12.2)
- Coverage rates: values (55%) > principles (53%) > policy areas (41%)
- Most covered: Peaceful/just societies (83%), Ethical governance (82%)
- Least covered: Communication (2%), Culture (9%), Gender (10%)
- **Figures needed:**
  - ✅ `fig_unesco_coverage_overview.png`
  - ✅ `fig_unesco_compliance_score.png`
  - ✅ `fig_unesco_gaps_top_bottom.png`
  - 🔲 `fig_unesco_coverage_by_layer.png` — grouped bar: values vs principles vs policy areas

#### 5.4.2 — Depth: Beyond Lip Service
- Depth distribution per UNESCO item
- Policy areas deepest (governance, economy); values shallowest (human rights!)
- "Paradox of proclamation": most-mentioned ≠ most-substantive
- **Figures needed:**
  - ✅ `fig_unesco_depth_heatmap.png`
  - ✅ `fig_unesco_values_depth.png`
  - ✅ `fig_unesco_principles_depth.png`
  - ✅ `fig_unesco_policy_areas_depth.png`
  - 🔲 `fig_unesco_coverage_vs_depth.png` — scatter: coverage % vs mean depth per UNESCO item

#### 5.4.3 — Income Divide
- Null finding on overall alignment (p=0.99, d=0.001) — striking
- BUT significant item-level gaps:
  - Health & well-being: Developing +19.3pp (p<0.001)
  - Gender: Developing +6.2pp (p=0.006)
- Developing countries address social dimensions MORE than HI countries
- **Figures needed:**
  - ✅ `fig_unesco_income_gap.png`
  - ✅ `fig_unesco_radar_income.png`
  - 🔲 `fig_unesco_income_alignment_dist.png` — violin/box: alignment score by income

#### 5.4.4 — Regional Patterns
- Regional variation in UNESCO alignment profiles
- Which regions lead on which UNESCO layers?
- **Figures needed:**
  - ✅ `fig_unesco_region_heatmap.png`
  - 🔲 `fig_unesco_region_scores.png` — bar: mean alignment score by region

#### 5.4.5 — Temporal Evolution: Before & After UNESCO
- Pre-UNESCO (≤2021, n=727) vs Post-UNESCO (≥2022, n=594)
- Did adoption of the Recommendation shift policy content?
- **Figures needed:**
  - ✅ `fig_unesco_score_vs_year.png`
  - 🔲 `fig_unesco_pre_post.png` — paired bar: coverage % before vs after, per UNESCO item
  - 🔲 `fig_unesco_pre_post_income.png` — pre/post × income interaction

#### 5.4.6 — Binding Force & Policy Type
- Hard law vs. soft law vs. non-binding: does binding nature predict depth?
- Policy type (legislation vs. strategy vs. guideline) and UNESCO alignment
- **Figures needed:**
  - 🔲 `fig_unesco_binding_nature.png` — boxplot: alignment by binding nature
  - 🔲 `fig_unesco_policy_type.png` — boxplot: alignment by policy type

#### 5.4.7 — Regression Analysis: What Predicts UNESCO Alignment?
- DV: UNESCO alignment score (0–100)
- IVs: income, region, year, GDP/capita, binding nature, policy type,
  capacity score, ethics score
- OLS + robustness with ordered logit
- **Tables needed:**
  - 🔲 `unesco_regression.csv` — main regression table
  - 🔲 `unesco_regression_stats.json` — coefficients, SEs, R²

#### 5.4.8 — Cluster Analysis: UNESCO Alignment Profiles
- K-means or hierarchical clustering on 25-item coverage vectors
- Identify country archetypes (e.g., "comprehensive aligners", "principle-heavy",
  "governance-focused", "minimal engagement")
- **Figures needed:**
  - 🔲 `fig_unesco_cluster_radar.png` — radar per cluster
  - 🔲 `fig_unesco_cluster_map.png` — world map coloured by cluster
  - 🔲 `fig_unesco_cluster_income.png` — cluster × income mosaic

### 5.5  Discussion  [~2,000 words]
- The "surprisingly equal" overall alignment — implications
- But the devil is in the details: item-level gaps reveal different priorities
- Developing countries' emphasis on health, gender, multi-stakeholder governance
  — reflecting UNESCO's influence in the Global South?
- The "principles-to-policy-areas" gap: values well-covered but implementation
  mechanisms lag behind
- Pre/post UNESCO shift (or lack thereof)
- Limitations: LLM-based extraction, English-medium bias, OECD.AI corpus
  completeness

### 5.6  Conclusion  [~500 words]

---

## 6. Analyses Still Needed

### 6.1 New Figures (🔲 = not yet built)

| # | Figure | Section | Complexity |
|---|--------|---------|------------|
| 12 | `fig_unesco_coverage_by_layer.png` | 5.4.1 | Simple |
| 13 | `fig_unesco_coverage_vs_depth.png` | 5.4.2 | Simple |
| 14 | `fig_unesco_income_alignment_dist.png` | 5.4.3 | Simple |
| 15 | `fig_unesco_region_scores.png` | 5.4.4 | Simple |
| 16 | `fig_unesco_pre_post.png` | 5.4.5 | Medium |
| 17 | `fig_unesco_pre_post_income.png` | 5.4.5 | Medium |
| 18 | `fig_unesco_binding_nature.png` | 5.4.6 | Simple |
| 19 | `fig_unesco_policy_type.png` | 5.4.6 | Medium (join needed) |
| 20 | `fig_unesco_cluster_radar.png` | 5.4.8 | Medium |
| 21 | `fig_unesco_cluster_map.png` | 5.4.8 | Medium |
| 22 | `fig_unesco_cluster_income.png` | 5.4.8 | Simple |

### 6.2 New Tables

| # | Table | Section |
|---|-------|---------|
| 6 | `unesco_regression.csv` | 5.4.7 |
| 7 | `unesco_pre_post.csv` | 5.4.5 |
| 8 | `unesco_binding.csv` | 5.4.6 |
| 9 | `unesco_clusters.csv` | 5.4.8 |

### 6.3 Data Joins Needed
- Join `phase_c_depth_normalised.jsonl` with `master_dataset.csv` to get
  `policy_type`, `binding_nature`, `capacity_score`, all 10 dimension scores
- This enables the regression and binding-nature/policy-type analyses

---

## 7. Quarto Book Integration

Add a new **Part IV** to `_quarto.yml`:

```yaml
    - part: "Part IV — UNESCO Alignment"
      chapters:
        - 17-unesco-landscape.qmd       # Coverage + depth descriptives
        - 18-unesco-determinants.qmd     # Income, region, binding, regression
        - 19-unesco-clusters.qmd         # Cluster analysis + typology
        - 20-unesco-dynamics.qmd         # Pre/post UNESCO temporal analysis
```

This mirrors the 4-chapter structure of Parts I and II.

---

## 8. Implementation Sequence

### Phase 1: Data Preparation
1. ✅ UNESCO taxonomy definition (4 + 10 + 11)
2. ✅ Canonical-to-UNESCO mapping (56 → 25)
3. 🔲 Join with master_dataset.csv for covariates

### Phase 2: Extended Analysis Script
4. 🔲 Extend `unesco_alignment.py` with:
   - Coverage-by-layer figure
   - Coverage-vs-depth scatter
   - Income violin/box
   - Region bar chart
   - Pre/post UNESCO comparison (with DiD-style test)
   - Binding nature & policy type breakdowns
   - OLS regression
   - Cluster analysis (K-means on 25-dim coverage vector)

### Phase 3: Quarto Chapters
5. 🔲 Write `17-unesco-landscape.qmd`
6. 🔲 Write `18-unesco-determinants.qmd`
7. 🔲 Write `19-unesco-clusters.qmd`
8. 🔲 Write `20-unesco-dynamics.qmd`

### Phase 4: Polish
9. 🔲 Update `_quarto.yml` with Part IV
10. 🔲 Git commit all Paper 3 work
11. 🔲 Clean up debug/temp scripts

---

## 9. Estimated Effort

| Task | Time | Cost |
|------|------|------|
| Extended analysis script | 1 session | $0 (local compute) |
| Quarto chapters (4) | 2 sessions | $0 |
| Revision & polish | 1 session | $0 |
| **Total** | **~4 sessions** | **$0 additional API cost** |

All LLM extraction is already done. Paper 3 is pure analysis & writing on
existing data.

---

## 10. Key Talking Points for the Paper

1. **First large-N empirical benchmark** of AI policies against UNESCO (n=1,326,
   79 jurisdictions) — no prior study has done this at scale
2. **Surprising null finding**: no overall income gap in UNESCO alignment
   (d=0.001) — counter to the "digital divide" narrative
3. **But item-level divergence**: developing countries significantly MORE aligned
   on health (+19pp) and gender (+6pp) — suggests UNESCO's norm diffusion may be
   more effective in the Global South
4. **The policy-area gap**: values and principles well-covered (53-55%) but
   policy action areas lag (41%) — the "implementation gap" is real
5. **Coverage ≠ depth**: most-mentioned items (transparency, governance) are not
   the deepest — "principle proclamation" without substantive engagement
6. **Communication & information** almost entirely absent (2%) — a blind spot
7. **Human rights & dignity** — the cornerstone UNESCO value — only appears in
   23% of policies, despite being the foundational value
