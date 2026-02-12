# 🌐 Global Observatory of AI Governance Capacity

A research infrastructure to systematically measure and compare AI governance capacity across 2,200+ jurisdictions worldwide, using the OECD.AI policy corpus.

## 🎯 Research Question

> **Do countries have the capacity to implement their AI policies, and how does this vary between high-income and developing countries?**

## 📁 Project Structure

```
observatory/
├── .env                           # API keys (OpenRouter)
├── requirements.txt               # Python dependencies
│
├── book1_capacity/                # 📘 Quarto Book: AI Governance Capacity
│   ├── _quarto.yml                # Book configuration
│   ├── index.qmd                  # Book landing page
│   ├── 01-introduction.qmd        # Introduction
│   ├── 02-literature.qmd          # Literature review
│   ├── 03-data-methods.qmd        # Data & methods
│   ├── 04-scoring.qmd             # Scoring methodology
│   ├── 05-capacity-landscape.qmd  # Capacity landscape analysis
│   ├── 06-capacity-determinants.qmd # Determinants of capacity
│   ├── 07-capacity-inequality.qmd # Inequality analysis
│   ├── 08-capacity-dynamics.qmd   # Temporal dynamics
│   └── appendix-*.qmd             # Appendices
│
├── book2_ethics/                  # 📗 Quarto Book: AI Ethics Governance
│   ├── _quarto.yml                # Book configuration
│   ├── 09-ethics-landscape.qmd    # Ethics landscape analysis
│   ├── 10-ethics-determinants.qmd # Determinants of ethics
│   ├── 11-ethics-inequality.qmd   # Inequality analysis
│   ├── 12-ethics-dynamics.qmd     # Temporal dynamics
│   └── ...                        # Shared chapters & appendices
│
├── book3_unesco/                  # 📕 Quarto Book: UNESCO Alignment
│   ├── _quarto.yml                # Book configuration
│   ├── 17-unesco-landscape.qmd    # UNESCO alignment landscape
│   ├── 18-unesco-determinants.qmd # Determinants of alignment
│   ├── 19-unesco-clusters.qmd     # Cluster analysis
│   ├── 20-unesco-dynamics.qmd     # Temporal dynamics
│   └── ...                        # Shared chapters & appendices
│
├── src/
│   ├── scrapers/                  # Data collection scripts
│   │   ├── retrieve_v3.py         # Final document retriever (+ Wayback Machine)
│   │   ├── download_all_pdfs.py   # Bulk PDF downloader
│   │   ├── find_pdfs_with_claude.py # Claude-assisted URL finder
│   │   ├── integrate_content.py   # Content file → corpus matcher
│   │   └── audit_matching.py      # PDF-to-corpus matching audit
│   ├── analysis/                  # Analysis pipeline
│   │   ├── extract_text.py        # Text extraction + quality flags
│   │   ├── score_policies.py      # 3-model LLM scoring (parallel)
│   │   ├── inter_rater.py         # Inter-rater reliability
│   │   ├── country_metadata.py    # Country → income/region/GDP mapping
│   │   ├── sota_analysis.py       # Core analyses (descriptive, regression, clustering)
│   │   ├── advanced_analysis.py   # Robustness, multilevel, PCA, convergence
│   │   ├── extended_analysis.py   # Inequality, portfolio, quantile & Tobit
│   │   ├── diffusion_frontier.py  # Policy diffusion & efficiency frontier
│   │   └── unesco_paper3.py       # UNESCO alignment analysis
│   └── collectors/                # Corpus building (completed)
│
├── data/
│   ├── corpus/                    # Master corpus (2,216 entries)
│   ├── pdfs/                      # Downloaded documents (~2,085 files)
│   ├── analysis/                  # Analysis outputs
│   │   ├── paper1_capacity/       # Capacity paper outputs
│   │   ├── paper2_ethics/         # Ethics paper outputs
│   │   ├── shared/                # Shared analysis outputs
│   │   ├── scores_raw.jsonl       # Raw scores (entry × model)
│   │   ├── scores_ensemble.json   # Merged median ensemble
│   │   └── inter_rater_report.json # ICC, kappa, correlations
│   └── _archive/                  # Archived raw/intermediate data
```

## 📊 Corpus Statistics

| Metric | Value |
|--------|-------|
| **Total policies** | 2,216 |
| **Documents downloaded** | ~2,085 (94%) |
| **Analysis-ready (full text)** | 1,754 (79.2%) |
| **Total words extracted** | 11.4 million |
| **Jurisdictions** | 70+ countries + EU/international |
| **Time span** | 2017–2025 |
| **Source** | OECD.AI Policy Observatory |

### Text Quality Distribution

| Quality | Count | % | Description |
|---------|-------|---|-------------|
| Good | 948 | 42.8% | ≥500 words, full analysis |
| Thin | 806 | 36.4% | 100–499 words, usable |
| Stub | 462 | 20.8% | <100 words, minimal text |

## 🔬 Scoring Framework

Each policy scored on **10 dimensions** (0–4 scale) by a **3-model LLM ensemble**:

### Capacity Dimensions (Mazmanian-Sabatier / Lipsky / Grindle / Fukuyama)

| Dim | Indicator | Mean Score |
|-----|-----------|------------|
| C1 | Clarity & Specificity | 0.94 |
| C2 | Resources & Budget | 0.68 |
| C3 | Authority & Enforcement | 1.04 |
| C4 | Accountability & M&E | 0.48 |
| C5 | Coherence & Coordination | 1.07 |
| | **Capacity composite** | **0.83/4** |

### Ethics Dimensions (Jobin / Floridi / OECD / UNESCO / EU AI Act)

| Dim | Indicator | Mean Score |
|-----|-----------|------------|
| E1 | Ethical Framework Depth | 0.67 |
| E2 | Rights Protection | 0.55 |
| E3 | Governance Mechanisms | 0.62 |
| E4 | Operationalisation | 0.59 |
| E5 | Inclusion & Participation | 0.65 |
| | **Ethics composite** | **0.61/4** |

### LLM Ensemble

| Model | Role | Entries Scored |
|-------|------|---------------|
| Claude Sonnet 4 (A) | Strictest scorer (mean 0.57) | 2,210/2,216 |
| GPT-4o (B) | Moderate scorer (mean 0.81) | 2,216/2,216 |
| Gemini Flash 2.0 (C) | Moderate scorer (mean 0.81) | 2,215/2,216 |

Final scores = **median** across 3 models. 99.7% of entries scored by all 3 models.

## 📐 Inter-Rater Reliability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ICC(2,1) overall** | **0.827** | Excellent |
| ICC(2,1) capacity | 0.824 | Excellent |
| ICC(2,1) ethics | 0.791 | Excellent |
| Mean dimension ICC | 0.734 (0.605–0.804) | Good–Excellent |
| Pairwise Pearson (avg) | 0.86 | Strong |
| Pairwise Spearman (avg) | 0.88 | Strong |
| Fleiss' κ (avg across dims) | 0.51 | Moderate |
| Mean overall spread | 0.40/4 | Low disagreement |
| Scores within 1 point | 95.4% | High consistency |

## 🏆 Top-Scoring Policies

| Score | Jurisdiction | Policy |
|-------|-------------|--------|
| 3.1 | European Union | General Data Protection Regulation (GDPR) |
| 3.0 | European Union | Artificial Intelligence Act (AI Act) |
| 2.7 | European Union | Digital Services Act Package |
| 2.7 | United States | National AI Initiative Office |
| 2.6 | Canada | Directive on Automated Decision-making |
| 2.5 | Colombia | CONPES 4144 (National AI Policy) |

### Score Distribution

| Range | Count | % |
|-------|-------|---|
| 0.0–0.9 | 1,415 | 63.9% |
| 1.0–1.9 | 722 | 32.6% |
| 2.0–2.9 | 77 | 3.5% |
| 3.0–4.0 | 2 | 0.1% |

> **Key finding:** The vast majority of AI policies worldwide (96.5%) score below 2/4 on implementation capacity and ethics operationalisation.

## 🚀 Project Phases

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for full details.

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Corpus construction & document download (2,216 policies) |
| **Phase 1** | ✅ Complete | Text extraction — 1,754 analysis-ready (79.2%), 11.4M words |
| **Phase 2** | ✅ Complete | LLM scoring — 3-model ensemble, 6,641 API calls, ICC=0.827 |
| **Phase 3a** | ✅ Complete | SOTA analysis — 10 analyses, 53 outputs (regression, clustering, temporal trends) |
| **Phase 3b** | ✅ Complete | Advanced analysis — robustness, multilevel models, PCA, convergence |
| **Phase 3c** | ✅ Complete | Extended analysis — inequality decomposition, portfolio breadth, quantile & Tobit regression (24 outputs) |
| **Phase 3d** | ✅ Complete | Diffusion & frontier — policy diffusion patterns, efficiency frontier (18 outputs) |
| **Phase 4** | ❌ Planned | Reporting & dissemination |

## 📋 Phase 3b: Advanced Analysis Results

### 🔴 1. Robustness Checks ⚠️

| Sample restriction | Capacity $d$ | Ethics $d$ |
|---|---|---|
| All texts | +0.30*** | +0.20*** |
| **Good-text only** | **+0.04 (n.s.)** | **−0.09 (n.s.)** |
| Good + thin | +0.23*** | +0.11 (p=.08) |
| Excl. stubs | +0.23*** | +0.11 (p=.08) |

> ⚠️ **Key finding:** The income-group gap largely vanishes when restricted to good-quality texts, suggesting text extraction quality may inflate the observed disparity.

- Bootstrap 95% CIs (1,000 reps): Capacity $d$ = 0.30 [0.19, 0.41]; Ethics $d$ = 0.20 [0.09, 0.30]
- Cluster stability: best $k=2$ by silhouette score (capacity 0.41, ethics 0.42)

### 🔴 2. Multilevel Models

| Metric | Capacity | Ethics |
|---|---|---|
| Country ICC | 0.091 (9.1%) | 0.125 (12.5%) |
| LR test vs OLS | $p = .007$** | $p < .001$*** |
| GDP β (mixed) | +0.066 ($p = .038$*) | +0.029 ($p = .38$) |
| GDP β (OLS) | +0.088 ($p < .001$) | +0.061 ($p = .002$) |

> Mixed model is the correct specification — OLS inflates the GDP effect by double-counting country-level variation.

### 🔴 3. PCA / Factor Analysis ✅

| Result | Value |
|---|---|
| Kaiser criterion | **Exactly 2 components** (λ = 6.59, 1.28) |
| PC1 (65.9%) | General governance factor — all 10 dimensions load equally |
| PC2 (12.8%) | **Separates capacity from ethics** (separation = 0.51) |
| Cronbach's α — Capacity (C1–C5) | **0.92** |
| Cronbach's α — Ethics (E1–E5) | **0.91** |
| Cronbach's α — All 10 dimensions | **0.94** |

> Two-factor structure empirically validated — PCA confirms capacity and ethics are distinct but related constructs.

### 🔴 4. Convergence / Divergence

| Metric | Capacity | Ethics |
|---|---|---|
| Income × Year interaction | β = +0.0003 ($p = .98$) | β = −0.031 ($p = .015$*) |
| HI temporal slope | −0.0001/yr (n.s.) | **−0.023/yr** ($p = .001$) |
| Developing slope | +0.010/yr (n.s.) | +0.016/yr (n.s.) |
| Gap trend | Stable | **Narrowing** (−0.038/yr, $p = .018$) |

> **Capacity:** No convergence — the gap is stable over time.
> **Ethics:** Significant convergence — but driven by HI countries *declining*, not developing countries improving.

## 📋 Phase 3c: Extended Analysis Results

### 🔴 5. Inequality Decomposition

| Metric | Capacity | Ethics |
|---|---|---|
| Gini (all countries) | 0.518 | 0.569 |
| Gini (HI only) | 0.499 | 0.553 |
| Gini (Developing) | 0.593 | 0.638 |
| Gini (country means) | 0.235 | 0.273 |
| Theil T — Between groups | **1.2%** | **0.5%** |
| Theil T — Within groups | **98.8%** | **99.5%** |

> **Key finding:** Within-group inequality overwhelmingly dominates (98–99%). The income-group gap explains only 1–2% of total inequality — variation within HI and within developing countries dwarfs the gap between them.

### 🔴 6. Policy Portfolio Breadth

| Metric | Capacity | Ethics |
|---|---|---|
| Countries with 5/5 coverage | 63 (93%) | 64 (94%) |
| HI mean breadth | 4.95/5 | 5.00/5 |
| Developing mean breadth | 4.52/5 | 4.36/5 |
| Breadth gap t-test | $p = .137$ (n.s.) | $p = .054$ (marginal) |
| Least covered (Capacity) | C4 Accountability (92.6%) | — |
| Least covered (Ethics) | E2 Rights / E5 Inclusion (94.1%) | — |

> **Key finding:** Most countries cover all 5 dimensions in at least one policy — the gap is not in breadth but in depth (score levels). C4 Accountability is the biggest gap.

### 🔴 7. Quantile Regression

| Quantile (τ) | GDP β Capacity | GDP β Ethics |
|---|---|---|
| 0.25 (positive subset) | +0.068** | 0.000 (n.s.) |
| 0.50 | +0.098*** | 0.000 (n.s.) |
| 0.75 | +0.064* | 0.000 (n.s.) |
| OLS (reference) | +0.086*** | +0.061** |

> **Key finding:** GDP matters for capacity at the median but not at the extremes (inverted-U pattern). For ethics, GDP has **zero effect across all quantiles** — the OLS significance is entirely driven by the extensive margin (whether any policy exists).

### 🔴 8. Tobit Regression (Left-Censored at 0)

| Variable | Capacity (Tobit β) | Ethics (Tobit β) |
|---|---|---|
| log(GDP pc) | +0.121 | +0.100 |
| Year | +0.008 | −0.015 |
| Binding regulation | +0.174 | +0.162 |
| Good text quality | +1.193 | +1.014 |
| σ | 0.742 | 0.700 |
| P(uncensored at mean) | 0.827 | 0.725 |
| Floor: score = 0 | 27.6% | 36.3% |
| Floor: score < 1 | 57.1% | 68.5% |

> **Key finding:** Tobit coefficients are ~40% larger than OLS for GDP (capacity: 0.121 vs 0.086; ethics: 0.100 vs 0.061), confirming OLS attenuates effects when floor effects are present. Text quality remains the dominant predictor in both models.

## 📋 Phase 3d: Diffusion & Efficiency Frontier Results

### 🔴 9. Policy Diffusion Patterns

| Metric | Capacity | Ethics |
|---|---|---|
| HI median first adoption | 2018 | 2018 |
| Developing median first adoption | 2019 | 2020 |
| Adoption lag (HI earlier by) | **1.3 yrs** ($p = .030$*) | **1.2 yrs** ($p = .021$*) |
| HI adoption by 2025 | 98% | 100% |
| Developing adoption by 2025 | 86% | 72% |
| Diffusion direction | 98% horizontal | 98% horizontal |

> **Key finding:** HI countries adopted ~1 year earlier, but diffusion is overwhelmingly **horizontal** (peer-to-peer within income groups, not top-down from rich to poor). SSA and MENA lag most — 14–29% adoption by 2019 vs 100% in NAM. Ethics adoption gap (72% developing vs 100% HI by 2025) is larger than capacity gap (86% vs 98%).

### 🔴 10. Governance Efficiency Frontier

| Metric | Capacity | Ethics |
|---|---|---|
| OLS R² (score ~ GDP) | 0.035 | 0.015 |
| Top overperformer | 🇧🇷 Brazil (+0.69) | 🇮🇸 Iceland (+0.61) |
| Top underperformer | 🇰🇿 Kazakhstan (−0.75) | 🇰🇿 Kazakhstan (−0.56) |
| Frontier countries (FDH) | Uganda → Rwanda → Kenya → Brazil | Uganda → Rwanda → Nigeria → Brazil → Iceland |
| Most efficient (score/$10k) | Rwanda (3.10), Kenya (1.91) | Rwanda (2.30), Nigeria (1.51) |
| Mean dist to frontier | 0.588 | 0.517 |

> **Key finding:** GDP explains only 1.5–3.5% of country-level score variation (R² ≈ 0.02–0.04). **GDP is not destiny** — Brazil, Kenya, Rwanda, and Tunisia punch far above their weight, while Korea, Portugal, and Kazakhstan underperform relative to resources. The efficiency frontier is anchored by African countries (Rwanda, Kenya, Uganda) with modest GDP but focused governance efforts.

## � Publications

This project produces three research outputs as Quarto books:

| Book | Focus | Key Chapters |
|------|-------|--------------|
| **📘 Book 1: Capacity** | AI governance implementation capacity | Landscape, determinants, inequality, dynamics |
| **📗 Book 2: Ethics** | AI ethics governance operationalisation | Landscape, determinants, inequality, dynamics |
| **📕 Book 3: UNESCO** | Alignment with UNESCO AI Recommendation | Landscape, determinants, clusters, dynamics |

### Building the Books

```bash
# Build individual books
cd book1_capacity && quarto render
cd book2_ethics && quarto render
cd book3_unesco && quarto render
```

## 🛠️ Setup

```bash
pip install -r requirements.txt
# Add OpenRouter API key to .env: OPENROUTER_API_KEY=sk-or-v1-...
```

## 📄 License

Research project — International Initiative for Impact Evaluation (3ie)

---

*"The question is not whether AI will be governed, but whether it will be governed well. That depends on the capacity we build today."*
