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
├── docs/                          # Core documentation
│   ├── PROJECT_PLAN.md            # Comprehensive 4-phase plan
│   ├── METHODOLOGY.md             # Research methodology
│   ├── THEORETICAL_FRAMEWORK.md   # Theoretical underpinnings
│   ├── INDICATOR_RUBRIC.md        # Capacity indicator definitions
│   ├── VALIDATION_PROTOCOL.md     # Validation methodology
│   └── MPHIL_MODULE.md           # Teaching module outline
│
├── src/
│   ├── scrapers/                  # Data collection scripts
│   │   ├── retrieve_v3.py         # Final document retriever (+ Wayback Machine)
│   │   ├── download_all_pdfs.py   # Phase 1 bulk downloader
│   │   ├── find_pdfs_with_claude.py # Claude-assisted URL finder
│   │   ├── integrate_content.py   # Content file → corpus matcher
│   │   ├── audit_matching.py      # PDF-to-corpus matching audit
│   │   └── ...                    # UNESCO/OECD specific scrapers
│   ├── analysis/                  # Analysis pipeline
│   │   ├── extract_text.py        # Phase 1: text extraction + quality flags
│   │   ├── score_policies.py      # Phase 2: 3-model LLM scoring (parallel)
│   │   ├── inter_rater.py         # Phase 2.3: inter-rater reliability
│   │   ├── country_metadata.py    # Country → income/region/GDP mapping
│   │   ├── sota_analysis.py       # Phase 3a: 10 core analyses (descriptive, regression, clustering)
│   │   └── advanced_analysis.py   # Phase 3b: robustness, multilevel, PCA, convergence
│   └── collectors/                # Corpus building (completed)
│
├── data/
│   ├── corpus/                    # Master corpus (2,216 entries)
│   │   └── corpus_enriched.json   # Enriched with full text + quality flags
│   ├── pdfs/                      # Downloaded documents (~2,085 files)
│   ├── analysis/                  # Analysis outputs
│   │   ├── scores_raw.jsonl       # Raw scores: 6,641 lines (entry × model)
│   │   ├── scores_ensemble.json   # Merged median ensemble (2,216 entries)
│   │   ├── inter_rater_report.json # ICC, kappa, correlations
│   │   ├── extraction_report.json # Phase 1 quality metrics
│   │   └── scoring_report.json    # Phase 2 run statistics
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
| **Phase 3b** | ⏳ In progress | Advanced analysis — robustness, multilevel models, PCA, convergence |
| **Phase 4** | ❌ Planned | Reporting & dissemination |

## 📋 Phase 3b: Advanced Analysis Plan

### 🔴 High Priority (analyses 1–4 — reviewer essentials)

| # | Analysis | Purpose | Method |
|---|----------|---------|--------|
| 1 | **Robustness checks** | Verify key findings hold under alternative specifications | Rerun income-gap tests excluding stubs/thin texts; bootstrap 95% CIs (1,000 reps); cluster solutions k=3–6 with silhouette scores |
| 2 | **Multilevel models** | Account for policies nested within countries | Random-intercepts model (country grouping); compare with pooled OLS; ICC for country-level variance |
| 3 | **PCA / Factor analysis** | Validate the two-construct framework (capacity vs ethics) | PCA on 10 dimensions; scree plot; loadings matrix; do C1–C5 and E1–E5 form two distinct factors? |
| 4 | **Convergence / divergence** | Are developing countries catching up or falling behind? | Income × year interaction; separate temporal slopes by income group; gap trajectory 2017–2025 |

### 🟡 Medium Priority (analyses 5–8 — strengthen contribution)

| # | Analysis | Purpose | Method |
|---|----------|---------|--------|
| 5 | **Inequality decomposition** | Between-group vs within-group inequality | Gini coefficient; Theil index decomposition (between income groups vs within) |
| 6 | **Policy portfolio breadth** | Do countries cover all dimensions or concentrate on a few? | Per-country coverage index (how many dimensions scored ≥1); portfolio gap identification |
| 7 | **Quantile regression** | Does GDP matter more at the bottom than the top? | Quantile regression at τ = 0.25, 0.50, 0.75 |
| 8 | **Tobit regression** | Handle floor effects (64% score 0–0.9) | Tobit model for bounded dependent variable [0, 4] |

### 🟢 Nice to Have (analyses 9–10 — differentiation)

| # | Analysis | Purpose | Method |
|---|----------|---------|--------|
| 9 | **Policy diffusion patterns** | Which countries led in each dimension? | Temporal leader-follower analysis by region |
| 10 | **Efficiency frontier** | Most governance capacity per GDP dollar | Score/GDP scatter with frontier envelope |

## 🛠️ Setup

```bash
pip install -r requirements.txt
# Add OpenRouter API key to .env: OPENROUTER_API_KEY=sk-or-v1-...
```

## 📚 Key Documentation

- **[Project Plan](docs/PROJECT_PLAN.md)** — Full roadmap with phases, deliverables, timelines
- **[Methodology](docs/METHODOLOGY.md)** — Research design and methods
- **[Indicator Rubric](docs/INDICATOR_RUBRIC.md)** — Scoring criteria for capacity dimensions
- **[Validation Protocol](docs/VALIDATION_PROTOCOL.md)** — Inter-rater reliability approach

## 📄 License

Research project — International Initiative for Impact Evaluation (3ie)

---

*"The question is not whether AI will be governed, but whether it will be governed well. That depends on the capacity we build today."*
