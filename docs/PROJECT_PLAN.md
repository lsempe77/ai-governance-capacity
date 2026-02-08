# Global Observatory of AI Governance Capacity — Project Plan

## Research Question

**Do countries have the capacity to implement their AI policies, and how does this vary between high-income and developing countries?**

This project builds a first-of-its-kind dataset measuring not just *what* AI policies exist, but *whether governments can actually implement them*. We analyse 2,216 policy documents from 80+ jurisdictions in the OECD.AI database, combining full-text analysis with LLM-based coding and state-of-the-art implementation science frameworks.

---

## Data

| Item | Detail |
|------|--------|
| **Source** | OECD.AI Policy Observatory — all policy initiatives |
| **Corpus** | 2,216 entries (policies, strategies, laws, executive orders, programs) |
| **Jurisdictions** | 80+ (countries, sub-national, international orgs) |
| **Time span** | 1956–2026 |
| **Document types** | PDFs, HTML pages, legislation texts, strategy documents |
| **Corpus file** | `data/corpus/corpus_master_20260127.json` |

### Corpus entry fields
Each entry contains: `title`, `jurisdiction`, `year`, `url`, `source_url`, `content` (OECD snippet), `target_sectors`, `ai_tags`, `word_count`, `status`, `responsible_organisation`, `stakeholder_involvement`, and other metadata.

---

## Project Phases

### Phase 0: Document Retrieval ✅ (Complete)

**Goal:** Download full-text source documents for all corpus entries.

**What was done:**
1. Phase 1 downloads: 402 PDFs from `source_url` fields in the corpus
2. Phase 2 downloads: 590 documents using Claude (via OpenRouter) to locate URLs
3. Retry round: +107 via 5-strategy retrieval (direct → OECD scrape → Wayback Machine → DuckDuckGo → Claude)
4. Integration of prior `data/content/` download campaign: +654 matched by title
5. Final retrieval round: ~200+ more via Wayback Machine

**Retrieval strategies (5-strategy pipeline in `src/scrapers/retrieve_v3.py`):**
1. Direct download from `source_url`
2. OECD.AI page scraping for embedded source links
3. Internet Archive Wayback Machine
4. DuckDuckGo search (`title + jurisdiction filetype:pdf`)
5. Claude API to locate official document URLs

**Current status:**
- ~1,950+ documents on disk (`data/pdfs/`)
- ~88% corpus coverage
- Remaining ~250 entries are snippet-only (no downloadable source found in any campaign)
- Matching is robust: deterministic MD5-based IDs, audited with zero mismatches

**Key files:**
- `data/pdfs/` — All downloaded documents (PDFs + HTMLs)
- `data/pdfs/download_progress.json` — Download tracking
- `src/scrapers/retrieve_v3.py` — Final retrieval script
- `src/scrapers/audit_matching.py` — Matching integrity audit
- `src/scrapers/integrate_content.py` — Content integration from prior campaign

---

### Phase 1: Parsing & Text Extraction

**Goal:** Extract clean, analysis-ready text from all downloaded documents and build an enriched corpus.

#### 1.1 Text extraction from PDFs
- Extract text from PDFs using `pdfplumber`, `PyMuPDF`, or `pdfminer`
- Handle scanned PDFs with OCR fallback (`pytesseract` or similar)
- Strip boilerplate (headers, footers, page numbers, TOCs)

#### 1.2 Text extraction from HTML
- Parse HTML files with `BeautifulSoup`
- Extract main content, strip navigation/chrome
- Handle Wayback Machine wrapper HTML (many files are archived OECD pages)

#### 1.3 Corpus enrichment
- For each corpus entry, replace the OECD snippet (`content` field, typically <500 words) with full extracted text
- Create `unified_text` field: full text if available, OECD snippet as fallback
- Save enriched corpus as `corpus_master_enriched.json`
- Log extraction quality: word count before/after, language detection, extraction method

#### 1.4 Quality control
- Flag entries where extraction failed or produced garbage text
- Compare extracted word counts to expected lengths
- Language detection (many non-English documents)
- Sample manual inspection (n=50) across jurisdictions and document types

**Output:**
- `data/corpus/corpus_master_enriched.json` — Full-text corpus
- `data/analysis/extraction_report.json` — Extraction quality metrics

---

### Phase 2: Classification & Model Comparison

**Goal:** Apply LLM-based coding to the full-text corpus and validate across models.

This phase repeats prior analysis (done on OECD snippets) but now with **full document text**, which should dramatically improve classification quality.

#### 2.1 Implementation Capacity Analysis (Full Text)

Score each policy on **5 capacity dimensions** grounded in implementation science (Mazmanian & Sabatier, 1983; Winter, 2012; Howlett, 2011):

| Dimension | What it measures | Score |
|-----------|-----------------|-------|
| **Clarity & Specificity** | Clear objectives, measurable targets, defined scope | 0–10 |
| **Resources & Budget** | Dedicated funding, staffing, infrastructure | 0–10 |
| **Authority & Enforcement** | Legal mandate, penalties, compliance mechanisms | 0–10 |
| **Accountability** | Reporting, evaluation, oversight bodies | 0–10 |
| **Coherence & Coordination** | Cross-agency alignment, international coordination | 0–10 |

**Method:** Multi-model ensemble via OpenRouter API
- Model A: `anthropic/claude-sonnet-4`
- Model B: `openai/gpt-4o`
- Model C: `google/gemini-2.0-flash-001`

Each model independently codes each policy. Final score = median of 3 models.

**Script:** `src/analysis/sota_capacity_analysis.py` (to be updated for full text)

#### 2.2 AI Ethics Analysis (Full Text)

Score each policy on **ethical governance maturity** grounded in Jobin et al. (2019), Floridi et al. (2018), OECD AI Principles, UNESCO Recommendation, EU AI Act:

| Dimension | What it measures | Score |
|-----------|-----------------|-------|
| **Ethical Framework Depth** | Grounding in principles, coherent ethical vision | 0–10 |
| **Rights Protection** | Privacy, non-discrimination, human oversight, transparency | 0–10 |
| **Governance Mechanisms** | Ethics boards, impact assessments, auditing | 0–10 |
| **Operationalisation** | Concrete requirements, standards, certification | 0–10 |
| **Inclusion & Participation** | Stakeholder processes, marginalised group representation | 0–10 |

**Script:** `src/analysis/ethics_sota_analysis.py` (to be updated for full text)

#### 2.3 Model Comparison & Inter-Rater Reliability

- Compute **inter-rater reliability** across 3 LLMs: Pearson, Spearman, ICC(2,1), Cohen's weighted kappa
- Identify systematic disagreements between models
- Compare snippet-based vs. full-text-based scores: how much does full text improve classification?
- Document bias patterns (model-specific tendencies)

**Scripts:**
- `src/analysis/inter_rater_reliability.py`
- `src/analysis/llm_validation.py`

#### 2.4 Human Validation

- Stratified sample (n=50): 10 high-capacity × 5 regions, 10 low-capacity × 5 regions
- Manual coding by 2 human coders using the same rubric
- Compute human-LLM agreement (kappa, ICC)
- Report in supplementary materials

**Output:**
- `data/analysis/capacity/capacity_scores_fulltext.json`
- `data/analysis/ethics/ethics_scores_fulltext.json`
- `data/analysis/validation/irr_fulltext_report.md`

---

### Phase 3: State-of-the-Art Analysis

**Goal:** Produce publication-quality findings grounded in the academic literature.

#### 3.1 Literature Foundation

Identify and synthesise key readings across three streams:

**Implementation science & state capacity:**
- Mazmanian, D. & Sabatier, P. (1983). *Implementation and Public Policy*
- Winter, S. (2012). *Implementation Perspectives: Status and Reconsideration*
- Howlett, M. (2011). *Designing Public Policies*
- Grindle, M. (2004). *Good Enough Governance Revisited*
- Fukuyama, F. (2013). *What Is Governance?*
- Lipsky, M. (1980). *Street-Level Bureaucracy*
- Andrews, M., Pritchett, L., & Woolcock, M. (2017). *Building State Capability*

**AI governance:**
- Jobin, A., Ienca, M., & Vayena, E. (2019). *The global landscape of AI ethics guidelines*
- Floridi, L. et al. (2018). *AI4People—An ethical framework for a good AI society*
- Cath, C. et al. (2018). *AI and the "good society"*
- Fjeld, J. et al. (2020). *Principled AI* (Berkman Klein Center)
- Stix, C. (2021). *Actionable principles for AI policy*
- Bradford, A. (2020). *The Brussels Effect* (regulatory diffusion)
- Radu, R. (2021). *Steering the governance of AI*

**Regulatory capacity in the Global South:**
- Poel, M. et al. (2018). *Data governance models*
- Hagendorff, T. (2020). *The ethics of AI ethics*
- Erman, E. & Furendal, M. (2022). *Artificial intelligence and the democratic challenge*
- Smuha, N. (2021). *From a "race to AI" to a "race to AI regulation"*

#### 3.2 AI Governance Capacity Analysis

**RQ1: The Implementation Gap**
- What share of policies are "implementable" vs. "aspirational"?
- Construct Ambition–Capacity Gap Index: policies with high stated ambition but low capacity scores
- Identify the "sweet spot" jurisdictions that balance ambition with implementation infrastructure

**RQ2: Income Divide**
- Multi-level regression: income group → capacity scores, controlling for GDP, AI readiness, governance quality
- Compare dimension profiles (HIC vs. LMIC): where are the biggest gaps?
- Test whether LMICs have different capacity "shapes" (e.g., strong authority but weak resources)
- Policy diffusion analysis: are LMIC policies "copies" of HIC frameworks?

**RQ3: Determinants of Capacity**
- What predicts implementation capacity? Merge with:
  - Oxford AI Readiness Index
  - Stanford HAI AI Index
  - World Bank Governance Indicators (WGI)
  - UN E-Government Development Index (EGDI)
  - V-Dem democracy indicators
- OLS/ordered logit regression with robustness checks
- Identify whether democratic institutions, economic resources, or prior governance quality matters most

**RQ4: Sector & Thematic Variation**
- Do health-sector AI policies have different capacity profiles than defense or education?
- Which sectors have the most/least implementable policies?
- Cross-tabulation with income groups

#### 3.3 AI Ethics SOTA Analysis

**RQ5: Beyond Principles**
- What share of ethics frameworks are "operationalised" vs. merely "principled"?
- Construct Ethics Maturity Index (4 levels):
  1. **Superficial** — mentions ethics keywords
  2. **Principled** — articulates ethical framework
  3. **Operationalised** — specifies mechanisms (impact assessments, audits)
  4. **Governed** — establishes oversight bodies with enforcement power
- Map global distribution of ethics maturity

**RQ6: The Ethics–Capacity Nexus**
- Are countries with strong ethics frameworks also implementation-capable?
- Scatter: ethics score × capacity score by jurisdiction
- Identify typology: "walk the walk" vs. "talk the talk" vs. "quiet implementers"

**RQ7: Rights Protection**
- Which rights are most/least protected globally? (privacy, non-discrimination, labour, human oversight)
- Income-group variation in rights protection emphasis
- Regional patterns (EU rights-heavy vs. US innovation-heavy vs. China state-led)

#### 3.4 Cross-Cutting Analyses

- **Temporal trends:** Is capacity improving over time? (2015–2026 panel)
- **Policy diffusion networks:** Semantic similarity between jurisdictions over time
- **Regulatory convergence:** Are approaches becoming more similar globally?
- **Regional clustering:** K-means or hierarchical clustering of countries by capacity + ethics profiles

**Output:**
- `data/analysis/capacity/SOTA_CAPACITY_REPORT_FULLTEXT.md`
- `data/analysis/ethics/SOTA_ETHICS_REPORT_FULLTEXT.md`
- `data/analysis/figures/` — Publication-quality visualisations

---

### Phase 4: Reporting & Publication

**Goal:** Produce a complete research paper and supporting materials.

#### 4.1 Paper Structure

| Section | Content | Approx. length |
|---------|---------|----------------|
| **Abstract** | Novel dataset, 2,216 policies, 80 jurisdictions, key finding on implementation gap + income divide | 250 words |
| **Introduction** | Motivation (AI governance as aspiration vs. reality), gap in literature (no capacity measures), contribution | 1,500 words |
| **Literature Review** | Implementation science + AI governance + Global South capacity, theoretical framework | 2,500 words |
| **Data & Methods** | OECD.AI corpus, document retrieval, LLM-based coding, validation, capacity index construction | 2,500 words |
| **Results** | Descriptive (landscape), capacity analysis (income divide), ethics analysis (maturity levels), determinants | 3,000 words |
| **Discussion** | Implications for policymakers, limitations, comparison with prior work | 1,500 words |
| **Conclusion** | Policy recommendations, future research | 500 words |
| **Supplementary** | Full rubrics, validation results, robustness checks, country scorecards | Appendix |

**Target word count:** ~12,000 words (main paper)

#### 4.2 Key Visualisations

1. **World map** — Capacity scores by jurisdiction (choropleth)
2. **Radar charts** — Capacity dimension profiles by income group
3. **Ambition–Capacity scatter** — 2D plot of all jurisdictions
4. **Ethics maturity stacked bar** — By region and income group
5. **Temporal trend** — Policy adoption + capacity scores over time
6. **Regression table** — Determinants of capacity (with controls)
7. **Heatmap** — Sector × dimension capacity scores
8. **Network graph** — Policy similarity clusters

#### 4.3 Target Outlets

| Priority | Outlet | Rationale |
|----------|--------|-----------|
| 1 | **Nature Human Behaviour** | Novel dataset, computational social science, global scope |
| 2 | **Regulation & Governance** | Core audience for regulatory capacity research |
| 3 | **World Development** | LMIC angle, implementation gap, development implications |
| 4 | **PNAS** | Interdisciplinary, data-driven, policy-relevant |
| Companion | **Working Paper** (3ie / Blavatnik) | Full methodology and country scorecards |

#### 4.4 Deliverables

- [ ] Research paper (draft → peer review → revision)
- [ ] Replication package (code + data)
- [ ] Country scorecards (individual capacity profiles for each jurisdiction)
- [ ] Policy brief (2-page summary for policymakers)
- [ ] Interactive dashboard (optional, if time permits)

---

## Repository Structure

```
observatory/
├── docs/                              # Documentation
│   ├── PROJECT_PLAN.md                # ← This file
│   ├── PROJECT_SUMMARY.md             # Research overview
│   ├── ANALYSIS_PLAN.md               # Detailed analysis plan
│   ├── METHODOLOGY.md                 # Methods documentation
│   ├── THEORETICAL_FRAMEWORK.md       # Theoretical grounding
│   ├── VALIDATION_PROTOCOL.md         # LLM validation protocol
│   └── INDICATOR_RUBRIC.md            # Capacity scoring rubric
│
├── src/
│   ├── scrapers/                      # Phase 0: Document retrieval
│   │   ├── retrieve_v3.py             #   Final 5-strategy retrieval
│   │   ├── integrate_content.py       #   Prior download integration
│   │   └── audit_matching.py          #   Matching quality audit
│   │
│   ├── collectors/                    # Data collection pipeline
│   │   ├── oecd_scraper.py            #   OECD.AI scraper
│   │   ├── build_corpus.py            #   Corpus builder
│   │   └── pipeline.py               #   Pipeline orchestrator
│   │
│   └── analysis/                      # Phase 2–3: Analysis
│       ├── sota_capacity_analysis.py  #   LLM-based capacity scoring
│       ├── ethics_sota_analysis.py    #   LLM-based ethics scoring
│       ├── inter_rater_reliability.py #   Multi-model agreement
│       ├── llm_validation.py          #   Human-LLM validation
│       ├── causal_analysis.py         #   Regression models
│       ├── external_validity.py       #   External index linkage
│       ├── country_aggregation.py     #   Country-level scores
│       └── visualize_*.py             #   Visualisation scripts
│
├── data/
│   ├── corpus/                        # Text corpus
│   │   └── corpus_master_*.json       #   Master corpus files
│   ├── pdfs/                          # Downloaded source documents
│   ├── content/                       # Prior download campaign
│   └── analysis/                      # Analysis outputs
│       ├── capacity/                  #   Capacity analysis results
│       ├── ethics/                    #   Ethics analysis results
│       ├── validation/                #   Validation outputs
│       ├── aggregated/                #   Country-level aggregations
│       └── figures/                   #   Publication charts
│
├── .env                               # API keys (OpenRouter)
├── requirements.txt                   # Python dependencies
└── README.md                          # Repository overview
```

---

## Current Progress

| Phase | Status | Detail |
|-------|--------|--------|
| **Phase 0** — Document Retrieval | ✅ Complete | ~1,950 documents, 88% coverage, matching audited |
| **Phase 1** — Parsing & Extraction | ⬜ Not started | Need to extract text from PDFs/HTMLs |
| **Phase 2** — Classification (snippets) | ✅ Done (on snippets) | Capacity + ethics scores from OECD content only |
| **Phase 2** — Classification (full text) | ⬜ Not started | Re-run with enriched corpus |
| **Phase 3** — SOTA Analysis | 🟡 Partial | Framework + initial runs done on snippets |
| **Phase 4** — Reporting | ⬜ Not started | |

---

## Technical Notes

- **LLM API:** OpenRouter (`OPENROUTER_API_KEY` in `.env`), supporting Claude, GPT-4o, Gemini
- **Corpus ID scheme:** `hashlib.md5(entry['url'].encode()).hexdigest()[:12]` — deterministic, unique per OECD URL
- **File encoding:** Always use `encoding='utf-8'` when reading/writing JSON (Windows cp1252 causes errors)
- **Rate limits:** ~30s per Claude call; batch processing uses periodic saves for crash resilience
- **Validation:** 3-LLM ensemble + stratified human coding sample

---

*Last updated: 8 February 2026*
