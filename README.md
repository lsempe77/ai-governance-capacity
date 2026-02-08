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
│   └── MPHIL_MODULE.md            # Teaching module outline
│
├── src/
│   ├── scrapers/                  # Data collection scripts (8 active)
│   │   ├── retrieve_v3.py         # Final document retriever (+ Wayback Machine)
│   │   ├── download_all_pdfs.py   # Phase 1 bulk downloader
│   │   ├── find_pdfs_with_claude.py # Claude-assisted URL finder
│   │   ├── integrate_content.py   # Content file → corpus matcher
│   │   ├── audit_matching.py      # PDF-to-corpus matching audit
│   │   └── ...                    # UNESCO/OECD specific scrapers
│   ├── analysis/                  # Analysis scripts (Phase 2-3, TBD)
│   └── collectors/                # Corpus building (completed)
│
├── data/
│   ├── corpus/                    # Master corpus (2,216 entries)
│   │   └── corpus_master_20260127.json
│   ├── pdfs/                      # Downloaded documents (~2,085 files)
│   ├── analysis/                  # Analysis outputs (Phase 2+)
│   └── _archive/                  # Archived raw/intermediate data
```

## 📊 Corpus Statistics

| Metric | Value |
|--------|-------|
| **Total policies** | 2,216 |
| **Documents downloaded** | ~2,085 (94%) |
| **Jurisdictions** | 70+ countries + EU/international |
| **Time span** | 2017–2025 |
| **Source** | OECD.AI Policy Observatory |

## 🔬 Capacity Indicators

We measure governance capacity across **5 dimensions**:

| Dimension | Weight |
|-----------|--------|
| **Institutional Architecture** — Dedicated AI unit, coordination mechanisms | 20% |
| **Legal Authority** — Enforcement powers, AI legislation, procurement rules | 25% |
| **Technical Expertise** — Staff qualifications, standards, research | 20% |
| **Resources** — Budget allocation, staffing levels | 15% |
| **Implementation Evidence** — Enforcement actions, guidance, complaints | 20% |

Each indicator scored 0–3 with documented evidence and confidence levels.

## 🚀 Project Phases

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for full details.

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Corpus construction & document download |
| **Phase 1** | ⏳ Next | Text extraction & parsing pipeline |
| **Phase 2** | ❌ Planned | AI-powered classification & scoring |
| **Phase 3** | ❌ Planned | SOTA analysis & validation |
| **Phase 4** | ❌ Planned | Reporting & dissemination |

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
