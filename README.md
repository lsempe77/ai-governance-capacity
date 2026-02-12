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
├── book2_ethics/                  # 📗 Quarto Book: AI Ethics Governance
├── book3_unesco/                  # 📕 Quarto Book: UNESCO Alignment
│
├── src/
│   ├── scrapers/                  # Data collection scripts
│   │   ├── retrieve_v3.py         # Document retriever (+ Wayback Machine)
│   │   ├── download_all_pdfs.py   # Bulk PDF downloader
│   │   ├── download_oecd_docs.py  # OECD document downloader
│   │   ├── download_unesco.py     # UNESCO document downloader
│   │   ├── find_pdfs_with_claude.py # Claude-assisted URL finder
│   │   ├── integrate_content.py   # Content file → corpus matcher
│   │   ├── audit_matching.py      # PDF-to-corpus matching audit
│   │   └── add_unesco_content.py  # UNESCO content integration
│   └── analysis/                  # Analysis pipeline
│       ├── extract_text.py        # Text extraction + quality flags
│       ├── score_policies.py      # 3-model LLM scoring (parallel)
│       ├── inter_rater.py         # Inter-rater reliability
│       ├── country_metadata.py    # Country → income/region/GDP mapping
│       ├── sota_analysis.py       # Core analyses
│       ├── advanced_analysis.py   # Robustness, multilevel, PCA
│       ├── extended_analysis.py   # Inequality, quantile & Tobit
│       ├── diffusion_frontier.py  # Policy diffusion & efficiency
│       └── unesco_*.py            # UNESCO alignment analysis
│
└── data/
    ├── corpus/                    # Master corpus
    ├── pdfs/                      # Downloaded documents
    └── analysis/                  # Analysis outputs
```

## 📚 Publications

This project produces three research outputs as Quarto books:

| Book | Focus |
|------|-------|
| **📘 Book 1** | AI governance implementation capacity |
| **📗 Book 2** | AI ethics governance operationalisation |
| **📕 Book 3** | Alignment with UNESCO AI Recommendation |

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add API key to .env
OPENROUTER_API_KEY=sk-or-v1-...
```

## 🚀 Usage

### Building the Books

```bash
cd book1_capacity && quarto render
cd book2_ethics && quarto render
cd book3_unesco && quarto render
```

### Running the Analysis Pipeline

```bash
# 1. Extract text from PDFs
python src/analysis/extract_text.py

# 2. Score policies with LLM ensemble
python src/analysis/score_policies.py

# 3. Run statistical analyses
python src/analysis/sota_analysis.py
python src/analysis/advanced_analysis.py
```

## 📊 Data

- **Source**: OECD.AI Policy Observatory
- **Corpus**: 2,216 AI policy documents
- **Coverage**: 70+ countries, 2017–2025
- **Scoring**: 10 dimensions (5 capacity + 5 ethics), 0–4 scale
- **Method**: 3-model LLM ensemble (Claude, GPT-4o, Gemini)

## 📄 License

Research project — International Initiative for Impact Evaluation (3ie)
