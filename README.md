# 🌐 Global Observatory of AI Governance Capacity

A research infrastructure project to systematically measure and compare AI governance capacity across jurisdictions worldwide.

## 🎯 Project Overview

This Observatory addresses a critical gap in AI governance research: while we can track *what* AI policies exist (via OECD.AI, IAPP, etc.), we lack systematic data on *whether states have the institutional capacity to implement them*.

**Research Question:** How do states build and deploy institutional capacity to govern artificial intelligence, and what explains variation in regulatory effectiveness across jurisdictions?

## 📁 Project Structure

```
observatory/
├── docs/                          # Documentation
│   ├── PROJECT_SUMMARY.md         # One-page research summary
│   ├── INDICATOR_RUBRIC.md        # Capacity indicator definitions
│   └── MPHIL_MODULE.md            # Teaching module outline
├── src/
│   └── collectors/
│       └── pipeline.py            # Data collection pipeline
├── data/
│   ├── schema/
│   │   └── observatory_schema.sql # PostgreSQL database schema
│   ├── sample/
│   │   └── jurisdictions_capacity.csv # Sample dataset (12 countries)
│   ├── raw/                       # Raw collected documents
│   └── cache/                     # HTTP response cache
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔬 Capacity Indicators

We measure governance capacity across **5 dimensions** with **15 indicators**:

| Dimension | Indicators | Weight |
|-----------|------------|--------|
| **Institutional Architecture** | Dedicated AI unit, coordination mechanisms, regulatory count | 20% |
| **Legal Authority** | Enforcement powers, AI legislation, procurement rules | 25% |
| **Technical Expertise** | Staff qualifications, standards participation, research partnerships | 20% |
| **Resources** | Budget allocation, staffing levels | 15% |
| **Implementation Evidence** | Enforcement actions, guidance issued, complaint mechanisms | 20% |

Each indicator is scored 0–3 with documented evidence and confidence levels.

## 🚀 Quick Start

### 1. Set up environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Playwright (JS rendering)
playwright install chromium
```

### 2. Run demo collection

```bash
cd observatory
python src/collectors/pipeline.py --demo
```

### 3. Explore sample data

```python
import pandas as pd

# Load sample capacity scores
df = pd.read_csv('data/sample/jurisdictions_capacity.csv')
print(df[['jurisdiction_name', 'composite_score', 'region']].head(10))
```

## 📊 Sample Data Preview

The sample dataset includes capacity scores for 12 jurisdictions:

| Jurisdiction | Region | Income | Score | Notable Features |
|--------------|--------|--------|-------|------------------|
| China | East Asia | Upper Middle | 86.7 | Comprehensive regulation, strong enforcement |
| United States | North America | High | 71.7 | Executive Order approach, NIST framework |
| France | Europe | High | 70.0 | Active DPA, strong standards participation |
| United Kingdom | Europe | High | 68.3 | AI Safety Institute, sectoral approach |
| Kenya | Sub-Saharan Africa | Lower Middle | 6.7 | Emerging policy discussion, capacity gaps |

## 🔐 Ethical Data Collection

This project follows responsible data collection principles:

1. **Check robots.txt** and Terms of Service before scraping
2. **Prefer APIs/exports** over web scraping when available
3. **Respect rate limits** with configurable delays
4. **Cache responses** to minimize server load
5. **Maintain provenance** for all data points
6. **No PII collection** or paywall circumvention

## 📚 Documentation

- **[Project Summary](docs/PROJECT_SUMMARY.md)** — Full research statement with methods and outputs
- **[Indicator Rubric](docs/INDICATOR_RUBRIC.md)** — Detailed scoring criteria for all 15 indicators
- **[MPhil Module](docs/MPHIL_MODULE.md)** — Teaching module "Governance Capacity for Digital Policy"

## 🛠️ Data Sources

| Priority | Source | Type | Coverage |
|----------|--------|------|----------|
| 1 | OECD.AI | API/Export | Global policy documents |
| 2 | IAPP Global Tracker | Scrape | Enforcement details |
| 3 | Stanford/Berkeley | Scrape | U.S. implementation |
| 4 | Government portals | Manual | Primary capacity data |
| 5 | Procurement portals | Scrape | Procurement rules |

## 📈 Outputs

### Academic
- Cross-national capacity dataset (80+ jurisdictions)
- Peer-reviewed articles on governance capacity determinants
- Working paper series with methodology documentation

### Policy
- Country scorecards for policymakers
- Best practice case studies
- Capacity diagnostic tools

### Technical
- Open-source data pipeline (this repository)
- API for programmatic data access
- Interactive dashboard

## 🤝 Contributing

This is a research project associated with [Institution]. We welcome:

- **Data contributions** for under-documented jurisdictions
- **Methodology feedback** on indicator definitions
- **Technical improvements** to collection pipelines

Please open an issue to discuss contributions.

## 📄 License

- **Code:** MIT License
- **Data:** CC-BY-NC 4.0 (attribution required; non-commercial use)
- **Documentation:** CC-BY 4.0

## 📧 Contact

[Your Name]  
[Institution]  
[Email]

---

*"The question is not whether AI will be governed, but whether it will be governed well. That depends on the capacity we build today."*
