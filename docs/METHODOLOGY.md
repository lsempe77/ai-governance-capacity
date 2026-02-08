# Implementation Capacity Analysis: Methodology Documentation

## Research Question
**"Do countries have the capacity to implement their AI policies, and how does this vary between high-income and developing countries?"**

---

## Two Approaches Compared

### Approach 1: Keyword Matching (Simple)
**File:** `src/analysis/capacity_analysis.py`

| Aspect | Description |
|--------|-------------|
| Method | Regex pattern matching for predefined keywords |
| Dimensions | 5 (Institutional, Enforcement, Resources, Operational, Expertise) |
| Scoring | Count matches → normalize to 0-1 |
| Weights | Arbitrary (25/25/20/15/15) |
| Evidence | None stored |
| Validation | None |

**Limitations:**
- Presence ≠ quality (mentioning "budget" ≠ adequate funding)
- False positives (keyword in wrong context)
- No theoretical grounding
- No evidence trail

---

### Approach 2: Rigorous NLP (Improved)
**File:** `src/analysis/rigorous_capacity_analysis.py`

| Aspect | Description |
|--------|-------------|
| Method | NER + Sentence classification + Pattern extraction |
| Dimensions | 5 (Clarity, Resources, Authority, Accountability, Coherence) |
| Scoring | 0-4 scale with explicit coding rules |
| Framework | Based on Mazmanian & Sabatier (1983), Winter (2012) |
| Evidence | Extracted quotes stored per dimension |
| Validation | 50-policy stratified sample generated |

**Improvements:**
- Extracts actual entities (amounts, dates, organizations)
- Sentence-level classification for context
- Evidence stored for each score
- Theoretically grounded in policy implementation literature
- Explicit coding scheme documented

---

## Theoretical Framework

Based on policy implementation research (Mazmanian & Sabatier 1983; Winter 2012; Howlett 2019):

### 1. CLARITY (0-4)
*The degree to which policy objectives, targets, scope, and definitions are precisely specified*

| Score | Criteria |
|-------|----------|
| 0 | No clear objectives stated |
| 1 | General objectives without specifics |
| 2 | Specific objectives but no measurable targets |
| 3 | Measurable targets for some objectives |
| 4 | Comprehensive targets with timelines |

### 2. RESOURCES (0-4)
*The degree to which financial, human, and technical resources are specified*

| Score | Criteria |
|-------|----------|
| 0 | No resources mentioned |
| 1 | General statement about need for resources |
| 2 | Commitment to allocate without specifics |
| 3 | Specific amounts for some resource types |
| 4 | Comprehensive allocation with sources |

### 3. AUTHORITY (0-4)
*The degree to which legal mandate, enforcement powers, and responsibilities are specified*

| Score | Criteria |
|-------|----------|
| 0 | No authority structures mentioned |
| 1 | General reference to government responsibility |
| 2 | Named agency without specific powers |
| 3 | Named agency with some defined powers |
| 4 | Clear authority, enforcement powers, and sanctions |

### 4. ACCOUNTABILITY (0-4)
*The degree to which monitoring, evaluation, and reporting mechanisms are specified*

| Score | Criteria |
|-------|----------|
| 0 | No accountability mechanisms |
| 1 | General commitment to monitoring |
| 2 | Monitoring mentioned without specifics |
| 3 | Specific monitoring with some reporting |
| 4 | Comprehensive M&E framework with review cycles |

### 5. COHERENCE (0-4)
*The degree to which the policy is internally consistent and aligned with other policies*

| Score | Criteria |
|-------|----------|
| 0 | Isolated policy with no references |
| 1 | Mentions other policies without integration |
| 2 | Some coordination mechanisms mentioned |
| 3 | Explicit alignment with specific policies |
| 4 | Comprehensive policy coherence framework |

---

## Results Comparison

### By Income Group (Approach 2 - Rigorous)

| Income Group | N | Total Score | Clarity | Resources | Authority | Accountability |
|--------------|---|-------------|---------|-----------|-----------|----------------|
| High Income | 1,743 | 0.093 | 0.27 | 0.47 | 0.48 | 0.33 |
| Upper Middle | 299 | 0.071 | 0.24 | 0.31 | 0.36 | 0.27 |
| Lower Middle | 126 | 0.083 | 0.18 | 0.39 | 0.42 | 0.33 |
| Low Income | 11 | 0.077 | 0.36 | 0.45 | 0.36 | 0.27 |

### Top Jurisdictions (Approach 2)

| Rank | Jurisdiction | Score | Income | N |
|------|--------------|-------|--------|---|
| 1 | United Kingdom | 0.233 | High | 72 |
| 2 | Nigeria | 0.210 | Lower Middle | 5 |
| 3 | United Nations | 0.200 | - | 1 |
| 4 | Armenia | 0.175 | Lower Middle | 2 |
| 5 | United States | 0.154 | High | 84 |
| 6 | Canada | 0.153 | High | 15 |
| 7 | Indonesia | 0.150 | Lower Middle | 2 |
| 8 | Cyprus | 0.150 | High | 9 |
| 9 | European Union | 0.132 | High | 60 |
| 10 | Ireland | 0.128 | High | 41 |

---

## Validation Protocol

### Sample Generated
- **50 policies** stratified by score level:
  - 15 high-scoring (≥0.2)
  - 15 medium-scoring (0.1-0.2)
  - 20 low-scoring (<0.1)

### Manual Validation Steps
1. Read full policy text
2. Score each dimension (0-4) independently
3. Compare to automated score
4. Calculate inter-rater reliability (if multiple coders)
5. Document discrepancies and refine scoring rules

### Files
- `data/analysis/rigorous_capacity/validation_sample.json` - 50 policies with evidence
- `data/analysis/rigorous_capacity/coding_scheme.json` - Full coding rules

---

## Limitations (Remaining)

Even with the improved methodology:

1. **Text availability**: Analysis quality depends on document length and detail
2. **English-only**: Non-English policies may be underrepresented or machine-translated
3. **Selection bias**: OECD.AI database may over-represent certain policy types
4. **Static snapshot**: Capacity evolves; this is a point-in-time measure
5. **Implementation ≠ Impact**: High capacity doesn't guarantee good outcomes

---

## Recommendations for Publication

### To Strengthen Claims:
1. **Manual validation**: Code 50-100 policies by hand, report Cohen's kappa
2. **External validation**: Correlate with World Bank governance indicators
3. **Robustness checks**: Alternative coding schemes, different thresholds
4. **Case studies**: Deep-dive on high vs. low scorers to validate findings
5. **Expert review**: Have AI governance practitioners review top/bottom 10

### Statistical Methods:
- Report confidence intervals for all scores
- Test income group differences with appropriate tests (Kruskal-Wallis)
- Control for policy length and document type
- Report effect sizes, not just significance

---

## Code and Data

| File | Description |
|------|-------------|
| `src/analysis/rigorous_capacity_analysis.py` | Main analysis script |
| `data/analysis/rigorous_capacity/rigorous_capacity_by_policy.json` | Policy-level results with evidence |
| `data/analysis/rigorous_capacity/rigorous_capacity_summary.json` | Aggregated results |
| `data/analysis/rigorous_capacity/coding_scheme.json` | Full coding definitions |
| `data/analysis/rigorous_capacity/validation_sample.json` | 50-policy validation sample |

---

## Ethics Analysis Methodology

### Framework Validation (February 2026)

The ethics analysis component scores policies against four internationally-recognized AI ethics frameworks. The primary normative framework is the **UNESCO Recommendation on the Ethics of Artificial Intelligence (2021)**, validated against the official source document (UNESDOC pf0000381137).

#### Framework Verification

| Framework | Source | Verified Against |
|-----------|--------|------------------|
| UNESCO AI Recommendation | 193 Member States, Nov 2021 | Official text from unesco.org/en/legal-affairs/recommendation-ethics-artificial-intelligence |
| Jobin et al. (2019) | "The global landscape of AI ethics guidelines" - Nature Machine Intelligence | Peer-reviewed publication |
| OECD AI Principles (2019) | OECD Council Recommendation | Official OECD.AI documentation |
| EU AI Act (2024) | Regulation (EU) 2024/1689 | Official Journal of the EU |

#### UNESCO Framework (Primary Reference)

**4 Core Values:**
1. Respect, protection and promotion of human rights and fundamental freedoms and human dignity
2. Environment and ecosystem flourishing
3. Ensuring diversity and inclusiveness
4. Living in peaceful, just and interconnected societies

**10 Core Principles:**
1. Proportionality and Do No Harm
2. Safety and Security
3. Fairness and Non-discrimination
4. Sustainability
5. Right to Privacy and Data Protection
6. Human Oversight and Determination
7. Transparency and Explainability
8. Responsibility and Accountability
9. Awareness and Literacy
10. Multi-stakeholder and Adaptive Governance

**11 Policy Action Areas:**
1. Ethical Impact Assessment
2. Ethical Governance and Stewardship
3. Data Policy
4. Development and International Cooperation
5. Environment and Ecosystems
6. Gender
7. Culture
8. Education and Research
9. Communication and Information
10. Economy and Labour
11. Health and Social Well-being

#### Scoring Methodology

Each policy is scored on a 0-3 scale per principle:
- **0**: Absent - principle not mentioned
- **1**: Mentioned - referenced but not elaborated
- **2**: Discussed - substantive treatment with context
- **3**: Central theme - core focus with implementation details

**Overall Ethics Score (0-100):**
- Jobin principles (11): 33 points max
- OECD principles (5): 15 points max
- UNESCO values (4): 12 points max
- UNESCO principles (10): 30 points max
- EU AI Act alignment (8): 10 points max (weighted lower as newer standard)

Normalized to 0-100 scale.

#### Implementation Classification

| Level | Description | Examples |
|-------|-------------|----------|
| L0 | Aspirational | Vision statements, principles without mechanisms |
| L1 | Voluntary | Industry guidelines, best practices |
| L2 | Soft Law | Recommendations, standards, non-binding frameworks |
| L3 | Co-regulatory | Industry codes with government oversight |
| L4 | Hard Law | Binding legislation with enforcement mechanisms |

#### Normative Approach Classification

Based on ethical theory traditions:
- **Deontological**: Duty-based, focusing on rights and rules
- **Consequentialist**: Outcome-focused, utilitarian considerations
- **Virtue Ethics**: Character-focused, emphasizing good practices
- **Care Ethics**: Relationship-focused, emphasizing stakeholder welfare
- **Rights-based**: Grounded in human rights frameworks
- **Procedural**: Process-focused, emphasizing governance mechanisms

#### Validation Results

The UNESCO Recommendation itself scores **92/100** using this methodology, confirming:
1. The scoring framework captures the intended normative content
2. A comprehensive policy can achieve high scores
3. The methodology discriminates between substantive and superficial ethics content

### Ethics Analysis Files

| File | Description |
|------|-------------|
| `src/analysis/ethics_sota_analysis.py` | Main SOTA analysis script |
| `src/analysis/analyze_unesco_ethics.py` | UNESCO document analysis |
| `data/analysis/ethics/sota_ethics_analysis.json` | Full corpus results (50 policies) |
| `data/analysis/ethics/unesco_ethics_analysis.json` | UNESCO framework validation |
| `docs/AI_ETHICS_READING_LIST.html` | Curated reading list |

---

*Methodology documentation updated: February 8, 2026*
