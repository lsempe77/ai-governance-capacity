# AI Governance Capacity Indicator Rubric

## Overview

This rubric defines 15 indicators across 5 dimensions of AI governance capacity. Each indicator has a standardized scoring scale (0–3), clear operationalization, and identified data sources.

---

## Dimension 1: Institutional Architecture

### 1.1 Dedicated AI Regulatory Unit

**Definition:** Existence of a government unit with explicit mandate for AI oversight/regulation.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No dedicated unit; AI handled ad-hoc by general tech/digital ministry | Most LMICs as of 2024 |
| 1 | AI mentioned in existing agency mandate but no dedicated staff/budget | UK pre-2023 (spread across regulators) |
| 2 | Dedicated AI unit within broader agency (≥3 FTE) | Singapore IMDA AI Governance Unit |
| 3 | Standalone AI agency or authority with statutory powers | Proposed under EU AI Act (national authorities) |

**Data sources:** Government organograms, agency websites, annual reports, budget documents  
**Extraction keywords:** "AI unit", "artificial intelligence office", "AI governance", staffing tables

---

### 1.2 Cross-Ministry Coordination Mechanism

**Definition:** Formal body coordinating AI policy across government departments.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No coordination mechanism | — |
| 1 | Informal working group or ad-hoc committee | — |
| 2 | Standing interministerial committee with regular meetings | UK AI Council (advisory) |
| 3 | Statutory coordination body with decision-making authority and secretariat | France Comité de l'IA |

**Data sources:** Strategy documents, executive orders, cabinet office publications  
**Extraction keywords:** "interministerial", "coordination committee", "national AI council", "task force"

---

### 1.3 Regulatory Institutional Count

**Definition:** Number of distinct agencies with formal AI-related regulatory responsibilities.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | 0 agencies with AI mandate | — |
| 1 | 1 agency (typically data protection authority) | Many European countries pre-AI Act |
| 2 | 2–3 agencies with complementary mandates | UK (ICO + FCA + CMA + Ofcom sectoral approach) |
| 3 | 4+ agencies with clear mandate delineation and coordination | EU post-AI Act (sectoral + horizontal) |

**Data sources:** OECD.AI policy inventory, national strategy documents, regulatory mapping exercises  
**Extraction method:** Entity extraction of agency names + mandate verification

---

## Dimension 2: Legal Authority

### 2.1 Enforcement Powers

**Definition:** Legal tools available to regulators for AI oversight.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No enforcement powers; purely advisory | Many early AI strategies |
| 1 | Power to issue guidance and recommendations only | UK current approach (soft law) |
| 2 | Power to impose administrative fines | GDPR-style enforcement for AI |
| 3 | Investigatory powers + fines + injunctive relief + criminal referral | EU AI Act high-risk enforcement |

**Data sources:** Enabling legislation, regulatory statutes, enforcement guidelines  
**Extraction keywords:** "fine", "penalty", "investigation", "audit", "injunction", "prohibition"

---

### 2.2 AI-Specific Legislation

**Definition:** Existence of binding legal instruments specifically addressing AI.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No AI-specific legislation | Many jurisdictions |
| 1 | Soft law only (strategies, guidelines, principles) | Most OECD countries 2018–2022 |
| 2 | Sectoral AI regulation (e.g., autonomous vehicles, healthcare AI) | US state-level AV laws |
| 3 | Comprehensive horizontal AI legislation | EU AI Act, China AI regulations |

**Data sources:** National gazettes, legislative databases, OECD.AI tracker  
**Extraction method:** Document classification (law vs. strategy vs. guidance)

---

### 2.3 Public Procurement Rules for AI

**Definition:** Specific requirements for government procurement of AI systems.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No AI-specific procurement rules | Most countries |
| 1 | General guidance on AI procurement (non-binding) | UK AI Procurement Guidelines |
| 2 | Mandatory risk assessment for AI procurement | Canada Algorithmic Impact Assessment |
| 3 | Comprehensive procurement framework (transparency, audit rights, bias testing) | Proposed in some US states |

**Data sources:** Procurement portals, government digital service websites, strategy documents  
**Extraction keywords:** "procurement", "acquisition", "algorithmic impact", "vendor", "government AI"

---

## Dimension 3: Technical Expertise

### 3.1 Staff Technical Qualifications

**Definition:** Proportion of AI governance staff with relevant technical background.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No evidence of technical staff | — |
| 1 | <25% staff with technical background (CS, data science, engineering) | — |
| 2 | 25–50% technical staff OR dedicated technical advisory board | Singapore PDPC |
| 3 | >50% technical staff AND in-house research capacity | UK AI Safety Institute |

**Data sources:** Agency annual reports, LinkedIn analysis (with caution), job postings, staff directories  
**Proxy indicators:** Job postings requiring technical skills, research publications by agency staff

---

### 3.2 Standards Body Participation

**Definition:** Active participation in international AI standards development.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No participation in AI standards bodies | — |
| 1 | Observer status or passive membership | — |
| 2 | Active participation in 1–2 bodies (ISO, IEEE, ITU) | Most OECD members |
| 3 | Leadership roles (working group chairs, conveners) in multiple bodies | US, China, Germany |

**Data sources:** ISO/IEEE/ITU participant lists, national standards body publications  
**Extraction method:** Entity matching against standards body membership rosters

---

### 3.3 Research Partnerships

**Definition:** Formal collaborations between regulators and research institutions.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No formal research partnerships | — |
| 1 | Ad-hoc consultations with academics | — |
| 2 | Formal MoUs with universities/research institutes | — |
| 3 | In-house research lab OR funded research programs | UK AI Safety Institute, NIST |

**Data sources:** Agency websites, MoU announcements, research grant databases  
**Extraction keywords:** "partnership", "collaboration", "research", "university", "institute"

---

## Dimension 4: Resources

### 4.1 Budget Allocation

**Definition:** Dedicated funding for AI governance activities.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No identifiable AI governance budget | — |
| 1 | <$1M annual budget for AI governance | — |
| 2 | $1–10M annual budget | — |
| 3 | >$10M annual budget with multi-year commitment | UK AI Safety Institute (£100M+) |

**Data sources:** Budget documents, appropriations acts, agency annual reports  
**Extraction method:** Currency extraction + normalization (PPP adjustment for comparisons)  
**Note:** Adjust thresholds by country income level for meaningful comparison

---

### 4.2 Staffing Levels

**Definition:** Full-time equivalent staff dedicated to AI governance.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | 0 FTE dedicated to AI | — |
| 1 | 1–5 FTE | — |
| 2 | 6–20 FTE | — |
| 3 | >20 FTE | UK AI Safety Institute, NIST AI |

**Data sources:** Annual reports, organograms, job posting analysis  
**Extraction method:** Staff count from reports; proxy via job postings if unavailable

---

## Dimension 5: Implementation Evidence

### 5.1 Enforcement Actions

**Definition:** Documented regulatory actions related to AI systems.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No enforcement actions | — |
| 1 | 1–2 enforcement actions (guidance letters, warnings) | — |
| 2 | 3–10 enforcement actions including fines | Italy Garante (Replika, ChatGPT) |
| 3 | >10 enforcement actions with significant penalties | — |

**Data sources:** Agency press releases, enforcement databases, news monitoring  
**Extraction method:** Event extraction from agency communications

---

### 5.2 Guidance Documents Issued

**Definition:** Non-binding guidance produced to support AI governance compliance.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No AI guidance issued | — |
| 1 | 1–3 guidance documents | — |
| 2 | 4–10 guidance documents covering multiple topics | UK ICO AI guidance suite |
| 3 | >10 guidance documents + regular updates + consultation processes | Singapore PDPC |

**Data sources:** Agency publications pages, document registries  
**Extraction method:** Document count + topic classification

---

### 5.3 Complaint/Oversight Mechanisms

**Definition:** Channels for public complaints and independent oversight of AI systems.

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No complaint mechanism for AI issues | — |
| 1 | General complaint channel that accepts AI-related complaints | — |
| 2 | Dedicated AI complaint channel OR ombudsman with AI remit | — |
| 3 | Dedicated AI complaint channel + tribunal/appeal mechanism + proactive audits | — |

**Data sources:** Agency websites, administrative procedure rules, ombudsman reports  
**Extraction keywords:** "complaint", "appeal", "tribunal", "ombudsman", "audit"

---

## Scoring Methodology

### Composite Index Calculation

**Dimension scores:** Average of constituent indicator scores (0–3 scale)

**Overall capacity score:** Weighted average of dimension scores

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Institutional Architecture | 20% | Foundation for governance |
| Legal Authority | 25% | Necessary for enforcement |
| Technical Expertise | 20% | Quality of oversight |
| Resources | 15% | Enabler, not determinant |
| Implementation Evidence | 20% | Actual effectiveness |

**Final score:** Converted to 0–100 scale for communication

### Confidence Scoring

Each indicator score includes a confidence level:

| Confidence | Criteria |
|------------|----------|
| High | Primary source (official document, legislation) |
| Medium | Secondary source (news, tracker) or inference |
| Low | Proxy indicator or expert estimate |

---

## Data Collection Protocol

### For each jurisdiction:

1. **OECD.AI baseline:** Extract policy documents and basic metadata
2. **Primary source search:** Agency websites, gazettes, budget documents
3. **Secondary validation:** IAPP, Stanford tracker, news search
4. **NLP extraction:** Apply entity recognition and pattern matching
5. **Manual review:** 20% random sample for quality control
6. **Expert validation:** Survey instrument for ambiguous cases

### Update frequency:

- **Quarterly:** Enforcement actions, guidance documents
- **Annually:** Budgets, staffing, institutional changes
- **Event-triggered:** New legislation, major policy announcements

---

## Limitations & Caveats

1. **Language barriers:** Non-English sources require translation; may miss nuance
2. **Formal vs. informal:** Scores capture formal capacity, not informal networks or political will
3. **Comparability:** Cross-national comparison requires careful normalization (income-adjusted thresholds)
4. **Lag:** Official data often lags reality by 12–24 months
5. **Gaming:** Once published, indicators may be gamed by jurisdictions

**Mitigation:** Triangulate sources, report uncertainty, prioritize implementation evidence over formal structures.
