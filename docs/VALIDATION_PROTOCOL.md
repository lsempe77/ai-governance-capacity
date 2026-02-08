# Manual Validation Protocol
## Implementation Capacity Analysis - Inter-Coder Reliability Study

---

## Purpose
Validate automated capacity scores by having human coders independently rate a stratified sample of 50 policies using the same coding scheme.

---

## Instructions for Coders

### Step 1: Read the Full Policy Text
For each policy, read the complete text provided. Focus on identifying concrete implementation details.

### Step 2: Score Each Dimension (0-4)
Using the coding rules below, assign a score for each of the 5 dimensions.

### Step 3: Record Evidence
For each dimension where you assign a score >0, note the specific text that supports your score.

### Step 4: Flag Uncertainties
If you're unsure about a score, mark it with "?" and explain your uncertainty.

---

## Coding Scheme

### DIMENSION 1: CLARITY (0-4)
*Does the policy clearly specify what it aims to achieve?*

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No clear objectives stated | Vague aspirational language only |
| 1 | General objectives without specifics | "Promote AI development" |
| 2 | Specific objectives but no measurable targets | "Increase AI adoption in healthcare" |
| 3 | Measurable targets for some objectives | "Train 10,000 AI specialists by 2025" |
| 4 | Comprehensive targets with timelines | Multiple quantified goals with dates |

**Look for:** Objectives, goals, targets, timelines, definitions, scope statements

---

### DIMENSION 2: RESOURCES (0-4)
*Does the policy specify what resources will be allocated?*

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No resources mentioned | - |
| 1 | General statement about need for resources | "Adequate resources will be provided" |
| 2 | Commitment to allocate without specifics | "Government will fund implementation" |
| 3 | Specific amounts for some resource types | "€50 million allocated for AI research" |
| 4 | Comprehensive allocation with funding sources | Budget, staff numbers, infrastructure, multi-year |

**Look for:** Budget amounts, funding, staff/FTE numbers, infrastructure, equipment

---

### DIMENSION 3: AUTHORITY (0-4)
*Does the policy specify who is responsible and what powers they have?*

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No authority structures mentioned | - |
| 1 | General reference to government responsibility | "Government will oversee" |
| 2 | Named agency without specific powers | "Ministry of Digital Affairs responsible" |
| 3 | Named agency with some defined powers | "Agency may issue guidance and conduct reviews" |
| 4 | Clear authority with enforcement and sanctions | Named body + investigation powers + penalties |

**Look for:** Agency names, ministries, commissions, enforcement powers, penalties, sanctions, legal basis

---

### DIMENSION 4: ACCOUNTABILITY (0-4)
*Does the policy specify how implementation will be monitored and reported?*

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | No accountability mechanisms | - |
| 1 | General commitment to monitoring | "Progress will be tracked" |
| 2 | Monitoring mentioned without specifics | "Regular reviews will be conducted" |
| 3 | Specific monitoring with some reporting | "Annual report to Parliament" |
| 4 | Comprehensive M&E framework | KPIs + review cycles + evaluation methodology + feedback |

**Look for:** Monitoring, evaluation, reporting requirements, review cycles, KPIs, audits, assessments

---

### DIMENSION 5: COHERENCE (0-4)
*Is the policy aligned with other policies and international standards?*

| Score | Criteria | Examples |
|-------|----------|----------|
| 0 | Isolated policy with no references | - |
| 1 | Mentions other policies without integration | "Consistent with national strategy" |
| 2 | Some coordination mechanisms mentioned | "Inter-ministerial working group" |
| 3 | Explicit alignment with specific policies | "Implements Article 5 of EU AI Act" |
| 4 | Comprehensive coherence framework | Cross-references + coordination body + international alignment |

**Look for:** References to other laws/policies, coordination mechanisms, international standards (ISO, OECD, EU)

---

## Validation Workflow

```
1. Coder A scores all 50 policies independently
2. Coder B scores all 50 policies independently  
3. Calculate inter-rater reliability (Cohen's kappa per dimension)
4. Discuss discrepancies (difference ≥2 points)
5. Reach consensus scores for discrepant cases
6. Compare consensus to automated scores
7. Calculate correlation and systematic bias
```

---

## Output Template

For each policy, record:

```
Policy ID: ___
Title: ___
Jurisdiction: ___

MANUAL SCORES (0-4):
  Clarity:        ___  Evidence: ___
  Resources:      ___  Evidence: ___
  Authority:      ___  Evidence: ___
  Accountability: ___  Evidence: ___
  Coherence:      ___  Evidence: ___

TOTAL: ___ / 20

AUTOMATED SCORE: ___ / 20

DIFFERENCE: ___

NOTES: ___
```

---

## Statistical Analysis Plan

### Inter-Rater Reliability
- Cohen's kappa (κ) for each dimension
- Interpretation: κ > 0.8 = excellent, 0.6-0.8 = substantial, 0.4-0.6 = moderate

### Validation of Automated Scores
- Pearson correlation between manual and automated scores
- Mean absolute error (MAE)
- Systematic bias (mean difference)
- Bland-Altman plot

### Acceptance Criteria
- κ > 0.6 for all dimensions (inter-rater)
- r > 0.7 between manual and automated (validity)
- MAE < 0.5 on normalized scale

---

## Files

| File | Purpose |
|------|---------|
| `validation_coding_sheet.csv` | Spreadsheet for entering manual scores |
| `validation_policies.md` | Full text of 50 policies for coding |
| `validation_sample.json` | Raw data with automated scores |
