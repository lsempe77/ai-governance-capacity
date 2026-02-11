# Writing Improvement Plan for AI Governance Capacity Book

**Created:** February 9, 2026  
**Status:** Phase 1 in progress

## Core Principles

- **NO bullet points** — Convert all lists to flowing prose
- **Academic/formal but accessible** — Avoid grandiosity and overcomplicated language
- **Define technical terms** — Create callout boxes for specialized terminology
- **Cross-reference tables and figures** — Link and explain all visual elements
- **Explain figures/tables** — Add 2-4 sentences of interpretation for each
- **Substantive section introductions** — Each major section needs a full paragraph (not just 1-2 sentences) that: (1) connects to the broader argument, (2) previews subsections, and (3) explains why this matters

---

## Issues Identified

### 1. Bullet Point Overuse (Critical — 18+ chapters)

**Locations:**
- index.qmd: Key findings list
- 01-introduction.qmd: Three numbered lists (gaps, contributions, findings)
- 02-literature.qmd: Mazmanian–Sabatier conditions
- 15-discussion.qmd: Limitations list
- 16-conclusion.qmd: "Five Takeaways"

**Fix:** Convert to prose paragraphs with italicized concept phrases

### 2. Grandiose/Promotional Language (Moderate — 12 chapters)

**Examples to fix:**
- "extraordinary volume" → "substantial volume"
- "deceptively simple question" → "a question that proves more complex than it appears"
- "fundamentally reoriented" → "significantly shifted the focus"
- "most unsettling finding" → "one finding with significant methodological implications"
- "more nuanced than the question suggests" → "a more complex picture than this framing implies"

### 3. Missing Term Definitions (High priority)

**Terms requiring callout boxes:**
- Cohen's *d* (01-introduction.qmd)
- ICC (Intraclass Correlation) (01-introduction.qmd)
- Welch's *t*-test (05-capacity-landscape.qmd)
- Quantile regression (06-capacity-determinants.qmd)
- Tobit model (06-capacity-determinants.qmd)
- Principal Component Analysis (13-pca-nexus.qmd)
- Cronbach's α (13-pca-nexus.qmd)
- Brussels Effect (02-literature.qmd)
- Isomorphic mimicry (02-literature.qmd)

### 4. Inadequate Figure/Table Explanations (High priority — 40+ figures)

**Pattern needed:** After each figure/table reference, add 2-4 sentences explaining:
1. What the reader should notice first
2. What the key numbers mean substantively
3. How this connects to the chapter's argument

### 5. Cross-Reference Gaps (Moderate)

**Missing connections:**
- 03-data-methods.qmd → appendix-validation.qmd
- 05-capacity-landscape.qmd → appendix-rubric.qmd
- 06-capacity-determinants.qmd → 07-capacity-inequality.qmd
- 09-ethics-landscape.qmd → 13-pca-nexus.qmd

### 6. Thin Section Introductions (High priority)

**Problem:** Many sections start with just 1-2 sentences before diving into subsections or technical content.

**Solution:** Add full introductory paragraphs (4-6 sentences) that:
- Connect the section to the broader research question
- Preview what subsections will cover
- Explain why this analysis matters
- Provide conceptual framing before technical details

**Example from 02-literature.qmd Implementation Science section:**
- Before: 1 sentence about Pressman & Wildavsky
- After: Full paragraph explaining relevance to AI governance + preview of three streams

### 7. Passive Voice Overuse (Minor)

**Preferred patterns:**
- "We find that..." (not "It was found that...")
- "The LLM ensemble scored..." (not "Policies were scored by...")
- "@fig-X shows that..." (not "This can be seen in...")

---

## Implementation Phases

### Phase 1: First/Last Impressions ✓ IN PROGRESS
**Files:** index.qmd, 01-introduction.qmd, 16-conclusion.qmd  
**Tasks:**
- Remove all bullet points
- Tone down grandiose language
- Improve flow and readability

### Phase 2: Foundational Chapters
**Files:** 02-literature.qmd, 03-data-methods.qmd, 04-scoring.qmd  
**Tasks:**
- Add definition boxes for technical terms
- Improve figure/table explanations
- Convert bullets to prose
- Add substantive section introductions

### Phase 3: Capacity Analysis (Part I)
**Files:** 05-capacity-landscape.qmd through 08-capacity-dynamics.qmd  
**Tasks:**
- Expand figure/table explanations
- Add cross-references
- Strengthen section introductions
- Add statistical term definitions

### Phase 4: Ethics Analysis (Part II)
**Files:** 09-ethics-landscape.qmd through 12-ethics-dynamics.qmd  
**Tasks:**
- Mirror capacity chapters' improvements
- Ensure parallel structure

### Phase 5: Advanced Analysis (Part III)
**Files:** 13-pca-nexus.qmd, 14-robustness.qmd  
**Tasks:**
- Add PCA/statistical term definitions
- Enhance technical explanations

### Phase 6: Synthesis & UNESCO
**Files:** 15-discussion.qmd, 17-unesco-landscape.qmd through 20-unesco-dynamics.qmd  
**Tasks:**
- Convert limitation/recommendation bullets to prose
- Improve policy recommendations

### Phase 7: Appendices
**Files:** All appendix files  
**Tasks:**
- Polish rubric presentation
- Ensure all cross-references work
- Add explanatory text

---

## Templates

### Definition Box Template

```markdown
::: {.callout-tip title="Definition: [Term]"}
[2-3 sentence plain-language explanation with context for this study]
:::
```

### Figure Explanation Pattern

```markdown
![Caption text.](path/to/figure.png){#fig-label}

@fig-label displays [what it shows]. The [key visual element] reveals [substantive finding]. 
This [interpretation] explains why [connection to argument]. [Additional context sentence].
```

### Prose Conversion Pattern (from bullets)

**Before:**
```markdown
Our contributions are:

1. Point one
2. Point two
3. Point three
```

**After:**
``Section introductions | ~50 | All chapters |
| Cross-reference additions | ~30 | All chapters |
| Language toning | ~40 | 18 chapters |
| **Total** | **~20. Third, [point three].
```

---

## Estimated Scope

| Task Category | Est. Edits | Chapters |
|:--|--:|:--|
| Bullet → prose conversion | ~25 | 12 chapters |
| Definition boxes | ~12 | 8 chapters |
| Figure/table explanations | ~45 | 15 chapters |
| Cross-reference additions | ~30 | All chapters |
| Language toning | ~40 | 18 chapters |
| **Total** | **~150 edits** | **24 files** |

---

## Progress Tracking

- [x] Phase 1: First/Last Impressions ✅ COMPLETED (Feb 9, 2026)
  - [x] index.qmd
  - [x] 01-introduction.qmd
  - [x] 16-conclusion.qmd
- [ ] Phase 2: Foundational Chapters
- [ ] Phase 3: Capacity Analysis
- [ ] Phase 4: Ethics Analysis
- [ ] Phase 5: Advanced Analysis
- [ ] Phase 6: Synthesis & UNESCO
- [ ] Phase 7: Appendices

---

## Notes

- Maintain parallel structure between capacity and ethics chapters
- Ensure all @sec-, @fig-, @tbl- references are valid
- Keep chapter summaries in callout boxes (these are good)
- Preserve all quantitative findings and citations
- Keep the academic rigor while improving accessibility
