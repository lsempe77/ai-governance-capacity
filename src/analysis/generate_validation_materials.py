"""
Generate validation materials:
1. CSV coding sheet for manual scoring
2. Document with full policy texts
"""
import json
import csv
from pathlib import Path

# Load validation sample
sample = json.load(open('data/analysis/rigorous_capacity/validation_sample.json', encoding='utf-8'))

# Load full corpus for text
corpus = json.load(open('data/corpus/corpus_master_20260128.json', encoding='utf-8'))
corpus_lookup = {e['url']: e for e in corpus['entries']}

# Load enriched for URLs
enriched = json.load(open('data/oecd/enriched/oecd_enriched_20260127_203406.json', encoding='utf-8'))
enriched_lookup = {p['title'] + '_' + p['jurisdiction']: p for p in enriched['policies']}

output_dir = Path('data/analysis/validation')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Create CSV coding sheet
print("Creating coding sheet...")
with open(output_dir / 'validation_coding_sheet.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Header
    writer.writerow([
        'Policy_ID', 'Title', 'Jurisdiction', 'Year', 'Word_Count',
        'AUTO_Clarity', 'AUTO_Resources', 'AUTO_Authority', 'AUTO_Accountability', 'AUTO_Coherence', 'AUTO_Total',
        'MANUAL_Clarity', 'MANUAL_Resources', 'MANUAL_Authority', 'MANUAL_Accountability', 'MANUAL_Coherence', 'MANUAL_Total',
        'Clarity_Evidence', 'Resources_Evidence', 'Authority_Evidence', 'Accountability_Evidence', 'Coherence_Evidence',
        'Coder_Notes'
    ])
    
    # Data rows
    for i, p in enumerate(sample, 1):
        auto_total = p['clarity_score'] + p['resources_score'] + p['authority_score'] + p['accountability_score'] + p['coherence_score']
        writer.writerow([
            f'P{i:03d}',
            p['title'][:80],
            p['jurisdiction'],
            p.get('year', ''),
            p.get('word_count', 0),
            p['clarity_score'],
            p['resources_score'],
            p['authority_score'],
            p['accountability_score'],
            p['coherence_score'],
            auto_total,
            '', '', '', '', '', '',  # Manual scores (to be filled)
            '', '', '', '', '',  # Evidence (to be filled)
            ''  # Notes
        ])

print(f"  Saved: {output_dir / 'validation_coding_sheet.csv'}")

# 2. Create document with full policy texts
print("\nCreating policy texts document...")

with open(output_dir / 'validation_policies.md', 'w', encoding='utf-8') as f:
    f.write("# Validation Sample: Policy Texts\n\n")
    f.write("This document contains the full text of 50 policies for manual coding.\n")
    f.write("Use the coding sheet (CSV) to record your scores.\n\n")
    f.write("---\n\n")
    
    for i, p in enumerate(sample, 1):
        f.write(f"## P{i:03d}: {p['title']}\n\n")
        f.write(f"**Jurisdiction:** {p['jurisdiction']}\n")
        f.write(f"**Year:** {p.get('year', 'Unknown')}\n")
        f.write(f"**Word Count:** {p.get('word_count', 0)}\n\n")
        
        # Automated scores for reference (hidden during blind coding)
        f.write("<details>\n<summary>Automated Scores (expand after manual coding)</summary>\n\n")
        f.write(f"- Clarity: {p['clarity_score']}/4\n")
        f.write(f"- Resources: {p['resources_score']}/4\n")
        f.write(f"- Authority: {p['authority_score']}/4\n")
        f.write(f"- Accountability: {p['accountability_score']}/4\n")
        f.write(f"- Coherence: {p['coherence_score']}/4\n")
        f.write(f"- **Total: {p['clarity_score'] + p['resources_score'] + p['authority_score'] + p['accountability_score'] + p['coherence_score']}/20**\n\n")
        
        # Evidence extracted
        f.write("**Automated Evidence:**\n")
        for dim in ['clarity', 'resources', 'authority', 'accountability', 'coherence']:
            evidence = p.get(f'{dim}_evidence', [])
            if evidence:
                f.write(f"- {dim.title()}: {'; '.join(evidence[:2])}\n")
        f.write("</details>\n\n")
        
        # Get full text from corpus
        key = p['title'] + '_' + p['jurisdiction']
        enriched_policy = enriched_lookup.get(key, {})
        url = enriched_policy.get('url', '')
        corpus_entry = corpus_lookup.get(url, {})
        
        full_text = corpus_entry.get('content', '') or enriched_policy.get('initiative_overview', '') or ''
        
        f.write("### Full Text\n\n")
        if full_text:
            # Limit to first 3000 words for readability
            words = full_text.split()
            if len(words) > 3000:
                f.write(f"*[Showing first 3000 of {len(words)} words]*\n\n")
                full_text = ' '.join(words[:3000]) + '...'
            f.write(f"{full_text}\n\n")
        else:
            f.write("*[No text available in corpus]*\n\n")
        
        f.write("---\n\n")

print(f"  Saved: {output_dir / 'validation_policies.md'}")

# 3. Summary stats
print("\n" + "=" * 60)
print("VALIDATION SAMPLE SUMMARY")
print("=" * 60)

score_dist = {'high': 0, 'medium': 0, 'low': 0}
income_dist = {}
for p in sample:
    if p['total_score'] >= 0.2:
        score_dist['high'] += 1
    elif p['total_score'] >= 0.1:
        score_dist['medium'] += 1
    else:
        score_dist['low'] += 1
    
    income = p.get('income_group', 'Unknown')
    income_dist[income] = income_dist.get(income, 0) + 1

print(f"\nTotal policies: {len(sample)}")
print(f"\nBy score level:")
print(f"  High (≥0.2):     {score_dist['high']}")
print(f"  Medium (0.1-0.2): {score_dist['medium']}")
print(f"  Low (<0.1):       {score_dist['low']}")

print(f"\nBy income group:")
for income, count in sorted(income_dist.items()):
    print(f"  {income}: {count}")

print(f"\nFiles created:")
print(f"  1. {output_dir / 'validation_coding_sheet.csv'}")
print(f"  2. {output_dir / 'validation_policies.md'}")
print(f"\nNext steps:")
print("  1. Distribute materials to 2+ independent coders")
print("  2. Each coder scores all 50 policies")
print("  3. Calculate inter-rater reliability (Cohen's kappa)")
print("  4. Resolve discrepancies through discussion")
print("  5. Compare consensus scores to automated scores")
