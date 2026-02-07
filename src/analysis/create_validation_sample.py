"""Create validation sample with evidence for manual verification."""
import json
import random

# Load rigorous results
results = json.load(open('data/analysis/rigorous_capacity/rigorous_capacity_by_policy.json', encoding='utf-8'))

# Create validation sample: stratified by score (high, medium, low) and income group
high_score = [r for r in results if r['total_score'] >= 0.2]
medium_score = [r for r in results if 0.1 <= r['total_score'] < 0.2]
low_score = [r for r in results if r['total_score'] < 0.1]

print('Score distribution:')
print(f'  High (>=0.2): {len(high_score)} policies')
print(f'  Medium (0.1-0.2): {len(medium_score)} policies')
print(f'  Low (<0.1): {len(low_score)} policies')

# Sample for validation
random.seed(42)
sample = []
sample.extend(random.sample(high_score, min(15, len(high_score))))
sample.extend(random.sample(medium_score, min(15, len(medium_score))))
sample.extend(random.sample(low_score, min(20, len(low_score))))

print(f'\nValidation sample: {len(sample)} policies')
print('=' * 80)

# Show examples with evidence
for i, p in enumerate(sample[:10], 1):
    title = p['title'][:60]
    print(f'\n{i}. {title}...')
    print(f'   Jurisdiction: {p["jurisdiction"]} | Score: {p["total_score"]:.3f}')
    print(f'   Dimensions: Clarity={p["clarity_score"]}, Resources={p["resources_score"]}, Authority={p["authority_score"]}, Account.={p["accountability_score"]}')
    
    # Show evidence
    all_evidence = (p['clarity_evidence'][:2] + p['resources_evidence'][:2] + 
                   p['authority_evidence'][:2] + p['accountability_evidence'][:2])
    if all_evidence:
        print('   Evidence:')
        for e in all_evidence[:4]:
            if len(e) > 80:
                print(f'     - {e[:80]}...')
            else:
                print(f'     - {e}')

# Save validation sample
with open('data/analysis/rigorous_capacity/validation_sample.json', 'w', encoding='utf-8') as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)

print(f'\n\nFull validation sample saved to data/analysis/rigorous_capacity/validation_sample.json')
