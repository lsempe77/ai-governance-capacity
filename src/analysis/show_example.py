"""Show example of evidence-based scoring."""
import json

# Load results
results = json.load(open('data/analysis/rigorous_capacity/rigorous_capacity_by_policy.json', encoding='utf-8'))

# Find UK AI Regulation White Paper
uk_policy = None
for r in results:
    if 'AI Regulation' in r['title'] and r['jurisdiction'] == 'United Kingdom':
        uk_policy = r
        break

if uk_policy:
    print('=' * 80)
    print('EXAMPLE: Evidence-Based Scoring')
    print('=' * 80)
    print(f"Policy: {uk_policy['title']}")
    print(f"Jurisdiction: {uk_policy['jurisdiction']}")
    print(f"Total Score: {uk_policy['total_score']:.3f} (out of 1.0)")
    print()
    
    print('DIMENSION SCORES (0-4 scale):')
    print('-' * 40)
    print(f"  Clarity:        {uk_policy['clarity_score']} / 4")
    print(f"  Resources:      {uk_policy['resources_score']} / 4")
    print(f"  Authority:      {uk_policy['authority_score']} / 4")
    print(f"  Accountability: {uk_policy['accountability_score']} / 4")
    print(f"  Coherence:      {uk_policy['coherence_score']} / 4")
    print()
    
    print('EVIDENCE EXTRACTED:')
    print('-' * 40)
    for dim in ['clarity', 'resources', 'authority', 'accountability']:
        evidence = uk_policy.get(f'{dim}_evidence', [])
        if evidence:
            print(f"\n{dim.upper()}:")
            for e in evidence[:3]:
                if len(e) > 100:
                    print(f"  - {e[:100]}...")
                else:
                    print(f"  - {e}")
    
    # Show extracted entities
    print()
    print('EXTRACTED ENTITIES:')
    print('-' * 40)
    entities_by_type = {}
    for e in uk_policy['entities'][:30]:
        etype = e['entity_type']
        if etype not in entities_by_type:
            entities_by_type[etype] = []
        entities_by_type[etype].append(e['text'][:50])
    
    for etype, texts in entities_by_type.items():
        unique = list(set(texts))[:3]
        print(f"{etype}: {unique}")
