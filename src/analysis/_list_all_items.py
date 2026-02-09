"""List all unique values, principles, and mechanisms from Phase C."""
import json
from collections import Counter

entries = []
with open('data/analysis/ethics_inventory/phase_c_depth.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

cats = {'values': Counter(), 'principles': Counter(), 'mechanisms': Counter()}
depth_scores = {'values': {}, 'principles': {}, 'mechanisms': {}}
depth_weights = {'word': 1, 'phrase': 2, 'sentence': 3, 'paragraph': 4, 'section': 5, 'not_found': 0}

for e in entries:
    dd = e.get('depth_data', {})
    for cat in cats:
        for item in dd.get(cat, []):
            name = item.get('name', '').strip()
            depth = item.get('depth', 'unknown').lower().strip()
            if name:
                cats[cat][name] += 1
                if name not in depth_scores[cat]:
                    depth_scores[cat][name] = []
                depth_scores[cat][name].append(depth_weights.get(depth, 0))

depth_label = ['?', 'word', 'phrase', 'sentence', 'paragraph', 'section']

for cat in ('values', 'principles', 'mechanisms'):
    n_unique = len(cats[cat])
    n_total = sum(cats[cat].values())
    print()
    print("=" * 78)
    print(f"  {cat.upper()} ({n_unique} unique, {n_total:,} total mentions)")
    print("=" * 78)
    header = f"  {'#':>3}  {'Name':<45} {'Count':>6}  {'Avg Depth':>10}"
    print(header)
    print(f"  {'---':>3}  {'-'*45}  {'------':>6}  {'----------':>10}")
    for i, (name, count) in enumerate(cats[cat].most_common(), 1):
        scores = depth_scores[cat][name]
        avg = sum(scores) / len(scores) if scores else 0
        r = round(avg)
        nearest = depth_label[r] if 0 < r <= 5 else '?'
        print(f"  {i:3d}  {name:<45} {count:6,}  {avg:5.2f} ({nearest})")
