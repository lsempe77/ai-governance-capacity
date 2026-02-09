"""Inspect Phase C depth output quality."""
import json, statistics, random
from collections import Counter

infile = 'data/analysis/ethics_inventory/phase_c_depth.jsonl'

entries = []
corrupt = 0
with open(infile, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            entries.append(json.loads(line))
        except:
            corrupt += 1

print("=== PHASE C OUTPUT INSPECTION ===")
print(f"Total valid entries: {len(entries)}")
print(f"Corrupt lines: {corrupt}")

# Unique check
eids = [e['entry_id'] for e in entries]
print(f"Unique entry_ids: {len(set(eids))}")
if len(eids) != len(set(eids)):
    print("  WARNING: duplicates found!")

# Depth distribution across all items
depth_counts = Counter()
cat_counts = Counter()
cat_depth = {}
total_items = 0
empty_verbatim = 0
verbatim_lengths = []

for e in entries:
    dd = e.get('depth_data', {})
    for cat in ('values', 'principles', 'mechanisms'):
        items = dd.get(cat, [])
        cat_counts[cat] += len(items)
        for item in items:
            total_items += 1
            d = item.get('depth', 'unknown')
            depth_counts[d] += 1
            key = (cat, d)
            cat_depth[key] = cat_depth.get(key, 0) + 1
            v = item.get('verbatim', '')
            if not v or v.strip() == '':
                empty_verbatim += 1
            else:
                verbatim_lengths.append(len(v))

print(f"\n--- Items extracted ---")
print(f"Total items: {total_items}")
for cat in ('values', 'principles', 'mechanisms'):
    print(f"  {cat}: {cat_counts[cat]}")

print(f"\n--- Depth distribution (all items) ---")
for d, c in sorted(depth_counts.items(), key=lambda x: -x[1]):
    pct = c / total_items * 100
    bar = '#' * int(pct / 2)
    print(f"  {d:15s}: {c:5d} ({pct:5.1f}%) {bar}")

print(f"\n--- Depth by category ---")
for cat in ('values', 'principles', 'mechanisms'):
    print(f"  {cat}:")
    for d in ('word', 'phrase', 'sentence', 'paragraph', 'section'):
        c = cat_depth.get((cat, d), 0)
        t = cat_counts[cat]
        pct = c / t * 100 if t else 0
        print(f"    {d:12s}: {c:5d} ({pct:5.1f}%)")

print(f"\n--- Verbatim quality ---")
print(f"Empty verbatim: {empty_verbatim} / {total_items} ({empty_verbatim/total_items*100:.1f}%)")
if verbatim_lengths:
    print(f"Verbatim length (chars):")
    print(f"  min:    {min(verbatim_lengths)}")
    print(f"  p25:    {sorted(verbatim_lengths)[len(verbatim_lengths)//4]}")
    print(f"  median: {statistics.median(verbatim_lengths):.0f}")
    print(f"  mean:   {statistics.mean(verbatim_lengths):.0f}")
    print(f"  p75:    {sorted(verbatim_lengths)[3*len(verbatim_lengths)//4]}")
    print(f"  max:    {max(verbatim_lengths)}")

# Jurisdiction coverage
jurisdictions = Counter(e.get('jurisdiction', '?') for e in entries)
print(f"\n--- Jurisdiction coverage ---")
print(f"Unique jurisdictions: {len(jurisdictions)}")
top10 = jurisdictions.most_common(10)
for j, c in top10:
    print(f"  {j:30s}: {c}")

# Sample entries
print(f"\n--- Sample entries (3 random) ---")
random.seed(42)
samples = random.sample(entries, min(3, len(entries)))
for s in samples:
    print(f"\n  [{s.get('jurisdiction','?')}] {s.get('title','?')[:70]}")
    dd = s.get('depth_data', {})
    for cat in ('values', 'principles', 'mechanisms'):
        items = dd.get(cat, [])
        if items:
            it = items[0]
            v = it.get('verbatim', '')[:150]
            loc = it.get('location_hint', '')
            print(f"    {cat}[0]: {it.get('name','?')} | depth={it.get('depth','?')} | loc={loc}")
            print(f'      "{v}..."')

# Token/cost summary
total_tokens = sum(e.get('usage', {}).get('total_tokens', 0) for e in entries)
prompt_tokens = sum(e.get('usage', {}).get('prompt_tokens', 0) for e in entries)
completion_tokens = sum(e.get('usage', {}).get('completion_tokens', 0) for e in entries)
cost = prompt_tokens * 5 / 1e6 + completion_tokens * 25 / 1e6
print(f"\n--- Token / cost summary ---")
print(f"Prompt tokens:     {prompt_tokens:>12,}")
print(f"Completion tokens: {completion_tokens:>12,}")
print(f"Total tokens:      {total_tokens:>12,}")
print(f"Est. cost:         ${cost:>10.2f}")
