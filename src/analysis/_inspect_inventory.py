"""Quick inspection of phase_b_inventory.jsonl evidence quality."""
import json, textwrap

with open("data/analysis/ethics_inventory/phase_b_inventory.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        d = json.loads(line)
        inv = d["inventory"]
        print(f"\n{'='*80}")
        print(f"POLICY: {d['title']}  |  {d['jurisdiction']}")
        print(f"{'='*80}")

        print(f"\n  VALUES ({len(inv['values'])}):")
        for v in inv["values"][:4]:
            ev = v.get("evidence", "")
            print(f"    [{v['strength']:>8}] {v['name']}")
            for ln in textwrap.wrap(ev, 100):
                print(f"             {ln}")

        print(f"\n  PRINCIPLES ({len(inv['principles'])}):")
        for p in inv["principles"][:4]:
            ev = p.get("evidence", "")
            print(f"    [{p['strength']:>8}] {p['name']}")
            for ln in textwrap.wrap(ev, 100):
                print(f"             {ln}")

        print(f"\n  MECHANISMS ({len(inv['mechanisms'])}):")
        for m in inv["mechanisms"][:4]:
            ev = m.get("evidence", "")
            print(f"    [{m['strength']:>8}] {m['name']}")
            for ln in textwrap.wrap(ev, 100):
                print(f"             {ln}")
