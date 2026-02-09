"""Quick inspection of Phase C depth output quality."""
import json, textwrap

with open("data/analysis/ethics_inventory/phase_c_depth.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        d = json.loads(line)
        dd = d["depth_data"]
        print(f"\n{'='*90}")
        print(f"POLICY: {d['title']}  |  {d['jurisdiction']}")
        print(f"{'='*90}")

        for category in ["values", "principles", "mechanisms"]:
            items = dd.get(category, [])
            print(f"\n  {category.upper()} ({len(items)}):")
            for item in items:
                name = item.get("name", "?")
                depth = item.get("depth", "?")
                verbatim = item.get("verbatim", "")
                loc = item.get("location_hint", "")
                vlen = len(verbatim)
                print(f"\n    [{depth:>9}] {name}  (loc: {loc})  [{vlen} chars]")
                # Show first 300 chars of verbatim
                wrapped = textwrap.wrap(verbatim[:400], 90)
                for ln in wrapped[:5]:
                    print(f"              {ln}")
                if vlen > 400:
                    print(f"              [... +{vlen - 400} chars]")
