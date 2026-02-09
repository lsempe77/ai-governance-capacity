"""Quick test of the fixed _parse_json_robust against the debug file."""
import sys, json
sys.path.insert(0, "src/analysis")
from ethics_depth import _parse_json_robust

with open("data/analysis/ethics_inventory/debug_depth_failed_response.txt", "r", encoding="utf-8") as f:
    raw = f.read()

try:
    result = _parse_json_robust(raw)
    print("SUCCESS!")
    print(f"Keys: {list(result.keys())}")
    for cat in ("values", "principles", "mechanisms"):
        items = result.get(cat, [])
        print(f"  {cat}: {len(items)} items")
        if items:
            first = items[0]
            name = first.get("name", "?")
            depth = first.get("depth", "?")
            verb = first.get("verbatim", "")[:80]
            print(f"    first: name={name}, depth={depth}")
            print(f"    verbatim: {verb}...")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
