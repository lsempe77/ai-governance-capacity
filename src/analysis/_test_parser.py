"""Test the fixed JSON parser against the actual failed response."""
import re, json, sys
sys.path.insert(0, ".")

with open("data/analysis/ethics_inventory/debug_depth_failed_response.txt", encoding="utf-8") as f:
    text = f.read()

start = text.index("=== RAW ===") + len("=== RAW ===")
end = text.index("=== END ===")
content = text[start:end].strip()

from src.analysis.ethics_depth import _parse_json_robust
try:
    result = _parse_json_robust(content)
    print("SUCCESS!")
    print("Keys:", list(result.keys()))
except Exception as e:
    print(f"FAILED: {e}")
    # Manual debug: strip fences, escape newlines, try parse
    import re
    c = content.strip()
    c = re.sub(r'^```\w*\s*', '', c)
    c = re.sub(r'\s*```\s*$', '', c)
    c = c.strip()
    print(f"After fence strip, starts with: {repr(c[:50])}")
    print(f"After fence strip, ends with: {repr(c[-50:])}")
    # Try direct parse
    try:
        json.loads(c)
        print("Direct parse after fence strip: OK")
    except json.JSONDecodeError as e2:
        print(f"Direct parse failed: {e2}")
        # Check for raw newlines
        in_str = False
        raw_nl = 0
        for i, ch in enumerate(c):
            if ch == '"' and (i == 0 or c[i-1] != '\\'):
                in_str = not in_str
            elif in_str and ch == '\n':
                raw_nl += 1
        print(f"Raw newlines inside strings: {raw_nl}")

        # Try the escape function manually
        def _escape_newlines_in_strings(text):
            result = []
            in_string = False
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == '"' and (i == 0 or text[i-1] != '\\'):
                    in_string = not in_string
                    result.append(ch)
                elif in_string and ch == '\n':
                    result.append('\\n')
                elif in_string and ch == '\r':
                    pass
                elif in_string and ch == '\t':
                    result.append('\\t')
                else:
                    result.append(ch)
                i += 1
            return ''.join(result)

        fixed = _escape_newlines_in_strings(c)
        try:
            json.loads(fixed)
            print("After newline escape: OK!")
        except json.JSONDecodeError as e3:
            print(f"After newline escape: {e3}")
            # Show context around error
            pos = e3.pos
            print(f"Context: ...{repr(fixed[max(0,pos-80):pos+80])}...")
