import requests, os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")
r = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"})
models = r.json()["data"]
claude = [m for m in models if "claude" in m["id"].lower() and ("opus" in m["id"].lower() or "sonnet" in m["id"].lower())]
for m in sorted(claude, key=lambda x: x["id"]):
    mid = m["id"]
    ctx = m.get("context_length", 0)
    pi = float(m.get("pricing", {}).get("prompt", 0)) * 1e6
    po = float(m.get("pricing", {}).get("completion", 0)) * 1e6
    print(f"{mid:55s}  ctx={ctx:>7,}  in=${pi:.2f}/M  out=${po:.2f}/M")
