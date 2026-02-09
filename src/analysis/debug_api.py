"""Quick debug: capture raw API response to diagnose JSON parse failures."""
import json, os, requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / '.env')
API_KEY = os.getenv('OPENROUTER_API_KEY')

with open(ROOT / 'data/analysis/scores_ensemble.json', encoding='utf-8') as f:
    data = json.load(f)

# Pick a simple policy
test = None
for e in data['entries']:
    if e.get('analysis_ready') and e.get('ethics_score', 0) > 0.5 and e.get('word_count', 0) < 1500:
        test = e
        break

print(f"Test: {test['title'][:60]}")

with open(ROOT / 'data/corpus/corpus_enriched.json', encoding='utf-8') as f:
    corpus = json.load(f)

text = ''
for ce in corpus['entries']:
    if ce.get('title') == test['title'] and ce.get('jurisdiction') == test['jurisdiction']:
        text = ce.get('text', '')[:3000]
        break

print(f"Text length: {len(text)}")

resp = requests.post(
    'https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={
        'model': 'anthropic/claude-sonnet-4',
        'messages': [
            {'role': 'system', 'content': 'You extract ethical content from AI policy documents. Always respond with valid JSON only. Never use markdown code fences. Escape all special characters in JSON strings properly.'},
            {'role': 'user', 'content': f"""Extract values, principles, mechanisms from this AI policy.

Title: {test['title']}
Jurisdiction: {test['jurisdiction']}

Text:
{text}

Return ONLY this JSON structure (no markdown):
{{"values": [{{"name": "string", "strength": "explicit", "evidence": "short quote"}}], "principles": [{{"name": "string", "strength": "explicit", "evidence": "short quote"}}], "mechanisms": [{{"name": "string", "strength": "proposed", "evidence": "short quote"}}], "referenced_frameworks": [], "summary": "two sentences"}}"""},
        ],
        'temperature': 0.1,
        'max_tokens': 2000,
    },
    timeout=120,
)

print(f"Status: {resp.status_code}")
result = resp.json()
content = result['choices'][0]['message']['content']

# Save raw response
out = ROOT / 'data' / 'analysis' / 'ethics_inventory' / 'debug_raw_response.txt'
with open(out, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Saved raw response to {out}")
print(f"Length: {len(content)} chars")
print()
print("=== RAW CONTENT ===")
print(content)
print("=== END ===")
print()

try:
    parsed = json.loads(content)
    print("PARSED OK!")
except json.JSONDecodeError as e:
    print(f"PARSE ERROR: {e}")
    print(f"Error at position {e.pos}: char = '{content[e.pos] if e.pos < len(content) else 'EOF'}'")
    print(f"Context: ...{content[max(0,e.pos-50):e.pos]}<<<HERE>>>{content[e.pos:e.pos+50]}...")
