"""Minimal test: save raw API response to file for inspection."""
import json, os, requests, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'data' / 'analysis' / 'ethics_inventory' / 'debug_output.txt'
load_dotenv(ROOT / '.env')
API_KEY = os.getenv('OPENROUTER_API_KEY')

with open(ROOT / 'data/analysis/scores_ensemble.json', encoding='utf-8') as f:
    data = json.load(f)

test = None
for e in data['entries']:
    if e.get('analysis_ready') and e.get('ethics_score', 0) > 0.5 and e.get('word_count', 0) < 1000:
        test = e
        break

lines = [f"Test: {test['title']}", f"Jurisdiction: {test['jurisdiction']}", ""]

with open(ROOT / 'data/corpus/corpus_enriched.json', encoding='utf-8') as f:
    corpus = json.load(f)

text = ''
for ce in corpus['entries']:
    if ce.get('title') == test['title'] and ce.get('jurisdiction') == test['jurisdiction']:
        text = ce.get('text', '')[:2000]
        break

lines.append(f"Text length: {len(text)}")

try:
    resp = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': 'anthropic/claude-sonnet-4',
            'messages': [
                {'role': 'system', 'content': 'Return valid JSON only. No markdown.'},
                {'role': 'user', 'content': f'List the ethical values and principles in this AI policy. Title: {test["title"]}. Text: {text[:1500]}. JSON format: {{"values":["name1"], "principles":["name2"], "summary":"text"}}'},
            ],
            'temperature': 0.0,
            'max_tokens': 1000,
        },
        timeout=60,
    )
    lines.append(f"Status: {resp.status_code}")
    result = resp.json()
    content = result['choices'][0]['message']['content']
    lines.append(f"Content length: {len(content)}")
    lines.append("")
    lines.append("=== RAW CONTENT ===")
    lines.append(content)
    lines.append("=== END RAW ===")
    lines.append("")
    lines.append(f"repr: {repr(content[:500])}")
    
    try:
        parsed = json.loads(content)
        lines.append("PARSED OK!")
    except json.JSONDecodeError as e:
        lines.append(f"PARSE ERROR: {e}")
        if hasattr(e, 'pos') and e.pos < len(content):
            lines.append(f"Char at error pos {e.pos}: {repr(content[e.pos])}")
            lines.append(f"Context: ...{repr(content[max(0,e.pos-80):e.pos+80])}...")
except Exception as ex:
    lines.append(f"EXCEPTION: {ex}")

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# Also print to confirm completion
print(f"Debug output written to {OUT}")
