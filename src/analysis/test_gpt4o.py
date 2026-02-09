"""Single API call test with GPT-4o + JSON mode"""
import json, os, requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / '.env')
API_KEY = os.getenv('OPENROUTER_API_KEY')
OUT = ROOT / 'data' / 'analysis' / 'ethics_inventory' / 'test_gpt4o.json'

resp = requests.post(
    'https://openrouter.ai/api/v1/chat/completions',
    headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    },
    json={
        'model': 'openai/gpt-4o',
        'messages': [
            {'role': 'system', 'content': 'You extract ethical content from AI policies. Always return valid JSON.'},
            {'role': 'user', 'content': 'Extract values, principles, mechanisms from this: "The EU AI Act establishes a risk-based approach to AI regulation, protecting fundamental rights including privacy, non-discrimination, and human dignity. It requires impact assessments, creates a European AI Board, and bans certain high-risk AI uses." Return JSON: {"values":["name"],"principles":["name"],"mechanisms":["name"]}'},
        ],
        'temperature': 0.0,
        'max_tokens': 500,
        'response_format': {'type': 'json_object'},
    },
    timeout=60,
)

result = resp.json()
content = result['choices'][0]['message']['content']
parsed = json.loads(content)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump({'status': 'ok', 'parsed': parsed, 'raw': content}, f, indent=2)

print(f"OK! Written to {OUT}")
print(json.dumps(parsed, indent=2))
