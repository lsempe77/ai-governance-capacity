import json

data = json.load(open('data/oecd/oecd_policies_20260126_201311.json', encoding='utf-8'))
no_source = [p for p in data['policies'] if not p.get('source_url')]

print('=== 3 EXAMPLES OF POLICIES WITHOUT SOURCE URLs ===\n')

for i, p in enumerate(no_source[:3], 1):
    print(f'EXAMPLE {i}:')
    print(f'  Title: {p["title"]}')
    print(f'  Jurisdiction: {p.get("jurisdiction")}')
    print(f'  Year: {p.get("start_year")}')
    print(f'  OECD URL: {p["url"]}')
    print(f'  Description ({len(p.get("description",""))} chars):')
    desc = p.get("description", "")
    print(f'    "{desc}"')
    print(f'  Policy Areas: {p.get("policy_areas", [])}')
    print('-' * 70)
    print()
