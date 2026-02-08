"""
Use Claude to find source PDFs for OECD.AI policy entries.
For entries without direct source links, Claude will reason about where to find the document.
"""

import asyncio
import json
import os
import re
import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

import httpx
from playwright.async_api import async_playwright

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "anthropic/claude-sonnet-4"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

CORPUS_PATH = Path("data/corpus/corpus_master_20260127.json")
OUTPUT_DIR = Path("data/pdfs")
PROGRESS_FILE = Path("data/pdfs/download_progress.json")

DELAY_BETWEEN_REQUESTS = 1.5
TIMEOUT = 60

# =============================================================================
# PROMPTS
# =============================================================================

FIND_PDF_PROMPT = """You are a research assistant helping to locate official government and policy documents.

Given the following AI policy/initiative metadata, provide the most likely URLs where the original source document (PDF, official webpage, or government publication) can be found.

## Policy Information
- **Title**: {title}
- **Jurisdiction**: {jurisdiction}
- **Year**: {year}
- **Description**: {description}

## Instructions
1. Based on the jurisdiction and title, identify the most likely government agency or organization that published this
2. Construct the most probable URLs where this document would be hosted
3. Consider official government domains (.gov, .gov.uk, .europa.eu, .gc.ca, etc.)
4. Consider international organization domains (oecd.org, unesco.org, un.org, etc.)
5. For EU countries, check EUR-Lex or national legislation databases
6. For strategies/frameworks, check relevant ministry websites

## Response Format
Return ONLY a JSON array of up to 5 most likely URLs, ordered by probability:

```json
[
  "https://most-likely-url.gov/document.pdf",
  "https://second-most-likely.gov/page",
  "https://third-option.org/doc"
]
```

If you cannot determine any likely URLs, return an empty array: []

Respond with ONLY the JSON array, no other text.
"""

# =============================================================================
# HELPERS
# =============================================================================

def load_corpus():
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"downloaded": {}, "failed": [], "no_source": [], "llm_found": {}}

def save_progress(progress):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def sanitize_filename(title, max_length=80):
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', '', title)
    safe = re.sub(r'\s+', '_', safe)
    return safe[:max_length]

def get_entry_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

async def call_claude(prompt):
    """Call Claude via OpenRouter to find PDF URLs."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-governance-observatory",
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        content = data['choices'][0]['message']['content']
        
        # Parse JSON from response
        try:
            # Remove markdown code blocks if present
            if '```' in content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)
            
            urls = json.loads(content.strip())
            return urls if isinstance(urls, list) else []
        except json.JSONDecodeError:
            return []

async def download_document(url, output_path):
    """Download a document from URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
    }
    
    async with httpx.AsyncClient(headers=headers, timeout=TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            ext = '.pdf'
        elif 'html' in content_type:
            ext = '.html'
        elif 'word' in content_type or 'docx' in content_type:
            ext = '.docx'
        else:
            parsed_ext = Path(urlparse(url).path).suffix
            ext = parsed_ext if parsed_ext else '.bin'
        
        final_path = Path(str(output_path) + ext)
        
        with open(final_path, 'wb') as f:
            f.write(response.content)
        
        return final_path, len(response.content)

async def try_download_with_playwright(url, output_path):
    """Try to download using Playwright for JavaScript-heavy sites."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Navigate to the page
            response = await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Check if it's a PDF
            content_type = response.headers.get('content-type', '')
            
            if 'pdf' in content_type:
                # Direct PDF download
                content = await response.body()
                final_path = Path(str(output_path) + '.pdf')
                with open(final_path, 'wb') as f:
                    f.write(content)
                await browser.close()
                return final_path, len(content)
            
            # Otherwise, look for PDF links on the page
            pdf_links = await page.evaluate('''() => {
                const links = document.querySelectorAll('a[href*=".pdf"], a[href*="download"], a[href*="document"]');
                return Array.from(links).map(l => l.href).filter(h => h.includes('.pdf'));
            }''')
            
            await browser.close()
            
            if pdf_links:
                # Try the first PDF link
                return await download_document(pdf_links[0], output_path)
            
            return None, 0
            
    except Exception as e:
        return None, 0

async def process_entry(entry, progress):
    """Process a single entry - use Claude to find source, then download."""
    title = entry.get('title', 'Unknown')
    url = entry.get('url', '')
    jurisdiction = entry.get('jurisdiction', 'Unknown')
    year = entry.get('year', 'Unknown')
    description = entry.get('content', '')[:500]  # First 500 chars
    
    entry_id = get_entry_id(url)
    
    # Skip if already processed
    if (entry_id in progress['downloaded'] or 
        entry_id in progress.get('llm_found', {}) or
        entry_id in progress['failed']):
        return "skipped", None
    
    print(f"\n  Asking Claude to find: {title[:50]}...")
    
    # Ask Claude where to find this document
    prompt = FIND_PDF_PROMPT.format(
        title=title,
        jurisdiction=jurisdiction,
        year=year,
        description=description
    )
    
    try:
        suggested_urls = await call_claude(prompt)
        
        if not suggested_urls:
            print(f"    Claude couldn't suggest URLs")
            return "no_urls", None
        
        print(f"    Claude suggested {len(suggested_urls)} URLs")
        
        # Try each suggested URL
        for i, doc_url in enumerate(suggested_urls[:5]):
            print(f"    Trying [{i+1}]: {doc_url[:60]}...")
            
            try:
                safe_title = sanitize_filename(title)
                output_path = OUTPUT_DIR / f"{entry_id}_{safe_title}"
                
                # Try direct download first
                try:
                    file_path, size = await download_document(doc_url, output_path)
                    
                    if size > 1024:  # More than 1KB
                        progress['llm_found'][entry_id] = {
                            'title': title,
                            'source_url': doc_url,
                            'file_path': str(file_path),
                            'size': size,
                            'method': 'claude_direct'
                        }
                        print(f"    ✓ Downloaded: {file_path.name} ({size:,} bytes)")
                        return "success", file_path
                        
                except Exception:
                    # Try with Playwright
                    file_path, size = await try_download_with_playwright(doc_url, output_path)
                    
                    if file_path and size > 1024:
                        progress['llm_found'][entry_id] = {
                            'title': title,
                            'source_url': doc_url,
                            'file_path': str(file_path),
                            'size': size,
                            'method': 'claude_playwright'
                        }
                        print(f"    ✓ Downloaded via browser: {file_path.name} ({size:,} bytes)")
                        return "success", file_path
                        
            except Exception as e:
                print(f"    ✗ Failed: {str(e)[:50]}")
                continue
        
        progress['failed'].append(entry_id)
        return "failed", None
        
    except Exception as e:
        print(f"    ✗ Claude error: {e}")
        return "error", None

async def main():
    print("=" * 70)
    print("CLAUDE-ASSISTED PDF FINDER")
    print("Using Claude to locate source documents")
    print("=" * 70)
    
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found")
        return
    
    # Load corpus and progress
    corpus = load_corpus()
    entries = corpus['entries']
    progress = load_progress()
    
    # Get entries that don't have sources yet
    entries_needing_search = []
    for entry in entries:
        entry_id = get_entry_id(entry.get('url', ''))
        if (entry_id not in progress['downloaded'] and 
            entry_id not in progress.get('llm_found', {}) and
            entry_id in progress.get('no_source', [])):
            entries_needing_search.append(entry)
    
    print(f"Total entries: {len(entries)}")
    print(f"Already downloaded: {len(progress['downloaded'])}")
    print(f"LLM-found: {len(progress.get('llm_found', {}))}")
    print(f"Entries needing LLM search: {len(entries_needing_search)}")
    
    if not entries_needing_search:
        print("\nNo entries need LLM search. Run download_oecd_docs.py first.")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process entries
    stats = {"success": 0, "failed": 0, "no_urls": 0, "skipped": 0, "error": 0}
    
    for i, entry in enumerate(entries_needing_search[:500]):  # Limit to 500 per run
        print(f"\n[{i+1}/{len(entries_needing_search)}] {entry.get('title', 'Unknown')[:50]}...")
        
        status, _ = await process_entry(entry, progress)
        stats[status] = stats.get(status, 0) + 1
        
        # Save progress every 10 entries
        if (i + 1) % 10 == 0:
            save_progress(progress)
            print(f"\n  [Progress saved: {len(progress.get('llm_found', {}))} found by Claude]")
        
        # Rate limiting for API
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Final save
    save_progress(progress)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"No URLs suggested: {stats['no_urls']}")
    print(f"Errors: {stats['error']}")
    print(f"\nTotal downloaded by Claude: {len(progress.get('llm_found', {}))}")
    print(f"Files in: {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
