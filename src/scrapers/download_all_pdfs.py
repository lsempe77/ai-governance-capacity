"""
Download source documents for all OECD.AI policy entries.

Strategy:
1. PHASE 1: Download from existing source_url field in corpus (642 entries, 29%)
2. PHASE 2: Use Claude to find source URLs for remaining entries (~1574)

Progress is saved incrementally so you can resume if interrupted.
"""

import asyncio
import json
import re
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import aiohttp

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "anthropic/claude-sonnet-4"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CORPUS_PATH = Path("data/corpus/corpus_master_20260127.json")
OUTPUT_DIR = Path("data/pdfs")
PROGRESS_FILE = Path("data/pdfs/download_progress.json")

TIMEOUT = 60
CONCURRENT_DOWNLOADS = 5

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
    return {
        "phase1_downloaded": {},  # From corpus source_url
        "phase2_downloaded": {},  # Found by Claude
        "failed": [],
        "no_source_in_corpus": [],
        "claude_no_url": [],
    }

def save_progress(progress):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def sanitize_filename(title, max_length=60):
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', '', title)
    safe = re.sub(r'\s+', '_', safe)
    return safe[:max_length]

def get_entry_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

async def download_file(session, url, output_path):
    """Download a file from URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
    }
    
    # Clean up Google redirect URLs
    if 'google.com/url' in url:
        import urllib.parse
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if 'url' in parsed:
            url = parsed['url'][0]
    
    try:
        async with session.get(url, headers=headers, 
                              timeout=aiohttp.ClientTimeout(total=TIMEOUT), 
                              allow_redirects=True) as resp:
            if resp.status == 200:
                content = await resp.read()
                
                # Determine extension from content-type
                ct = resp.headers.get('content-type', '').lower()
                if 'pdf' in ct or url.lower().endswith('.pdf'):
                    ext = '.pdf'
                elif 'html' in ct:
                    ext = '.html'
                elif 'word' in ct or 'docx' in ct:
                    ext = '.docx'
                else:
                    parsed_ext = Path(urlparse(url).path).suffix
                    ext = parsed_ext if parsed_ext else '.html'
                
                final_path = Path(str(output_path) + ext)
                with open(final_path, 'wb') as f:
                    f.write(content)
                
                return final_path, len(content)
            else:
                return None, resp.status
    except Exception as e:
        return None, str(e)

async def call_claude(session, title, jurisdiction, year, description):
    """Call Claude to find document URLs."""
    if not OPENROUTER_API_KEY:
        return []
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    prompt = f"""Find the official source URL for this AI policy document.

Title: {title}
Country/Jurisdiction: {jurisdiction}
Year: {year}
Description: {description[:400]}

Instructions:
1. Think about which government ministry or agency would publish this
2. Consider official government domains (.gov, .gov.xx, .gob.xx, .gouv.xx, europa.eu)
3. Check for PDF versions when possible
4. For international documents, check OECD, UNESCO, UN, EU sources

Return ONLY a JSON array of 1-3 most likely URLs. No explanation, just the array.
Example: ["https://ministry.gov.xx/policy.pdf", "https://gov.xx/ai-strategy"]
If truly unknown, return: []"""
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload,
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
            data = await resp.json()
            content = data['choices'][0]['message']['content']
            
            # Parse JSON
            if '```' in content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)
            
            urls = json.loads(content.strip())
            return urls if isinstance(urls, list) else []
    except:
        return []

async def process_phase1_batch(session, entries, progress, start_idx):
    """Process a batch of entries with existing source URLs."""
    tasks = []
    batch_entries = []
    
    for entry in entries:
        entry_id = get_entry_id(entry.get('url', ''))
        source_url = entry.get('source_url', '')
        
        # Skip if already processed
        if entry_id in progress['phase1_downloaded']:
            continue
        if entry_id in progress['failed']:
            continue
        if not source_url:
            if entry_id not in progress['no_source_in_corpus']:
                progress['no_source_in_corpus'].append(entry_id)
            continue
        
        title = entry.get('title', 'Unknown')
        safe_title = sanitize_filename(title)
        output_path = OUTPUT_DIR / f"{entry_id}_{safe_title}"
        
        tasks.append(download_file(session, source_url, output_path))
        batch_entries.append((entry_id, title, source_url))
    
    if not tasks:
        return 0
    
    # Execute downloads concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    downloaded = 0
    for (entry_id, title, source_url), result in zip(batch_entries, results):
        if isinstance(result, Exception):
            progress['failed'].append(entry_id)
            print(f"  ✗ {title[:40]}: Exception")
        elif result[0] is not None and result[1] > 500:
            file_path, size = result
            progress['phase1_downloaded'][entry_id] = {
                'title': title,
                'source_url': source_url,
                'file_path': str(file_path),
                'size': size
            }
            print(f"  ✓ {title[:40]}: {size:,} bytes")
            downloaded += 1
        else:
            if entry_id not in progress['failed']:
                progress['failed'].append(entry_id)
            print(f"  ✗ {title[:40]}: Failed ({result[1]})")
    
    return downloaded

async def main():
    print("=" * 70)
    print("OECD.AI DOCUMENT DOWNLOADER")
    print("=" * 70)
    
    # Load data
    corpus = load_corpus()
    entries = corpus['entries']
    progress = load_progress()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Separate entries
    entries_with_source = [e for e in entries if e.get('source_url')]
    entries_without_source = [e for e in entries if not e.get('source_url')]
    
    print(f"\nTotal entries: {len(entries)}")
    print(f"With source_url in corpus: {len(entries_with_source)}")
    print(f"Without source_url: {len(entries_without_source)}")
    print(f"Already downloaded (Phase 1): {len(progress['phase1_downloaded'])}")
    print(f"Already downloaded (Phase 2): {len(progress['phase2_downloaded'])}")
    
    # ==========================================================================
    # PHASE 1: Download from existing source URLs
    # ==========================================================================
    remaining_phase1 = [e for e in entries_with_source 
                        if get_entry_id(e.get('url', '')) not in progress['phase1_downloaded']
                        and get_entry_id(e.get('url', '')) not in progress['failed']]
    
    if remaining_phase1:
        print(f"\n{'='*70}")
        print(f"PHASE 1: Downloading {len(remaining_phase1)} entries with existing source URLs")
        print("="*70)
        
        async with aiohttp.ClientSession() as session:
            # Process in batches
            batch_size = CONCURRENT_DOWNLOADS
            for i in range(0, len(remaining_phase1), batch_size):
                batch = remaining_phase1[i:i+batch_size]
                print(f"\nBatch {i//batch_size + 1} ({i+1}-{min(i+batch_size, len(remaining_phase1))}):")
                
                await process_phase1_batch(session, batch, progress, i)
                
                # Save progress
                save_progress(progress)
        
        print(f"\nPhase 1 complete: {len(progress['phase1_downloaded'])} downloaded")
    else:
        print("\nPhase 1: All entries with source URLs already processed")
    
    # ==========================================================================
    # PHASE 2: Use Claude to find sources for remaining entries
    # ==========================================================================
    if not OPENROUTER_API_KEY:
        print("\n⚠ OPENROUTER_API_KEY not set - skipping Phase 2 (Claude search)")
    else:
        # Get entries that need Claude search
        remaining_phase2 = []
        for e in entries_without_source:
            entry_id = get_entry_id(e.get('url', ''))
            if (entry_id not in progress['phase2_downloaded'] and
                entry_id not in progress['claude_no_url'] and
                entry_id not in progress['failed']):
                remaining_phase2.append(e)
        
        if remaining_phase2:
            print(f"\n{'='*70}")
            print(f"PHASE 2: Using Claude to find {len(remaining_phase2)} missing sources")
            print("="*70)
            
            async with aiohttp.ClientSession() as session:
                for i, entry in enumerate(remaining_phase2):
                    title = entry.get('title', 'Unknown')
                    entry_id = get_entry_id(entry.get('url', ''))
                    jurisdiction = entry.get('jurisdiction', '')
                    year = entry.get('year', '')
                    description = entry.get('content', '')
                    
                    print(f"\n[{i+1}/{len(remaining_phase2)}] {title[:50]}...")
                    
                    # Ask Claude
                    urls = await call_claude(session, title, jurisdiction, year, description)
                    
                    if not urls:
                        progress['claude_no_url'].append(entry_id)
                        print(f"  → Claude found no URLs")
                        continue
                    
                    print(f"  Claude suggested: {urls[0][:60]}...")
                    
                    # Try to download
                    for doc_url in urls[:2]:
                        safe_title = sanitize_filename(title)
                        output_path = OUTPUT_DIR / f"{entry_id}_{safe_title}"
                        
                        file_path, result = await download_file(session, doc_url, output_path)
                        
                        if file_path and result > 500:
                            progress['phase2_downloaded'][entry_id] = {
                                'title': title,
                                'source_url': doc_url,
                                'file_path': str(file_path),
                                'size': result
                            }
                            print(f"  ✓ Downloaded: {result:,} bytes")
                            break
                    else:
                        progress['failed'].append(entry_id)
                        print(f"  ✗ All URLs failed")
                    
                    # Rate limit & save
                    await asyncio.sleep(0.5)
                    if (i + 1) % 20 == 0:
                        save_progress(progress)
                        print(f"\n  [Progress saved]")
            
            save_progress(progress)
        else:
            print("\nPhase 2: All remaining entries already processed")
    
    # ==========================================================================
    # FINAL REPORT
    # ==========================================================================
    save_progress(progress)
    
    total_downloaded = len(progress['phase1_downloaded']) + len(progress['phase2_downloaded'])
    
    print(f"\n{'='*70}")
    print("FINAL REPORT")
    print("="*70)
    print(f"Downloaded from corpus URLs (Phase 1): {len(progress['phase1_downloaded'])}")
    print(f"Found by Claude (Phase 2): {len(progress['phase2_downloaded'])}")
    print(f"Total documents downloaded: {total_downloaded}")
    print(f"Failed downloads: {len(progress['failed'])}")
    print(f"Claude found no URL: {len(progress['claude_no_url'])}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
