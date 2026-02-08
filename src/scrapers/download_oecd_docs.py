"""
Download source documents from OECD.AI policy pages.
Uses Playwright to render JavaScript and find source document links.

Key patterns found:
1. PDF links: https://oecd-ai.case-api.buddyweb.fr/storage//policy-initiatives/...
2. External source URLs in the page
3. Government website links (.gov, .gv, .europa.eu)
"""

import asyncio
import json
import re
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

# =============================================================================
# CONFIGURATION
# =============================================================================

CORPUS_PATH = Path("data/corpus/corpus_master_20260127.json")
OUTPUT_DIR = Path("data/pdfs")
PROGRESS_FILE = Path("data/pdfs/download_progress.json")

DELAY_BETWEEN_REQUESTS = 1.0
TIMEOUT = 60000

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
    return {"downloaded": {}, "failed": [], "no_source": []}

def save_progress(progress):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

def sanitize_filename(title, max_length=80):
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', '', title)
    safe = re.sub(r'\s+', '_', safe)
    return safe[:max_length]

def get_entry_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

async def find_source_links(page, url):
    """Find source document links on an OECD.AI page."""
    try:
        await page.goto(url, wait_until='networkidle', timeout=TIMEOUT)
        await page.wait_for_timeout(2000)
        
        content = await page.content()
        source_links = []
        
        # Pattern 1: OECD.AI storage URLs (PDFs hosted by OECD)
        storage_pattern = r'https://oecd-ai\.case-api\.buddyweb\.fr/storage[^"\'>\s]+'
        storage_matches = re.findall(storage_pattern, content)
        for match in storage_matches:
            source_links.append({
                'url': match,
                'type': 'oecd_storage',
                'priority': 1
            })
        
        # Pattern 2: Direct PDF links
        pdf_pattern = r'href="(https?://[^"]+\.pdf)"'
        pdf_matches = re.findall(pdf_pattern, content, re.IGNORECASE)
        for match in pdf_matches:
            if 'oecd.ai' not in match:
                source_links.append({
                    'url': match,
                    'type': 'pdf',
                    'priority': 1
                })
        
        # Pattern 3: Government and official source links via JavaScript
        external_links = await page.evaluate('''() => {
            const results = [];
            const allLinks = document.querySelectorAll('a[href]');
            
            for (const link of allLinks) {
                const href = link.href;
                const text = link.innerText.trim();
                
                // Skip oecd.ai internal
                if (href.includes('oecd.ai') && !href.includes('case-api')) continue;
                if (href.startsWith('#') || href.startsWith('javascript')) continue;
                if (!href.startsWith('http')) continue;
                
                // Prioritize government and official sources
                const isGov = href.match(/\.(gov|gv|gc|gouv|europa\.eu|unesco|un\.org)/i);
                const isPdf = href.toLowerCase().includes('.pdf');
                const isDoc = href.match(/\.(doc|docx|pdf|html)/i) || text.toLowerCase().includes('document');
                
                if (isGov || isPdf || isDoc) {
                    results.push({
                        url: href,
                        text: text.substring(0, 100),
                        isGov: !!isGov,
                        isPdf: isPdf
                    });
                }
            }
            
            return results;
        }''')
        
        for link in external_links:
            priority = 2
            if link.get('isPdf'):
                priority = 1
            if link.get('isGov'):
                priority = 1
            
            source_links.append({
                'url': link['url'],
                'type': 'external',
                'text': link.get('text', ''),
                'priority': priority
            })
        
        # Sort by priority
        source_links.sort(key=lambda x: x['priority'])
        
        # Remove duplicates
        seen = set()
        unique = []
        for link in source_links:
            if link['url'] not in seen:
                seen.add(link['url'])
                unique.append(link)
        
        return unique
        
    except Exception as e:
        print(f"  Error finding links: {e}")
        return []

async def download_document(url, output_path):
    """Download a document from URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    async with httpx.AsyncClient(headers=headers, timeout=60, follow_redirects=True) as client:
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

async def process_batch(entries, progress, start_idx, batch_size):
    """Process a batch of entries."""
    stats = {"success": 0, "no_source": 0, "failed": 0, "skipped": 0}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        end_idx = min(start_idx + batch_size, len(entries))
        
        for i in range(start_idx, end_idx):
            entry = entries[i]
            title = entry.get('title', 'Unknown')[:50]
            url = entry.get('url', '')
            entry_id = get_entry_id(url)
            
            # Skip already processed
            if entry_id in progress['downloaded'] or entry_id in progress['failed'] or entry_id in progress['no_source']:
                stats['skipped'] += 1
                continue
            
            print(f"[{i+1}/{len(entries)}] {title}...")
            
            try:
                # Find source links
                links = await find_source_links(page, url)
                
                if not links:
                    progress['no_source'].append(entry_id)
                    stats['no_source'] += 1
                    print(f"  - No source links found")
                    continue
                
                # Try to download
                downloaded = False
                for link in links[:5]:
                    try:
                        safe_title = sanitize_filename(title)
                        output_path = OUTPUT_DIR / f"{entry_id}_{safe_title}"
                        
                        file_path, size = await download_document(link['url'], output_path)
                        
                        # Only count as success if file is substantial (>1KB)
                        if size > 1024:
                            progress['downloaded'][entry_id] = {
                                'title': entry.get('title'),
                                'source_url': link['url'],
                                'file_path': str(file_path),
                                'size': size,
                                'type': link.get('type', 'unknown')
                            }
                            print(f"  ✓ {file_path.name} ({size:,} bytes)")
                            downloaded = True
                            stats['success'] += 1
                            break
                        else:
                            file_path.unlink()  # Remove tiny files
                            
                    except Exception as e:
                        # Continue to next link
                        pass
                
                if not downloaded:
                    progress['failed'].append(entry_id)
                    stats['failed'] += 1
                    print(f"  ✗ Download failed")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                progress['failed'].append(entry_id)
                stats['failed'] += 1
            
            # Save progress periodically
            if (i + 1) % 10 == 0:
                save_progress(progress)
            
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
        
        await browser.close()
    
    return stats

async def main():
    print("=" * 70)
    print("OECD.AI DOCUMENT DOWNLOADER")
    print("=" * 70)
    
    corpus = load_corpus()
    entries = corpus['entries']
    print(f"Total entries: {len(entries)}")
    
    progress = load_progress()
    done = len(progress['downloaded']) + len(progress['failed']) + len(progress['no_source'])
    print(f"Already processed: {done}")
    print(f"  Downloaded: {len(progress['downloaded'])}")
    print(f"  No source: {len(progress['no_source'])}")
    print(f"  Failed: {len(progress['failed'])}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process in batches of 100
    batch_size = 100
    start_idx = 0
    
    total_stats = {"success": 0, "no_source": 0, "failed": 0, "skipped": 0}
    
    while start_idx < len(entries):
        print(f"\n--- Batch {start_idx//batch_size + 1}: entries {start_idx+1}-{min(start_idx+batch_size, len(entries))} ---")
        
        stats = await process_batch(entries, progress, start_idx, batch_size)
        
        for k, v in stats.items():
            total_stats[k] += v
        
        save_progress(progress)
        print(f"Batch complete. Downloaded: {len(progress['downloaded'])}")
        
        start_idx += batch_size
        
        if start_idx < len(entries):
            print("Pausing 3 seconds...")
            await asyncio.sleep(3)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Downloaded: {len(progress['downloaded'])}")
    print(f"No source found: {len(progress['no_source'])}")
    print(f"Failed: {len(progress['failed'])}")
    print(f"\nFiles in: {OUTPUT_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
