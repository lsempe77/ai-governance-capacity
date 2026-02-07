"""
Download content from ALL URLs discovered during enrichment.

This downloads from the source_urls field in the enriched data,
which contains 3,228 unique URLs across 1,950 policies.

Usage:
    python download_enriched_urls.py --enriched data/oecd/enriched/oecd_enriched_20260127_203406.json
"""

import json
import os
import re
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Set

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 30
MAX_FILE_SIZE = 50 * 1024 * 1024

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
}


def get_already_downloaded(content_dir: Path) -> Set[str]:
    """Get set of URLs already downloaded."""
    downloaded = set()
    
    # Check text files
    text_dir = content_dir / "text"
    if text_dir.exists():
        for f in text_dir.glob("*.txt"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    for line in fp.read().split('\n')[:5]:
                        if line.startswith('Source:'):
                            downloaded.add(line.replace('Source:', '').strip())
                            break
            except:
                pass
    
    # Check download results
    for results_file in content_dir.glob("download_results_*.json"):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for r in data.get('results', []):
                    if r.get('success'):
                        downloaded.add(r.get('source_url', ''))
        except:
            pass
    
    return downloaded


def sanitize_filename(title: str, url: str, max_length: int = 80) -> str:
    """Create safe filename from title and URL hash."""
    safe = re.sub(r'[<>:"/\\|?*]', '', title)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe[:max_length]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{safe}_{url_hash}"


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF."""
    return '.pdf' in url.lower() or 'pdf' in url.lower().split('/')[-1]


def download_pdf(session: requests.Session, url: str, output_dir: Path, filename: str) -> dict:
    """Download PDF file."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        pdf_path = output_dir / "pdfs" / f"{filename}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = 0
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE:
                        raise Exception("File too large")
                    f.write(chunk)
        
        return {
            'success': True,
            'type': 'pdf',
            'path': str(pdf_path),
            'size': total_size
        }
    except Exception as e:
        return {'success': False, 'error': str(e)[:100]}


def download_webpage(session: requests.Session, url: str, output_dir: Path, filename: str, title: str) -> dict:
    """Download and extract text from webpage."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if len(text) < 100:
            return {'success': False, 'error': 'No meaningful content'}
        
        # Save HTML
        html_path = output_dir / "html" / f"{filename}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Save text
        text_path = output_dir / "text" / f"{filename}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""Source: {url}
Title: {title}
Downloaded: {datetime.now().isoformat()}
{'='*80}

{text}
"""
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            'success': True,
            'type': 'webpage',
            'path': str(text_path),
            'size': len(text)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)[:100]}


def main():
    parser = argparse.ArgumentParser(description='Download content from enriched URLs')
    parser.add_argument('--enriched', required=True, help='Enriched JSON file')
    parser.add_argument('--output', default='./data/content', help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit downloads')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load enriched data
    logger.info(f"Loading enriched data from {args.enriched}")
    with open(args.enriched, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build list of (url, policy_title) pairs
    url_policy_map = []
    for policy in data['policies']:
        title = policy['title']
        for url in policy.get('source_urls', []):
            url_policy_map.append((url, title))
    
    logger.info(f"Total URL-policy pairs: {len(url_policy_map)}")
    
    # Get already downloaded
    downloaded = get_already_downloaded(output_dir)
    logger.info(f"Already downloaded: {len(downloaded)}")
    
    # Filter to new URLs
    to_download = [(url, title) for url, title in url_policy_map if url not in downloaded]
    # Deduplicate by URL
    seen = set()
    unique_to_download = []
    for url, title in to_download:
        if url not in seen:
            seen.add(url)
            unique_to_download.append((url, title))
    
    logger.info(f"New URLs to download: {len(unique_to_download)}")
    
    if args.limit:
        unique_to_download = unique_to_download[:args.limit]
        logger.info(f"Limited to: {args.limit}")
    
    # Download
    session = requests.Session()
    session.headers.update(HEADERS)
    
    results = []
    success_count = 0
    fail_count = 0
    pdf_count = 0
    web_count = 0
    
    logger.info("=" * 60)
    logger.info("DOWNLOADING ENRICHED URLs")
    logger.info("=" * 60)
    
    for i, (url, title) in enumerate(unique_to_download, 1):
        logger.info(f"[{i}/{len(unique_to_download)}] {title[:50]}...")
        
        filename = sanitize_filename(title, url)
        
        if is_pdf_url(url):
            result = download_pdf(session, url, output_dir, filename)
            if result['success']:
                pdf_count += 1
        else:
            result = download_webpage(session, url, output_dir, filename, title)
            if result['success']:
                web_count += 1
        
        result['url'] = url
        result['title'] = title
        results.append(result)
        
        if result['success']:
            success_count += 1
            size_str = f"{result['size']/1024:.1f} KB" if result.get('size') else ''
            logger.info(f"  ✓ Downloaded {result['type']} {size_str}")
        else:
            fail_count += 1
            logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown')[:50]}")
        
        # Save checkpoint every 100
        if i % 100 == 0:
            checkpoint_file = output_dir / f"enriched_download_checkpoint_{i}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({'results': results, 'success': success_count, 'failed': fail_count}, f)
            logger.info(f"Checkpoint saved: {i} URLs processed")
        
        time.sleep(REQUEST_DELAY)
    
    # Save final results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"enriched_download_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total': len(unique_to_download),
            'success': success_count,
            'failed': fail_count,
            'pdfs': pdf_count,
            'webpages': web_count,
            'results': results
        }, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total attempted: {len(unique_to_download)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"PDFs: {pdf_count}")
    logger.info(f"Webpages: {web_count}")


if __name__ == '__main__':
    main()
