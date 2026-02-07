"""
OECD Policy Content Downloader
Downloads PDFs and web page content for text analysis.

Usage:
    python content_downloader.py --input data/oecd/oecd_policies_20260126_201311.json
    python content_downloader.py --input data/oecd/oecd_policies_20260126_201311.json --pdfs-only
    python content_downloader.py --input data/oecd/oecd_policies_20260126_201311.json --limit 50
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
from dataclasses import dataclass, asdict
from typing import Optional, List
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REQUEST_DELAY = 2.0  # Seconds between requests (polite crawling)
REQUEST_TIMEOUT = 30  # Seconds
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB max per file

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}


@dataclass
class DownloadResult:
    """Result of a content download attempt."""
    policy_title: str
    source_url: str
    success: bool
    content_type: str  # 'pdf', 'html', 'text', 'error'
    file_path: Optional[str]
    file_size: Optional[int]
    error_message: Optional[str]
    downloaded_at: str


class ContentDownloader:
    """Downloads and saves policy document content."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.pdfs_dir = self.output_dir / "pdfs"
        self.html_dir = self.output_dir / "html"
        self.text_dir = self.output_dir / "text"
        
        # Create directories
        for d in [self.pdfs_dir, self.html_dir, self.text_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.results: List[DownloadResult] = []
    
    def _sanitize_filename(self, title: str, max_length: int = 80) -> str:
        """Create a safe filename from policy title."""
        # Remove invalid characters
        safe = re.sub(r'[<>:"/\\|?*]', '', title)
        safe = re.sub(r'\s+', '_', safe)
        safe = safe[:max_length]
        return safe
    
    def _get_url_hash(self, url: str) -> str:
        """Generate short hash for URL to ensure unique filenames."""
        return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def _resolve_google_redirect(self, url: str) -> str:
        """Extract actual URL from Google redirect links."""
        if 'google.com/url' in url:
            try:
                from urllib.parse import parse_qs, urlparse
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                if 'url' in params:
                    return params['url'][0]
            except:
                pass
        return url
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF."""
        url_lower = url.lower()
        return '.pdf' in url_lower or 'application/pdf' in url_lower
    
    def download_pdf(self, url: str, title: str) -> DownloadResult:
        """Download a PDF file."""
        url = self._resolve_google_redirect(url)
        filename = f"{self._sanitize_filename(title)}_{self._get_url_hash(url)}.pdf"
        filepath = self.pdfs_dir / filename
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                # Not actually a PDF, save as HTML instead
                return self.download_webpage(url, title)
            
            # Check size
            content_length = int(response.headers.get('Content-Length', 0))
            if content_length > MAX_FILE_SIZE:
                return DownloadResult(
                    policy_title=title,
                    source_url=url,
                    success=False,
                    content_type='pdf',
                    file_path=None,
                    file_size=content_length,
                    error_message=f"File too large: {content_length / 1024 / 1024:.1f} MB",
                    downloaded_at=datetime.now().isoformat()
                )
            
            # Download with timeout protection
            try:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                # Clean up partial file
                if filepath.exists():
                    filepath.unlink()
                raise e
            
            file_size = filepath.stat().st_size
            logger.info(f"  Downloaded PDF: {filename} ({file_size / 1024:.1f} KB)")
            
            return DownloadResult(
                policy_title=title,
                source_url=url,
                success=True,
                content_type='pdf',
                file_path=str(filepath),
                file_size=file_size,
                error_message=None,
                downloaded_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"  PDF download failed: {str(e)[:60]}")
            return DownloadResult(
                policy_title=title,
                source_url=url,
                success=False,
                content_type='pdf',
                file_path=None,
                file_size=None,
                error_message=str(e),
                downloaded_at=datetime.now().isoformat()
            )
    
    def download_webpage(self, url: str, title: str) -> DownloadResult:
        """Download a webpage and extract text content."""
        url = self._resolve_google_redirect(url)
        base_filename = f"{self._sanitize_filename(title)}_{self._get_url_hash(url)}"
        html_filepath = self.html_dir / f"{base_filename}.html"
        text_filepath = self.text_dir / f"{base_filename}.txt"
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Check if we actually got a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type.lower():
                return self.download_pdf(url, title)
            
            html_content = response.text
            
            # Save raw HTML
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Extract and save text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)
            
            # Add metadata header
            text_with_metadata = f"""Source: {url}
Title: {title}
Downloaded: {datetime.now().isoformat()}
{'='*80}

{clean_text}
"""
            
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text_with_metadata)
            
            file_size = text_filepath.stat().st_size
            logger.info(f"  Downloaded webpage: {base_filename} ({file_size / 1024:.1f} KB text)")
            
            return DownloadResult(
                policy_title=title,
                source_url=url,
                success=True,
                content_type='html',
                file_path=str(text_filepath),
                file_size=file_size,
                error_message=None,
                downloaded_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"  Webpage download failed: {str(e)[:60]}")
            return DownloadResult(
                policy_title=title,
                source_url=url,
                success=False,
                content_type='html',
                file_path=None,
                file_size=None,
                error_message=str(e),
                downloaded_at=datetime.now().isoformat()
            )
    
    def download_content(self, url: str, title: str) -> DownloadResult:
        """Download content (auto-detect PDF vs webpage)."""
        if self._is_pdf_url(url):
            return self.download_pdf(url, title)
        else:
            return self.download_webpage(url, title)
    
    def process_policies(self, policies: List[dict], pdfs_only: bool = False,
                        webpages_only: bool = False,
                        limit: Optional[int] = None) -> List[DownloadResult]:
        """Process multiple policies and download their content."""
        # Filter to policies with source URLs
        with_urls = [p for p in policies if p.get('source_url')]
        
        if pdfs_only:
            with_urls = [p for p in with_urls if self._is_pdf_url(p['source_url'])]
            logger.info(f"Found {len(with_urls)} policies with PDF source URLs")
        elif webpages_only:
            with_urls = [p for p in with_urls if not self._is_pdf_url(p['source_url'])]
            logger.info(f"Found {len(with_urls)} policies with webpage source URLs")
        else:
            logger.info(f"Found {len(with_urls)} policies with source URLs")
        
        if limit:
            with_urls = with_urls[:limit]
            logger.info(f"Processing limited to {limit} policies")
        
        logger.info("=" * 60)
        
        for i, policy in enumerate(with_urls, 1):
            title = policy['title']
            url = policy['source_url']
            
            logger.info(f"[{i}/{len(with_urls)}] {title[:50]}...")
            
            result = self.download_content(url, title)
            self.results.append(result)
            
            time.sleep(REQUEST_DELAY)
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save download results to JSON."""
        summary = {
            'download_date': datetime.now().isoformat(),
            'total_attempted': len(self.results),
            'successful': sum(1 for r in self.results if r.success),
            'failed': sum(1 for r in self.results if not r.success),
            'pdfs_downloaded': sum(1 for r in self.results if r.success and r.content_type == 'pdf'),
            'webpages_downloaded': sum(1 for r in self.results if r.success and r.content_type == 'html'),
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        return summary


def main():
    parser = argparse.ArgumentParser(description='Download OECD policy document content')
    parser.add_argument('--input', required=True, help='Input JSON file with policies')
    parser.add_argument('--output', default='./data/content', help='Output directory')
    parser.add_argument('--pdfs-only', action='store_true', help='Only download PDFs')
    parser.add_argument('--webpages-only', action='store_true', help='Only download webpages (non-PDFs)')
    parser.add_argument('--limit', type=int, help='Limit number of downloads')
    args = parser.parse_args()
    
    # Load policies
    logger.info(f"Loading policies from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    policies = data['policies']
    logger.info(f"Loaded {len(policies)} policies")
    
    # Initialize downloader
    downloader = ContentDownloader(args.output)
    
    # Process
    logger.info("=" * 60)
    logger.info("OECD Policy Content Downloader")
    logger.info("=" * 60)
    
    results = downloader.process_policies(
        policies, 
        pdfs_only=args.pdfs_only,
        webpages_only=args.webpages_only,
        limit=args.limit
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Path(args.output) / f"download_results_{timestamp}.json"
    summary = downloader.save_results(str(results_file))
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total attempted: {summary['total_attempted']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"PDFs downloaded: {summary['pdfs_downloaded']}")
    logger.info(f"Webpages downloaded: {summary['webpages_downloaded']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
