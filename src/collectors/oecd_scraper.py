"""
OECD.AI Policy Initiatives Scraper
===================================

Scrapes policy initiatives from https://oecd.ai/en/dashboards/policy-initiatives

Requirements:
    pip install playwright beautifulsoup4 pandas
    playwright install chromium

Usage:
    python oecd_scraper.py --pages 5        # Scrape first 5 pages
    python oecd_scraper.py --all            # Scrape all pages
    python oecd_scraper.py --test           # Test with 1 page

Author: AI Governance Observatory
"""

import json
import time
import re
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PolicyInitiative:
    """Data model for a single policy initiative."""
    title: str
    url: str
    description: str
    jurisdiction: str
    start_year: Optional[int] = None
    
    # Additional fields from detail page (populated later)
    end_year: Optional[int] = None
    policy_areas: List[str] = field(default_factory=list)
    target_groups: List[str] = field(default_factory=list)
    governance_stage: str = ""
    full_description: str = ""
    source_url: str = ""
    
    # Metadata
    scraped_at: str = ""
    detail_scraped: bool = False
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OECDPolicyScraper:
    """
    Scraper for OECD.AI Policy Initiatives.
    
    Respects robots.txt (English pages allowed) and implements polite delays.
    """
    
    BASE_URL = "https://oecd.ai"
    INDEX_URL = "https://oecd.ai/en/dashboards/policy-initiatives"
    
    # Rate limiting
    PAGE_DELAY = 3.0  # seconds between page requests
    DETAIL_DELAY = 2.0  # seconds between detail page requests
    
    def __init__(self, output_dir: str = "./data/oecd"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.policies: List[PolicyInitiative] = []
        self._browser = None
        self._playwright = None
    
    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        logger.info("Browser launched")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        logger.info("Browser closed")
    
    def _new_page(self):
        """Create a new page with proper headers."""
        page = self._browser.new_page()
        page.set_extra_http_headers({
            'User-Agent': 'AIGovernanceObservatory/1.0 (Academic Research; contact@university.edu)',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        return page
    
    def get_total_pages(self) -> int:
        """Get the total number of pages in the policy index."""
        page = self._new_page()
        try:
            logger.info(f"Fetching index to count pages: {self.INDEX_URL}")
            page.goto(self.INDEX_URL, wait_until='networkidle', timeout=60000)
            
            # Wait for pagination to load
            page.wait_for_selector('.pagination, nav[aria-label*="pagination"], [class*="pagination"]', timeout=10000)
            
            # Try to find the last page number
            # Look for pagination links
            pagination_text = page.content()
            
            # Common patterns for page numbers
            # Pattern 1: Look for numbered links in pagination
            page_numbers = page.query_selector_all('a[href*="page="], button[class*="page"], [class*="pagination"] a')
            
            max_page = 1
            for elem in page_numbers:
                text = elem.inner_text().strip()
                if text.isdigit():
                    max_page = max(max_page, int(text))
            
            # Also check for "111" or similar in the page content (from earlier we know there are ~111 pages)
            import re
            numbers = re.findall(r'\b(\d{2,3})\b', pagination_text[-5000:])  # Check last part of page
            for num in numbers:
                if 50 < int(num) < 200:  # Reasonable page count range
                    max_page = max(max_page, int(num))
            
            logger.info(f"Detected {max_page} total pages")
            return max_page
            
        except Exception as e:
            logger.warning(f"Could not determine total pages: {e}. Defaulting to 111.")
            return 111
        finally:
            page.close()
    
    def scrape_index_page(self, page_num: int = 1) -> List[PolicyInitiative]:
        """Scrape a single page of the policy index."""
        
        url = f"{self.INDEX_URL}?orderBy=startYearDesc&page={page_num}"
        page = self._new_page()
        policies = []
        
        try:
            logger.info(f"Scraping index page {page_num}: {url}")
            page.goto(url, wait_until='networkidle', timeout=60000)
            
            # Wait for policy cards to load
            page.wait_for_selector('h4 a, .policy-card, [class*="initiative"], article', timeout=15000)
            
            # Get page content
            content = page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find policy entries - based on the structure we saw earlier
            # Each policy has: title (h4 with link), description, start year, country/organisation
            
            # Try multiple selector strategies
            policy_cards = []
            
            # Strategy 1: Look for h4 tags with links (seen in the data)
            for h4 in soup.find_all('h4'):
                link = h4.find('a', href=True)
                if link and '/policy-initiatives/' in link.get('href', ''):
                    # Found a policy entry, now get its container
                    container = h4.find_parent(['article', 'div', 'li'])
                    if container:
                        policy_cards.append(container)
                    else:
                        policy_cards.append(h4.parent)
            
            # Strategy 2: If no h4 found, try article or card divs
            if not policy_cards:
                policy_cards = soup.select('article, .card, [class*="policy"], [class*="initiative"]')
            
            for card in policy_cards:
                try:
                    # Extract title and URL
                    title_elem = card.find('a', href=lambda x: x and '/policy-initiatives/' in x)
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get('href', '')
                    
                    # Make URL absolute
                    if href.startswith('/'):
                        url = self.BASE_URL + href
                    elif not href.startswith('http'):
                        url = self.BASE_URL + '/' + href
                    else:
                        url = href
                    
                    # Extract description (usually in a paragraph or following text)
                    desc_elem = card.find(['p', 'span', 'div'], class_=lambda x: x and ('desc' in str(x).lower() or 'summary' in str(x).lower()))
                    if not desc_elem:
                        # Try getting text after the title
                        desc_elem = card.find('p')
                    description = desc_elem.get_text(strip=True) if desc_elem else ""
                    
                    # Extract jurisdiction/country
                    jurisdiction = ""
                    # Look for text containing "Country/Organisation:" or flag emoji
                    card_text = card.get_text()
                    
                    # Pattern: "Country/Organisation: X" or "🌐 Country/Organisation: X"
                    country_match = re.search(r'Country/Organisation:\s*([^\n]+)', card_text)
                    if country_match:
                        jurisdiction = country_match.group(1).strip()
                    else:
                        # Look for country names after flag emoji or in specific span
                        country_span = card.find(['span', 'div'], class_=lambda x: x and 'country' in str(x).lower())
                        if country_span:
                            jurisdiction = country_span.get_text(strip=True)
                    
                    # Extract start year
                    start_year = None
                    year_match = re.search(r'Start year:\s*(\d{4})', card_text)
                    if year_match:
                        start_year = int(year_match.group(1))
                    else:
                        # Try to find any 4-digit year
                        years = re.findall(r'\b(20\d{2})\b', card_text)
                        if years:
                            start_year = int(years[0])
                    
                    # Create policy object
                    policy = PolicyInitiative(
                        title=title,
                        url=url,
                        description=description[:500] if description else "",  # Truncate long descriptions
                        jurisdiction=jurisdiction,
                        start_year=start_year
                    )
                    
                    policies.append(policy)
                    logger.debug(f"  Found: {title[:50]}... ({jurisdiction})")
                    
                except Exception as e:
                    logger.warning(f"  Error parsing policy card: {e}")
                    continue
            
            logger.info(f"  Found {len(policies)} policies on page {page_num}")
            return policies
            
        except Exception as e:
            logger.error(f"Error scraping page {page_num}: {e}")
            return []
        finally:
            page.close()
    
    def scrape_detail_page(self, policy: PolicyInitiative) -> PolicyInitiative:
        """Scrape additional details from a policy's detail page."""
        
        page = self._new_page()
        
        try:
            logger.info(f"Scraping detail: {policy.title[:50]}...")
            page.goto(policy.url, wait_until='networkidle', timeout=60000)
            
            # Wait for content
            page.wait_for_selector('h1, h2, article, main', timeout=10000)
            
            content = page.content()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract full description
            main_content = soup.find(['article', 'main', 'div'], class_=lambda x: x and ('content' in str(x).lower() or 'body' in str(x).lower()))
            if main_content:
                paragraphs = main_content.find_all('p')
                policy.full_description = '\n\n'.join(p.get_text(strip=True) for p in paragraphs[:5])
            
            # Extract policy areas/categories
            policy_areas = soup.find_all(['span', 'a', 'div'], class_=lambda x: x and ('tag' in str(x).lower() or 'category' in str(x).lower() or 'area' in str(x).lower()))
            policy.policy_areas = [pa.get_text(strip=True) for pa in policy_areas if pa.get_text(strip=True)]
            
            # Extract source URL if present
            source_link = soup.find('a', href=True, string=lambda x: x and ('source' in str(x).lower() or 'official' in str(x).lower() or 'view' in str(x).lower()))
            if source_link:
                policy.source_url = source_link.get('href', '')
            
            # Look for external links that might be the official source
            external_links = soup.find_all('a', href=lambda x: x and x.startswith('http') and 'oecd.ai' not in x)
            for link in external_links[:3]:  # Check first 3 external links
                href = link.get('href', '')
                if '.gov' in href or 'government' in href or 'official' in href:
                    policy.source_url = href
                    break
            
            policy.detail_scraped = True
            
        except Exception as e:
            logger.warning(f"Error scraping detail page for {policy.title[:30]}: {e}")
        finally:
            page.close()
        
        return policy
    
    def scrape_all(self, max_pages: Optional[int] = None, scrape_details: bool = False):
        """
        Scrape all policy initiatives.
        
        Args:
            max_pages: Maximum number of index pages to scrape (None = all)
            scrape_details: Whether to also scrape individual detail pages
        """
        
        total_pages = self.get_total_pages()
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        logger.info(f"Starting scrape of {total_pages} pages")
        
        for page_num in range(1, total_pages + 1):
            # Scrape index page
            policies = self.scrape_index_page(page_num)
            self.policies.extend(policies)
            
            # Rate limiting
            if page_num < total_pages:
                logger.info(f"Waiting {self.PAGE_DELAY}s before next page...")
                time.sleep(self.PAGE_DELAY)
            
            # Save progress periodically
            if page_num % 10 == 0:
                self.save_progress()
        
        logger.info(f"Index scraping complete. Found {len(self.policies)} policies.")
        
        # Optionally scrape detail pages
        if scrape_details:
            logger.info("Scraping detail pages...")
            for i, policy in enumerate(self.policies):
                if not policy.detail_scraped:
                    self.scrape_detail_page(policy)
                    time.sleep(self.DETAIL_DELAY)
                    
                    if (i + 1) % 20 == 0:
                        self.save_progress()
                        logger.info(f"  Progress: {i + 1}/{len(self.policies)} details scraped")
        
        # Final save
        self.save_results()
    
    def save_progress(self):
        """Save current progress to a temporary file."""
        filepath = self.output_dir / "oecd_policies_progress.json"
        data = {
            'scraped_at': datetime.now().isoformat(),
            'count': len(self.policies),
            'policies': [p.to_dict() for p in self.policies]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Progress saved: {len(self.policies)} policies")
    
    def save_results(self):
        """Save final results to JSON and CSV."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = self.output_dir / f"oecd_policies_{timestamp}.json"
        data = {
            'source': 'OECD.AI Policy Navigator',
            'source_url': self.INDEX_URL,
            'scraped_at': datetime.now().isoformat(),
            'total_count': len(self.policies),
            'citation': 'OECD.AI (2026), OECD.AI Policy Navigator, https://oecd.ai/dashboards',
            'policies': [p.to_dict() for p in self.policies]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # Save as CSV
        try:
            import pandas as pd
            csv_path = self.output_dir / f"oecd_policies_{timestamp}.csv"
            df = pd.DataFrame([p.to_dict() for p in self.policies])
            
            # Flatten list columns for CSV
            for col in ['policy_areas', 'target_groups']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Saved CSV: {csv_path}")
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")
        
        # Also save a "latest" version for easy access
        latest_json = self.output_dir / "oecd_policies_latest.json"
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Scraping complete! Total policies: {len(self.policies)}")
        return json_path


def main():
    parser = argparse.ArgumentParser(description='Scrape OECD.AI Policy Initiatives')
    parser.add_argument('--pages', type=int, default=None, help='Number of pages to scrape (default: all)')
    parser.add_argument('--all', action='store_true', help='Scrape all pages')
    parser.add_argument('--test', action='store_true', help='Test mode: scrape only 1 page')
    parser.add_argument('--details', action='store_true', help='Also scrape detail pages')
    parser.add_argument('--output', type=str, default='./data/oecd', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine pages to scrape
    if args.test:
        max_pages = 1
    elif args.all:
        max_pages = None
    else:
        max_pages = args.pages or 5  # Default to 5 pages
    
    logger.info("=" * 60)
    logger.info("OECD.AI Policy Initiatives Scraper")
    logger.info("=" * 60)
    logger.info(f"Pages to scrape: {max_pages or 'all'}")
    logger.info(f"Scrape details: {args.details}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    with OECDPolicyScraper(output_dir=args.output) as scraper:
        scraper.scrape_all(max_pages=max_pages, scrape_details=args.details)


if __name__ == "__main__":
    main()
