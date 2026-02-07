"""
OECD Detail Page Scraper
Scrapes full content from OECD.AI policy detail pages for content analysis.

This handles the 1,569 policies that don't have external source URLs but
have rich content on their OECD.AI detail pages.

Usage:
    python oecd_detail_scraper.py --input data/oecd/oecd_policies_20260126_201311.json
    python oecd_detail_scraper.py --input data/oecd/oecd_policies_20260126_201311.json --limit 50
"""

import json
import os
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PAGE_DELAY = 1.5  # Faster since it's the same domain


@dataclass
class DetailContent:
    """Content extracted from OECD detail page."""
    policy_title: str
    url: str
    jurisdiction: str
    start_year: Optional[int]
    
    # Extracted content
    full_description: str
    policy_areas: List[str]
    target_groups: List[str]
    governance_stage: str
    related_policies: List[str]
    
    # Metadata
    word_count: int
    scraped_at: str


class OECDDetailScraper:
    """Scrapes full content from OECD.AI detail pages."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[DetailContent] = []
        
    def _extract_detail_content(self, page, url: str, policy: dict) -> DetailContent:
        """Extract all content from a detail page."""
        try:
            page.goto(url, wait_until='networkidle', timeout=30000)
            time.sleep(1)  # Let dynamic content load
            
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract full description - look for main content area
            full_desc = ""
            
            # Try multiple selectors for description
            desc_selectors = [
                'div[class*="description"]',
                'div[class*="content"]',
                'article',
                'main',
                '.policy-detail',
                '.initiative-detail'
            ]
            
            for selector in desc_selectors:
                elem = soup.select_one(selector)
                if elem:
                    text = elem.get_text(separator=' ', strip=True)
                    if len(text) > len(full_desc):
                        full_desc = text
            
            # If no luck with selectors, get body text
            if len(full_desc) < 100:
                body = soup.find('body')
                if body:
                    # Remove nav, footer, script elements
                    for tag in body.find_all(['nav', 'footer', 'script', 'style', 'header']):
                        tag.decompose()
                    full_desc = body.get_text(separator=' ', strip=True)
            
            # Extract policy areas (tags/categories)
            policy_areas = []
            for tag in soup.find_all(['span', 'a', 'div'], class_=re.compile(r'tag|badge|label|chip')):
                text = tag.get_text(strip=True)
                if text and len(text) < 100:
                    policy_areas.append(text)
            
            # Extract target groups
            target_groups = []
            target_section = soup.find(text=re.compile(r'target|audience|stakeholder', re.I))
            if target_section:
                parent = target_section.find_parent()
                if parent:
                    for li in parent.find_all('li'):
                        target_groups.append(li.get_text(strip=True))
            
            # Extract governance stage
            governance_stage = ""
            stage_elem = soup.find(text=re.compile(r'stage|phase|status', re.I))
            if stage_elem:
                governance_stage = stage_elem.find_parent().get_text(strip=True)[:100]
            
            # Extract related policies
            related = []
            for link in soup.find_all('a', href=re.compile(r'/policy-initiatives/')):
                text = link.get_text(strip=True)
                if text and text != policy['title']:
                    related.append(text)
            
            # Clean up description
            full_desc = re.sub(r'\s+', ' ', full_desc).strip()
            word_count = len(full_desc.split())
            
            return DetailContent(
                policy_title=policy['title'],
                url=url,
                jurisdiction=policy.get('jurisdiction', ''),
                start_year=policy.get('start_year'),
                full_description=full_desc,
                policy_areas=list(set(policy_areas))[:20],
                target_groups=list(set(target_groups))[:10],
                governance_stage=governance_stage,
                related_policies=list(set(related))[:10],
                word_count=word_count,
                scraped_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"Error extracting content: {e}")
            return DetailContent(
                policy_title=policy['title'],
                url=url,
                jurisdiction=policy.get('jurisdiction', ''),
                start_year=policy.get('start_year'),
                full_description=policy.get('description', ''),
                policy_areas=policy.get('policy_areas', []),
                target_groups=[],
                governance_stage='',
                related_policies=[],
                word_count=len(policy.get('description', '').split()),
                scraped_at=datetime.now().isoformat()
            )
    
    def scrape_details(self, policies: List[dict], limit: Optional[int] = None):
        """Scrape detail pages for policies without source URLs."""
        # Filter to policies without source URLs
        no_source = [p for p in policies if not p.get('source_url')]
        logger.info(f"Found {len(no_source)} policies without external source URLs")
        
        if limit:
            no_source = no_source[:limit]
            logger.info(f"Limited to {limit} policies")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            for i, policy in enumerate(no_source, 1):
                url = policy['url']
                logger.info(f"[{i}/{len(no_source)}] {policy['title'][:50]}...")
                
                content = self._extract_detail_content(page, url, policy)
                self.results.append(content)
                
                if i % 50 == 0:
                    self._save_checkpoint(i)
                
                time.sleep(PAGE_DELAY)
            
            browser.close()
        
        return self.results
    
    def _save_checkpoint(self, count: int):
        """Save progress checkpoint."""
        checkpoint_file = self.output_dir / "detail_content_checkpoint.json"
        data = {
            'scraped_count': count,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.results]
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved: {count} policies")
    
    def save_results(self):
        """Save final results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = self.output_dir / f"oecd_detail_content_{timestamp}.json"
        data = {
            'scrape_date': datetime.now().isoformat(),
            'total_policies': len(self.results),
            'total_words': sum(r.word_count for r in self.results),
            'results': [asdict(r) for r in self.results]
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save as text corpus (one file per policy)
        corpus_dir = self.output_dir / "corpus"
        corpus_dir.mkdir(exist_ok=True)
        
        for r in self.results:
            safe_name = re.sub(r'[<>:"/\\|?*]', '', r.policy_title)[:60]
            txt_file = corpus_dir / f"{safe_name}.txt"
            
            content = f"""Title: {r.policy_title}
Jurisdiction: {r.jurisdiction}
Year: {r.start_year or 'N/A'}
URL: {r.url}
Policy Areas: {', '.join(r.policy_areas)}
{'='*80}

{r.full_description}
"""
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Saved {len(self.results)} detail pages to {json_file}")
        logger.info(f"Text corpus saved to {corpus_dir}")
        
        return json_file


def main():
    parser = argparse.ArgumentParser(description='Scrape OECD detail pages for full content')
    parser.add_argument('--input', required=True, help='Input JSON with policies')
    parser.add_argument('--output', default='./data/content/oecd_details', help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of pages')
    args = parser.parse_args()
    
    # Load policies
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    policies = data['policies']
    logger.info(f"Loaded {len(policies)} policies")
    
    # Scrape
    scraper = OECDDetailScraper(args.output)
    
    logger.info("=" * 60)
    logger.info("OECD Detail Page Content Scraper")
    logger.info("=" * 60)
    
    scraper.scrape_details(policies, limit=args.limit)
    scraper.save_results()
    
    # Summary
    total_words = sum(r.word_count for r in scraper.results)
    logger.info("=" * 60)
    logger.info(f"Scraped {len(scraper.results)} detail pages")
    logger.info(f"Total words extracted: {total_words:,}")
    logger.info(f"Average words per policy: {total_words // len(scraper.results)}")


if __name__ == '__main__':
    main()
