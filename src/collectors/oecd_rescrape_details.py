"""
OECD Detail Page Re-Scraper
============================

Re-scrapes the 1,569 OECD policies that don't have external source URLs
to extract the FULL content from their detail pages, including:

- Initiative overview (longer than description)
- Responsible organisation
- Status (Proposed/Active/Completed)
- Binding status
- Target sectors
- OECD AI Principles mapping
- Related/external URLs
- Stakeholder involvement mechanisms

Usage:
    python oecd_rescrape_details.py --input data/oecd/oecd_policies_20260126_201311.json --limit 5
    python oecd_rescrape_details.py --input data/oecd/oecd_policies_20260126_201311.json --all
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

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PAGE_DELAY = 1.5  # seconds between requests


@dataclass
class EnrichedPolicy:
    """Enhanced data model with all detail page fields."""
    # Original fields
    title: str
    url: str
    description: str
    jurisdiction: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    
    # Enriched fields from detail page
    initiative_overview: str = ""  # The longer description on detail page
    original_language_name: str = ""
    responsible_organisation: str = ""
    stakeholder_involvement: str = ""
    
    # Structured metadata
    status: str = ""  # "Proposed or under development", "Active", "Completed", etc.
    binding: str = ""  # "Binding", "Non-binding", "Mixed"
    initiative_type: str = ""  # "National AI strategy", "AI use cases in public sector", etc.
    category: str = ""  # "AI policy initiatives, programmes and projects", etc.
    
    # Tags and classifications
    policy_areas: List[str] = field(default_factory=list)
    target_sectors: List[str] = field(default_factory=list)
    oecd_ai_principles: List[str] = field(default_factory=list)
    ai_tags: List[str] = field(default_factory=list)
    
    # External links
    source_urls: List[str] = field(default_factory=list)  # Multiple possible URLs
    related_policies: List[str] = field(default_factory=list)
    
    # Text stats
    word_count: int = 0
    
    # Metadata
    scraped_at: str = ""
    enrichment_success: bool = False
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()


class OECDDetailEnricher:
    """Re-scrapes OECD detail pages to extract full content."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enriched: List[EnrichedPolicy] = []
        self._browser = None
        self._context = None
    
    def _extract_section_text(self, soup, header_text: str) -> str:
        """Extract text following a specific header (h3, h4, etc.)."""
        # Find header containing the text
        header = soup.find(['h1', 'h2', 'h3', 'h4', 'h5'], 
                          string=lambda x: x and header_text.lower() in x.lower())
        if not header:
            return ""
        
        # Get following sibling content until next header
        content = []
        for sibling in header.find_next_siblings():
            if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                break
            text = sibling.get_text(strip=True)
            if text and text != '-':
                content.append(text)
        
        return ' '.join(content)
    
    def _extract_list_items(self, soup, header_text: str) -> List[str]:
        """Extract list items following a header."""
        header = soup.find(['h1', 'h2', 'h3', 'h4', 'h5'],
                          string=lambda x: x and header_text.lower() in x.lower())
        if not header:
            return []
        
        items = []
        # Look for next ul or links
        next_elem = header.find_next(['ul', 'div'])
        if next_elem:
            for link in next_elem.find_all('a'):
                text = link.get_text(strip=True)
                if text:
                    items.append(text)
        return items
    
    def _is_valid_external_url(self, url: str) -> bool:
        """Check if URL is a valid external link worth capturing."""
        if not url or not url.startswith('http'):
            return False
        # Exclude OECD domains and social media
        excluded = ['oecd.ai', 'oecd.org', 'twitter.com', 'facebook.com', 'linkedin.com', 
                   'youtube.com', 'instagram.com', 'google.com/url', 'goingdigital.oecd']
        if any(exc in url.lower() for exc in excluded):
            return False
        return True
    
    def _extract_field_value(self, soup, field_name: str) -> str:
        """Extract a field value from label: value pairs."""
        # Look for the field label and get the next text element
        label = soup.find(string=lambda x: x and field_name.lower() in x.lower() and ':' in x)
        if label:
            # Get the parent and find the value
            parent = label.find_parent()
            if parent:
                # Get text after the label, stop at next label or newline
                full_text = parent.get_text(strip=True)
                # Extract just the value after "Field:"
                pattern = rf'{re.escape(field_name)}:\s*([^:]+?)(?:\s+\w+:|$)'
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Clean up - remove common trailing patterns
                    value = re.sub(r'\s*(Start Year|Binding|Status|Category|Initiative type|Target|OECD).*$', '', value, flags=re.IGNORECASE)
                    return value[:200]  # Cap length
        return ""
    
    def enrich_policy(self, page, original: dict) -> EnrichedPolicy:
        """Scrape detail page and extract all available content."""
        url = original['url']
        
        enriched = EnrichedPolicy(
            title=original['title'],
            url=url,
            description=original.get('description', ''),
            jurisdiction=original.get('jurisdiction', ''),
            start_year=original.get('start_year'),
            end_year=original.get('end_year'),
            policy_areas=original.get('policy_areas', []),
        )
        
        try:
            page.goto(url, wait_until='networkidle', timeout=45000)
            time.sleep(0.5)  # Let dynamic content render
            
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            # === EXTRACT INITIATIVE OVERVIEW (the longer description) ===
            overview = self._extract_section_text(soup, 'Initiative overview')
            enriched.initiative_overview = overview or enriched.description
            
            # === EXTRACT STRUCTURED METADATA ===
            enriched.original_language_name = self._extract_section_text(soup, 'Name in original language')
            enriched.responsible_organisation = self._extract_section_text(soup, 'responsible organisation')
            enriched.stakeholder_involvement = self._extract_section_text(soup, 'stakeholder')
            
            # Extract from "About the policy initiative" section
            enriched.status = self._extract_field_value(soup, 'Status')
            enriched.binding = self._extract_field_value(soup, 'Binding')
            enriched.initiative_type = self._extract_field_value(soup, 'Initiative type')
            enriched.category = self._extract_field_value(soup, 'Category')
            
            # === EXTRACT TAGS AND CLASSIFICATIONS ===
            # Target sectors - look for links after the header
            sectors_header = soup.find(string=lambda x: x and 'Target Sectors' in str(x))
            if sectors_header:
                parent = sectors_header.find_parent()
                if parent:
                    next_div = parent.find_next_sibling() or parent.find_next()
                    if next_div:
                        for link in next_div.find_all('a', href=True)[:10]:
                            text = link.get_text(strip=True)
                            if text and len(text) < 100 and not text.startswith('http'):
                                enriched.target_sectors.append(text)
            
            # OECD AI Principles
            principles_header = soup.find(string=lambda x: x and 'OECD AI Principles' in str(x))
            if principles_header:
                parent = principles_header.find_parent()
                if parent:
                    next_div = parent.find_next_sibling() or parent.find_next()
                    if next_div:
                        for link in next_div.find_all('a', href=True)[:10]:
                            text = link.get_text(strip=True)
                            if text and len(text) < 150 and not text.startswith('http'):
                                enriched.oecd_ai_principles.append(text)
            
            # AI Tags (from the tag section - these are usually Data, Innovation, etc.)
            ai_tags_header = soup.find(['h3', 'h4'], string=lambda x: x and 'AI Tags' in str(x))
            if ai_tags_header:
                parent = ai_tags_header.find_parent()
                if parent:
                    for link in parent.find_all('a', href=True):
                        text = link.get_text(strip=True)
                        # Only add if it's a tag name (short text, not URL, not country)
                        if text and len(text) < 50 and not text.startswith('http'):
                            if text not in [enriched.jurisdiction] and text not in enriched.ai_tags:
                                enriched.ai_tags.append(text)
            
            # === EXTRACT EXTERNAL URLS ===
            # Look for "Other relevant urls" section specifically
            all_external_urls = []
            
            # Method 1: Find links near "relevant url" text
            urls_text = soup.find(string=lambda x: x and 'relevant url' in str(x).lower())
            if urls_text:
                # Search in nearby elements
                container = urls_text.find_parent(['div', 'section', 'article'])
                if container:
                    for link in container.find_all('a', href=True):
                        href = link.get('href', '')
                        if self._is_valid_external_url(href):
                            all_external_urls.append(href)
            
            # Method 2: Find any external links on the page that look like source docs
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if self._is_valid_external_url(href):
                    # Prioritize government and official sources
                    if any(domain in href for domain in ['.gov', '.gob', '.gouv', 'government', 'official', 'europa.eu', 'un.org']):
                        if href not in all_external_urls:
                            all_external_urls.insert(0, href)  # Add to front
                    elif href not in all_external_urls:
                        all_external_urls.append(href)
            
            # Deduplicate and limit
            enriched.source_urls = list(dict.fromkeys(all_external_urls))[:10]
            
            # === COMPUTE STATS ===
            all_text = ' '.join([
                enriched.initiative_overview,
                enriched.description,
                enriched.responsible_organisation,
                enriched.stakeholder_involvement
            ])
            enriched.word_count = len(all_text.split())
            
            enriched.enrichment_success = True
            
        except Exception as e:
            logger.warning(f"Error enriching {original['title'][:40]}: {e}")
            enriched.enrichment_success = False
        
        return enriched
    
    def process_policies(self, policies: List[dict], 
                        only_without_source: bool = True,
                        limit: Optional[int] = None):
        """Process multiple policies and enrich them."""
        
        # Filter policies
        if only_without_source:
            to_process = [p for p in policies if not p.get('source_url')]
            logger.info(f"Found {len(to_process)} policies without source URLs")
        else:
            to_process = policies
            logger.info(f"Processing all {len(to_process)} policies")
        
        if limit:
            to_process = to_process[:limit]
            logger.info(f"Limited to {limit} policies")
        
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            for i, policy in enumerate(to_process, 1):
                logger.info(f"[{i}/{len(to_process)}] {policy['title'][:50]}...")
                
                enriched = self.enrich_policy(page, policy)
                self.enriched.append(enriched)
                
                # Save checkpoint every 50 policies
                if i % 50 == 0:
                    self._save_checkpoint(i)
                
                time.sleep(PAGE_DELAY)
            
            browser.close()
        
        return self.enriched
    
    def _save_checkpoint(self, count: int):
        """Save progress checkpoint."""
        checkpoint_file = self.output_dir / "enrichment_checkpoint.json"
        data = {
            'count': count,
            'timestamp': datetime.now().isoformat(),
            'policies': [asdict(p) for p in self.enriched]
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved: {count} policies")
    
    def save_results(self):
        """Save final enriched results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate stats
        successful = sum(1 for p in self.enriched if p.enrichment_success)
        with_overview = sum(1 for p in self.enriched if len(p.initiative_overview) > len(p.description))
        with_new_urls = sum(1 for p in self.enriched if p.source_urls)
        total_words = sum(p.word_count for p in self.enriched)
        
        # Save JSON
        output_file = self.output_dir / f"oecd_enriched_{timestamp}.json"
        data = {
            'scrape_date': datetime.now().isoformat(),
            'total_policies': len(self.enriched),
            'successfully_enriched': successful,
            'with_longer_overview': with_overview,
            'with_new_source_urls': with_new_urls,
            'total_words': total_words,
            'policies': [asdict(p) for p in self.enriched]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.enriched)} enriched policies to {output_file}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("ENRICHMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total processed: {len(self.enriched)}")
        logger.info(f"Successfully enriched: {successful}")
        logger.info(f"With longer overview: {with_overview}")
        logger.info(f"With new source URLs: {with_new_urls}")
        logger.info(f"Total words: {total_words:,}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Re-scrape OECD detail pages for full content')
    parser.add_argument('--input', required=True, help='Input JSON with policies')
    parser.add_argument('--output', default='./data/oecd/enriched', help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number to process')
    parser.add_argument('--all', action='store_true', help='Process all policies (not just those without source)')
    args = parser.parse_args()
    
    # Load policies
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    policies = data['policies']
    logger.info(f"Loaded {len(policies)} policies")
    
    # Process
    enricher = OECDDetailEnricher(args.output)
    
    logger.info("=" * 60)
    logger.info("OECD Detail Page Enrichment")
    logger.info("=" * 60)
    
    enricher.process_policies(
        policies, 
        only_without_source=not args.all,
        limit=args.limit
    )
    
    enricher.save_results()


if __name__ == '__main__':
    main()
