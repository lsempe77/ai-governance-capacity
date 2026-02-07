"""
AI Governance Observatory - Data Collection Pipeline
=====================================================

A modular, ethical scraping pipeline for collecting AI governance policy data
from OECD.AI and other authoritative sources.

IMPORTANT: Before running this code:
1. Review robots.txt for each source
2. Check Terms of Service for data reuse permissions
3. Use official APIs/exports where available (preferred over scraping)
4. Respect rate limits and implement polite crawling

Author: [Your Name]
Project: Global Observatory of AI Governance Capacity
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('observatory')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScraperConfig:
    """Configuration for ethical, responsible scraping."""
    
    # Rate limiting
    request_delay: float = 2.0  # Seconds between requests
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # Timeouts
    request_timeout: int = 30
    
    # Caching
    cache_dir: Path = Path("./data/cache")
    cache_expiry_days: int = 7
    
    # Output
    output_dir: Path = Path("./data/raw")
    
    # User agent (be transparent about who you are)
    user_agent: str = "AIGovernanceObservatory/1.0 (Research Project; contact@example.edu)"
    
    # Respect robots.txt
    respect_robots: bool = True


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PolicyDocument:
    """Represents a single policy document from any source."""
    
    # Identifiers
    doc_id: str
    source: str  # e.g., "oecd_ai", "iapp", "government"
    
    # Basic metadata
    title: str
    jurisdiction: str
    jurisdiction_iso: Optional[str] = None  # ISO 3166-1 alpha-3
    
    # Classification
    category: str = ""  # law, strategy, guidance, standard
    scope: str = ""  # ai_general, sectoral
    
    # Dates
    date_published: Optional[str] = None
    date_collected: str = ""
    
    # URLs and content
    source_url: str = ""
    full_text_url: Optional[str] = None
    
    # Extracted content
    summary: str = ""
    full_text: Optional[str] = None
    
    # Capacity indicators (to be filled by analysis)
    capacity_indicators: Dict[str, Any] = None
    
    # Provenance
    extraction_confidence: float = 1.0
    extraction_notes: str = ""
    
    def __post_init__(self):
        if self.capacity_indicators is None:
            self.capacity_indicators = {}
        if not self.date_collected:
            self.date_collected = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class CapacityIndicator:
    """A single capacity indicator measurement for a jurisdiction."""
    
    jurisdiction: str
    jurisdiction_iso: str
    indicator_id: str  # e.g., "inst_1_1" for Dedicated AI Unit
    indicator_name: str
    
    score: int  # 0-3
    confidence: str  # "high", "medium", "low"
    
    evidence_sources: List[str]  # URLs or document IDs
    evidence_text: str = ""
    extraction_date: str = ""
    notes: str = ""
    
    def __post_init__(self):
        if not self.extraction_date:
            self.extraction_date = datetime.now().isoformat()


# =============================================================================
# HTTP Client with Ethical Safeguards
# =============================================================================

class EthicalHTTPClient:
    """HTTP client with rate limiting, caching, and polite crawling."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.last_request_time = 0
        
        # Ensure cache directory exists
        config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.config.cache_dir / f"{url_hash}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached response is still valid."""
        if not cache_path.exists():
            return False
        
        cache_age = time.time() - cache_path.stat().st_mtime
        max_age = self.config.cache_expiry_days * 24 * 60 * 60
        return cache_age < max_age
    
    def _load_from_cache(self, url: str) -> Optional[Dict]:
        """Load response from cache if valid."""
        cache_path = self._get_cache_path(url)
        if self._is_cache_valid(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                logger.debug(f"Cache hit: {url}")
                return json.load(f)
        return None
    
    def _save_to_cache(self, url: str, data: Dict):
        """Save response to cache."""
        cache_path = self._get_cache_path(url)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.request_delay:
            sleep_time = self.config.request_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def get(self, url: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Fetch URL with caching, rate limiting, and retries.
        
        Returns dict with 'content', 'status_code', 'headers', 'url'.
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(url)
            if cached:
                return cached
        
        # Rate limit
        self._rate_limit()
        
        # Fetch with retries
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Fetching: {url}")
                response = self.session.get(
                    url,
                    timeout=self.config.request_timeout
                )
                
                result = {
                    'url': url,
                    'final_url': response.url,
                    'status_code': response.status_code,
                    'content': response.text,
                    'headers': dict(response.headers),
                    'fetched_at': datetime.now().isoformat()
                }
                
                if response.status_code == 200:
                    self._save_to_cache(url, result)
                    return result
                elif response.status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limited (429). Waiting {self.config.retry_delay * 2}s")
                    time.sleep(self.config.retry_delay * 2)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    return result
                    
            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        return None


# =============================================================================
# OECD.AI Collector
# =============================================================================

class OECDAICollector:
    """
    Collector for OECD.AI Policy Observatory.
    
    IMPORTANT: Check https://oecd.ai/robots.txt and terms of use before running.
    Prefer using any official API or data export if available.
    """
    
    BASE_URL = "https://oecd.ai"
    
    # Known endpoints (verify these are current)
    POLICY_INDEX_URL = "https://oecd.ai/en/dashboards/policy-initiatives"
    
    def __init__(self, client: EthicalHTTPClient, config: ScraperConfig):
        self.client = client
        self.config = config
        self.output_dir = config.output_dir / "oecd_ai"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def check_robots_txt(self) -> Dict[str, Any]:
        """
        Check robots.txt to understand crawling permissions.
        Returns parsed rules (simplified).
        """
        robots_url = f"{self.BASE_URL}/robots.txt"
        response = self.client.get(robots_url, use_cache=False)
        
        if response and response['status_code'] == 200:
            logger.info("robots.txt found. Please review manually for compliance.")
            return {
                'found': True,
                'content': response['content'][:2000],  # First 2000 chars
                'url': robots_url
            }
        else:
            logger.warning("Could not fetch robots.txt")
            return {'found': False}
    
    def check_for_api_or_export(self) -> str:
        """
        Document any API or data export options.
        
        NOTE: This should be manually verified. OECD.AI may offer:
        - CSV/Excel exports on dashboard pages
        - Official API (check developer documentation)
        - Data download portal
        
        ALWAYS prefer official exports over scraping.
        """
        guidance = """
        OECD.AI Data Access Options (verify current availability):
        
        1. DASHBOARD EXPORTS: Some dashboards allow CSV/Excel download.
           Check for download buttons on policy initiative pages.
        
        2. OECD DATA PORTAL: Check data.oecd.org for structured datasets.
        
        3. API ACCESS: Contact OECD directly to inquire about API access
           for research purposes.
        
        4. PARTNERSHIP: For systematic research, consider reaching out
           to OECD.AI team for data sharing agreement.
        
        Recommendation: Before scraping, exhaust all official channels.
        """
        logger.info(guidance)
        return guidance
    
    def fetch_policy_index(self) -> Optional[List[Dict]]:
        """
        Fetch the main policy index page and extract policy entries.
        
        NOTE: OECD.AI uses JavaScript rendering. This basic version
        may not capture all content. For JS-heavy sites, use Playwright.
        """
        response = self.client.get(self.POLICY_INDEX_URL)
        
        if not response or response['status_code'] != 200:
            logger.error("Failed to fetch policy index")
            return None
        
        soup = BeautifulSoup(response['content'], 'html.parser')
        
        # NOTE: These selectors are illustrative and need verification
        # against the actual OECD.AI page structure
        policies = []
        
        # Try to find policy cards/entries (adjust selectors as needed)
        # This is a placeholder - actual structure needs inspection
        for card in soup.select('.policy-card, .initiative-card, [data-policy]'):
            try:
                title_elem = card.select_one('h2, h3, .title')
                link_elem = card.select_one('a[href]')
                country_elem = card.select_one('.country, .jurisdiction')
                
                if title_elem:
                    policy = {
                        'title': title_elem.get_text(strip=True),
                        'url': urljoin(self.BASE_URL, link_elem['href']) if link_elem else None,
                        'jurisdiction': country_elem.get_text(strip=True) if country_elem else 'Unknown',
                        'source': 'oecd_ai'
                    }
                    policies.append(policy)
            except Exception as e:
                logger.warning(f"Error parsing policy card: {e}")
        
        logger.info(f"Found {len(policies)} policy entries (basic parsing)")
        
        # If no policies found with basic parsing, note that JS rendering may be needed
        if not policies:
            logger.warning(
                "No policies found with basic HTML parsing. "
                "OECD.AI likely requires JavaScript rendering. "
                "Use Playwright-based collector for full content."
            )
        
        return policies
    
    def parse_policy_page(self, url: str) -> Optional[PolicyDocument]:
        """
        Parse an individual policy page into a PolicyDocument.
        """
        response = self.client.get(url)
        
        if not response or response['status_code'] != 200:
            return None
        
        soup = BeautifulSoup(response['content'], 'html.parser')
        
        # Extract fields (selectors need verification)
        title = soup.select_one('h1, .policy-title')
        title_text = title.get_text(strip=True) if title else "Unknown"
        
        # Look for jurisdiction
        jurisdiction = "Unknown"
        for selector in ['.country', '.jurisdiction', '[data-country]']:
            elem = soup.select_one(selector)
            if elem:
                jurisdiction = elem.get_text(strip=True)
                break
        
        # Look for category/type
        category = ""
        for selector in ['.policy-type', '.category', '[data-type]']:
            elem = soup.select_one(selector)
            if elem:
                category = elem.get_text(strip=True).lower()
                break
        
        # Look for date
        date_published = None
        for selector in ['.date', 'time', '[datetime]']:
            elem = soup.select_one(selector)
            if elem:
                date_published = elem.get('datetime') or elem.get_text(strip=True)
                break
        
        # Look for description/summary
        summary = ""
        for selector in ['.description', '.summary', '.policy-body p:first-child']:
            elem = soup.select_one(selector)
            if elem:
                summary = elem.get_text(strip=True)[:500]
                break
        
        # Generate document ID
        doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
        
        return PolicyDocument(
            doc_id=f"oecd_{doc_id}",
            source="oecd_ai",
            title=title_text,
            jurisdiction=jurisdiction,
            category=category,
            date_published=date_published,
            source_url=url,
            summary=summary,
            extraction_notes="Basic HTML parsing; verify completeness"
        )
    
    def save_document(self, doc: PolicyDocument):
        """Save a policy document to JSON."""
        filename = f"{doc.doc_id}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved: {filepath}")


# =============================================================================
# Playwright-based Collector (for JS-heavy sites)
# =============================================================================

class PlaywrightCollector:
    """
    Collector using Playwright for JavaScript-rendered pages.
    
    Requires: pip install playwright
    Then run: playwright install chromium
    """
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self._browser = None
        self._playwright = None
    
    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
    
    def fetch_rendered_page(self, url: str, wait_for: str = None) -> str:
        """
        Fetch a page with full JavaScript rendering.
        
        Args:
            url: URL to fetch
            wait_for: Optional CSS selector to wait for before capturing content
        
        Returns:
            Rendered HTML content
        """
        # Rate limiting
        time.sleep(self.config.request_delay)
        
        page = self._browser.new_page()
        page.set_extra_http_headers({
            'User-Agent': self.config.user_agent
        })
        
        try:
            logger.info(f"Fetching (Playwright): {url}")
            page.goto(url, timeout=self.config.request_timeout * 1000)
            
            if wait_for:
                page.wait_for_selector(wait_for, timeout=10000)
            else:
                # Wait for network to be mostly idle
                page.wait_for_load_state('networkidle')
            
            return page.content()
        finally:
            page.close()


# =============================================================================
# NLP Extraction Helpers
# =============================================================================

class CapacityExtractor:
    """
    Extract capacity indicators from policy document text.
    
    Uses rule-based patterns and keyword matching.
    For production, consider using spaCy NER or transformer models.
    """
    
    # Keywords for enforcement powers
    ENFORCEMENT_KEYWORDS = {
        'high': ['fine', 'penalty', 'investigation', 'audit', 'injunction', 
                 'prohibition', 'sanction', 'enforcement action'],
        'medium': ['guidance', 'recommendation', 'advisory', 'best practice'],
        'low': ['voluntary', 'self-regulation', 'principles']
    }
    
    # Keywords for institutional capacity
    INSTITUTION_KEYWORDS = [
        'agency', 'authority', 'commission', 'office', 'board',
        'ministry', 'department', 'regulator', 'council', 'committee'
    ]
    
    # Keywords for technical expertise
    EXPERTISE_KEYWORDS = [
        'technical staff', 'data scientist', 'engineer', 'researcher',
        'phd', 'expertise', 'laboratory', 'research unit'
    ]
    
    def extract_enforcement_score(self, text: str) -> Dict[str, Any]:
        """
        Score enforcement powers (0-3) based on keyword presence.
        """
        text_lower = text.lower()
        
        high_count = sum(1 for kw in self.ENFORCEMENT_KEYWORDS['high'] 
                        if kw in text_lower)
        medium_count = sum(1 for kw in self.ENFORCEMENT_KEYWORDS['medium'] 
                          if kw in text_lower)
        
        if high_count >= 3:
            score = 3
        elif high_count >= 1:
            score = 2
        elif medium_count >= 2:
            score = 1
        else:
            score = 0
        
        return {
            'score': score,
            'confidence': 'low',  # Rule-based = low confidence
            'high_keywords_found': high_count,
            'medium_keywords_found': medium_count,
            'method': 'keyword_matching'
        }
    
    def extract_institution_mentions(self, text: str) -> List[str]:
        """
        Extract mentions of institutional bodies.
        
        For production, use spaCy NER with custom training.
        """
        import re
        
        institutions = []
        text_lower = text.lower()
        
        # Simple pattern: capitalized words followed by institution keywords
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:' + '|'.join(self.INSTITUTION_KEYWORDS) + ')'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            institutions.append(match.group(0))
        
        return list(set(institutions))
    
    def extract_budget_mentions(self, text: str) -> List[Dict]:
        """
        Extract budget/funding amounts from text.
        """
        import re
        
        # Patterns for currency amounts
        patterns = [
            r'[\$€£]\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|m|bn)?',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion)\s*(?:dollars|euros|pounds)',
        ]
        
        amounts = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amounts.append({
                    'raw': match.group(0),
                    'amount': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else None
                })
        
        return amounts


# =============================================================================
# Main Pipeline
# =============================================================================

def run_oecd_collection_demo(output_dir: str = "./data"):
    """
    Demonstration of the OECD.AI collection pipeline.
    
    This is a minimal working example. For production:
    1. Verify robots.txt compliance
    2. Check for official API/export options
    3. Use Playwright for JS-rendered content
    4. Implement more robust parsing
    """
    config = ScraperConfig(
        output_dir=Path(output_dir) / "raw",
        cache_dir=Path(output_dir) / "cache"
    )
    
    client = EthicalHTTPClient(config)
    collector = OECDAICollector(client, config)
    
    # Step 1: Check robots.txt
    logger.info("=" * 60)
    logger.info("STEP 1: Checking robots.txt")
    logger.info("=" * 60)
    robots = collector.check_robots_txt()
    if robots.get('found'):
        print("\nrobots.txt content (first 1000 chars):")
        print(robots['content'][:1000])
    
    # Step 2: Document API/export options
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: API/Export options")
    logger.info("=" * 60)
    collector.check_for_api_or_export()
    
    # Step 3: Attempt basic index fetch (may require Playwright)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Fetching policy index (basic HTML)")
    logger.info("=" * 60)
    policies = collector.fetch_policy_index()
    
    if policies:
        print(f"\nFound {len(policies)} policies")
        for p in policies[:5]:
            print(f"  - {p['title']} ({p['jurisdiction']})")
    else:
        print("\nNo policies extracted with basic parsing.")
        print("This site likely requires JavaScript rendering.")
        print("Use PlaywrightCollector for full content.")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo complete. Review output and logs.")
    logger.info("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Governance Observatory - Data Collection Pipeline"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration of OECD.AI collection"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_oecd_collection_demo(args.output)
    else:
        print("Use --demo to run OECD.AI collection demonstration")
        print("Or import this module and use collectors programmatically")
