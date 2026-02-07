"""
Build Unified Text Corpus for Content Analysis
Combines all available text content from:
1. OECD policy descriptions (all 2,211 policies)
2. Downloaded PDFs (extracted text)
3. Downloaded webpages (extracted text)

Output: One text file per policy + master corpus file

Usage:
    python build_corpus.py --policies data/oecd/oecd_policies_20260126_201311.json
    python build_corpus.py --policies data/oecd/oecd_policies_20260126_201311.json --extract-pdfs
"""

import json
import os
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        logger.warning("pypdf not installed. Run: pip install pypdf")
        return ""
    except Exception as e:
        logger.warning(f"Error extracting PDF text: {e}")
        return ""


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    return text.strip()


def build_corpus(policies_file: str, content_dir: str, output_dir: str, 
                 extract_pdfs: bool = False, enriched_file: str = None) -> Dict:
    """Build unified text corpus from all sources."""
    
    output_path = Path(output_dir)
    corpus_dir = output_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    content_path = Path(content_dir)
    
    # Load policies
    with open(policies_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    policies = data['policies']
    
    # Load enriched data if provided
    enriched_lookup = {}
    if enriched_file and Path(enriched_file).exists():
        logger.info(f"Loading enriched data from {enriched_file}")
        with open(enriched_file, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
        for ep in enriched_data.get('policies', []):
            enriched_lookup[ep['url']] = ep
        logger.info(f"Loaded {len(enriched_lookup)} enriched policies")
    
    logger.info(f"Processing {len(policies)} policies")
    
    # Index downloaded content by URL
    downloaded_text = {}
    
    # Load text from downloaded webpages
    text_dir = content_path / "text"
    if text_dir.exists():
        for txt_file in text_dir.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract URL from file
                for line in content.split('\n')[:5]:
                    if line.startswith('Source:'):
                        url = line.replace('Source:', '').strip()
                        downloaded_text[url] = content
                        break
    
    # Load text from PDFs if requested
    if extract_pdfs:
        pdf_dir = content_path / "pdfs"
        if pdf_dir.exists():
            logger.info("Extracting text from PDFs...")
            for pdf_file in pdf_dir.glob("*.pdf"):
                text = extract_text_from_pdf(str(pdf_file))
                if text:
                    # Map back to policy by filename
                    downloaded_text[str(pdf_file)] = text
    
    # Build corpus
    corpus_entries = []
    stats = {
        'total': len(policies),
        'with_external_content': 0,
        'oecd_only': 0,
        'total_words': 0,
        'by_jurisdiction': {}
    }
    
    for policy in policies:
        title = policy['title']
        jurisdiction = policy.get('jurisdiction', 'Unknown')
        year = policy.get('start_year', '')
        url = policy.get('url', '')
        source_url = policy.get('source_url', '')
        description = policy.get('description', '')
        
        # Determine best content source (priority: downloaded > enriched > original)
        content = ""
        content_source = "oecd_description"
        target_sectors = []
        ai_tags = []
        
        # Check if we have downloaded external content
        if source_url and source_url in downloaded_text:
            content = downloaded_text[source_url]
            content_source = "external_download"
            stats['with_external_content'] += 1
        # Check if we have enriched data (longer overview)
        elif url in enriched_lookup:
            enriched = enriched_lookup[url]
            content = enriched.get('initiative_overview', '') or description
            content_source = "oecd_enriched"
            target_sectors = enriched.get('target_sectors', [])
            ai_tags = enriched.get('ai_tags', [])
            stats['with_external_content'] += 1  # Count enriched as "external" improvement
        else:
            content = description
            stats['oecd_only'] += 1
        
        content = clean_text(content)
        word_count = len(content.split())
        stats['total_words'] += word_count
        
        # Track by jurisdiction
        if jurisdiction not in stats['by_jurisdiction']:
            stats['by_jurisdiction'][jurisdiction] = {'count': 0, 'words': 0}
        stats['by_jurisdiction'][jurisdiction]['count'] += 1
        stats['by_jurisdiction'][jurisdiction]['words'] += word_count
        
        # Create corpus entry
        entry = {
            'title': title,
            'jurisdiction': jurisdiction,
            'year': year,
            'url': url,
            'source_url': source_url,
            'content_source': content_source,
            'target_sectors': target_sectors,
            'ai_tags': ai_tags,
            'word_count': word_count,
            'content': content
        }
        corpus_entries.append(entry)
        
        # Save individual file
        safe_name = re.sub(r'[<>:"/\\|?*]', '', title)[:60]
        safe_name = re.sub(r'\s+', '_', safe_name)
        txt_file = corpus_dir / f"{safe_name}.txt"
        
        sectors_str = ', '.join(target_sectors) if target_sectors else 'N/A'
        tags_str = ', '.join(ai_tags) if ai_tags else 'N/A'
        
        file_content = f"""TITLE: {title}
JURISDICTION: {jurisdiction}
YEAR: {year}
URL: {url}
SOURCE: {source_url or 'OECD.AI'}
CONTENT_SOURCE: {content_source}
TARGET_SECTORS: {sectors_str}
AI_TAGS: {tags_str}
WORD_COUNT: {word_count}
{'='*80}

{content}
"""
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(file_content)
    
    # Save master corpus JSON
    master_file = output_path / f"corpus_master_{datetime.now().strftime('%Y%m%d')}.json"
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump({
            'created': datetime.now().isoformat(),
            'stats': stats,
            'entries': corpus_entries
        }, f, indent=2, ensure_ascii=False)
    
    # Save corpus as single text file (for tools like topic modeling)
    combined_file = output_path / f"corpus_combined_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for entry in corpus_entries:
            f.write(f"### {entry['title']} | {entry['jurisdiction']} | {entry['year']}\n")
            f.write(entry['content'])
            f.write("\n\n---\n\n")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("CORPUS BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total policies: {stats['total']}")
    logger.info(f"With external content: {stats['with_external_content']}")
    logger.info(f"OECD description only: {stats['oecd_only']}")
    logger.info(f"Total words: {stats['total_words']:,}")
    logger.info(f"Average words/policy: {stats['total_words'] // stats['total']}")
    logger.info(f"Individual files: {corpus_dir}")
    logger.info(f"Master JSON: {master_file}")
    logger.info(f"Combined text: {combined_file}")
    
    # Top jurisdictions by content
    logger.info("\nTop 10 jurisdictions by word count:")
    sorted_j = sorted(stats['by_jurisdiction'].items(), key=lambda x: x[1]['words'], reverse=True)
    for j, data in sorted_j[:10]:
        logger.info(f"  {j}: {data['count']} policies, {data['words']:,} words")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Build unified text corpus')
    parser.add_argument('--policies', required=True, help='Policies JSON file')
    parser.add_argument('--enriched', help='Enriched policies JSON file (from oecd_rescrape_details.py)')
    parser.add_argument('--content', default='./data/content', help='Downloaded content directory')
    parser.add_argument('--output', default='./data/corpus', help='Output directory')
    parser.add_argument('--extract-pdfs', action='store_true', help='Extract text from PDFs')
    args = parser.parse_args()
    
    build_corpus(args.policies, args.content, args.output, args.extract_pdfs, args.enriched)


if __name__ == '__main__':
    main()
