"""
Phase 1: Text Extraction Pipeline
===================================
Extracts clean text from all downloaded policy documents (HTML, PDF, DOCX, CFM)
and produces an enriched corpus with full-text content.

Input:
  - data/corpus/corpus_master_20260127.json  (2,216 entries with OECD snippets)
  - data/pdfs/  (~2,085 files: .html, .pdf, .docx, .cfm)

Output:
  - data/corpus/corpus_enriched.json  (corpus with full text)
  - data/analysis/extraction_report.json  (quality metrics)

Usage:
  python src/analysis/extract_text.py
  python src/analysis/extract_text.py --limit 50          # test on first 50
  python src/analysis/extract_text.py --resume             # resume from checkpoint
"""

import json
import hashlib
import os
import re
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = ROOT / 'data' / 'corpus' / 'corpus_master_20260127.json'
PDFS_DIR = ROOT / 'data' / 'pdfs'
OUTPUT_CORPUS = ROOT / 'data' / 'corpus' / 'corpus_enriched.json'
REPORT_PATH = ROOT / 'data' / 'analysis' / 'extraction_report.json'
CHECKPOINT_PATH = ROOT / 'data' / 'analysis' / 'extraction_checkpoint.json'


# ─── Lazy imports (only load heavy libs when needed) ───────────────────────────
def get_trafilatura():
    """Extract main content from HTML using trafilatura."""
    import trafilatura
    return trafilatura

def get_bs4():
    from bs4 import BeautifulSoup
    return BeautifulSoup

def get_fitz():
    import fitz  # PyMuPDF
    return fitz

def get_docx():
    import docx
    return docx

def detect_language(text):
    """Detect language of text, returns ISO 639-1 code or 'unknown'."""
    try:
        from langdetect import detect
        # Use first 2000 chars for speed
        return detect(text[:2000])
    except Exception:
        return 'unknown'


# ─── Quality thresholds ───────────────────────────────────────────────────────
QUALITY_THRESHOLDS = {
    'good': 500,       # >500 words — suitable for deep AI analysis
    'thin': 100,       # 100-500 words — usable but limited depth
    'stub': 1,         # 1-99 words — too short, flag for review
    'empty': 0,        # 0 words — extraction failed entirely
}

def classify_quality(word_count: int) -> str:
    """Classify text quality based on word count."""
    if word_count >= QUALITY_THRESHOLDS['good']:
        return 'good'
    elif word_count >= QUALITY_THRESHOLDS['thin']:
        return 'thin'
    elif word_count >= QUALITY_THRESHOLDS['stub']:
        return 'stub'
    return 'empty'

def quality_usable(quality: str) -> bool:
    """Return True if text quality is sufficient for deep analysis."""
    return quality in ('good', 'thin')


# ─── ID generation (must match scraper convention) ─────────────────────────────
def entry_id(url: str) -> str:
    """Generate 12-char hex ID from URL, matching the scraper convention."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ─── HTML Extraction ──────────────────────────────────────────────────────────
def extract_html(filepath: Path) -> dict:
    """
    Extract text from an HTML file. Handles two variants:
    1. Text-with-metadata-header (starts with "Source:") — already extracted text
    2. Raw HTML pages — use trafilatura for main content extraction
    """
    try:
        raw = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        return {'text': '', 'method': 'html_error', 'error': str(e)}

    # Variant 1: Pre-extracted text with metadata header
    if raw.startswith('Source:'):
        # Format: Source: ...\nTitle: ...\nDownloaded: ...\n===...===\n<text>
        lines = raw.split('\n')
        text_start = 0
        for i, line in enumerate(lines):
            if line.startswith('=' * 10):
                text_start = i + 1
                break
        text = '\n'.join(lines[text_start:]).strip()
        if text:
            return {'text': text, 'method': 'html_preextracted'}

    # Variant 2: Raw HTML — use trafilatura (best at extracting article content)
    try:
        trafilatura = get_trafilatura()
        text = trafilatura.extract(
            raw,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True
        )
        if text and len(text.strip()) > 50:
            return {'text': text.strip(), 'method': 'html_trafilatura'}
    except Exception:
        pass

    # Fallback: BeautifulSoup get_text()
    try:
        BeautifulSoup = get_bs4()
        soup = BeautifulSoup(raw, 'lxml')
        # Remove script, style, nav, footer, header elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        if text and len(text.strip()) > 50:
            return {'text': text.strip(), 'method': 'html_beautifulsoup'}
    except Exception as e:
        return {'text': '', 'method': 'html_error', 'error': str(e)}

    return {'text': raw.strip()[:5000] if raw.strip() else '', 'method': 'html_raw_truncated'}


# ─── PDF Extraction ───────────────────────────────────────────────────────────
def extract_pdf(filepath: Path) -> dict:
    """
    Extract text from PDF using PyMuPDF.
    Falls back to reporting if the PDF appears to be scanned (image-only).
    """
    try:
        fitz = get_fitz()
        doc = fitz.open(str(filepath))
        pages_text = []
        page_count = len(doc)
        for page in doc:
            text = page.get_text('text')
            if text:
                pages_text.append(text)
        doc.close()

        full_text = '\n\n'.join(pages_text).strip()

        # Clean common PDF artifacts
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r'[ \t]{2,}', ' ', full_text)
        # Remove page numbers (standalone lines with just a number)
        full_text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', full_text)

        if len(full_text) > 50:
            return {
                'text': full_text,
                'method': 'pdf_pymupdf',
                'pages': len(pages_text)
            }
        else:
            # Likely a scanned PDF — mark for OCR
            return {
                'text': '',
                'method': 'pdf_scanned',
                'pages': page_count,
                'note': 'Scanned PDF, needs OCR'
            }
    except Exception as e:
        return {'text': '', 'method': 'pdf_error', 'error': str(e)}


# ─── DOCX Extraction ──────────────────────────────────────────────────────────
def extract_docx(filepath: Path) -> dict:
    """Extract text from DOCX using python-docx."""
    try:
        docx = get_docx()
        doc = docx.Document(str(filepath))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = '\n\n'.join(paragraphs).strip()
        return {'text': text, 'method': 'docx_python_docx'} if text else {'text': '', 'method': 'docx_empty'}
    except Exception as e:
        return {'text': '', 'method': 'docx_error', 'error': str(e)}


# ─── CFM Extraction (ColdFusion → treat as HTML) ──────────────────────────────
def extract_cfm(filepath: Path) -> dict:
    """CFM files are ColdFusion output — treat as HTML."""
    result = extract_html(filepath)
    result['method'] = 'cfm_as_' + result.get('method', 'unknown')
    return result


# ─── Router ────────────────────────────────────────────────────────────────────
EXTRACTORS = {
    '.html': extract_html,
    '.pdf': extract_pdf,
    '.docx': extract_docx,
    '.cfm': extract_cfm,
}

def extract_file(filepath: Path) -> dict:
    """Route file to appropriate extractor based on extension."""
    ext = filepath.suffix.lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        return {'text': '', 'method': 'unsupported', 'error': f'No extractor for {ext}'}
    return extractor(filepath)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────
def build_file_index(pdfs_dir: Path) -> dict:
    """
    Build a mapping: entry_id → filepath for all document files.
    Filename convention: {entry_id}_{title_slug}.{ext}
    If multiple files for same ID, prefer PDF > HTML > DOCX > CFM.
    """
    index = {}
    priority = {'.pdf': 1, '.docx': 2, '.html': 3, '.cfm': 4}

    for f in pdfs_dir.iterdir():
        if f.name == 'download_progress.json':
            continue
        if not f.is_file():
            continue
        # Extract ID from filename (first 12 chars before underscore)
        parts = f.name.split('_', 1)
        if len(parts) < 2:
            continue
        fid = parts[0]
        if len(fid) != 12:
            continue

        ext = f.suffix.lower()
        file_priority = priority.get(ext, 99)

        if fid not in index or file_priority < priority.get(index[fid].suffix.lower(), 99):
            index[fid] = f

    return index


def run_pipeline(limit=None, resume=False):
    """Main extraction pipeline."""
    log.info("=" * 70)
    log.info("PHASE 1: TEXT EXTRACTION PIPELINE")
    log.info("=" * 70)

    # Load corpus
    log.info(f"Loading corpus from {CORPUS_PATH}")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    entries = corpus_data['entries']
    log.info(f"  Corpus: {len(entries)} entries")

    # Build file index
    log.info(f"Building file index from {PDFS_DIR}")
    file_index = build_file_index(PDFS_DIR)
    log.info(f"  Files indexed: {len(file_index)}")

    # Load checkpoint if resuming
    completed_ids = set()
    results_cache = {}
    if resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        completed_ids = set(checkpoint.get('completed_ids', []))
        results_cache = checkpoint.get('results', {})
        log.info(f"  Resuming: {len(completed_ids)} already processed")

    # Counters for reporting
    stats = Counter()
    method_counts = Counter()
    results = dict(results_cache)  # copy from checkpoint
    errors = []

    # Process entries
    start_time = time.time()
    entries_to_process = entries if limit is None else entries[:limit]
    total = len(entries_to_process)

    for i, entry in enumerate(entries_to_process):
        eid = entry_id(entry['url'])

        # Skip if already processed (resume mode)
        if eid in completed_ids:
            stats['skipped_checkpoint'] += 1
            continue

        # Progress logging
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1 - stats['skipped_checkpoint']) / max(elapsed, 1)
            log.info(f"  [{i+1}/{total}] ({rate:.1f} files/sec)")

        # Check if we have a file for this entry
        filepath = file_index.get(eid)

        try:
            if filepath is None:
                # No file — use OECD snippet as fallback
                snippet = entry.get('content', '').strip()
                wc = len(snippet.split()) if snippet else 0
                result = {
                    'text': snippet,
                    'method': 'oecd_snippet_only',
                    'word_count_before': wc,
                    'word_count_after': wc,
                    'text_quality': classify_quality(wc),
                }
                stats['snippet_only'] += 1
            else:
                # Extract text from file
                result = extract_file(filepath)
                result['source_file'] = filepath.name
                result['file_ext'] = filepath.suffix.lower()
                try:
                    result['file_size_kb'] = round(filepath.stat().st_size / 1024, 1)
                except OSError:
                    result['file_size_kb'] = 0

                # Word counts
                snippet_wc = len(entry.get('content', '').split()) if entry.get('content') else 0
                extracted_wc = len(result['text'].split()) if result['text'] else 0
                result['word_count_before'] = snippet_wc
                result['word_count_after'] = extracted_wc

                # Detect language
                if result['text'] and len(result['text']) > 100:
                    result['language'] = detect_language(result['text'])

                # Classify quality
                quality = classify_quality(extracted_wc)
                result['text_quality'] = quality
                stats[f'quality_{quality}'] += 1

                # Classify outcome
                if extracted_wc > 100:
                    stats['success'] += 1
                elif extracted_wc > 0:
                    stats['partial'] += 1
                else:
                    stats['failed'] += 1
                    errors.append({
                        'entry_id': eid,
                        'title': entry.get('title', ''),
                        'file': filepath.name,
                        'method': result.get('method', ''),
                        'error': result.get('error', 'Empty extraction')
                    })

        except KeyboardInterrupt:
            log.info(f"\n  Interrupted at entry {i+1}/{total}. Saving checkpoint...")
            save_checkpoint(completed_ids, results)
            raise
        except Exception as exc:
            log.warning(f"  Entry {eid} failed: {exc}")
            result = {
                'text': entry.get('content', '').strip(),
                'method': 'exception_fallback',
                'error': str(exc),
                'text_quality': classify_quality(len(entry.get('content', '').split()) if entry.get('content') else 0),
            }
            stats['failed'] += 1

        method_counts[result.get('method', 'unknown')] += 1
        results[eid] = result
        completed_ids.add(eid)

        # Save checkpoint every 200 entries
        if (i + 1) % 200 == 0:
            save_checkpoint(completed_ids, results)

    elapsed = time.time() - start_time
    log.info(f"\nExtraction complete in {elapsed:.0f}s")

    # ─── Build enriched corpus ─────────────────────────────────────────────
    log.info("Building enriched corpus...")
    enriched_entries = []
    for entry in entries:
        eid = entry_id(entry['url'])
        enriched = dict(entry)  # copy

        if eid in results and results[eid].get('text'):
            enriched['full_text'] = results[eid]['text']
            enriched['extraction_method'] = results[eid].get('method', '')
            enriched['extraction_word_count'] = results[eid].get('word_count_after', 0)
            enriched['text_quality'] = results[eid].get('text_quality', classify_quality(results[eid].get('word_count_after', 0)))
            enriched['analysis_ready'] = quality_usable(enriched['text_quality'])
            if 'language' in results[eid]:
                enriched['detected_language'] = results[eid]['language']
        else:
            # Fallback to OECD snippet
            snippet_text = entry.get('content', '')
            snippet_wc = len(snippet_text.split()) if snippet_text else 0
            enriched['full_text'] = snippet_text
            enriched['extraction_method'] = 'oecd_snippet_fallback'
            enriched['extraction_word_count'] = snippet_wc
            enriched['text_quality'] = classify_quality(snippet_wc)
            enriched['analysis_ready'] = quality_usable(enriched['text_quality'])

        enriched_entries.append(enriched)

    # Compute quality distribution across all enriched entries
    quality_dist = Counter(e.get('text_quality', 'empty') for e in enriched_entries)
    analysis_ready_count = sum(1 for e in enriched_entries if e.get('analysis_ready'))

    # Save enriched corpus
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    enriched_corpus = {
        'created': datetime.now().isoformat(),
        'source': str(CORPUS_PATH),
        'extraction_date': datetime.now().strftime('%Y-%m-%d'),
        'stats': {
            'total_entries': len(entries),
            'with_full_text': stats.get('success', 0),
            'partial_extraction': stats.get('partial', 0),
            'failed_extraction': stats.get('failed', 0),
            'snippet_only': stats.get('snippet_only', 0),
            'analysis_ready': analysis_ready_count,
            'quality_distribution': {
                'good_gt500w': quality_dist.get('good', 0),
                'thin_100_500w': quality_dist.get('thin', 0),
                'stub_lt100w': quality_dist.get('stub', 0),
                'empty_0w': quality_dist.get('empty', 0),
            },
        },
        'entries': enriched_entries
    }

    log.info(f"Saving enriched corpus to {OUTPUT_CORPUS}")
    with open(OUTPUT_CORPUS, 'w', encoding='utf-8') as f:
        json.dump(enriched_corpus, f, ensure_ascii=False, indent=1)

    # ─── Build extraction report ───────────────────────────────────────────
    word_counts = [r.get('word_count_after', 0) for r in results.values() if r.get('word_count_after', 0) > 0]
    languages = Counter(r.get('language', 'unknown') for r in results.values() if r.get('language'))

    report = {
        'generated': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'summary': {
            'total_entries': len(entries),
            'files_found': len(file_index),
            'successful_extractions': stats.get('success', 0),
            'partial_extractions': stats.get('partial', 0),
            'failed_extractions': stats.get('failed', 0),
            'snippet_only': stats.get('snippet_only', 0),
            'coverage_pct': round(100 * (stats.get('success', 0) + stats.get('partial', 0)) / max(len(entries), 1), 1),
        },
        'quality_distribution': {
            'good_gt500w': quality_dist.get('good', 0),
            'thin_100_500w': quality_dist.get('thin', 0),
            'stub_lt100w': quality_dist.get('stub', 0),
            'empty_0w': quality_dist.get('empty', 0),
            'analysis_ready': analysis_ready_count,
            'analysis_ready_pct': round(100 * analysis_ready_count / max(len(entries), 1), 1),
        },
        'by_method': dict(method_counts.most_common()),
        'word_counts': {
            'mean': round(sum(word_counts) / max(len(word_counts), 1)),
            'median': sorted(word_counts)[len(word_counts) // 2] if word_counts else 0,
            'min': min(word_counts) if word_counts else 0,
            'max': max(word_counts) if word_counts else 0,
            'total': sum(word_counts),
        },
        'languages': dict(languages.most_common(20)),
        'errors': errors[:50],  # cap at 50 for readability
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving extraction report to {REPORT_PATH}")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # ─── Print summary ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("EXTRACTION SUMMARY")
    log.info("=" * 70)
    log.info(f"  Total entries:        {len(entries)}")
    log.info(f"  Files found:          {len(file_index)}")
    log.info(f"  Successful (>100w):   {stats.get('success', 0)}")
    log.info(f"  Partial (1-100w):     {stats.get('partial', 0)}")
    log.info(f"  Failed (0w):          {stats.get('failed', 0)}")
    log.info(f"  Snippet-only:         {stats.get('snippet_only', 0)}")
    log.info(f"  Coverage:             {report['summary']['coverage_pct']}%")
    log.info(f"\n  Quality distribution:")
    log.info(f"    Good  (>500w):      {quality_dist.get('good', 0)}")
    log.info(f"    Thin  (100-500w):   {quality_dist.get('thin', 0)}")
    log.info(f"    Stub  (<100w):      {quality_dist.get('stub', 0)}  ⚠ too small for analysis")
    log.info(f"    Empty (0w):         {quality_dist.get('empty', 0)}  ⚠ extraction failed")
    log.info(f"    Analysis-ready:     {analysis_ready_count} ({report['quality_distribution']['analysis_ready_pct']}%)")
    log.info(f"\n  Avg word count:       {report['word_counts']['mean']}")
    log.info(f"  Total words:          {report['word_counts']['total']:,}")
    log.info(f"\n  Top methods:")
    for method, count in method_counts.most_common(10):
        log.info(f"    {method}: {count}")
    log.info(f"\n  Top languages:")
    for lang, count in languages.most_common(10):
        log.info(f"    {lang}: {count}")
    log.info("=" * 70)

    return report


def save_checkpoint(completed_ids, results):
    """Save progress checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'completed_ids': list(completed_ids),
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'text'} for k, v in results.items()},
    }
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    log.info(f"  Checkpoint saved: {len(completed_ids)} entries")


# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 1: Extract text from policy documents')
    parser.add_argument('--limit', type=int, default=None, help='Process only first N entries (for testing)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()

    report = run_pipeline(limit=args.limit, resume=args.resume)

    sys.exit(0 if report['summary']['successful_extractions'] > 0 else 1)
