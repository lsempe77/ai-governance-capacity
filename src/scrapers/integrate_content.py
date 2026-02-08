"""
Integrate prior downloads from data/content/ with current data/pdfs/ downloads.

The data/content/ folder has files from a prior download campaign with different ID schemes:
- 8-char hex IDs based on source URL hash
- Files in text/, html/, pdfs/ subfolders

The data/pdfs/ folder has files from current campaign:
- 12-char hex IDs based on md5(entry['url'])[:12]

This script:
1. Loads the corpus and builds lookup by title
2. Scans data/content/text/ for extracted text (most useful)
3. Matches content files to corpus entries
4. For entries NOT covered in data/pdfs/, copies or creates content from data/content/
5. Updates the corpus with the text content for entries that only had snippets
"""

import json
import os
import hashlib
import shutil
import re
from pathlib import Path
from collections import defaultdict

BASE = r"c:\Users\LucasSempe\OneDrive - International Initiative for Impact Evaluation\Desktop\Gen AI tools\AI_policies\observatory"
CORPUS_PATH = os.path.join(BASE, "data", "corpus", "corpus_master_20260127.json")
CONTENT_DIR = os.path.join(BASE, "data", "content")
PDFS_DIR = os.path.join(BASE, "data", "pdfs")
PROGRESS_PATH = os.path.join(PDFS_DIR, "download_progress.json")

def entry_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

def normalize_title(t):
    """Normalize title for fuzzy matching."""
    t = t.lower().strip()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t

def extract_title_from_filename(fname):
    """Extract title from filename like 'Title_Name_abc12345.ext'"""
    base = os.path.splitext(fname)[0]
    # Remove the 8-char hex ID at the end
    match = re.match(r'^(.+)_([a-f0-9]{8})$', base)
    if match:
        title_part = match.group(1)
        file_id = match.group(2)
        # Convert underscores back to spaces
        title = title_part.replace('_', ' ')
        return title, file_id
    return base.replace('_', ' '), None

def main():
    print("=" * 70)
    print("  INTEGRATING data/content/ WITH data/pdfs/")
    print("=" * 70)
    
    # 1. Load corpus
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    corpus = corpus_data['entries']
    print(f"\nCorpus: {len(corpus)} entries")
    
    # Build lookup: normalized_title -> list of corpus entries
    title_to_entries = defaultdict(list)
    url_to_entry = {}
    id_to_entry = {}
    for entry in corpus:
        eid = entry_id(entry['url'])
        id_to_entry[eid] = entry
        url_to_entry[entry['url']] = entry
        nt = normalize_title(entry['title'])
        title_to_entries[nt].append(entry)
    
    # 2. Find entries already covered in data/pdfs/
    pdfs_files = os.listdir(PDFS_DIR)
    covered_ids = set()
    for f in pdfs_files:
        if f.endswith('.json'):
            continue
        # Extract 12-char ID from filename: "id_title.ext"
        parts = f.split('_', 1)
        if len(parts) >= 1 and len(parts[0]) == 12:
            covered_ids.add(parts[0])
    
    print(f"Already covered in data/pdfs/: {len(covered_ids)} entries")
    
    # 3. Scan data/content/text/ files (extracted text - most useful!)
    text_dir = os.path.join(CONTENT_DIR, "text")
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    print(f"\ndata/content/text/ files: {len(text_files)}")
    
    # 4. Match content text files to corpus entries by title
    matched = 0
    unmatched = 0
    new_content = 0
    upgraded = 0
    too_small = 0
    already_covered = 0
    
    content_matches = []  # (corpus_entry, text_file_path, file_size, content_file_id)
    
    for tf in text_files:
        title, file_id = extract_title_from_filename(tf)
        nt = normalize_title(title)
        
        # Try exact normalized title match
        if nt in title_to_entries:
            entries = title_to_entries[nt]
            for entry in entries:
                eid = entry_id(entry['url'])
                text_path = os.path.join(text_dir, tf)
                fsize = os.path.getsize(text_path)
                content_matches.append((entry, text_path, fsize, file_id, eid))
            matched += 1
        else:
            unmatched += 1
    
    print(f"Title matches: {matched}, Unmatched: {unmatched}")
    print(f"Total match pairs (entry, file): {len(content_matches)}")
    
    # 5. Also check data/content/pdfs/ for entries not covered
    pdf_content_dir = os.path.join(CONTENT_DIR, "pdfs")
    pdf_content_files = [f for f in os.listdir(pdf_content_dir) if f.endswith('.pdf')]
    print(f"\ndata/content/pdfs/ files: {len(pdf_content_files)}")
    
    pdf_matches = []
    pdf_matched = 0
    for pf in pdf_content_files:
        title, file_id = extract_title_from_filename(pf)
        nt = normalize_title(title)
        if nt in title_to_entries:
            entries = title_to_entries[nt]
            for entry in entries:
                eid = entry_id(entry['url'])
                pdf_path = os.path.join(pdf_content_dir, pf)
                fsize = os.path.getsize(pdf_path)
                pdf_matches.append((entry, pdf_path, fsize, file_id, eid))
            pdf_matched += 1
    
    print(f"PDF title matches: {pdf_matched}")
    print(f"PDF match pairs: {len(pdf_matches)}")
    
    # 6. Analyze: which corpus entries can we fill from content?
    # Only care about entries NOT in covered_ids
    uncovered_ids = set(id_to_entry.keys()) - covered_ids
    print(f"\nUncovered corpus entries: {len(uncovered_ids)}")
    
    # Group content matches by entry ID
    text_by_eid = defaultdict(list)
    for entry, path, size, fid, eid in content_matches:
        text_by_eid[eid].append((path, size, fid))
    
    pdf_by_eid = defaultdict(list)
    for entry, path, size, fid, eid in pdf_matches:
        pdf_by_eid[eid].append((path, size, fid))
    
    # Entries that are uncovered but have content matches
    fillable_text = set(text_by_eid.keys()) & uncovered_ids
    fillable_pdf = set(pdf_by_eid.keys()) & uncovered_ids
    fillable_any = fillable_text | fillable_pdf
    
    print(f"\nFillable from content/text/: {len(fillable_text)}")
    print(f"Fillable from content/pdfs/: {len(fillable_pdf)}")
    print(f"Fillable from either: {len(fillable_any)}")
    
    # Also check: entries already covered that might get BETTER content
    # (current data/pdfs might have small files, content/text might have full text)
    upgradeable = set(text_by_eid.keys()) & covered_ids
    print(f"Already covered but also in content/text/: {len(upgradeable)}")
    
    # 7. Copy useful PDFs from content/pdfs/ to data/pdfs/ for uncovered entries
    copied_pdfs = 0
    copied_texts = 0
    skipped_small = 0
    
    MIN_SIZE = 500  # Skip files smaller than 500 bytes (too small to be useful)
    
    for eid in fillable_any:
        entry = id_to_entry[eid]
        safe_title = re.sub(r'[^\w\s-]', '', entry['title'])[:80].strip().replace(' ', '_')
        
        # Prefer PDF if available and large enough
        best_source = None
        best_type = None
        
        if eid in pdf_by_eid:
            # Pick largest PDF
            pdfs = sorted(pdf_by_eid[eid], key=lambda x: x[1], reverse=True)
            if pdfs[0][1] >= MIN_SIZE:
                best_source = pdfs[0][0]
                best_type = 'pdf'
        
        if best_source is None and eid in text_by_eid:
            # Pick largest text
            texts = sorted(text_by_eid[eid], key=lambda x: x[1], reverse=True)
            if texts[0][1] >= MIN_SIZE:
                best_source = texts[0][0]
                best_type = 'txt'
        
        if best_source is None:
            skipped_small += 1
            continue
        
        # Copy to data/pdfs/ with the correct naming convention
        ext = '.pdf' if best_type == 'pdf' else '.html'
        dest = os.path.join(PDFS_DIR, f"{eid}_{safe_title}{ext}")
        
        if not os.path.exists(dest):
            shutil.copy2(best_source, dest)
            if best_type == 'pdf':
                copied_pdfs += 1
            else:
                copied_texts += 1
    
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"Copied PDFs to data/pdfs/: {copied_pdfs}")
    print(f"Copied texts to data/pdfs/: {copied_texts}")
    print(f"Skipped (too small <{MIN_SIZE}b): {skipped_small}")
    print(f"Total new files added: {copied_pdfs + copied_texts}")
    
    # 8. Update progress tracking
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    else:
        progress = {}
    
    progress['content_integrated'] = copied_pdfs + copied_texts
    progress['content_pdfs'] = copied_pdfs
    progress['content_texts'] = copied_texts
    
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)
    
    # 9. Final count
    final_files = len([f for f in os.listdir(PDFS_DIR) if not f.endswith('.json')])
    total_size = sum(os.path.getsize(os.path.join(PDFS_DIR, f)) 
                     for f in os.listdir(PDFS_DIR) if not f.endswith('.json'))
    
    print(f"\nFinal data/pdfs/ count: {final_files} files, {total_size/1024/1024:.0f} MB")
    print(f"Corpus coverage: {final_files}/{len(corpus)} ({final_files/len(corpus)*100:.1f}%)")
    
    # 10. Also output: which entries are STILL missing (no content anywhere)
    all_covered_now = set()
    for f in os.listdir(PDFS_DIR):
        if f.endswith('.json'):
            continue
        parts = f.split('_', 1)
        if len(parts) >= 1 and len(parts[0]) == 12:
            all_covered_now.add(parts[0])
    
    still_missing = set(id_to_entry.keys()) - all_covered_now
    print(f"Still missing: {len(still_missing)} entries")
    
    # Show breakdown of missing entries by word_count
    missing_entries = [id_to_entry[eid] for eid in still_missing]
    snippet_missing = [e for e in missing_entries if e.get('word_count', 0) < 500]
    substantial_missing = [e for e in missing_entries if e.get('word_count', 0) >= 500]
    
    print(f"  - Missing with snippet content (<500 words): {len(snippet_missing)}")
    print(f"  - Missing with substantial content (>=500 words): {len(substantial_missing)}")
    
    # Jurisdictions of missing
    from collections import Counter
    jur_counts = Counter(e.get('jurisdiction', 'Unknown') for e in missing_entries)
    print(f"\nTop 15 missing jurisdictions:")
    for j, c in jur_counts.most_common(15):
        print(f"  {j}: {c}")

if __name__ == "__main__":
    main()
