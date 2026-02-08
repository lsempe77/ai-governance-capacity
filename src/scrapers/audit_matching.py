"""Audit the robustness of PDF-to-corpus matching."""
import os, json, hashlib
from collections import defaultdict, Counter

BASE = r"c:\Users\LucasSempe\OneDrive - International Initiative for Impact Evaluation\Desktop\Gen AI tools\AI_policies\observatory"
CORPUS_PATH = os.path.join(BASE, "data/corpus/corpus_master_20260127.json")
PDFS_DIR = os.path.join(BASE, "data/pdfs")

corpus = json.load(open(CORPUS_PATH, "r", encoding="utf-8"))["entries"]
id_to_entry = {}
for e in corpus:
    eid = hashlib.md5(e["url"].encode()).hexdigest()[:12]
    id_to_entry[eid] = e
    e["_id"] = eid

# ── 1. Check for duplicate titles in corpus ──
title_entries = defaultdict(list)
for e in corpus:
    title_entries[e["title"]].append(e["_id"])

dup_titles = {t: ids for t, ids in title_entries.items() if len(ids) > 1}
print(f"Unique titles in corpus: {len(title_entries)}")
print(f"Duplicate titles (>1 entry): {len(dup_titles)} titles, {sum(len(v) for v in dup_titles.values())} entries")

if dup_titles:
    print("\nDuplicate titles (potential mis-assignment from title matching):")
    for t, ids in sorted(dup_titles.items()):
        entries = [id_to_entry[eid] for eid in ids]
        jurisdictions = [e.get("jurisdiction", "?") for e in entries]
        print(f'  "{t[:70]}"')
        print(f"    IDs: {ids}, Jurisdictions: {jurisdictions}")
        for eid in ids:
            matching_files = [f for f in os.listdir(PDFS_DIR) if f.startswith(eid + "_")]
            print(f"    {eid}: {len(matching_files)} file(s)")

# ── 2. Check provenance: which files came from which source ──
# Read the progress file to see what came from phase1/phase2/retry3 vs content integration
try:
    prog = json.load(open(os.path.join(PDFS_DIR, "download_progress.json"), "r", encoding="utf-8"))
except:
    prog = {}

phase1 = set(prog.get("phase1_downloaded", {}).keys())
phase2 = set(prog.get("phase2_downloaded", {}).keys())
retry3 = set(prog.get("retry3_downloaded", {}).keys())
tracked = phase1 | phase2 | retry3

# Files on disk
all_files = [f for f in os.listdir(PDFS_DIR) if not f.endswith(".json") and not f.endswith(".txt") and f != "UNESCOrecomendation.pdf"]
file_ids = set()
for f in all_files:
    parts = f.split("_", 1)
    if len(parts) >= 1 and len(parts[0]) == 12 and all(c in "0123456789abcdef" for c in parts[0]):
        file_ids.add(parts[0])

untracked = file_ids - tracked
tracked_only = tracked - file_ids  # in progress but no file on disk

print(f"\n── PROVENANCE ──")
print(f"Phase 1 tracked: {len(phase1)}")
print(f"Phase 2 tracked: {len(phase2)}")
print(f"Retry3 tracked:  {len(retry3)}")
print(f"All tracked:     {len(tracked)}")
print(f"On disk (unique IDs): {len(file_ids)}")
print(f"Untracked on disk (from content integration): {len(untracked)}")
print(f"Tracked but no file: {len(tracked_only)}")

# ── 3. Verify content integration matches were correct ──
# For untracked files (came from content integration via title matching),
# spot-check by comparing titles
print(f"\n── CONTENT INTEGRATION VERIFICATION ──")
print(f"Checking {len(untracked)} files that came from title-based matching...")

wrong_matches = []
for eid in untracked:
    entry = id_to_entry.get(eid)
    if not entry:
        wrong_matches.append((eid, "NO ENTRY IN CORPUS", ""))
        continue
    
    # Find the file
    matching = [f for f in all_files if f.startswith(eid + "_")]
    if not matching:
        continue
    
    fname = matching[0]
    # Extract title from filename
    name_part = fname.split("_", 1)[1] if "_" in fname else fname
    file_title = os.path.splitext(name_part)[0].replace("_", " ")
    
    corpus_title = entry["title"]
    
    # Normalize for comparison
    def norm(s):
        import re
        return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
    
    fn = norm(file_title)
    cn = norm(corpus_title)
    
    # Check if they're similar enough
    if fn[:25] != cn[:25]:
        wrong_matches.append((eid, file_title[:50], corpus_title[:50]))

print(f"Potential mismatches: {len(wrong_matches)}")
for eid, ft, ct in wrong_matches[:15]:
    print(f"  {eid}: File='{ft}' vs Corpus='{ct}'")

# ── 4. Check files that are actually usable (not empty/tiny) ──
print(f"\n── USABILITY SUMMARY ──")
usable = 0
problematic = 0
for f in all_files:
    fpath = os.path.join(PDFS_DIR, f)
    sz = os.path.getsize(fpath)
    if sz >= 1000:
        usable += 1
    else:
        problematic += 1

unique_usable_ids = set()
for f in all_files:
    fpath = os.path.join(PDFS_DIR, f)
    sz = os.path.getsize(fpath)
    parts = f.split("_", 1)
    if len(parts) >= 1 and len(parts[0]) == 12:
        eid = parts[0]
        if sz >= 1000:
            unique_usable_ids.add(eid)

print(f"Total files: {len(all_files)}")
print(f"Usable (>=1KB): {usable}")
print(f"Problematic (<1KB): {problematic}")
print(f"Unique corpus entries with usable file: {len(unique_usable_ids)} / {len(id_to_entry)} ({100*len(unique_usable_ids)/len(id_to_entry):.1f}%)")

# ── 5. Final coverage breakdown ──
print(f"\n── FINAL COVERAGE ──")
covered_ids = file_ids & set(id_to_entry.keys())
missing_ids = set(id_to_entry.keys()) - file_ids
print(f"Corpus entries with file:    {len(covered_ids)} ({100*len(covered_ids)/len(id_to_entry):.1f}%)")
print(f"Corpus entries without file: {len(missing_ids)} ({100*len(missing_ids)/len(id_to_entry):.1f}%)")
print(f"Corpus entries with USABLE file: {len(unique_usable_ids)} ({100*len(unique_usable_ids)/len(id_to_entry):.1f}%)")
