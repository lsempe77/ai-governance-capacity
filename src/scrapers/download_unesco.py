"""Download UNESCO AI Ethics PDFs and add them to the corpus."""

import json
import requests
import os
from datetime import datetime
import re

# UNESCO documents to download
UNESCO_DOCS = [
    {
        "id": "pf0000381137",
        "title": "Recommendation on the Ethics of Artificial Intelligence",
        "year": 2021,
        "url": "https://unesdoc.unesco.org/ark:/48223/pf0000381137",
        "pdf_url": "https://unesdoc.unesco.org/ark:/48223/pf0000381137/PDF/381137eng.pdf.multi",
        "description": "The first-ever global standard on AI ethics, adopted by all 193 UNESCO Member States. Establishes 4 core values and 10 principles."
    },
    {
        "id": "pf0000385082",
        "title": "Implementing the Recommendation on the Ethics of Artificial Intelligence: Methodological Guidance",
        "year": 2023,
        "url": "https://unesdoc.unesco.org/ark:/48223/pf0000385082",
        "pdf_url": "https://unesdoc.unesco.org/ark:/48223/pf0000385082/PDF/385082eng.pdf.multi",
        "description": "Guidance document for implementing the UNESCO Recommendation, covering 11 key policy areas."
    },
    {
        "id": "pf0000385198",
        "title": "Readiness Assessment Methodology (RAM): A Tool of the Recommendation on the Ethics of Artificial Intelligence",
        "year": 2023,
        "url": "https://unesdoc.unesco.org/ark:/48223/pf0000385198",
        "pdf_url": "https://unesdoc.unesco.org/ark:/48223/pf0000385198/PDF/385198eng.pdf.multi",
        "description": "Methodology to assess whether Member States are prepared to implement AI ethics recommendations."
    },
    {
        "id": "pf0000386276",
        "title": "Ethical Impact Assessment (EIA): A Tool of the Recommendation on the Ethics of Artificial Intelligence",
        "year": 2023,
        "url": "https://unesdoc.unesco.org/ark:/48223/pf0000386276",
        "pdf_url": "https://unesdoc.unesco.org/ark:/48223/pf0000386276/PDF/386276eng.pdf.multi",
        "description": "Tool to evaluate whether a specific AI system aligns with UNESCO's Recommendation values and principles."
    },
    {
        "id": "pf0000390793",
        "title": "Governing AI for Humanity: Final Report of the UN High-Level Advisory Body on AI",
        "year": 2024,
        "url": "https://unesdoc.unesco.org/ark:/48223/pf0000390793",
        "pdf_url": "https://unesdoc.unesco.org/ark:/48223/pf0000390793/PDF/390793eng.pdf.multi",
        "description": "UN High-Level Advisory Body recommendations for global AI governance."
    },
]

# Alternative direct PDF URLs (UNESDOC sometimes requires different formats)
ALTERNATIVE_URLS = {
    "pf0000381137": [
        "https://unesdoc.unesco.org/ark:/48223/pf0000381137_eng",
        "https://unesdoc.unesco.org/in/documentViewer.xhtml?v=2.1.196&id=p::usmarcdef_0000381137&file=/in/rest/annotationSVC/DownloadWatermarkedAttachment/attach_import_17e45ab0-2b70-400e-b186-35726f133bdb%3F_%3D381137eng.pdf",
    ],
    "pf0000385082": [
        "https://unesdoc.unesco.org/ark:/48223/pf0000385082_eng",
    ],
    "pf0000385198": [
        "https://unesdoc.unesco.org/ark:/48223/pf0000385198_eng",
    ],
    "pf0000386276": [
        "https://unesdoc.unesco.org/ark:/48223/pf0000386276_eng",
    ],
}

def download_pdf(url, output_path):
    """Download PDF from URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/pdf,*/*',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        if response.status_code == 200 and len(response.content) > 1000:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        print("  PyMuPDF not installed. Trying pdfplumber...")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text.strip()
        except ImportError:
            print("  Neither PyMuPDF nor pdfplumber installed.")
            return None

def main():
    # Create output directory
    pdf_dir = "data/unesco_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Load existing corpus
    corpus_path = "data/corpus/corpus_master_20260127.json"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print("=" * 80)
    print("UNESCO AI ETHICS DOCUMENTS - DOWNLOAD AND EXTRACTION")
    print("=" * 80)
    
    new_entries = []
    
    for doc in UNESCO_DOCS:
        print(f"\n📄 {doc['title']}")
        print(f"   Year: {doc['year']}")
        
        pdf_path = os.path.join(pdf_dir, f"{doc['id']}.pdf")
        
        # Try to download
        downloaded = False
        if not os.path.exists(pdf_path):
            print(f"   Downloading from primary URL...")
            downloaded = download_pdf(doc['pdf_url'], pdf_path)
            
            # Try alternatives if primary fails
            if not downloaded and doc['id'] in ALTERNATIVE_URLS:
                for alt_url in ALTERNATIVE_URLS[doc['id']]:
                    print(f"   Trying alternative URL...")
                    downloaded = download_pdf(alt_url, pdf_path)
                    if downloaded:
                        break
            
            if downloaded:
                print(f"   ✓ Downloaded successfully")
            else:
                print(f"   ✗ Download failed - will add metadata only")
        else:
            print(f"   ✓ PDF already exists")
            downloaded = True
        
        # Extract text if PDF exists
        text_content = ""
        if os.path.exists(pdf_path):
            print(f"   Extracting text...")
            text_content = extract_text_from_pdf(pdf_path)
            if text_content:
                print(f"   ✓ Extracted {len(text_content):,} characters")
            else:
                print(f"   ✗ Text extraction failed")
        
        # Create corpus entry
        entry = {
            "id": f"unesco_{doc['id']}",
            "title": doc['title'],
            "country": "International",
            "source": "UNESCO",
            "date_published": str(doc['year']),
            "url": doc['url'],
            "body": text_content if text_content else doc['description'],
            "description": doc['description'],
            "document_type": "International Standard",
            "added_date": datetime.now().isoformat(),
            "has_full_text": bool(text_content),
        }
        
        new_entries.append(entry)
    
    # Check for duplicates and add new entries
    existing_ids = {e.get('id', '') for e in corpus['entries']}
    existing_urls = {e.get('url', '') for e in corpus['entries']}
    
    added = 0
    for entry in new_entries:
        if entry['id'] not in existing_ids and entry['url'] not in existing_urls:
            corpus['entries'].append(entry)
            added += 1
            print(f"\n✓ Added to corpus: {entry['title']}")
        else:
            print(f"\n⚠ Already in corpus: {entry['title']}")
    
    # Save updated corpus
    if added > 0:
        # Create backup
        backup_path = corpus_path.replace('.json', '_backup.json')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        # Save updated corpus
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Added {added} UNESCO documents to corpus")
        print(f"✓ Total corpus size: {len(corpus['entries'])} entries")
        print(f"✓ Backup saved to: {backup_path}")
        print("="*80)
    else:
        print(f"\n{'='*80}")
        print("No new documents added (all already in corpus)")
        print("="*80)

if __name__ == "__main__":
    main()
