"""
Retrieve remaining missing policies - v3
Simple, robust, no fancy encoding tricks.
Writes all output to a log file to avoid console issues.
"""
import json, hashlib, os, sys, re, time, random, urllib.parse, traceback, signal
from pathlib import Path
from dotenv import load_dotenv
import requests
import urllib3
urllib3.disable_warnings()

# Ignore SIGINT (Ctrl+C) - Windows terminal sends spurious interrupts
signal.signal(signal.SIGINT, signal.SIG_IGN)

load_dotenv()

BASE = Path(r"c:\Users\LucasSempe\OneDrive - International Initiative for Impact Evaluation\Desktop\Gen AI tools\AI_policies\observatory")
CORPUS_PATH = BASE / "data" / "corpus" / "corpus_master_20260127.json"
PDF_DIR = BASE / "data" / "pdfs"
PROGRESS_PATH = PDF_DIR / "download_progress.json"
LOG_PATH = PDF_DIR / "retrieve_v3_log.txt"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
]

def log(msg, logf):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    logf.write(line + "\n")
    logf.flush()
    # Safe console print
    try:
        print(line, flush=True)
    except Exception:
        try:
            print(line.encode('ascii', 'replace').decode('ascii'), flush=True)
        except Exception:
            pass

def eid_from_url(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

def safe_name(title, maxlen=60):
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', '', title or 'x')
    return re.sub(r'\s+', '_', s).strip('_')[:maxlen]

def hdrs():
    return {
        'User-Agent': random.choice(UAS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
    }

def try_download(sess, url, eid, title, timeout=10):
    if not url or not url.startswith('http'):
        return None
    try:
        r = sess.get(url, headers=hdrs(), timeout=timeout, verify=False, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 500:
            ct = r.headers.get('Content-Type', '').lower()
            ext = '.pdf' if ('pdf' in ct or url.lower().endswith('.pdf')) else '.html'
            fp = PDF_DIR / f"{eid}_{safe_name(title)}{ext}"
            fp.write_bytes(r.content)
            return {'size': len(r.content), 'ext': ext}
    except Exception:
        pass
    return None


def strategy_direct(sess, src, eid, title):
    return try_download(sess, src, eid, title, timeout=10)

def strategy_oecd(sess, oecd_url, eid, title):
    if not oecd_url:
        return None
    try:
        r = sess.get(oecd_url, headers=hdrs(), timeout=8, verify=False)
        if r.status_code != 200:
            return None
        patterns = [
            r'href=["\']([^"\']+)["\'][^>]*>\s*(?:Source|View\s+source|Original)',
            r'<a[^>]*href=["\']([^"\']+\.pdf)["\']',
            r'"source_url"\s*:\s*"([^"]+)"',
        ]
        for pat in patterns:
            for m in re.findall(pat, r.text, re.IGNORECASE):
                if 'oecd.ai' not in m and m.startswith('http'):
                    res = try_download(sess, m, eid, title, timeout=8)
                    if res:
                        return res
    except Exception:
        pass
    return None

def strategy_wayback(sess, url, eid, title):
    if not url:
        return None
    try:
        api = f"https://archive.org/wayback/available?url={urllib.parse.quote_plus(url)}"
        r = sess.get(api, timeout=8)
        if r.status_code != 200:
            return None
        snap = r.json().get('archived_snapshots', {}).get('closest', {})
        if snap.get('available') and snap.get('url'):
            return try_download(sess, snap['url'], eid, title, timeout=12)
    except Exception:
        pass
    return None

def strategy_ddg(sess, title, jurisdiction, eid):
    q = urllib.parse.quote_plus(f'{title} {jurisdiction} filetype:pdf')
    try:
        r = sess.get(f"https://html.duckduckgo.com/html/?q={q}",
                      headers=hdrs(), timeout=8, verify=False)
        if r.status_code != 200:
            return None
        urls = re.findall(r'uddg=([^&"\']+)', r.text)
        for raw in urls[:5]:
            u = urllib.parse.unquote(raw)
            if u.startswith('http') and 'duckduckgo' not in u and 'oecd.ai' not in u:
                res = try_download(sess, u, eid, title, timeout=8)
                if res:
                    return res
    except Exception:
        pass
    return None

def strategy_claude(sess, entry, eid, title):
    if not OPENROUTER_API_KEY:
        return None
    jurisdiction = entry.get('jurisdiction', 'Unknown')
    year = entry.get('year', '')
    desc = (entry.get('content', '') or '')[:400]
    
    prompt = f"""Find the direct download URL for this AI policy document.
Title: {title}
Jurisdiction: {jurisdiction}
Year: {year}
Description: {desc}

Return ONLY a JSON array of up to 3 URLs. Example: ["https://example.gov/doc.pdf"]
If impossible: []"""

    try:
        r = sess.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "anthropic/claude-sonnet-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
            },
            timeout=30,
        )
        if r.status_code == 429:
            time.sleep(10)
            return None
        if r.status_code != 200:
            return None
        
        content = r.json()['choices'][0]['message']['content'].strip()
        if '```' in content:
            m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if m:
                content = m.group(1)
        
        try:
            urls = json.loads(content)
        except json.JSONDecodeError:
            urls = re.findall(r'https?://[^\s"\'<>\]]+', content)
        
        if not isinstance(urls, list):
            return None
        
        for u in urls[:3]:
            u = u.strip().rstrip('.,;:)')
            res = try_download(sess, u, eid, title, timeout=10)
            if res:
                return res
    except Exception:
        pass
    return None


def save_progress(progress):
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def main():
    corpus = json.load(open(CORPUS_PATH, 'r', encoding='utf-8'))
    progress = json.load(open(PROGRESS_PATH, 'r', encoding='utf-8')) if PROGRESS_PATH.exists() else {}
    
    id_map = {}
    for e in corpus['entries']:
        eid = eid_from_url(e.get('url', ''))
        id_map[eid] = e
    
    # What's on disk already
    on_disk = set()
    for f in PDF_DIR.iterdir():
        if f.suffix not in ('.json', '.txt') and f.is_file():
            parts = f.stem.split('_', 1)
            if len(parts) >= 1 and len(parts[0]) == 12:
                on_disk.add(parts[0])
    
    missing = sorted(set(id_map.keys()) - on_disk)
    
    if 'retry3_downloaded' not in progress:
        progress['retry3_downloaded'] = {}
    if 'retry3_failed' not in progress:
        progress['retry3_failed'] = []
    
    with open(LOG_PATH, 'w', encoding='utf-8') as logf:
        log(f"=== RETRIEVAL v3 ===", logf)
        log(f"Corpus: {len(id_map)}, On disk: {len(on_disk)}, Missing: {len(missing)}", logf)
        
        if not missing:
            log("Nothing to do!", logf)
            return
        
        sess = requests.Session()
        ok = 0
        fail = 0
        stats = {'direct': 0, 'oecd': 0, 'wayback': 0, 'ddg': 0, 'claude': 0}
        
        for i, eid in enumerate(missing):
            try:
                e = id_map[eid]
                title = e.get('title', '?')
                src = e.get('source_url', '')
                oecd = e.get('url', '')
                jur = e.get('jurisdiction', '')
                
                result = None
                method = ''
                
                # S1: direct
                if src:
                    result = strategy_direct(sess, src, eid, title)
                    if result:
                        method = 'direct'
                
                # S2: OECD scrape
                if not result and oecd:
                    result = strategy_oecd(sess, oecd, eid, title)
                    if result:
                        method = 'oecd'
                
                # S3: wayback (try source_url and oecd_url)
                if not result:
                    for u in [src, oecd]:
                        if u:
                            result = strategy_wayback(sess, u, eid, title)
                            if result:
                                method = 'wayback'
                                break
                
                # S4: DuckDuckGo
                if not result:
                    result = strategy_ddg(sess, title, jur, eid)
                    if result:
                        method = 'ddg'
                
                # S5: Claude
                if not result:
                    result = strategy_claude(sess, e, eid, title)
                    if result:
                        method = 'claude'
                
                if result:
                    progress['retry3_downloaded'][eid] = {
                        'title': title, 'method': method, 'size': result['size']
                    }
                    ok += 1
                    stats[method] += 1
                    log(f"[{i+1}/{len(missing)}] + {title[:50]} [{method}] ({result['size']:,}b)", logf)
                    save_progress(progress)
                else:
                    progress['retry3_failed'].append(eid)
                    fail += 1
                
                if (i+1) % 20 == 0:
                    log(f"--- [{i+1}/{len(missing)}] +{ok} new, {fail} fail | Stats: {stats}", logf)
                    save_progress(progress)
                
                time.sleep(random.uniform(0.1, 0.3))
                
            except KeyboardInterrupt:
                # Ignore keyboard interrupts and continue
                log(f"[{i+1}] KeyboardInterrupt caught, continuing...", logf)
                continue
            except Exception:
                tb = traceback.format_exc()
                log(f"[{i+1}] EXCEPTION on {eid}: {tb[-200:]}", logf)
                progress['retry3_failed'].append(eid)
                fail += 1
        
        save_progress(progress)
        
        total_on_disk = len([f for f in PDF_DIR.iterdir() if f.suffix not in ('.json', '.txt') and f.is_file()])
        log(f"=== DONE ===", logf)
        log(f"+{ok} new, {fail} failed | Total files: {total_on_disk}/{len(id_map)}", logf)
        log(f"Strategies: {stats}", logf)


if __name__ == "__main__":
    main()
