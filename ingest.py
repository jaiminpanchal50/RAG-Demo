# import os
# import json
# import pdfplumber
# import chromadb
# from sentence_transformers import SentenceTransformer

# # ── Config ──────────────────────────────────────────
# PDF_DIR = "./pdfs"
# JSONL_DIR = "./jsonl"          # ← put your .jsonl files here
# CHUNK_SIZE = 300
# CHUNK_OVERLAP = 50
# COLLECTION_NAME = "praangan_elitus"
# # ────────────────────────────────────────────────────

# model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_or_create_collection(
#     name=COLLECTION_NAME,
#     metadata={"hnsw:space": "cosine"}
# )

# # ── PDF Ingestion (existing) ─────────────────────────
# def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         if chunk:
#             chunks.append(chunk)
#     return chunks

# def is_valid_chunk(text):
#     words = text.split()
#     if len(words) < 20:
#         return False
#     noise = ["LIFT","FOYER","DN","UP","DAOR","SSECCA","NIGRAM","LOBBY"]
#     noise_count = sum(1 for w in words if w.upper() in noise)
#     return (noise_count / len(words)) <= 0.15

# def ingest_pdfs(all_chunks, all_ids, all_metadata, chunk_id):
#     for filename in os.listdir(PDF_DIR):
#         if not filename.endswith(".pdf"):
#             continue
#         print(f"📄 PDF: {filename}")
#         with pdfplumber.open(os.path.join(PDF_DIR, filename)) as pdf:
#             full_text = "".join(
#                 p.extract_text() or "" for p in pdf.pages
#             )
#         for chunk in chunk_text(full_text):
#             if is_valid_chunk(chunk):
#                 all_chunks.append(chunk)
#                 all_ids.append(f"pdf_chunk_{chunk_id}")
#                 all_metadata.append({"source": filename, "type": "pdf"})
#                 chunk_id += 1
#     return chunk_id

# # ── JSONL Ingestion (new) ────────────────────────────
# def ingest_jsonl(all_chunks, all_ids, all_metadata, chunk_id):
#     for filename in os.listdir(JSONL_DIR):
#         if not filename.endswith(".jsonl"):
#             continue
#         print(f"📋 JSONL: {filename}")

#         with open(os.path.join(JSONL_DIR, filename), "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     entry = json.loads(line)
#                 except json.JSONDecodeError:
#                     continue

#                 instruction = entry.get("instruction", "")
#                 variants = entry.get("variants", [])
#                 responses = entry.get("response", "")

#                 # Handle response as string or list
#                 if isinstance(responses, str):
#                     responses = [responses]

#                 # Build chunk = question context + all answers (EN + HI + GU)
#                 # This makes it match queries in any language
#                 questions_text = instruction
#                 if variants:
#                     questions_text += " | " + " | ".join(variants)

#                 answers_text = " | ".join(responses)

#                 chunk = f"Q: {questions_text}\nA: {answers_text}"

#                 all_chunks.append(chunk)
#                 all_ids.append(f"jsonl_chunk_{chunk_id}")
#                 all_metadata.append({
#                     "source": filename,
#                     "type": "jsonl",
#                     "instruction": instruction
#                 })
#                 chunk_id += 1

#     return chunk_id

# # ── Main ─────────────────────────────────────────────
# def main():
#     all_chunks, all_ids, all_metadata = [], [], []
#     chunk_id = 0

#     if os.path.exists(PDF_DIR):
#         chunk_id = ingest_pdfs(all_chunks, all_ids, all_metadata, chunk_id)

#     if os.path.exists(JSONL_DIR):
#         chunk_id = ingest_jsonl(all_chunks, all_ids, all_metadata, chunk_id)

#     print(f"\nTotal chunks: {len(all_chunks)}")
#     print("Embedding... (CPU may take a few minutes)")

#     prefixed = [f"passage: {c}" for c in all_chunks]
#     # embeddings = model.encode(prefixed, batch_size=32, show_progress_bar=True)
#     embeddings = model.encode(prefixed, batch_size=8, show_progress_bar=True)

#     collection.add(
#         documents=all_chunks,
#         embeddings=embeddings.tolist(),
#         ids=all_ids,
#         metadatas=all_metadata
#     )
#     print(f"\n✅ Done! {len(all_chunks)} chunks stored.")

# if __name__ == "__main__":
#     main()

import os
import json
import pdfplumber
import chromadb
import requests
import feedparser
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────
def resolve_dir(new_path, old_path):
    if os.path.exists(new_path) and os.listdir(new_path):
        return new_path
    if os.path.exists(old_path) and os.listdir(old_path):
        print(f"  ℹ️  Using legacy path: {old_path}")
        return old_path
    return new_path

DATA_DIR       = "./data"
PDF_DIR        = resolve_dir(f"{DATA_DIR}/pdfs",  "./pdfs")
JSONL_DIR      = resolve_dir(f"{DATA_DIR}/jsonl", "./jsonl")
DOCX_DIR       = resolve_dir(f"{DATA_DIR}/docx",  "./docx")
TXT_DIR        = resolve_dir(f"{DATA_DIR}/txt",   "./txt")
URLS_FILE      = f"{DATA_DIR}/urls.txt" if os.path.exists(f"{DATA_DIR}/urls.txt") else "./urls.txt"
RSS_FILE       = f"{DATA_DIR}/rss.txt"  if os.path.exists(f"{DATA_DIR}/rss.txt")  else "./rss.txt"
DB_CONFIG_FILE = "./db_config.json"

CHUNK_SIZE      = 300
CHUNK_OVERLAP   = 50
COLLECTION_NAME = "praangan_elitus"
# ─────────────────────────────────────────────────────────────

print("🔄 Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ── Shared Utilities ─────────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
        if words[i:i + chunk_size]
    ]

def is_valid_chunk(text, min_words=15):
    words = text.split()
    if len(words) < min_words:
        return False
    noise = ["LIFT","FOYER","DN","UP","DAOR","SSECCA","NIGRAM","LOBBY"]
    noise_count = sum(1 for w in words if w.upper() in noise)
    return (noise_count / len(words)) <= 0.15

def add_chunks(all_chunks, all_ids, all_metadata,
               chunks, source, source_type, chunk_id, min_words=15):
    added = 0
    for chunk in chunks:
        if is_valid_chunk(chunk, min_words=min_words):
            all_chunks.append(chunk)
            all_ids.append(f"{source_type}_{chunk_id}")
            all_metadata.append({"source": source, "type": source_type})
            chunk_id += 1
            added += 1
    if added == 0:
        print(f"  ⚠️  No valid chunks from: {source}")
    return chunk_id

# ── 1. PDF + Brochures ───────────────────────────────────────
def ingest_pdfs(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(PDF_DIR):
        print(f"  ⏭️  Folder not found: {PDF_DIR}")
        return chunk_id
    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not files:
        print(f"  ⏭️  No PDF files in {PDF_DIR}")
        return chunk_id
    for filename in files:
        print(f"  📄 PDF: {filename}")
        try:
            with pdfplumber.open(os.path.join(PDF_DIR, filename)) as pdf:
                full_text = "".join(p.extract_text() or "" for p in pdf.pages)
            print(f"      → Extracted {len(full_text.split())} words")
            chunk_id = add_chunks(all_chunks, all_ids, all_metadata,
                                   chunk_text(full_text), filename, "pdf", chunk_id)
        except Exception as e:
            print(f"  ⚠️  Error in {filename}: {e}")
    return chunk_id

# ── 2. JSONL Q&A ─────────────────────────────────────────────
def ingest_jsonl(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(JSONL_DIR):
        print(f"  ⏭️  Folder not found: {JSONL_DIR}")
        return chunk_id
    files = [f for f in os.listdir(JSONL_DIR) if f.endswith(".jsonl")]
    if not files:
        print(f"  ⏭️  No JSONL files in {JSONL_DIR}")
        return chunk_id
    for filename in files:
        print(f"  📋 JSONL: {filename}")
        count = 0
        with open(os.path.join(JSONL_DIR, filename), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = entry.get("instruction", "")
                variants    = entry.get("variants", [])
                responses   = entry.get("response", "")
                if isinstance(responses, str):
                    responses = [responses]
                questions_text = instruction
                if variants:
                    questions_text += " | " + " | ".join(
                        v for v in variants if isinstance(v, str)
                    )
                chunk = f"Q: {questions_text}\nA: {' | '.join(responses)}"
                all_chunks.append(chunk)
                all_ids.append(f"jsonl_{chunk_id}")
                all_metadata.append({
                    "source": filename,
                    "type": "jsonl",
                    "instruction": instruction
                })
                chunk_id += 1
                count += 1
        print(f"      → Loaded {count} Q&A pairs")
    return chunk_id

# ── 3. Word Manuals (.docx) ──────────────────────────────────
def ingest_docx(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(DOCX_DIR):
        print(f"  ⏭️  Folder not found: {DOCX_DIR}")
        return chunk_id
    files = [f for f in os.listdir(DOCX_DIR) if f.lower().endswith(".docx")]
    if not files:
        print(f"  ⏭️  No DOCX files in {DOCX_DIR}")
        return chunk_id
    for filename in files:
        print(f"  📝 DOCX: {filename}")
        try:
            doc = Document(os.path.join(DOCX_DIR, filename))
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            print(f"      → Extracted {len(full_text.split())} words")
            chunk_id = add_chunks(all_chunks, all_ids, all_metadata,
                                   chunk_text(full_text), filename, "docx", chunk_id)
        except Exception as e:
            print(f"  ⚠️  Error in {filename}: {e}")
    return chunk_id

# ── 4. Plain Text (.txt) ─────────────────────────────────────
def ingest_txt(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(TXT_DIR):
        print(f"  ⏭️  Folder not found: {TXT_DIR}")
        return chunk_id
    files = [f for f in os.listdir(TXT_DIR) if f.lower().endswith(".txt")]
    if not files:
        print(f"  ⏭️  No TXT files in {TXT_DIR}")
        return chunk_id
    for filename in files:
        print(f"  📃 TXT: {filename}")
        try:
            with open(os.path.join(TXT_DIR, filename), "r", encoding="utf-8") as f:
                full_text = f.read()
            print(f"      → Extracted {len(full_text.split())} words")
            chunk_id = add_chunks(all_chunks, all_ids, all_metadata,
                                   chunk_text(full_text), filename, "txt", chunk_id)
        except Exception as e:
            print(f"  ⚠️  Error in {filename}: {e}")
    return chunk_id

# ── 5. Websites (URL scraping) ───────────────────────────────
# def scrape_url(url):
#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
#             "Chrome/120.0.0.0 Safari/537.36"
#         ),
#         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#         "Accept-Language": "en-US,en;q=0.5",
#     }
#     try:
#         res = requests.get(url, headers=headers, timeout=20)
#         print(f"      → HTTP {res.status_code} | {len(res.text)} chars raw HTML")
#         if res.status_code == 200:
#             soup = BeautifulSoup(res.text, "html.parser")
#             for tag in soup(["script","style","nav","footer","header","noscript"]):
#                 tag.decompose()
#             text = soup.get_text(separator=" ", strip=True)
#             print(f"      → Extracted {len(text.split())} words after parsing")
#             if len(text.split()) > 50:
#                 return text
#             print(f"      → JS-rendered site detected, trying Selenium...")
#     except Exception as e:
#         print(f"      → requests failed: {e}")

#     try:
#         from selenium import webdriver
#         from selenium.webdriver.chrome.options import Options
#         from selenium.webdriver.chrome.service import Service
#         import time

#         print(f"      → Launching headless Chromium...")
#         options = Options()
#         options.add_argument("--headless")
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         options.add_argument("--disable-gpu")
#         options.add_argument("--window-size=1920,1080")
#         options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64)")
#         options.binary_location = "/usr/bin/chromium-browser"

#         service = Service("/usr/bin/chromedriver")
#         driver = webdriver.Chrome(service=service, options=options)
#         driver.get(url)
#         time.sleep(5)

#         soup = BeautifulSoup(driver.page_source, "html.parser")
#         driver.quit()

#         for tag in soup(["script","style","nav","footer","header","noscript"]):
#             tag.decompose()
#         text = soup.get_text(separator=" ", strip=True)
#         print(f"      → Selenium extracted {len(text.split())} words")
#         return text

#     except ImportError:
#         print(f"      → Selenium not installed: pip install selenium")
#         print(f"      → Tip: Copy website text manually to data/txt/website.txt")
#         return ""
#     except Exception as e:
#         print(f"      → Selenium failed: {e}")
#         print(f"      → Tip: Copy website text manually to data/txt/website.txt")
#         return ""

# def ingest_urls(all_chunks, all_ids, all_metadata, chunk_id):
#     if not os.path.exists(URLS_FILE):
#         print(f"  ⏭️  urls.txt not found at {URLS_FILE}")
#         return chunk_id
#     with open(URLS_FILE, "r") as f:
#         urls = [
#             line.strip() for line in f
#             if line.strip() and not line.startswith("#")
#         ]
#     if not urls:
#         print(f"  ⏭️  No URLs in {URLS_FILE}")
#         return chunk_id
#     for url in urls:
#         print(f"  🌐 Scraping: {url}")
#         text = scrape_url(url)
#         if text:
#             before = len(all_chunks)
#             # Smaller chunks for website content for better diversity
#             chunk_id = add_chunks(all_chunks, all_ids, all_metadata,
#                                    chunk_text(text, chunk_size=150, overlap=20),
#                                    url, "website", chunk_id, min_words=15)
#             print(f"      → Added {len(all_chunks) - before} chunks")
#     return chunk_id




# ── 5. Websites (URL scraping) ───────────────────────────────
def scrape_url(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    try:
        res = requests.get(url, headers=headers, timeout=20)
        print(f"      → HTTP {res.status_code} | {len(res.text)} chars")
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script","style","nav","footer","header","noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            if len(text.split()) > 50:
                return text
            print(f"      → JS-rendered, trying Selenium...")
    except Exception as e:
        print(f"      → requests failed: {e}")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        import time

        print(f"      → Launching headless Chromium...")
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.binary_location = "/usr/bin/chromium-browser"

        driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=options)
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        for tag in soup(["script","style","nav","footer","header","noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        print(f"      → Selenium extracted {len(text.split())} words")
        return text

    except ImportError:
        print(f"      → Selenium not installed: pip install selenium")
        return ""
    except Exception as e:
        print(f"      → Selenium failed: {e}")
        return ""


def chunk_website_text(text):
    """
    Split website text by meaningful boundaries:
    sentences, then enforce max 60 words per chunk.
    This prevents dumping entire page sections into one chunk.
    """
    import re

    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks   = []
    current  = []
    word_cnt = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue

        # If adding this sentence exceeds 60 words, save current and start new
        if word_cnt + len(words) > 60 and current:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= 12:     # min 12 words
                chunks.append(chunk)
            current  = []
            word_cnt = 0

        current.append(sent)
        word_cnt += len(words)

    # Save last chunk
    if current:
        chunk = " ".join(current).strip()
        if len(chunk.split()) >= 12:
            chunks.append(chunk)

    return chunks


def ingest_urls(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(URLS_FILE):
        print(f"  ⏭️  urls.txt not found at {URLS_FILE}")
        return chunk_id
    with open(URLS_FILE, "r") as f:
        urls = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
    if not urls:
        print(f"  ⏭️  No URLs in {URLS_FILE}")
        return chunk_id
    for url in urls:
        print(f"  🌐 Scraping: {url}")
        text = scrape_url(url)
        if not text:
            continue

        # Use sentence-aware chunking for websites
        chunks = chunk_website_text(text)
        print(f"      → Created {len(chunks)} sentence-based chunks")

        before = len(all_chunks)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"website_{chunk_id}")
            all_metadata.append({"source": url, "type": "website"})
            chunk_id += 1

        print(f"      → Added {len(all_chunks) - before} chunks")
    return chunk_id

# ── 6. Blogs / RSS Feeds ─────────────────────────────────────
def ingest_rss(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(RSS_FILE):
        print(f"  ⏭️  rss.txt not found at {RSS_FILE}")
        return chunk_id
    with open(RSS_FILE, "r") as f:
        rss_urls = [line.strip() for line in f if line.strip()]
    if not rss_urls:
        print(f"  ⏭️  No RSS URLs in {RSS_FILE}")
        return chunk_id
    for rss_url in rss_urls:
        print(f"  📰 RSS: {rss_url}")
        feed = feedparser.parse(rss_url)
        count = 0
        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            link    = entry.get("link", rss_url)
            text    = f"{title}. {summary}"
            if len(text.split()) > 15:
                chunk_id = add_chunks(all_chunks, all_ids, all_metadata,
                                       chunk_text(text), link, "blog",
                                       chunk_id, min_words=15)
                count += 1
        print(f"      → Loaded {count} blog entries")
    return chunk_id

# ── 7. Database (MySQL / PostgreSQL) ─────────────────────────
def ingest_database(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(DB_CONFIG_FILE):
        print(f"  ⏭️  db_config.json not found")
        return chunk_id
    with open(DB_CONFIG_FILE) as f:
        cfg = json.load(f)
    if not cfg.get("enabled", False):
        print("  ⏭️  Database ingestion disabled in db_config.json")
        return chunk_id

    print(f"  🗄️  Database: {cfg['type']} — {cfg['database']}")
    try:
        if cfg["type"] == "mysql":
            import mysql.connector
            conn = mysql.connector.connect(
                host=cfg["host"], port=cfg["port"],
                user=cfg["user"], password=cfg["password"],
                database=cfg["database"]
            )
        elif cfg["type"] == "postgresql":
            import psycopg2
            conn = psycopg2.connect(
                host=cfg["host"], port=cfg["port"],
                user=cfg["user"], password=cfg["password"],
                dbname=cfg["database"]
            )
        else:
            print(f"  ⚠️  Unknown DB type: {cfg['type']}")
            return chunk_id

        cursor = conn.cursor()
        cursor.execute(cfg["query"])
        rows = cursor.fetchall()
        text_cols = cfg.get("text_columns", [])
        print(f"      → {len(rows)} rows fetched")

        for row in rows:
            row_dict = dict(zip(text_cols, row))
            combined = " | ".join(str(v) for v in row_dict.values() if v)
            if combined.strip():
                chunk_id = add_chunks(all_chunks, all_ids, all_metadata,
                                       chunk_text(combined),
                                       cfg["database"], "database", chunk_id)
        conn.close()
    except Exception as e:
        print(f"  ⚠️  DB error: {e}")
    return chunk_id


# ── 8. CSV Files ─────────────────────────────────────────────
import csv

CSV_DIR = resolve_dir(f"{DATA_DIR}/csv", "./csv")

def ingest_csv(all_chunks, all_ids, all_metadata, chunk_id):
    if not os.path.exists(CSV_DIR):
        print(f"  ⏭️  Folder not found: {CSV_DIR}")
        return chunk_id
    files = [f for f in os.listdir(CSV_DIR) if f.lower().endswith(".csv")]
    if not files:
        print(f"  ⏭️  No CSV files in {CSV_DIR}")
        return chunk_id

    for filename in files:
        print(f"  📊 CSV: {filename}")
        count = 0
        try:
            with open(os.path.join(CSV_DIR, filename), "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # ── Pull fields safely ──────────────────
                    row_id      = row.get("id", "").strip()
                    project     = row.get("project", "").strip()
                    category    = row.get("category", "").strip()
                    subcategory = row.get("subcategory", "").strip()
                    title       = row.get("title", "").strip()
                    document    = row.get("document", "").strip()
                    keywords    = row.get("keywords", "").strip()
                    source      = row.get("source", "").strip()

                    # ── Skip empty rows ─────────────────────
                    if not document:
                        continue

                    # ── Build rich chunk text ───────────────
                    # Format: "Title: X. Category: Y. Answer: Z. Keywords: A B C"
                    # This makes semantic search highly accurate
                    parts = []
                    if title:
                        parts.append(f"Title: {title}.")
                    if category and subcategory:
                        parts.append(f"Category: {category} > {subcategory}.")
                    elif category:
                        parts.append(f"Category: {category}.")
                    if project:
                        parts.append(f"Project: {project}.")
                    parts.append(f"Info: {document}")
                    if keywords:
                        parts.append(f"Keywords: {keywords}.")

                    chunk = " ".join(parts)

                    all_chunks.append(chunk)
                    all_ids.append(f"csv_{chunk_id}")
                    all_metadata.append({
                        "source":      filename,
                        "type":        "csv",
                        "id":          row_id,
                        "project":     project,
                        "category":    category,
                        "subcategory": subcategory,
                        "title":       title,
                        "origin":      source     # website+pdf / pdf / synthetic
                    })
                    chunk_id += 1
                    count += 1

        except Exception as e:
            print(f"  ⚠️  Error in {filename}: {e}")

        print(f"      → Loaded {count} rows as chunks")
    return chunk_id
    
# ── Main ─────────────────────────────────────────────────────
def main():
    all_chunks, all_ids, all_metadata = [], [], []
    chunk_id = 0

    print("\n📥 Starting ingestion from all sources...\n")

    print("── PDFs & Brochures ──")
    chunk_id = ingest_pdfs(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── JSONL Q&A ──")
    chunk_id = ingest_jsonl(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── Word Manuals ──")
    chunk_id = ingest_docx(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── Text Files ──")
    chunk_id = ingest_txt(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── Websites ──")
    chunk_id = ingest_urls(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── Blogs / RSS ──")
    chunk_id = ingest_rss(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── Database ──")
    chunk_id = ingest_database(all_chunks, all_ids, all_metadata, chunk_id)

    print("\n── CSV Files ──")
    chunk_id = ingest_csv(all_chunks, all_ids, all_metadata, chunk_id)
    
    print("\n" + "─" * 50)

    if not all_chunks:
        print("❌ No valid chunks found. Check warnings above.")
        return

    print(f"✅ Total chunks ready: {len(all_chunks)}")
    print("⚙️  Embedding... (may take a few minutes on CPU)\n")

    prefixed   = [f"passage: {c}" for c in all_chunks]
    embeddings = model.encode(prefixed, batch_size=8, show_progress_bar=True)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        ids=all_ids,
        metadatas=all_metadata
    )
    print(f"\n🎉 Ingestion complete! {len(all_chunks)} chunks stored in ChromaDB.")

    from collections import Counter
    types = Counter(m["type"] for m in all_metadata)
    print("\n📊 Chunks by source type:")
    for t, count in types.items():
        print(f"   {t:12s} → {count} chunks")

if __name__ == "__main__":
    main()