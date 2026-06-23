# 🏠 Praangan Elitus — Multilingual RAG Demo

A Retrieval-Augmented Generation (RAG) system for the Praangan Elitus real estate project.
Supports PDF brochures, JSONL Q&A datasets, CSV structured data, Word manuals, plain text, websites, RSS blogs, and databases — with multilingual support for English, Hindi, and Gujarati.

---

## 📁 Project Structure

```
RAG Demo/
├── data/
│   ├── pdfs/           ← PDF brochures (Praangan_Elitus_Update_01.pdf)
│   ├── jsonl/          ← Q&A datasets (data.jsonl)
│   ├── csv/            ← Structured data (data.csv)
│   ├── docx/           ← Word manuals
│   ├── txt/            ← Plain text files
│   ├── urls.txt        ← Website URLs to scrape (one per line)
│   └── rss.txt         ← RSS/blog feed URLs (one per line)
├── chroma_db/          ← Auto-created vector database (do not edit)
├── db_config.json      ← MySQL/PostgreSQL connection config
├── ingest.py           ← Step 1: Parse all sources → embed → store in ChromaDB
├── query.py            ← Step 2: Search ChromaDB → retrieve → display results
└── README.md           ← This file
```

---

## ⚙️ System Requirements

| Component | Requirement |
|---|---|
| OS | Ubuntu 22.04 LTS or higher |
| Python | 3.10+ |
| RAM | 16 GB minimum |
| GPU | Not required (CPU mode) |
| Disk | 2 GB free (model + ChromaDB) |

---

## 🚀 Step 1 — Setup

### 1.1 Create Virtual Environment

```bash
cd ~/Desktop/RAG\ Demo
python3 -m venv venv
source venv/bin/activate
```

> Always activate the venv before running any scripts:
> ```bash
> source venv/bin/activate
> ```

### 1.2 Install CPU-Only PyTorch First

This is important — installing regular PyTorch downloads 2GB+ of CUDA files unnecessarily.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1.3 Install All Dependencies

```bash
pip install --timeout 300 --retries 5 \
    chromadb \
    pdfplumber \
    sentence-transformers \
    python-docx \
    requests \
    beautifulsoup4 \
    feedparser \
    selenium \
    numpy
```

### 1.4 Verify Installation

```bash
python3 -c "import torch; print('Torch:', torch.__version__)"
python3 -c "import chromadb; print('ChromaDB: OK')"
python3 -c "import pdfplumber; print('pdfplumber: OK')"
python3 -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers: OK')"
```

All 4 should print without errors.

---

## 📂 Step 2 — Add Your Data

Place your files in the correct folders:

```bash
# PDF brochures
cp Praangan_Elitus_Update_01.pdf data/pdfs/

# JSONL Q&A dataset
cp data.jsonl data/jsonl/

# CSV structured data
cp data.csv data/csv/

# Website to scrape (one URL per line)
echo "https://praanganinfra.in/" > data/urls.txt
```

### Data Format Guide

#### PDF / Brochures
Any standard PDF. The system extracts all text automatically and filters out noise (floor plan coordinates, lift labels, etc.).

#### JSONL Q&A Dataset
Each line is one JSON object with this structure:

```json
{
  "instruction": "What is the possession date?",
  "variants": [
    "When will flats be handed over?",
    "पजेशन कब मिलेगा?",
    "પઝેશન ક્યારે મળશે?"
  ],
  "response": "The expected possession date is December 2029."
}
```

- `instruction` — the main question in English
- `variants` — same question in other languages or phrasings
- `response` — the answer (string or list of strings for multilingual answers)

The system combines all variants and the response into one searchable chunk so queries in any language match the same answer.

#### CSV Structured Data
Required columns: `id, project, category, subcategory, title, document, keywords, source`

```
id,project,category,subcategory,title,document,keywords,source
ELITUS_AMENITY_001,Praangan Elitus,amenity,fitness,Gym,"Modern gym with premium equipment.",gym fitness,pdf
ELITUS_LOCATION_001,Praangan Elitus,location,address,Address,"Near Dehgam Circle, Naroda Ahmedabad.",address,pdf
```

Each row becomes one searchable chunk formatted as:
```
Title: Gym. Category: amenity > fitness. Project: Praangan Elitus.
Info: Modern gym with premium equipment. Keywords: gym fitness.
```

> **Note:** If your CSV has no header row, the system auto-detects and adds default headers.

#### Website URLs (`data/urls.txt`)
```
https://praanganinfra.in/
https://praanganinfra.in/elitus
# Lines starting with # are ignored
```

The scraper tries `requests` first. If the site is JavaScript-rendered (React/Vue SPA), it automatically falls back to Selenium headless Chrome.

#### RSS / Blog Feeds (`data/rss.txt`)
```
https://example.com/feed.xml
https://realestateblog.com/rss
```

#### Database (`db_config.json`)
```json
{
  "enabled": false,
  "type": "mysql",
  "host": "localhost",
  "port": 3306,
  "user": "root",
  "password": "your_password",
  "database": "praangan_db",
  "query": "SELECT title, description FROM properties",
  "text_columns": ["title", "description"]
}
```

Set `"enabled": true` when ready to use.

---

## 🔄 Step 3 — Ingest Data

This script reads all your data sources, splits them into chunks, generates embeddings, and stores them in ChromaDB.

```bash
python3 ingest.py
```

### What Happens Internally

```
PDF / JSONL / CSV / TXT / Website / RSS / DB
          ↓
    Text Extraction
          ↓
    Chunking (300 words, 50 word overlap)
          ↓
    Noise Filter (removes floor plan garbage)
          ↓
    Multilingual E5 Embedding Model
    (converts each chunk to a 768-dim vector)
          ↓
    ChromaDB (stores vectors + original text + metadata)
```

### Expected Output

```
🔄 Loading embedding model...

📥 Starting ingestion from all sources...

── PDFs & Brochures ──
  📄 PDF: Praangan_Elitus_Update_01.pdf
      → Extracted 4821 words
      → Added 18 chunks

── JSONL Q&A ──
  📋 JSONL: data.jsonl
      → Loaded 407 Q&A pairs

── CSV Files ──
  📊 CSV: data.csv
      → Loaded 68 rows | Skipped 0 empty rows

── Websites ──
  🌐 Scraping: https://praanganinfra.in/
      → Selenium extracted 1240 words
      → Added 31 sentence chunks

── Database ──
  ⏭️  Database ingestion disabled in db_config.json

──────────────────────────────────────────────
✅ Total chunks ready: 524
⚙️  Embedding... (may take 5-10 minutes on CPU)

🎉 Ingestion complete! 524 chunks stored in ChromaDB.

📊 Chunks by source type:
   jsonl        → 407 chunks
   csv          →  68 chunks
   website      →  31 chunks
   pdf          →  18 chunks
```

> **Re-ingesting:** If you add new files or change existing ones, always clear the old database first:
> ```bash
> rm -rf chroma_db
> python3 ingest.py
> ```

---

## 🔍 Step 4 — Query the RAG

```bash
python3 query.py
```

Type your question and press Enter. Type `quit` to exit.

### How Retrieval Works

```
User Query
    ↓
E5 Embedding ("query: What is the possession date?")
    ↓
ChromaDB Semantic Search (fetch top 12 candidates)
    ↓
Similarity Score Check
    ├── Score < 0.83 → REJECTED (out of domain)
    └── Score ≥ 0.83 → PASSED
                ↓
        MMR Reranking (pick 3 diverse chunks)
                ↓
        Display Results with Source Type
```

### MMR (Maximal Marginal Relevance) Explained

Without MMR, all 3 returned chunks could be nearly identical (just different sentences from the same webpage). MMR solves this by balancing two factors:

- **Relevance** — how similar is the chunk to your query?
- **Diversity** — how different is this chunk from already-selected chunks?

`lambda_param = 0.7` means 70% relevance + 30% diversity.

### Sample Output

```
────────────────────────────────────────────────────────────
🔍 Query: Does Praangan Elitus have a swimming pool?
📊 Best similarity score : 0.912  (threshold: 0.83)
✅ PASSED — Returning top 3 diverse chunks

┌─ Chunk 1 ────────────────────────────────────────────────
│  📊  Source Type : CSV
│  📂 File       : data.csv
│  💯 Score      : 0.912
│  📝 Content    :
│     Title: Swimming Pool. Category: amenity > water.
│     Info: A premium swimming pool is available for residents.
└──────────────────────────────────────────────────────────

┌─ Chunk 2 ────────────────────────────────────────────────
│  📋  Source Type : JSONL
│  📂 File       : data.jsonl
│  💯 Score      : 0.887
│  📝 Content    :
│     Q: Does Praangan Elitus have swimming pool?
│     A: Yes, a premium pool is available on the upper level.
└──────────────────────────────────────────────────────────

┌─ Chunk 3 ────────────────────────────────────────────────
│  🌐  Source Type : WEBSITE
│  📂 File       : https://praanganinfra.in/
│  💯 Score      : 0.861
│  📝 Content    :
│     Splash pool Swimming pool (on upper level) Poolside leisure areas...
└──────────────────────────────────────────────────────────
```

### Rejection Example

```
🔍 Query: What is the weather today?
📊 Best similarity score : 0.721  (threshold: 0.83)
🚫 REJECTED — Out of domain / Low relevance
```

---

## 🌐 Website Scraping Notes

### JavaScript-Rendered Sites (React/Vue SPA)

Sites like `praanganinfra.in` load content via JavaScript, so a simple HTTP request returns almost no text. The scraper handles this automatically:

```
requests.get() → returns ~2 words (just JS bundle)
      ↓
Auto-detects JS-rendered site
      ↓
Launches headless Chromium via Selenium
      ↓
Waits 5 seconds for JS to execute
      ↓
Extracts full rendered text (~1200+ words)
```

To install Selenium + Chrome support:

```bash
pip install selenium
sudo apt install -y chromium-browser chromium-chromedriver
```

### Website Chunking

Websites are chunked differently from PDFs — by sentence boundaries (max 60 words per chunk) instead of fixed word count. This prevents one giant chunk from matching every query.

---

## 🔧 Configuration Reference

### `query.py` Settings

```python
SIMILARITY_THRESHOLD = 0.83   # Minimum score to pass retrieval gate
                               # Lower = more permissive (more false positives)
                               # Higher = stricter (may miss valid questions)
                               # Recommended range: 0.80 – 0.87

TOP_K = 3                      # Number of chunks to return
                               # Increase to 5 for more context
```

### `ingest.py` Settings

```python
CHUNK_SIZE    = 300   # Words per chunk (PDFs, JSONL, CSV, TXT)
CHUNK_OVERLAP = 50    # Overlap between chunks to avoid cutting mid-sentence
```

Website chunks use separate settings (60 words max, sentence-aware).

---

## 🌍 Multilingual Support

The system uses `intfloat/multilingual-e5-base` which natively supports 100+ languages including English, Hindi, and Gujarati.

### Test Multilingual Queries

```
# English
Does Praangan Elitus have a gym?

# Hindi
क्या प्रांगण एलीटस में जिम है?

# Gujarati
પ્રાણ એલિટસમાં જિમ છે?
```

All three will match the same chunks because:
1. The JSONL format stores all language variants together in one chunk
2. The E5 model understands semantic similarity across languages

### E5 Prefix Convention

The E5 model requires specific prefixes:
- Documents stored in ChromaDB: `passage: {chunk text}`
- Queries at search time: `query: {user question}`

This is handled automatically — you do not need to add these manually.

---

## 🐛 Troubleshooting

### "No valid chunks found"

```
❌ No valid chunks found. Check warnings above.
```

**Causes and fixes:**

| Cause | Fix |
|---|---|
| Files in wrong folder | Check folder names match `data/pdfs/`, `data/jsonl/` etc. |
| CSV has no header row | System auto-detects, but check the warning message |
| PDF has only images (scanned) | PDF needs OCR — pdfplumber cannot extract image-based text |
| All chunks are noise | PDF contains mostly floor plan coordinates — add JSONL/CSV data |

### Download Timeout During pip install

```bash
pip install --timeout 300 --retries 5 package-name
```

### ChromaDB Already Has Data Error

```bash
rm -rf chroma_db
python3 ingest.py
```

### Selenium / Chrome Not Found

```bash
sudo apt install -y chromium-browser chromium-chromedriver
which chromium-browser    # should print /usr/bin/chromium-browser
which chromedriver        # should print /usr/bin/chromedriver
```

### All Queries Return Same Results

1. Raise `SIMILARITY_THRESHOLD` to `0.87` in `query.py`
2. Re-ingest with smaller website chunk size (already set to 150 words)
3. Ensure JSONL data has specific, targeted Q&A pairs

### Slow Ingestion on CPU

Normal on i3/i5 without GPU. Expected times:

| Data Size | Estimated Time |
|---|---|
| 50 chunks | ~1 minute |
| 200 chunks | ~4 minutes |
| 500 chunks | ~10 minutes |
| 1000+ chunks | ~20+ minutes |

Use `batch_size=8` (already set) — increasing this will cause memory errors on 16GB RAM.

---

## 📊 RAG Type Used

This project implements **Advanced RAG** with elements of **Modular RAG**:

| Feature | Status |
|---|---|
| Basic semantic search | ✅ Implemented |
| Cosine similarity retrieval gate | ✅ Implemented |
| MMR diversity reranking | ✅ Implemented |
| Multi-source ingestion (8 sources) | ✅ Implemented |
| Source type tagging in results | ✅ Implemented |
| Multilingual embedding | ✅ Implemented |
| Conversation memory | 🔲 Not yet |
| Query routing by category | 🔲 Next step |
| LLM answer generation | 🔲 Next step |

---

## 🗺️ Next Steps

### Connect an LLM for Full RAG Pipeline

Once retrieval is working, connect to an LLM to generate final answers from retrieved chunks:

```python
# Pseudocode for full RAG pipeline
chunks = query_rag(user_question)           # retrieve
context = "\n".join([c["chunk"] for c in chunks])  # combine
prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"
response = llm.generate(prompt)             # generate
```

LLM options (in order of recommendation for this project):
1. **Groq API** (free tier, fast, multilingual) — `llama-3.1-8b-instant`
2. **OpenAI API** — `gpt-4o-mini`
3. **Local Ollama** — `qwen2.5:7b` (works on 16GB RAM)
4. **Your fine-tuned Shuka-1** — best for Praangan-specific knowledge

### Add Query Router (Modular RAG)

Route different question types to the best source:

```python
if "price" in query or "cost" in query:
    search_in = ["csv"]       # price data is in CSV
elif "amenity" in query or "gym" in query:
    search_in = ["jsonl", "csv"]
else:
    search_in = ["all"]
```

---

## 🙏 Credits

- **Embedding Model:** `intfloat/multilingual-e5-base` (HuggingFace)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **PDF Parsing:** [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Web Scraping:** BeautifulSoup4 + Selenium
- **Project:** Praangan Elitus, Naroda Ahmedabad

