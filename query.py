import re
import torch
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tts import SvaraTTS


# ── Config ───────────────────────────────────────────────────
COLLECTION_NAME      = "praangan_elitus"
SIMILARITY_THRESHOLD = 0.55
TOP_K                = 2
COMPRESS_THRESHOLD   = 0.55
MIN_SENTENCES_KEEP   = 1
# LLM_MODEL            = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS       = 700

SOURCE_EMOJI = {
    "pdf":      "📄",
    "jsonl":    "📋",
    "txt":      "📃",
    "docx":     "📝",
    "website":  "🌐",
    "blog":     "📰",
    "database": "🗄️",
    "csv":      "📊",
}
# ─────────────────────────────────────────────────────────────

# ── Runtime state ───────────────────────────────────────────
embed_model = None
tokenizer = None
llm = None
tts = None
client = None
collection = None


def detect_language(text: str) -> str:
    if not text:
        return "English"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "Gujarati"
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    return "English"


def build_answer_prompt(user_query: str, context: str, language: str) -> str:
    return f"""You are a helpful real estate assistant for Praangan Elitus, a luxury residential project in Naroda, Ahmedabad.

IDENTITY RULES:
- You are the Praangan Elitus assistant. Never reveal you are an AI model, never mention Alibaba Cloud, Qwen, or any technology.
- If asked who you are, say: "I am the Praangan Elitus assistant, here to help you with information about our project."

LANGUAGE RULES:
- Detect the user language from the question and answer in exactly the same language.
- If the question is Hindi, answer in Hindi.
- If the question is Gujarati, answer in Gujarati.
- If the question is English, answer in English.
- If the retrieved context is in another language, translate the answer into {language} while keeping the meaning.
- Never switch languages in the middle of the answer.

PRICING RULES:
- If the customer asks about price, cost, rate, payment plan, EMI, or budget:
  Reply ONLY: "For pricing details, please contact our developer directly or visit our site for a detailed discussion."
- Never guess or mention any price figures.

OUT OF SCOPE RULES:
- If the question is not related to Praangan Elitus:
  Reply ONLY: "I don't have information outside of Praangan Elitus. If you have any questions about our project, I'm here to help!"
- Never answer questions outside the project scope.

ACCURACY RULES:
- Answer ONLY from the context provided below.
- If a detail like phone number, email, address, price, or date is NOT present in the context — do NOT mention it.
- Never assume, guess, or fabricate any contact details, numbers, emails, or specifications.
- If the answer is not in the context, reply ONLY:
  "I don't have that detail right now. Please contact our developer directly or visit praanganinfra.in for accurate information."

FORMAT RULES:
- Keep answers SHORT and DIRECT — maximum 3-4 sentences.
- No bold text, no markdown formatting.
- Speak like a helpful sales assistant, not a document writer.
- One clear answer, nothing extra.

Context:
{context}

Question: {user_query}

Answer in {language} (short, direct, no extra details):"""


def initialize_runtime():
    global embed_model, tokenizer, llm, tts, client, collection

    if embed_model is not None:
        return

    print("🔄 Loading embedding model (E5)...")
    embed_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
    print("   ✅ E5 embedding model ready\n")

    # print(f"🔄 Loading LLM: {LLM_MODEL}")
    # print("   (First run downloads the model on first use — please wait...)")
    # tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    # llm = AutoModelForCausalLM.from_pretrained(
    #     LLM_MODEL,
    #     torch_dtype=torch.float32,
    #     device_map="cpu"
    # )
    # llm.eval()
    # print("   ✅ LLM ready\n")

    # print("TTS model loading")
    # tts = SvaraTTS()
    # print(" ✅  TTS model loaded")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(COLLECTION_NAME)

# ── MMR Re-ranker ────────────────────────────────────────────
def mmr_rerank(query_embedding, documents, embeddings, top_k=3, lambda_param=0.7):
    query_vec = np.array(query_embedding)
    doc_vecs  = np.array(embeddings)

    relevance = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )

    selected_idx  = []
    remaining_idx = list(range(len(documents)))

    while len(selected_idx) < top_k and remaining_idx:
        if not selected_idx:
            best = max(remaining_idx, key=lambda i: relevance[i])
        else:
            selected_vecs = doc_vecs[selected_idx]
            scores = []
            for i in remaining_idx:
                rel = relevance[i]
                sim_to_selected = max(
                    np.dot(doc_vecs[i], sv) /
                    (np.linalg.norm(doc_vecs[i]) * np.linalg.norm(sv) + 1e-10)
                    for sv in selected_vecs
                )
                scores.append((i, lambda_param * rel - (1 - lambda_param) * sim_to_selected))
            best = max(scores, key=lambda x: x[1])[0]

        selected_idx.append(best)
        remaining_idx.remove(best)

    return selected_idx

# ── Context Compressor ───────────────────────────────────────
def compress_context(user_query: str, retrieved: list) -> list:
    """
    Sentence-level compressor using E5 model.
    Keeps only sentences relevant to the query.
    """
    query_vec = np.array(
        embed_model.encode(f"query: {user_query}", normalize_embeddings=True)
    )

    compressed_results = []

    for chunk_data in retrieved:
        doc = chunk_data["chunk"]

        # Split into sentences (handles English + Hindi । + Gujarati)
        raw_sentences = re.split(r'(?<=[।.!?])\s+', doc)
        sentences = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

        # Too short to compress — keep as is
        if len(sentences) <= 2:
            compressed_results.append({
                **chunk_data,
                "compressed":        doc,
                "original_words":    len(doc.split()),
                "compressed_words":  len(doc.split()),
                "compression_ratio": 1.0,
                "sentences_kept":    len(sentences),
                "sentences_total":   len(sentences),
            })
            continue

        # Embed sentences with E5 passage prefix
        prefixed = [f"passage: {s}" for s in sentences]
        sent_vecs = embed_model.encode(prefixed, normalize_embeddings=True)

        # Score each sentence vs query
        scores = np.dot(sent_vecs, query_vec)

        # Keep sentences above threshold
        keep_indices = [i for i, s in enumerate(scores) if s >= COMPRESS_THRESHOLD]

        # Always keep at least MIN_SENTENCES_KEEP
        if len(keep_indices) < MIN_SENTENCES_KEEP:
            top_indices = np.argsort(scores)[::-1][:MIN_SENTENCES_KEEP]
            keep_indices = sorted(top_indices.tolist())

        keep_indices = sorted(set(keep_indices))
        compressed_text = " ".join(sentences[i] for i in keep_indices)

        original_words   = len(doc.split())
        compressed_words = len(compressed_text.split())

        compressed_results.append({
            **chunk_data,
            "compressed":        compressed_text,
            "original_words":    original_words,
            "compressed_words":  compressed_words,
            "compression_ratio": round(compressed_words / max(original_words, 1), 2),
            "sentences_kept":    len(keep_indices),
            "sentences_total":   len(sentences),
        })

    return compressed_results

# ── LLM Answer Generator ─────────────────────────────────────
def _generate_text(prompt: str) -> str:
    initialize_runtime()

    messages = [{"role": "user", "content": prompt}]

    # Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_answer(user_query: str, compressed_chunks: list) -> str:
    """
    Build prompt from compressed context and generate
    answer using the multilingual LLM.
    """
    initialize_runtime()

    language = detect_language(user_query)

    context_parts = []
    for c in compressed_chunks:
        source = f"[{c['source_type'].upper()} | {c['source_file']}]"
        context_parts.append(f"{source}\n{c['compressed']}")
    context = "\n\n".join(context_parts)


    prompt = build_answer_prompt(user_query, context, language)
    answer = _generate_text(prompt)

    if language != "English" and answer:
        translate_prompt = f"""Translate the following answer into {language}. Keep it short, natural, and do not add extra information.

Answer:
{answer}

Translated {language} answer:"""
        answer = _generate_text(translate_prompt)

    return answer.strip()

# ── Main Query Function ───────────────────────────────────────
def query_rag(user_query: str):
    initialize_runtime()

    print(f"\n{'─'*60}")
    print(f"🔍 Query: {user_query}")

    # Step 1 — Embed query
    query_embedding = embed_model.encode(
        f"query: {user_query}",
        normalize_embeddings=True
    ).tolist()

    # print(f"query_embedding {query_embedding}")
    # Step 2 — ChromaDB semantic search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K * 4,
        include=["documents", "metadatas", "distances", "embeddings"]
    )

    print("=" * 60 + "\n")
    # print(f"semantic search Result {results}")
    print("=" * 60 + "\n")


    documents  = results["documents"][0]
    distances  = results["distances"][0]
    metadatas  = results["metadatas"][0]
    embeddings = results["embeddings"][0]

    similarities = [round(1 - d, 3) for d in distances]
    best_score   = similarities[0]

    print(f"📊 Best embed score   : {best_score}  (threshold: {SIMILARITY_THRESHOLD})")

    if best_score < SIMILARITY_THRESHOLD:
        if best_score >= 0.45:
            print("⚠️  Low relevance, but continuing with the best available context\n")
        else:
            print("🚫 REJECTED — Out of domain / Low relevance\n")
            return None

    # Step 3 — MMR diversity filter
    mmr_indices = mmr_rerank(
        query_embedding, documents, embeddings,
        top_k=TOP_K * 2,
        lambda_param=0.7
    )
    mmr_docs  = [documents[i]  for i in mmr_indices]
    mmr_metas = [metadatas[i]  for i in mmr_indices]
    mmr_sims  = [similarities[i] for i in mmr_indices]

    print(f"✅ MMR selected {len(mmr_docs)} diverse candidates")

    # Step 4 — Build retrieved list
    retrieved = []
    for doc, meta, score in zip(mmr_docs[:TOP_K], mmr_metas[:TOP_K], mmr_sims[:TOP_K]):
        retrieved.append({
            "chunk":       doc,
            "embed_score": score,
            "source_type": meta.get("type",   "unknown"),
            "source_file": meta.get("source", "unknown"),
        })

    print("=" * 60 + "\n")
    print(f"retrieved {retrieved}")
    print("=" * 60 + "\n")


    # Step 5 — Compress context
    print(f"🗜️  Compressing context (threshold={COMPRESS_THRESHOLD})...")
    compressed = compress_context(user_query, retrieved)

    total_before = sum(c["original_words"]   for c in compressed)
    total_after  = sum(c["compressed_words"] for c in compressed)
    saved_pct    = round((1 - total_after / max(total_before, 1)) * 100)

    print(f"   → {total_before} → {total_after} words ({saved_pct}% reduced)\n")

    # Show compressed chunks
    print(f"{'─'*60}")
    for rank, c in enumerate(compressed):
        emoji = SOURCE_EMOJI.get(c["source_type"], "📁")
        print(f"┌─ Chunk {rank+1} {'─'*42}")
        print(f"│  {emoji}  {c['source_type'].upper()} | {c['source_file']}")
        print(f"│  💯 Score : {c['embed_score']}  "
              f"🗜️  {c['sentences_kept']}/{c['sentences_total']} sentences kept")
        print(f"│  📝 {c['compressed']}")
        print(f"└{'─'*50}\n")

    return compressed

# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    initialize_runtime()
    total = collection.count()
    print(f"📦 ChromaDB — {total} chunks total")
    print(f"⚙️  Embed: E5 | Threshold: {SIMILARITY_THRESHOLD} | "
          f"Top-K: {TOP_K} | Compress: {COMPRESS_THRESHOLD}")
    print(f"🤖 LLM: {LLM_MODEL}\n")

    while True:
        q = input("Enter your query (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        if not q:
            continue

        compressed = query_rag(q)

        print(f"compressed {compressed}")

        if compressed:
            print("🤖 Generating answer (may take 15-30 sec on CPU)...\n")
            # answer = generate_answer(q, compressed)
            print("=" * 60)
            print("💬 ANSWER:")
            print("=" * 60)
            # print(answer)
            print("=" * 60 + "\n")

            # ── Speak the answer ──────────────────────────
            # tts.speak(answer)   # auto-detects Hindi/Gujarati/English


#### 


# import re
# import torch
# import numpy as np
# import chromadb
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tts import SvaraTTS

# # ── Config ───────────────────────────────────────────────────
# COLLECTION_NAME         = "praangan_elitus"
# SIMILARITY_THRESHOLD_EN = 0.78   # English — strict
# SIMILARITY_THRESHOLD_HI = 0.74   # Hindi — relaxed
# SIMILARITY_THRESHOLD_GU = 0.74   # Gujarati — relaxed
# TOP_K                   = 5
# COMPRESS_THRESHOLD      = 0.55
# MIN_SENTENCES_KEEP      = 1
# LLM_MODEL               = "Qwen/Qwen2.5-0.5B-Instruct"
# MAX_NEW_TOKENS          = 120

# SOURCE_EMOJI = {
#     "pdf":      "📄",
#     "jsonl":    "📋",
#     "txt":      "📃",
#     "docx":     "📝",
#     "website":  "🌐",
#     "blog":     "📰",
#     "database": "🗄️",
#     "csv":      "📊",
# }

# # ── Hard Rule Keyword Lists ───────────────────────────────────
# PRICING_KW = [
#     "price","cost","rate","budget","emi","payment","charge","fees",
#     "amount","lakh","crore","afford","expensive","cheap","discount",
#     "किंमत","कीमत","ભाવ","दाम","रेट","कितना","कितने",
#     "पैसा","रुपए","किमत","registration cost","stamp duty",
#     "maintenance charge","loan","resale","pmay","subsidy",
#     "installment","down payment","booking amount","token"
# ]
# CONTACT_KW = [
#     "contact","phone","number","email","address","reach","call",
#     "whatsapp","helpline","enquiry","inquiry","book","register",
#     "site visit","appointment","meet","sales","office",
#     "संपर्क","फोन","नंबर","ईमेल","bata do"
# ]
# IDENTITY_KW = [
#     "who are you","what are you","are you ai","are you robot",
#     "are you chatgpt","are you human","made by","created by",
#     "which model","what model","your name","who made",
#     "forget your instructions","ignore instructions","system prompt",
#     "act as","pretend you are","you are now","new persona",
#     "तुम कौन","आप कौन","तुम क्या","किसने बनाया",
# ]
# OUT_OF_SCOPE_KW = [
#     "weather","temperature","rain","forecast","mausam","climate",
#     "humidity","sunny","cloudy","storm","aaj ka mausam",
#     "cricket","ipl","bcci","football","match","score","fifa",
#     "sports","team","player","tournament",
#     "bollywood","film","actor","actress","series","netflix",
#     "politics","election","government","minister","prime minister",
#     "president","party","vote","bjp","congress",
#     "recipe","cook","biryani","restaurant","dal","sabzi",
#     "khana","cuisine","menu","dinner","lunch","breakfast",
#     "stock","share","market","sensex","nifty","bitcoin","crypto",
#     "mutual fund","gold price","silver","forex","dollar","rupee rate",
#     "capital","country","population","history","science","math",
#     "joke","funny","story","poem","song","lyrics","music",
#     "news","headline","breaking","latest news","today news",
#     "मौसम","बारिश","क्रिकेट","राजनीति","खाना","फिल्म","शेयर",
#     "मज़ाक","कहानी","गाना","समाचार",
#     "હavaman","वरसाद","राजकारण","खावानु",
# ]
# COMPETITOR_KW = [
#     "compare","better than","vs","versus","adani","godrej",
#     "lodha","sobha","prestige","brigade","other project",
#     "which is better","difference between"
# ]

# # ── Hallucination Signals (only truly fake patterns) ─────────
# HALLUCINATION_SIGNALS = [
#     "i think","i believe","i'm not sure","i am not sure",
#     "probably","approximately","around","estimated",
#     "maybe","could be","might be","generally","typically","usually",
#     "as per my knowledge","based on my training",
#     "various options","multiple courses","courses per person",
#     "additionally they","these facilities ensure",
#     "enjoyable experience","diverse needs","accommodating",
#     "larger-scale","tailored for larger",
#     "separate tables or booths","specifically tailored",
# ]

# # ── Canned Responses (EN / HI / GU) ──────────────────────────
# RESPONSES = {
#     "pricing": {
#         "English":  "For pricing details, please contact our sales team directly or visit praanganinfra.in for a detailed discussion.",
#         "Hindi":    "कीमत की जानकारी के लिए कृपया हमारी सेल्स टीम से संपर्क करें या praanganinfra.in पर जाएं।",
#         "Gujarati": "Bhav maate krupa karine sales team no sampark karo athva praanganinfra.in ni mulakat lo.",
#     },
#     "contact": {
#         "English":  "Please contact our sales team directly or visit praanganinfra.in to connect with us.",
#         "Hindi":    "कृपया हमारी सेल्स टीम से सीधे संपर्क करें या praanganinfra.in पर जाएं।",
#         "Gujarati": "Krupa karine sales team no sampark karo athva praanganinfra.in ni mulakat lo.",
#     },
#     "identity": {
#         "English":  "I am the Praangan Elitus assistant, here to help you with information about our luxury project in Ahmedabad.",
#         "Hindi":    "मैं प्रांगण एलीटस का सहायक हूं। अहमदाबाद में हमारे लक्जरी प्रोजेक्ट के बारे में मदद के लिए यहां हूं।",
#         "Gujarati": "Hu Praangan Elitus assistant chu. Ahmedabad ma amara luxury project vishe tamne madad karva hazu chu.",
#     },
#     "out_of_scope": {
#         "English":  "I only have information about Praangan Elitus. For any questions about our project, I am here to help!",
#         "Hindi":    "मुझे केवल प्रांगण एलीटस के बारे में जानकारी है। हमारे प्रोजेक्ट के बारे में कुछ पूछना हो तो जरूर पूछें!",
#         "Gujarati": "Mane fakt Praangan Elitus vishe jankari chhe. Amara project vishe koi prashn hoy to jarur pucho!",
#     },
#     "competitor": {
#         "English":  "I can only provide information about Praangan Elitus. We believe our project speaks for itself with its unique features and luxury offerings.",
#         "Hindi":    "मैं केवल प्रांगण एलीटस के बारे में जानकारी दे सकता हूं। हमारा प्रोजेक्ट अपनी खूबियों से खुद बोलता है।",
#         "Gujarati": "Hu fakt Praangan Elitus vishe jankari aapi sakhu chu. Amaro project pote j tamne impress karse.",
#     },
#     "not_found": {
#         "English":  "I don't have that information right now. Please contact our sales team or visit praanganinfra.in for accurate details.",
#         "Hindi":    "अभी यह जानकारी मेरे पास नहीं है। सटीक जानकारी के लिए हमारी सेल्स टीम से संपर्क करें या praanganinfra.in पर जाएं।",
#         "Gujarati": "Abhi aa jankari mara pase nathi. Sachi jankari mate sales team no sampark karo athva praanganinfra.in ni mulakat lo.",
#     },
#     "nonsense": {
#         "English":  "I didn't understand that. Could you please ask me something about Praangan Elitus?",
#         "Hindi":    "मुझे समझ नहीं आया। क्या आप प्रांगण एलीटस के बारे में कुछ पूछ सकते हैं?",
#         "Gujarati": "Mane samjayun nahi. Shya tame Praangan Elitus vishe koi prashn puchhsho?",
#     },
# }

# # ── Runtime State ─────────────────────────────────────────────
# embed_model = None
# tokenizer   = None
# llm         = None
# tts         = None
# client      = None
# collection  = None
# TTS_ENABLED = False


# # ── Language Detection ────────────────────────────────────────
# def detect_language(text: str) -> str:
#     if not text:
#         return "English"
#     gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
#     hindi    = sum(1 for c in text if '\u0900' <= c <= '\u097F')
#     if gujarati > hindi and gujarati > 1:
#         return "Gujarati"
#     if hindi > 1:
#         return "Hindi"
#     return "English"


# def get_response(rule: str, lang: str) -> str:
#     return RESPONSES.get(rule, {}).get(lang, RESPONSES[rule]["English"])


# def get_threshold(lang: str) -> float:
#     return {
#         "English":  SIMILARITY_THRESHOLD_EN,
#         "Hindi":    SIMILARITY_THRESHOLD_HI,
#         "Gujarati": SIMILARITY_THRESHOLD_GU,
#     }.get(lang, SIMILARITY_THRESHOLD_EN)


# # ── Safe TTS Speaker ──────────────────────────────────────────
# def speak(text: str):
#     if TTS_ENABLED and tts and text:
#         try:
#             tts.speak(text)
#         except Exception as e:
#             print(f"   ⚠️  TTS error: {e}")


# # ── Hard Rule Checker ─────────────────────────────────────────
# def check_hard_rules(query: str, lang: str):
#     q = query.lower().strip()

#     if len(q) <= 2 or not any(c.isalpha() for c in q):
#         return "nonsense", get_response("nonsense", lang)
#     if any(kw in q for kw in IDENTITY_KW):
#         return "identity", get_response("identity", lang)
#     if any(kw in q for kw in PRICING_KW):
#         return "pricing", get_response("pricing", lang)
#     if any(kw in q for kw in CONTACT_KW):
#         return "contact", get_response("contact", lang)
#     if any(kw in q for kw in COMPETITOR_KW):
#         return "competitor", get_response("competitor", lang)
#     if any(kw in q for kw in OUT_OF_SCOPE_KW):
#         return "out_of_scope", get_response("out_of_scope", lang)

#     return None, None


# # ── Model Initialization ──────────────────────────────────────
# def initialize_runtime():
#     global embed_model, tokenizer, llm, tts, client, collection, TTS_ENABLED

#     if embed_model is not None:
#         return

#     print("🔄 Loading E5 embedding model...")
#     embed_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
#     print("   ✅ E5 ready\n")

#     print(f"🔄 Loading LLM: {LLM_MODEL}...")
#     tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
#     llm = AutoModelForCausalLM.from_pretrained(
#         LLM_MODEL, torch_dtype=torch.float32, device_map="cpu"
#     )
#     llm.eval()
#     print(f"   ✅ LLM ready\n")

#     try:
#         print("🔄 Loading TTS models...")
#         tts = SvaraTTS()
#         TTS_ENABLED = True
#         print("   ✅ TTS ready\n")
#     except Exception as e:
#         print(f"   ⚠️  TTS not loaded: {e} — running without voice\n")
#         tts = None
#         TTS_ENABLED = False

#     client     = chromadb.PersistentClient(path="./chroma_db")
#     collection = client.get_collection(COLLECTION_NAME)
#     print(f"   ✅ ChromaDB ready ({collection.count()} chunks)\n")


# # ── MMR Re-ranker ─────────────────────────────────────────────
# def mmr_rerank(query_embedding, documents, embeddings, top_k=3, lambda_param=0.7):
#     query_vec = np.array(query_embedding)
#     doc_vecs  = np.array(embeddings)

#     relevance = np.dot(doc_vecs, query_vec) / (
#         np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-10
#     )

#     selected_idx  = []
#     remaining_idx = list(range(len(documents)))

#     while len(selected_idx) < top_k and remaining_idx:
#         if not selected_idx:
#             best = max(remaining_idx, key=lambda i: relevance[i])
#         else:
#             selected_vecs = doc_vecs[selected_idx]
#             scores = []
#             for i in remaining_idx:
#                 rel = relevance[i]
#                 sim = max(
#                     np.dot(doc_vecs[i], sv) /
#                     (np.linalg.norm(doc_vecs[i]) * np.linalg.norm(sv) + 1e-10)
#                     for sv in selected_vecs
#                 )
#                 scores.append((i, lambda_param * rel - (1 - lambda_param) * sim))
#             best = max(scores, key=lambda x: x[1])[0]
#         selected_idx.append(best)
#         remaining_idx.remove(best)

#     return selected_idx


# # ── Context Compressor ────────────────────────────────────────
# def compress_context(user_query: str, retrieved: list) -> list:
#     query_vec = np.array(
#         embed_model.encode(f"query: {user_query}", normalize_embeddings=True)
#     )
#     compressed_results = []

#     for chunk_data in retrieved:
#         doc           = chunk_data["chunk"]
#         raw_sentences = re.split(r'(?<=[।.!?])\s+', doc)
#         sentences     = [s.strip() for s in raw_sentences if len(s.strip().split()) >= 4]

#         if len(sentences) <= 2:
#             compressed_results.append({
#                 **chunk_data,
#                 "compressed":        doc,
#                 "original_words":    len(doc.split()),
#                 "compressed_words":  len(doc.split()),
#                 "compression_ratio": 1.0,
#                 "sentences_kept":    len(sentences),
#                 "sentences_total":   len(sentences),
#             })
#             continue

#         prefixed  = [f"passage: {s}" for s in sentences]
#         sent_vecs = embed_model.encode(prefixed, normalize_embeddings=True)
#         scores    = np.dot(sent_vecs, query_vec)

#         keep_indices = [i for i, s in enumerate(scores) if s >= COMPRESS_THRESHOLD]
#         if len(keep_indices) < MIN_SENTENCES_KEEP:
#             top_idx      = np.argsort(scores)[::-1][:MIN_SENTENCES_KEEP]
#             keep_indices = sorted(top_idx.tolist())

#         keep_indices    = sorted(set(keep_indices))
#         compressed_text = " ".join(sentences[i] for i in keep_indices)

#         compressed_results.append({
#             **chunk_data,
#             "compressed":        compressed_text,
#             "original_words":    len(doc.split()),
#             "compressed_words":  len(compressed_text.split()),
#             "compression_ratio": round(len(compressed_text.split()) / max(len(doc.split()), 1), 2),
#             "sentences_kept":    len(keep_indices),
#             "sentences_total":   len(sentences),
#         })

#     return compressed_results


# # ── LLM Call ─────────────────────────────────────────────────
# def _run_llm(prompt: str) -> str:
#     messages = [{"role": "user", "content": prompt}]
#     text     = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(text, return_tensors="pt")

#     with torch.no_grad():
#         outputs = llm.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=0.1,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             repetition_penalty=1.3,
#         )

#     new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
#     response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

#     response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
#     response = re.sub(r'\*(.*?)\*',     r'\1', response)
#     response = re.sub(r'#{1,6}\s*',     '',    response)
#     response = re.sub(r'^\s*[-*]\s+',   '',    response, flags=re.MULTILINE)
#     response = re.sub(r'^\d+\.\s+',     '',    response, flags=re.MULTILINE)
#     response = re.sub(r'\n{2,}',        ' ',   response)
#     return response.strip()


# def translate_to(text: str, target_lang: str) -> str:
#     if target_lang == "English":
#         return text

#     prompt = f"""Translate to {target_lang}. Output ONLY the translation, nothing else.

# Text: {text}

# {target_lang}:"""

#     translated = _run_llm(prompt)

#     if target_lang == "Hindi" and not re.search(r'[\u0900-\u097F]', translated):
#         return text
#     if target_lang == "Gujarati" and not re.search(r'[\u0A80-\u0AFF]', translated):
#         return text

#     return translated


# # ── Answer Generator ──────────────────────────────────────────
# def generate_answer(user_query: str, compressed_chunks: list,
#                     lang: str = "English") -> str:

#     rule, response = check_hard_rules(user_query, lang)
#     if rule:
#         print(f"   ⚡ Hard rule [{rule}] caught at generation stage")
#         return response

#     context = "\n".join(c["compressed"] for c in compressed_chunks if c["compressed"])

#     prompt = f"""You are a sales assistant for Praangan Elitus luxury apartments in Ahmedabad.
# Answer using ONLY the facts below. Do NOT add any info not in the context.
# If context does not have the answer, say exactly: NOT_FOUND
# Write 1-2 sentences only. Plain text only. No markdown. No lists. No bold.

# Context:
# {context}

# Question: {user_query}
# Answer:"""

#     english_answer = _run_llm(prompt)

#     if (not english_answer
#             or "NOT_FOUND" in english_answer
#             or len(english_answer.split()) < 3):
#         return get_response("not_found", lang)

#     if any(sig in english_answer.lower() for sig in HALLUCINATION_SIGNALS):
#         print(f"   ⚠️  Hallucination detected — returning not_found")
#         return get_response("not_found", lang)

#     if re.search(r'\+?\d[\d\s\-]{8,}', english_answer):
#         return get_response("contact", lang)
#     if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', english_answer):
#         return get_response("contact", lang)

#     if any(w in english_answer.lower() for w in
#            ["alibaba","qwen","openai","chatgpt","language model","created by"]):
#         return get_response("identity", lang)

#     if lang != "English":
#         print(f"   🔄 Translating to {lang}...")
#         return translate_to(english_answer, lang)

#     return english_answer


# # ── Main Query Function ───────────────────────────────────────
# def query_rag(user_query: str):
#     initialize_runtime()

#     lang      = detect_language(user_query)
#     threshold = get_threshold(lang)

#     print(f"\n{'─'*60}")
#     print(f"🔍 Query : {user_query}")
#     print(f"🌐 Lang  : {lang}")

#     # Layer 1 — Hard rules
#     rule, hard_response = check_hard_rules(user_query, lang)
#     if rule:
#         print(f"⚡ Hard rule [{rule}] — no RAG needed")
#         return hard_response, None

#     # Layer 2 — ChromaDB
#     query_embedding = embed_model.encode(
#         f"query: {user_query}", normalize_embeddings=True
#     ).tolist()

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=TOP_K * 4,
#         include=["documents", "metadatas", "distances", "embeddings"]
#     )

#     documents  = results["documents"][0]
#     distances  = results["distances"][0]
#     metadatas  = results["metadatas"][0]
#     embeddings = results["embeddings"][0]

#     similarities = [round(1 - d, 3) for d in distances]
#     best_score   = similarities[0]

#     print(f"📊 Best score : {best_score}  (threshold: {threshold})")

#     # Layer 3 — Similarity gate
#     if best_score < threshold:
#         print(f"🚫 REJECTED — score {best_score} below threshold {threshold}")
#         return get_response("out_of_scope", lang), None

#     # Layer 4 — MMR
#     mmr_indices = mmr_rerank(
#         query_embedding, documents, embeddings,
#         top_k=TOP_K * 2, lambda_param=0.7
#     )
#     mmr_docs  = [documents[i]  for i in mmr_indices]
#     mmr_metas = [metadatas[i]  for i in mmr_indices]
#     mmr_sims  = [similarities[i] for i in mmr_indices]

#     print(f"✅ MMR → {len(mmr_docs)} diverse candidates")

#     retrieved = []
#     for doc, meta, score in zip(mmr_docs[:TOP_K], mmr_metas[:TOP_K], mmr_sims[:TOP_K]):
#         retrieved.append({
#             "chunk":       doc,
#             "embed_score": score,
#             "source_type": meta.get("type",   "unknown"),
#             "source_file": meta.get("source", "unknown"),
#         })

#     # Layer 5 — Compress
#     print(f"🗜️  Compressing...")
#     compressed = compress_context(user_query, retrieved)

#     total_before = sum(c["original_words"]   for c in compressed)
#     total_after  = sum(c["compressed_words"] for c in compressed)
#     saved_pct    = round((1 - total_after / max(total_before, 1)) * 100)
#     print(f"   → {total_before} → {total_after} words ({saved_pct}% reduced)\n")

#     print(f"{'─'*60}")
#     for rank, c in enumerate(compressed):
#         emoji = SOURCE_EMOJI.get(c["source_type"], "📁")
#         print(f"┌─ Chunk {rank+1} {'─'*42}")
#         print(f"│  {emoji}  {c['source_type'].upper()} | {c['source_file']}")
#         print(f"│  💯 {c['embed_score']}  🗜️  {c['sentences_kept']}/{c['sentences_total']} sentences")
#         print(f"│  📝 {c['compressed']}")
#         print(f"└{'─'*50}\n")

#     return None, compressed


# # ── Entry Point ───────────────────────────────────────────────
# if __name__ == "__main__":
#     initialize_runtime()

#     print(f"📦 ChromaDB — {collection.count()} chunks")
#     print(f"⚙️  Threshold EN:{SIMILARITY_THRESHOLD_EN} HI/GU:{SIMILARITY_THRESHOLD_HI} | Top-K: {TOP_K} | Compress: {COMPRESS_THRESHOLD}")
#     print(f"🤖 LLM: {LLM_MODEL} | TTS: {'ON' if TTS_ENABLED else 'OFF'}\n")

#     while True:
#         q = input("Enter your query (or 'quit'): ").strip()
#         if q.lower() == "quit":
#             break
#         if not q:
#             continue

#         lang                      = detect_language(q)
#         hard_response, compressed = query_rag(q)

#         if hard_response:
#             print("=" * 60)
#             print("💬 ANSWER:")
#             print("=" * 60)
#             print(hard_response)
#             print("=" * 60 + "\n")
#             speak(hard_response)

#         elif compressed:
#             print("🤖 Generating answer...\n")
#             answer = generate_answer(q, compressed, lang=lang)
#             print("=" * 60)
#             print("💬 ANSWER:")
#             print("=" * 60)
#             print(answer)
#             print("=" * 60 + "\n")
#             speak(answer)

#         else:
#             fallback = get_response("out_of_scope", lang)
#             print(fallback + "\n")
#             speak(fallback)