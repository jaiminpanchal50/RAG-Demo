import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────
COLLECTION_NAME      = "praangan_elitus"
SIMILARITY_THRESHOLD = 0.83
TOP_K                = 3

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

print("🔄 Loading model...")
model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(COLLECTION_NAME)

# ── MMR Re-ranker ────────────────────────────────────────────
def mmr_rerank(query_embedding, documents, embeddings, top_k=3, lambda_param=0.7):
    """
    Maximal Marginal Relevance:
    Picks chunks that are relevant AND different from each other.
    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
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
                mmr_score = lambda_param * rel - (1 - lambda_param) * sim_to_selected
                scores.append((i, mmr_score))
            best = max(scores, key=lambda x: x[1])[0]

        selected_idx.append(best)
        remaining_idx.remove(best)

    return selected_idx

# ── Main Query Function ───────────────────────────────────────
def query_rag(user_query: str):
    print(f"\n{'─'*60}")
    print(f"🔍 Query: {user_query}")

    query_embedding = model.encode(f"query: {user_query}").tolist()

    # Fetch 4x more candidates so MMR has room to pick diverse ones
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K * 4,
        include=["documents", "metadatas", "distances", "embeddings"]
    )

    documents  = results["documents"][0]
    distances  = results["distances"][0]
    metadatas  = results["metadatas"][0]
    embeddings = results["embeddings"][0]

    similarities = [round(1 - d, 3) for d in distances]
    best_score   = similarities[0]

    print(f"📊 Best similarity score : {best_score}  (threshold: {SIMILARITY_THRESHOLD})")

    # ── Retrieval Gate ───────────────────────────────────────
    if best_score < SIMILARITY_THRESHOLD:
        print("🚫 REJECTED — Out of domain / Low relevance\n")
        return None

    # ── MMR Diversity Reranking ──────────────────────────────
    mmr_indices = mmr_rerank(
        query_embedding, documents, embeddings,
        top_k=TOP_K, lambda_param=0.7
    )

    print(f"✅ PASSED — Returning top {TOP_K} diverse chunks\n")

    retrieved = []
    for rank, idx in enumerate(mmr_indices):
        doc         = documents[idx]
        score       = similarities[idx]
        meta        = metadatas[idx]
        source_type = meta.get("type", "unknown")
        source_file = meta.get("source", "unknown")
        emoji       = SOURCE_EMOJI.get(source_type, "📁")

        print(f"┌─ Chunk {rank+1} {'─'*40}")
        print(f"│  {emoji}  Source Type : {source_type.upper()}")
        print(f"│  📂 File       : {source_file}")
        print(f"│  💯 Score      : {score}")
        print(f"│  📝 Content    :")
        print(f"│     {doc[:100300]}")
        print(f"└{'─'*50}\n")

        retrieved.append({
            "chunk":       doc,
            "score":       score,
            "source_type": source_type,
            "source_file": source_file
        })

    return retrieved

# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    total = collection.count()
    print(f"\n📦 ChromaDB loaded — {total} chunks total")
    print(f"⚙️  Threshold : {SIMILARITY_THRESHOLD} | Top-K : {TOP_K}\n")

    while True:
        q = input("Enter your query (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        if not q:
            continue
        query_rag(q)