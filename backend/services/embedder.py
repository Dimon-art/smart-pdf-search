import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word windows.

    Words come from text.split(). Each chunk has at most chunk_size words;
    the window advances by chunk_size - overlap. A short text becomes a
    single chunk. Empty text or empty chunks are not returned.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        chunk = " ".join(words)
        return [chunk] if chunk else []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def build_embeddings(chunks: list[str]) -> tuple:
    """Encode text chunks with a multilingual MiniLM model.

    Loads SentenceTransformer inside this function. Returns (embeddings, model),
    or (None, None) when chunks is empty.
    """
    if not chunks:
        return (None, None)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return (embeddings, model)


def build_faiss_index(embeddings) -> faiss.Index:
    """Build an L2 FAISS index from embedding vectors.

    Uses IndexFlatL2 with dimension embeddings.shape[1]. Returns None when
    embeddings is None or empty.
    """
    if embeddings is None:
        return None

    vectors = np.asarray(embeddings)
    if vectors.size == 0:
        return None

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype("float32"))
    return index


def search(index, model, query: str, chunks: list[str], top_k: int = 5) -> list[dict]:
    """Find the nearest chunks to query by L2 distance.

    Returns dicts with chunk, score (L2 distance, lower is better), and
    1-based rank. Returns [] when index, model, query, or chunks is empty.
    """
    if index is None or model is None or not query or not chunks:
        return []

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype("float32"), top_k)

    results = []
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(
            {
                "chunk": chunks[idx],
                "score": float(distance),
                "rank": rank,
            }
        )
    return results
