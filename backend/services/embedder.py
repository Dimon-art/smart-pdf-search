import json
import os

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


def save_cache(pdf_hash: str, chunks: list[str], embeddings, index, cache_dir: str = "cache") -> None:
    """Save chunks, embeddings, and FAISS index to disk.

    Writes {pdf_hash}_chunks.json, {pdf_hash}_embeddings.npy, and
    {pdf_hash}_faiss.index under cache_dir. Creates the directory if needed.
    Prints a warning on failure instead of raising.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        chunks_path = os.path.join(cache_dir, f"{pdf_hash}_chunks.json")
        embeddings_path = os.path.join(cache_dir, f"{pdf_hash}_embeddings.npy")
        index_path = os.path.join(cache_dir, f"{pdf_hash}_faiss.index")
        with open(chunks_path, "w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=2)
        np.save(embeddings_path, embeddings)
        faiss.write_index(index, index_path)
    except Exception as exc:
        print(f"Warning: failed to save cache: {exc}")


def load_cache(pdf_hash: str, cache_dir: str = "cache"):
    """Load cached chunks, embeddings, and FAISS index from disk.

    Returns (chunks, embeddings, index) when all three files exist,
    otherwise None. Returns None on any load error.
    """
    chunks_path = os.path.join(cache_dir, f"{pdf_hash}_chunks.json")
    embeddings_path = os.path.join(cache_dir, f"{pdf_hash}_embeddings.npy")
    index_path = os.path.join(cache_dir, f"{pdf_hash}_faiss.index")
    if not (
        os.path.exists(chunks_path)
        and os.path.exists(embeddings_path)
        and os.path.exists(index_path)
    ):
        return None

    try:
        with open(chunks_path, encoding="utf-8") as file:
            chunks = json.load(file)
        embeddings = np.load(embeddings_path)
        index = faiss.read_index(index_path)
        return (chunks, embeddings, index)
    except Exception:
        return None


def chunk_blocks_with_pages(
    blocks: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Разбивает блоки PDF на чанки с сохранением привязки к страницам.

    Args:
        blocks: список блоков от extract_blocks_with_pages — каждый
            {"text": str, "bbox": [x0, y0, x1, y1], "page": int}.
        chunk_size: количество слов в чанке.
        overlap: перекрытие между соседними чанками (в словах).

    Returns:
        Список чанков вида:
        {
            "text": str,
            "pages": list[int],       # уникальные номера страниц, отсортированные
            "bboxes": list[dict],     # [{"page": int, "bbox": [...]}, ...]
        }
        Пустые чанки не добавляются.
    """
    if not blocks:
        return []

    # Собираем плоский список слов с привязкой к блоку
    words = []
    word_block_idx = []
    for block_idx, block in enumerate(blocks):
        block_words = block["text"].split()
        for word in block_words:
            words.append(word)
            word_block_idx.append(block_idx)

    if not words:
        return []

    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    chunks = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        window_words = words[start:end]
        if not window_words:
            continue

        chunk_text = " ".join(window_words).strip()
        if not chunk_text:
            continue

        # Какие блоки попали в окно
        block_indices_in_window = sorted(set(word_block_idx[start:end]))

        pages = []
        bboxes = []
        for idx in block_indices_in_window:
            block = blocks[idx]
            if block["page"] not in pages:
                pages.append(block["page"])
            bboxes.append({
                "page": block["page"],
                "bbox": block["bbox"],
            })

        chunks.append({
            "text": chunk_text,
            "pages": sorted(pages),
            "bboxes": bboxes,
        })

        if end >= len(words):
            break

    return chunks
