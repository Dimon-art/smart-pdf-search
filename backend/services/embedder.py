import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Если в документе меньше этого числа слов — чанкуем по блокам
# (1 блок = 1 чанк), чтобы подсветка была точечной, а не на весь файл.
SMALL_DOC_WORDS_THRESHOLD = 1500


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


def _block_refs_for_block(block_idx_on_page: int, page: int, block: dict) -> tuple:
    """Возвращает (pages, bboxes, block_refs) для одного блока."""
    pages = [page]
    bboxes = [{"page": page, "bbox": block.get("bbox")}]
    block_refs = [[page, block_idx_on_page]]
    return pages, bboxes, block_refs


def chunk_blocks_with_pages(
    blocks: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Разбивает блоки документа на чанки с сохранением привязки к страницам.

    Логика:
      - если всего слов <= SMALL_DOC_WORDS_THRESHOLD → 1 блок = 1 чанк
        (мелкая гранулярность, точная подсветка);
      - иначе → окна по chunk_size слов с overlap.

    Args:
        blocks: список блоков от extract_blocks_with_pages (PDF) или
            document_parser (DOCX/TXT/MD). Каждый: {"text": str,
            "bbox": [x0, y0, x1, y1] | None, "page": int}.
        chunk_size: количество слов в чанке (для больших документов).
        overlap: перекрытие между соседними чанками (для больших документов).

    Returns:
        Список чанков вида:
        {
            "text": str,
            "pages": list[int],
            "bboxes": list[dict],
            "block_refs": list[list[int]],  # [[page, block_idx_in_page], ...]
        }
    """
    if not blocks:
        return []

    # Общее число слов — определяет режим чанкинга
    total_words = sum(len(b.get("text", "").split()) for b in blocks)
    small_doc = total_words <= SMALL_DOC_WORDS_THRESHOLD

    # Общий префикс: ссылки на блоки (page, idx_on_page)
    page_counters = {}
    block_refs_flat = []  # для каждого блока: (page, idx_on_page)
    for block in blocks:
        page = block.get("page", 0)
        idx_on_page = page_counters.get(page, 0)
        page_counters[page] = idx_on_page + 1
        block_refs_flat.append((page, idx_on_page))

    # ---------- Режим «маленький документ»: 1 блок = 1 чанк ----------
    if small_doc:
        chunks = []
        for i, block in enumerate(blocks):
            text = block.get("text", "").strip()
            if not text:
                continue
            page, idx_on_page = block_refs_flat[i]
            pages, bboxes, block_refs = _block_refs_for_block(idx_on_page, page, block)
            chunks.append({
                "text": text,
                "pages": pages,
                "bboxes": bboxes,
                "block_refs": block_refs,
            })
        return chunks

    # ---------- Режим «большой документ»: окна по chunk_size слов ----------
    words = []
    word_block_pos = []  # индекс блока в blocks[] для каждого слова
    for i, block in enumerate(blocks):
        for word in block.get("text", "").split():
            words.append(word)
            word_block_pos.append(i)

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

        # Уникальные блоки в окне (сохраняем порядок первого появления)
        seen = set()
        refs_in_window = []
        for i in word_block_pos[start:end]:
            if i not in seen:
                seen.add(i)
                refs_in_window.append(i)

        pages = []
        bboxes = []
        block_refs = []
        for i in refs_in_window:
            block = blocks[i]
            page, idx_on_page = block_refs_flat[i]
            if page not in pages:
                pages.append(page)
            bboxes.append({"page": page, "bbox": block.get("bbox")})
            block_refs.append([page, idx_on_page])

        chunks.append({
            "text": chunk_text,
            "pages": sorted(pages),
            "bboxes": bboxes,
            "block_refs": block_refs,
        })

        if end >= len(words):
            break

    return chunks