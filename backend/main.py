import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from services.pdf_parser import extract_pages_with_bbox, clean_text
from services.embedder import (
    chunk_text,
    build_embeddings,
    build_faiss_index,
    search,
    save_cache,
    load_cache,
)

PDF_PATH = Path(__file__).resolve().parent.parent / "sample.pdf"
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNKS = None
INDEX = None
MODEL = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def startup() -> None:
    """Load cache or parse sample.pdf into CHUNKS, INDEX, and MODEL."""
    global CHUNKS, INDEX, MODEL

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF не найден: {PDF_PATH}. Положите sample.pdf в корень проекта."
        )

    pdf_hash = hashlib.md5(PDF_PATH.read_bytes()).hexdigest()
    print(f"PDF hash: {pdf_hash[:8]}...")

    cached = load_cache(pdf_hash)
    if cached is not None:
        chunks, _embeddings, index = cached
        print("Загружено из кэша")
        model = SentenceTransformer(MODEL_NAME)
    else:
        print("Кэш не найден, парсим PDF...")
        pages = extract_pages_with_bbox(str(PDF_PATH))
        full_text = clean_text(" ".join(b["text"] for p in pages for b in p["blocks"]))
        print(f"Длина текста: {len(full_text)}")
        chunks = chunk_text(full_text, chunk_size=500, overlap=100)
        print(f"Чанков: {len(chunks)}")
        embeddings, model = build_embeddings(chunks)
        print(f"Размер эмбеддингов: {embeddings.shape}")
        index = build_faiss_index(embeddings)
        print(f"FAISS-индекс: {index.ntotal} векторов")
        save_cache(pdf_hash, chunks, embeddings, index)
        print("Кэш сохранён")

    CHUNKS = chunks
    INDEX = index
    MODEL = model
    print("Сервер готов к поиску")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup() before serving requests."""
    startup()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    """Serve frontend/index.html."""
    return FileResponse(FRONTEND_PATH)


@app.post("/search")
def search_pdf(request: SearchRequest):
    """Return top_k semantic search hits for the given query."""
    if MODEL is None or INDEX is None or CHUNKS is None:
        raise HTTPException(status_code=503, detail="Индекс ещё не готов")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query не должен быть пустым")

    print(f"Поиск: {request.query!r}, top_k={request.top_k}")
    results = search(INDEX, MODEL, request.query, CHUNKS, top_k=request.top_k)
    return {"query": request.query, "results": results}
