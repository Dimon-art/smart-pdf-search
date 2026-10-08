import hashlib
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from services.pdf_parser import extract_blocks_with_pages, clean_text
from services.embedder import (
    chunk_blocks_with_pages,
    build_embeddings,
    build_faiss_index,
    search,
    save_cache,
    load_cache,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
CURRENT_PDF = UPLOADS_DIR / "current.pdf"
DEFAULT_PDF = BASE_DIR / "sample.pdf"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

CHUNKS = None
INDEX = None
MODEL = None
CURRENT_PDF_NAME = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


def _build_index_for_pdf(pdf_path: Path, pdf_name: str) -> None:
    """Parse pdf_path and populate CHUNKS, INDEX, MODEL."""
    global CHUNKS, INDEX, MODEL, CURRENT_PDF_NAME

    pdf_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    print(f"PDF hash: {pdf_hash[:8]}... ({pdf_name})")

    cached = load_cache(pdf_hash)
    if cached is not None:
        chunks, _embeddings, index = cached
        print("Загружено из кэша")
        model = SentenceTransformer(MODEL_NAME)
    else:
        print("Кэш не найден, парсим PDF...")
        blocks = extract_blocks_with_pages(str(pdf_path))
        print(f"Блоков: {len(blocks)}")
        chunks = chunk_blocks_with_pages(blocks, chunk_size=500, overlap=100)
        print(f"Чанков: {len(chunks)}")
        texts = [c["text"] for c in chunks]
        embeddings, model = build_embeddings(texts)
        print(f"Размер эмбеддингов: {embeddings.shape}")
        index = build_faiss_index(embeddings)
        print(f"FAISS-индекс: {index.ntotal} векторов")
        save_cache(pdf_hash, chunks, embeddings, index)
        print("Кэш сохранён")

    CHUNKS = chunks
    INDEX = index
    MODEL = model
    CURRENT_PDF_NAME = pdf_name
    print(f"PDF готов: {pdf_name}")


def startup() -> None:
    """On server start, load the default sample.pdf."""
    if not DEFAULT_PDF.exists():
        raise FileNotFoundError(
            f"sample.pdf не найден: {DEFAULT_PDF}. Положите его в корень проекта."
        )
    _build_index_for_pdf(DEFAULT_PDF, "sample.pdf")
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


@app.get("/static/current.pdf")
def serve_current_pdf():
    """Serve the currently active PDF (either default or uploaded)."""
    if CURRENT_PDF.exists():
        return FileResponse(CURRENT_PDF, media_type="application/pdf")
    if DEFAULT_PDF.exists():
        return FileResponse(DEFAULT_PDF, media_type="application/pdf")
    raise HTTPException(status_code=404, detail="PDF не найден")


@app.get("/status")
def status():
    """Return info about the currently active PDF."""
    return {
        "pdf_name": CURRENT_PDF_NAME,
        "chunks_count": len(CHUNKS) if CHUNKS else 0,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a new PDF, parse it, build its index, and make it active."""
    global CURRENT_PDF_NAME

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF-файлы")

    # Читаем в память с ограничением размера
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой: {len(content) // 1024 // 1024} МБ (лимит 100 МБ)",
        )

    print(f"Загрузка: {file.filename} ({len(content) // 1024} КБ)")

    # Сохраняем как current.pdf
    CURRENT_PDF.write_bytes(content)

    # Индексируем
    try:
        _build_index_for_pdf(CURRENT_PDF, file.filename)
    except Exception as exc:
        print(f"Ошибка индексации: {exc}")
        raise HTTPException(status_code=500, detail=f"Ошибка индексации: {exc}")

    return {
        "status": "ok",
        "pdf_name": file.filename,
        "chunks_count": len(CHUNKS) if CHUNKS else 0,
    }


@app.post("/search")
def search_pdf(request: SearchRequest):
    """Return top_k semantic search hits for the given query."""
    if MODEL is None or INDEX is None or CHUNKS is None:
        raise HTTPException(status_code=503, detail="Индекс ещё не готов")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query не должен быть пустым")

    print(f"Поиск: {request.query!r}, top_k={request.top_k}")

    texts = [c["text"] for c in CHUNKS]
    raw_results = search(INDEX, MODEL, request.query, texts, top_k=request.top_k)

    results = []
    for r in raw_results:
        idx = next(
            (i for i, c in enumerate(CHUNKS) if c["text"] == r["chunk"]),
            None,
        )
        item = {
            "rank": r["rank"],
            "score": r["score"],
            "chunk": r["chunk"],
        }
        if idx is not None:
            item["pages"] = CHUNKS[idx]["pages"]
            item["bboxes"] = CHUNKS[idx]["bboxes"]
        results.append(item)

    return {"query": request.query, "results": results}