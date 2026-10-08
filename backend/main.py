import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
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
from services.pdf_editor import find_matches, replace_matches

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
CURRENT_PDF = UPLOADS_DIR / "current.pdf"
EDITED_PDF = UPLOADS_DIR / "edited.pdf"
DEFAULT_PDF = BASE_DIR / "sample.pdf"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

CHUNKS = None
INDEX = None
MODEL = None
CURRENT_PDF_NAME = None

JOB_STATE = {
    "running": False,
    "stage": "idle",
    "progress": 0,
    "error": None,
}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class FindMatchesRequest(BaseModel):
    old_text: str


class ReplaceRequest(BaseModel):
    old_text: str
    new_text: str
    selected_ids: list[int]


def _set_stage(stage: str, progress: int) -> None:
    """Update JOB_STATE with current stage and progress."""
    JOB_STATE["stage"] = stage
    JOB_STATE["progress"] = progress
    print(f"[job] {stage} — {progress}%")


def _build_index_for_pdf(pdf_path: Path, pdf_name: str) -> None:
    """Parse pdf_path and populate CHUNKS, INDEX, MODEL. Updates JOB_STATE."""
    global CHUNKS, INDEX, MODEL, CURRENT_PDF_NAME

    _set_stage("parsing", 5)
    pdf_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    print(f"PDF hash: {pdf_hash[:8]}... ({pdf_name})")

    cached = load_cache(pdf_hash)
    if cached is not None:
        _set_stage("loading_cache", 50)
        chunks, _embeddings, index = cached
        print("Загружено из кэша")
        model = SentenceTransformer(MODEL_NAME)
    else:
        _set_stage("parsing", 15)
        blocks = extract_blocks_with_pages(str(pdf_path))
        print(f"Блоков: {len(blocks)}")

        _set_stage("chunking", 30)
        chunks = chunk_blocks_with_pages(blocks, chunk_size=500, overlap=100)
        print(f"Чанков: {len(chunks)}")

        _set_stage("embedding", 40)
        texts = [c["text"] for c in chunks]
        embeddings, model = build_embeddings(texts)
        print(f"Размер эмбеддингов: {embeddings.shape}")

        _set_stage("saving", 90)
        index = build_faiss_index(embeddings)
        print(f"FAISS-индекс: {index.ntotal} векторов")
        save_cache(pdf_hash, chunks, embeddings, index)
        print("Кэш сохранён")

    CHUNKS = chunks
    INDEX = index
    MODEL = model
    CURRENT_PDF_NAME = pdf_name
    _set_stage("ready", 100)
    print(f"PDF готов: {pdf_name}")


def startup() -> None:
    """On server start, load the default sample.pdf."""
    if not DEFAULT_PDF.exists():
        raise FileNotFoundError(
            f"sample.pdf не найден: {DEFAULT_PDF}. Положите его в корень проекта."
        )
    _build_index_for_pdf(DEFAULT_PDF, "sample.pdf")
    JOB_STATE["stage"] = "ready"
    JOB_STATE["progress"] = 100
    print("Сервер готов к поиску")


def _process_upload(pdf_bytes: bytes, pdf_name: str) -> None:
    """Background task: save uploaded PDF and build its index."""
    try:
        CURRENT_PDF.write_bytes(pdf_bytes)
        _build_index_for_pdf(CURRENT_PDF, pdf_name)
    except Exception as exc:
        JOB_STATE["stage"] = "error"
        JOB_STATE["error"] = str(exc)
        print(f"Ошибка обработки upload: {exc}")
    finally:
        JOB_STATE["running"] = False


def _active_pdf_path() -> Path:
    """Return the currently active PDF path (uploaded or default)."""
    if CURRENT_PDF.exists():
        return CURRENT_PDF
    return DEFAULT_PDF


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
    """Serve the currently active PDF."""
    return FileResponse(_active_pdf_path(), media_type="application/pdf")


@app.get("/static/edited.pdf")
def serve_edited_pdf():
    """Serve the most recently edited PDF (if exists)."""
    if not EDITED_PDF.exists():
        raise HTTPException(status_code=404, detail="Обработанный PDF ещё не создан")
    return FileResponse(EDITED_PDF, media_type="application/pdf")


@app.get("/status")
def status():
    """Return info about the currently active PDF and job progress."""
    return {
        "pdf_name": CURRENT_PDF_NAME,
        "chunks_count": len(CHUNKS) if CHUNKS else 0,
        "stage": JOB_STATE["stage"],
        "progress": JOB_STATE["progress"],
        "running": JOB_STATE["running"],
        "error": JOB_STATE["error"],
    }


@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Accept PDF upload, return immediately, process in background."""
    if JOB_STATE["running"]:
        raise HTTPException(status_code=409, detail="Другая загрузка уже идёт")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF-файлы")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой: {len(content) // 1024 // 1024} МБ (лимит 100 МБ)",
        )

    print(f"Загрузка: {file.filename} ({len(content) // 1024} КБ)")

    JOB_STATE["running"] = True
    JOB_STATE["stage"] = "uploading"
    JOB_STATE["progress"] = 0
    JOB_STATE["error"] = None

    background_tasks.add_task(_process_upload, content, file.filename)

    return {
        "status": "started",
        "pdf_name": file.filename,
        "size_kb": len(content) // 1024,
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


@app.post("/find_matches")
def find_matches_endpoint(request: FindMatchesRequest):
    """Find all occurrences of old_text in the current PDF."""
    if not request.old_text or not request.old_text.strip():
        raise HTTPException(status_code=400, detail="old_text не должен быть пустым")

    pdf_path = _active_pdf_path()
    print(f"Поиск вхождений: {request.old_text!r} в {pdf_path.name}")

    matches = find_matches(str(pdf_path), request.old_text)
    print(f"Найдено совпадений: {len(matches)}")

    return {
        "old_text": request.old_text,
        "matches": matches,
        "total": len(matches),
    }


@app.post("/replace")
def replace_endpoint(request: ReplaceRequest):
    """Replace selected occurrences of old_text with new_text."""
    if not request.old_text or not request.old_text.strip():
        raise HTTPException(status_code=400, detail="old_text не должен быть пустым")
    if not request.new_text:
        raise HTTPException(status_code=400, detail="new_text не должен быть пустым")
    if not request.selected_ids:
        raise HTTPException(status_code=400, detail="Не выбрано ни одного совпадения")

    pdf_path = _active_pdf_path()
    print(
        f"Замена: {request.old_text!r} → {request.new_text!r}, "
        f"ids={request.selected_ids}"
    )

    try:
        report = replace_matches(
            str(pdf_path),
            str(EDITED_PDF),
            request.old_text,
            request.new_text,
            request.selected_ids,
        )
    except Exception as exc:
        print(f"Ошибка замены: {exc}")
        raise HTTPException(status_code=500, detail=f"Ошибка замены: {exc}")

    print(
        f"Заменено: {report['replaced_count']}, пропущено: {report['skipped_count']}"
    )

    return {
        "status": "ok",
        "replaced_count": report["replaced_count"],
        "skipped_count": report["skipped_count"],
        "skipped": report["skipped"],
        "download_url": "/static/edited.pdf",
    }