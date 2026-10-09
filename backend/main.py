import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from services.pdf_parser import extract_blocks_with_pages, clean_text
from services.document_parser import parse_document, get_document_metadata
from services.embedder import (
    chunk_blocks_with_pages,
    build_embeddings,
    build_faiss_index,
    search,
    save_cache,
    load_cache,
)
from services import pdf_editor

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
DEFAULT_PDF = BASE_DIR / "sample.pdf"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

CHUNKS = None
INDEX = None
MODEL = None
CURRENT_PDF_NAME = None
DOCUMENT_TYPE = "pdf"          # pdf | docx | txt | md
SUPPORTS_HIGHLIGHT = True
PAGES_CACHE: Optional[list] = None   # структура pages для не-PDF (и для PDF тоже пригодится)

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
    JOB_STATE["stage"] = stage
    JOB_STATE["progress"] = progress
    print(f"[job] {stage} — {progress}%")


# ---------------------------------------------------------------------------
# Пути для активного документа
# ---------------------------------------------------------------------------

def _current_path() -> Path:
    """Активный исходный файл (для PDF/DOCX/TXT/MD)."""
    ext = DOCUMENT_TYPE if DOCUMENT_TYPE else "pdf"
    p = UPLOADS_DIR / f"current.{ext}"
    if p.exists():
        return p
    return DEFAULT_PDF


def _edited_path() -> Path:
    ext = DOCUMENT_TYPE if DOCUMENT_TYPE else "pdf"
    return UPLOADS_DIR / f"edited.{ext}"


# ---------------------------------------------------------------------------
# PDF-путь (как было)
# ---------------------------------------------------------------------------

def _blocks_from_pdf(pdf_path: Path) -> list[dict]:
    blocks = extract_blocks_with_pages(str(pdf_path))
    print(f"PDF-блоков: {len(blocks)}")
    return blocks


# ---------------------------------------------------------------------------
# Не-PDF путь (DOCX/TXT/MD) — блоки без bbox
# ---------------------------------------------------------------------------

def _blocks_from_document(doc_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Возвращает (blocks, pages).
    blocks — плоский список в формате pdf_parser: {"text", "page", "bbox"}.
    pages  — структура из document_parser для find_matches/replace.
    """
    parsed = parse_document(str(doc_path))
    pages = parsed["pages"]
    blocks: list[dict] = []
    for page in pages:
        for b in page["blocks"]:
            blocks.append(
                {
                    "text": b["text"],
                    "page": page["page"],
                    "bbox": b.get("bbox"),  # None для не-PDF
                }
            )
    print(f"Блоков ({parsed['type']}): {len(blocks)}")
    return blocks, pages


# ---------------------------------------------------------------------------
# Сборка индекса
# ---------------------------------------------------------------------------

def _build_index_for_document(doc_path: Path, doc_name: str) -> None:
    """Универсальная сборка индекса: PDF и DOCX/TXT/MD."""
    global CHUNKS, INDEX, MODEL, CURRENT_PDF_NAME
    global DOCUMENT_TYPE, SUPPORTS_HIGHLIGHT, PAGES_CACHE

    ext = doc_path.suffix.lower().lstrip(".")
    DOCUMENT_TYPE = ext if ext in {"pdf", "docx", "txt", "md"} else "pdf"
    SUPPORTS_HIGHLIGHT = (DOCUMENT_TYPE == "pdf")

    _set_stage("parsing", 5)
    doc_hash = hashlib.md5(doc_path.read_bytes()).hexdigest()
    print(f"Doc hash: {doc_hash[:8]}... ({doc_name}, type={DOCUMENT_TYPE})")

    cached = load_cache(doc_hash)
    if cached is not None:
        _set_stage("loading_cache", 50)
        chunks, _embeddings, index = cached
        print("Загружено из кэша")
        model = SentenceTransformer(MODEL_NAME)

        # pages всё равно пересоберём (нужны для /find_matches и /replace)
        if DOCUMENT_TYPE == "pdf":
            PAGES_CACHE = None  # для PDF pages берём из pdf_editor
        else:
            _, PAGES_CACHE = _blocks_from_document(doc_path)
    else:
        _set_stage("parsing", 15)
        if DOCUMENT_TYPE == "pdf":
            blocks = _blocks_from_pdf(doc_path)
            PAGES_CACHE = None
        else:
            blocks, PAGES_CACHE = _blocks_from_document(doc_path)

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
        save_cache(doc_hash, chunks, embeddings, index)
        print("Кэш сохранён")

    CHUNKS = chunks
    INDEX = index
    MODEL = model
    CURRENT_PDF_NAME = doc_name
    _set_stage("ready", 100)
    print(f"Документ готов: {doc_name}")


def startup() -> None:
    if not DEFAULT_PDF.exists():
        raise FileNotFoundError(
            f"sample.pdf не найден: {DEFAULT_PDF}. Положите его в корень проекта."
        )
    _build_index_for_document(DEFAULT_PDF, "sample.pdf")
    JOB_STATE["stage"] = "ready"
    JOB_STATE["progress"] = 100
    print("Сервер готов к поиску")


def _process_upload(doc_bytes: bytes, doc_name: str) -> None:
    """Background task: сохраняет файл с правильным расширением и строит индекс."""
    global DOCUMENT_TYPE
    try:
        ext = Path(doc_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Неподдерживаемое расширение: {ext}")

        # Сначала выставляем тип, чтобы _current_path() знал имя
        DOCUMENT_TYPE = ext.lstrip(".")

        # Сохраняем как uploads/current.<ext>
        target = UPLOADS_DIR / f"current{ext}"
        # Удаляем старые current.* чтобы не путались
        for old in UPLOADS_DIR.glob("current.*"):
            try:
                old.unlink()
            except OSError:
                pass

        target.write_bytes(doc_bytes)
        _build_index_for_document(target, doc_name)
    except Exception as exc:
        JOB_STATE["stage"] = "error"
        JOB_STATE["error"] = str(exc)
        print(f"Ошибка обработки upload: {exc}")
    finally:
        JOB_STATE["running"] = False


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(FRONTEND_PATH)


@app.get("/status")
def status():
    return {
        "pdf_name": CURRENT_PDF_NAME,
        "document_type": DOCUMENT_TYPE,
        "supports_highlight": SUPPORTS_HIGHLIGHT,
        "chunks_count": len(CHUNKS) if CHUNKS else 0,
        "stage": JOB_STATE["stage"],
        "progress": JOB_STATE["progress"],
        "running": JOB_STATE["running"],
        "error": JOB_STATE["error"],
    }


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if JOB_STATE["running"]:
        raise HTTPException(status_code=409, detail="Другая загрузка уже идёт")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Разрешены: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой: {len(content) // 1024 // 1024} МБ (лимит 100 МБ)",
        )

    print(f"Загрузка: {file.filename} ({len(content) // 1024} КБ, {ext})")

    JOB_STATE["running"] = True
    JOB_STATE["stage"] = "uploading"
    JOB_STATE["progress"] = 0
    JOB_STATE["error"] = None

    background_tasks.add_task(_process_upload, content, file.filename)

    return {
        "status": "started",
        "pdf_name": file.filename,
        "document_type": ext.lstrip("."),
        "size_kb": len(content) // 1024,
    }


@app.post("/search")
def search_document(request: SearchRequest):
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
            item["pages"] = CHUNKS[idx].get("pages")
            item["bboxes"] = CHUNKS[idx].get("bboxes")
            item["block_refs"] = CHUNKS[idx].get("block_refs") or []
        results.append(item)

    return {
        "query": request.query,
        "document_type": DOCUMENT_TYPE,
        "supports_highlight": SUPPORTS_HIGHLIGHT,
        "results": results,
    }


# ---------------------------------------------------------------------------
# find_matches
# ---------------------------------------------------------------------------

def _find_matches_text(old_text: str) -> list[dict]:
    """Поиск old_text по блокам не-PDF документа."""
    matches: list[dict] = []
    needle = old_text
    lower = needle.lower()
    counter = 0
    for page in (PAGES_CACHE or []):
        for bi, block in enumerate(page["blocks"]):
            text = block["text"]
            start = 0
            while True:
                pos = text.lower().find(lower, start)
                if pos == -1:
                    break
                ctx_from = max(0, pos - 40)
                ctx_to = min(len(text), pos + len(needle) + 40)
                matches.append(
                    {
                        "id": counter,
                        "page": page["page"],
                        "block": bi,
                        "bbox": None,
                        "context": text[ctx_from:ctx_to],
                    }
                )
                counter += 1
                start = pos + max(1, len(needle))
    return matches


@app.post("/find_matches")
def find_matches_endpoint(request: FindMatchesRequest):
    if not request.old_text or not request.old_text.strip():
        raise HTTPException(status_code=400, detail="old_text не должен быть пустым")

    if DOCUMENT_TYPE == "pdf":
        pdf_path = _current_path()
        print(f"Поиск вхождений (pdf): {request.old_text!r} в {pdf_path.name}")
        matches = pdf_editor.find_matches(str(pdf_path), request.old_text)
    else:
        print(f"Поиск вхождений ({DOCUMENT_TYPE}): {request.old_text!r}")
        matches = _find_matches_text(request.old_text)

    print(f"Найдено совпадений: {len(matches)}")
    return {
        "old_text": request.old_text,
        "matches": matches,
        "total": len(matches),
        "document_type": DOCUMENT_TYPE,
        "supports_highlight": SUPPORTS_HIGHLIGHT,
    }


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

def _replace_text(old_text: str, new_text: str, selected_ids: list[int]) -> dict:
    """Замена в PAGES_CACHE + пересборка page['text']. Возвращает отчёт."""
    global PAGES_CACHE
    selected = set(selected_ids)
    replaced = 0
    skipped = 0
    counter = 0
    new_pages = []

    for page in (PAGES_CACHE or []):
        new_blocks = []
        for block in page["blocks"]:
            t = block["text"]
            if old_text in t:
                # сколько раз old_text встречается в этом блоке
                occurrences = t.count(old_text)
                for _ in range(occurrences):
                    if counter in selected:
                        replaced += 1
                    else:
                        skipped += 1
                    counter += 1
                if any(
                    (i in selected)
                    for i in range(counter - occurrences, counter)
                ):
                    # заменяем все вхождения (упрощённо)
                    t = t.replace(old_text, new_text)
            new_blocks.append({"text": t, "bbox": block.get("bbox")})
        new_pages.append(
            {
                "page": page["page"],
                "text": "\n".join(b["text"] for b in new_blocks),
                "blocks": new_blocks,
            }
        )

    PAGES_CACHE = new_pages
    _write_replaced_file()
    return {
        "replaced_count": replaced,
        "skipped_count": skipped,
        "skipped": [],
    }


def _write_replaced_file() -> None:
    out = _edited_path()
    ext = DOCUMENT_TYPE

    if ext == "docx":
        import docx
        doc = docx.Document()
        for page in (PAGES_CACHE or []):
            for block in page["blocks"]:
                doc.add_paragraph(block["text"])
        doc.save(str(out))
    else:  # txt / md
        text = "\n\n".join(p["text"] for p in (PAGES_CACHE or []))
        out.write_text(text, encoding="utf-8")


@app.post("/replace")
def replace_endpoint(request: ReplaceRequest):
    if not request.old_text or not request.old_text.strip():
        raise HTTPException(status_code=400, detail="old_text не должен быть пустым")
    if not request.new_text:
        raise HTTPException(status_code=400, detail="new_text не должен быть пустым")
    if not request.selected_ids:
        raise HTTPException(status_code=400, detail="Не выбрано ни одного совпадения")

    if DOCUMENT_TYPE == "pdf":
        pdf_path = _current_path()
        out = UPLOADS_DIR / "edited.pdf"
        print(
            f"Замена (pdf): {request.old_text!r} → {request.new_text!r}, "
            f"ids={request.selected_ids}"
        )
        try:
            report = pdf_editor.replace_matches(
                str(pdf_path),
                str(out),
                request.old_text,
                request.new_text,
                request.selected_ids,
            )
        except Exception as exc:
            print(f"Ошибка замены: {exc}")
            raise HTTPException(status_code=500, detail=f"Ошибка замены: {exc}")

        print(
            f"Заменено: {report['replaced_count']}, "
            f"пропущено: {report['skipped_count']}"
        )
        return {
            "status": "ok",
            "replaced_count": report["replaced_count"],
            "skipped_count": report["skipped_count"],
            "skipped": report["skipped"],
            "download_url": "/static/edited.pdf",
            "document_type": "pdf",
        }

    # DOCX / TXT / MD
    print(
        f"Замена ({DOCUMENT_TYPE}): {request.old_text!r} → {request.new_text!r}, "
        f"ids={request.selected_ids}"
    )
    report = _replace_text(request.old_text, request.new_text, request.selected_ids)
    print(
        f"Заменено: {report['replaced_count']}, "
        f"пропущено: {report['skipped_count']}"
    )
    return {
        "status": "ok",
        "replaced_count": report["replaced_count"],
        "skipped_count": report["skipped_count"],
        "skipped": report["skipped"],
        "download_url": f"/static/edited.{DOCUMENT_TYPE}",
        "document_type": DOCUMENT_TYPE,
    }


# ---------------------------------------------------------------------------
# Статика
# ---------------------------------------------------------------------------

def _serve_current(ext: str):
    p = UPLOADS_DIR / f"current.{ext}"
    if not p.exists() and ext == "pdf":
        p = DEFAULT_PDF
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"current.{ext} не найден")
    return FileResponse(p)


def _serve_edited(ext: str):
    p = UPLOADS_DIR / f"edited.{ext}"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"edited.{ext} ещё не создан")
    return FileResponse(p)


@app.get("/static/current.pdf")
def serve_current_pdf():
    return FileResponse(_current_path(), media_type="application/pdf")


@app.get("/static/current.docx")
def serve_current_docx():
    return _serve_current("docx")


@app.get("/static/current.txt")
def serve_current_txt():
    return _serve_current("txt")


@app.get("/static/current.md")
def serve_current_md():
    return _serve_current("md")


@app.get("/static/edited.pdf")
def serve_edited_pdf():
    return _serve_edited("pdf")


@app.get("/static/edited.docx")
def serve_edited_docx():
    return _serve_edited("docx")


@app.get("/static/edited.txt")
def serve_edited_txt():
    return _serve_edited("txt")


@app.get("/static/edited.md")
def serve_edited_md():
    return _serve_edited("md")