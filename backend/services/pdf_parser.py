import re

import pymupdf


def extract_pages_with_bbox(pdf_path: str) -> list[dict]:
    """Extract text blocks with bounding boxes from every page of a PDF.

    Each page dict contains 1-based page_number, width, height, and blocks
    with text, bbox [x0, y0, x1, y1], and block_type. Empty or whitespace-only
    blocks are skipped. Returns [] if the file is missing or cannot be read.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return []

    try:
        pages = []
        for page_index, page in enumerate(doc):
            page_dict = page.get_text("dict")
            blocks = []
            for block in page_dict.get("blocks", []):
                line_texts = []
                for line in block.get("lines", []):
                    span_text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    )
                    line_texts.append(span_text)
                text = "\n".join(line_texts)
                if not text.strip():
                    continue
                x0, y0, x1, y1 = block["bbox"]
                blocks.append(
                    {
                        "text": text,
                        "bbox": [
                            round(x0, 2),
                            round(y0, 2),
                            round(x1, 2),
                            round(y1, 2),
                        ],
                        "block_type": int(block.get("type", 0)),
                    }
                )
            pages.append(
                {
                    "page_number": page_index + 1,
                    "width": float(page_dict.get("width", page.rect.width)),
                    "height": float(page_dict.get("height", page.rect.height)),
                    "blocks": blocks,
                }
            )
        return pages
    except Exception:
        return []
    finally:
        doc.close()


def clean_text(text: str) -> str:
    """Normalize whitespace: collapse spaces and newlines, then strip."""
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def get_pdf_metadata(pdf_path: str) -> dict:
    """Return page_count, title, and author for a PDF.

    Missing title/author become None. Returns {} if the file cannot be opened.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return {}

    try:
        metadata = doc.metadata or {}
        title = metadata.get("title") or None
        author = metadata.get("author") or None
        return {
            "page_count": int(doc.page_count),
            "title": title,
            "author": author,
        }
    except Exception:
        return {}
    finally:
        doc.close()


def extract_blocks_with_pages(pdf_path: str) -> list[dict]:
    """Извлекает текстовые блоки из PDF с указанием номера страницы.

    Args:
        pdf_path: Путь к PDF-файлу.

    Returns:
        Плоский список текстовых блоков вида:
        {"text": str, "bbox": [x0, y0, x1, y1], "page": int}.

        При пустом или повреждённом PDF возвращается пустой список.
    """
    pages = extract_pages_with_bbox(pdf_path)

    blocks = []
    for page in pages:
        page_number = page["page_number"]
        for block in page.get("blocks", []):
            if block.get("block_type") != 0:
                continue
            blocks.append({
                "text": block["text"],
                "bbox": block["bbox"],
                "page": page_number,
            })

    return blocks