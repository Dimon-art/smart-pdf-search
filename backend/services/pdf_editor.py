"""PDF text editing: find text matches and replace them via redaction."""

import pymupdf

MIN_FONTSIZE = 6
DEFAULT_FONTSIZE = 11
FONT_NAME = "helv"
FILL_COLOR = (1, 1, 1)
CONTEXT_WINDOW = 30


def fit_fontsize(
    font: pymupdf.Font,
    text: str,
    bbox_width: float,
    start: float = DEFAULT_FONTSIZE,
    min_size: float = MIN_FONTSIZE,
    step: float = 0.5,
) -> float:
    """Find the largest fontsize at which text fits into bbox_width.

    Starts at `start` pt and decreases by `step` until the text fits or
    reaches `min_size`.
    """
    fontsize = start
    while font.text_length(text, fontsize) > bbox_width and fontsize > min_size:
        fontsize -= step
    return fontsize


def _extract_context(page: pymupdf.Page, rect: pymupdf.Rect, window: int = CONTEXT_WINDOW) -> str:
    """Return a short text snippet around the given rect on the page.

    Uses page.get_text("words") to find the word closest to rect and
    returns `window` characters before and after.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return ""

    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2

    closest_idx = None
    closest_dist = None
    for i, w in enumerate(words):
        wx0, wy0, wx1, wy1 = w[:4]
        wx_c = (wx0 + wx1) / 2
        wy_c = (wy0 + wy1) / 2
        dist = (wx_c - cx) ** 2 + (wy_c - cy) ** 2
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    if closest_idx is None:
        return ""

    start_i = max(0, closest_idx - 5)
    end_i = min(len(words), closest_idx + 6)
    snippet_words = [w[4] for w in words[start_i:end_i]]
    snippet = " ".join(snippet_words)

    if len(snippet) > window * 2:
        snippet = snippet[: window * 2] + "..."

    return snippet


def find_matches(input_path: str, old_text: str) -> list[dict]:
    """Find all occurrences of old_text in the PDF.

    Returns a list of dicts:
    {
        "id": int,
        "page": int,
        "bbox": [x0, y0, x1, y1],
        "context": str,
        "width_pt": float,
        "height_pt": float,
    }
    IDs are assigned in the same order that replace_matches uses.
    """
    doc = pymupdf.open(input_path)
    matches = []
    match_id = 0

    try:
        for page_num, page in enumerate(doc):
            for rect in page.search_for(old_text):
                matches.append({
                    "id": match_id,
                    "page": page_num,
                    "bbox": [round(rect.x0, 2), round(rect.y0, 2),
                             round(rect.x1, 2), round(rect.y1, 2)],
                    "context": _extract_context(page, rect),
                    "width_pt": round(rect.width, 2),
                    "height_pt": round(rect.height, 2),
                })
                match_id += 1
    finally:
        doc.close()

    return matches


def replace_matches(
    input_path: str,
    output_path: str,
    old_text: str,
    new_text: str,
    selected_ids: list[int],
) -> dict:
    """Replace selected occurrences of old_text with new_text via redaction.

    `selected_ids` refers to IDs returned by find_matches — same iteration
    order (page → rect) is used to keep IDs consistent.

    Returns a report:
    {
        "output_path": str,
        "replaced_count": int,
        "skipped_count": int,
        "replaced": [{"page": int, "bbox": [...], "fontsize": float}],
        "skipped": [{"page": int, "bbox": [...], "reason": str,
                     "needed_pt": float, "available_pt": float}],
    }
    """
    doc = pymupdf.open(input_path)
    font = pymupdf.Font(FONT_NAME)
    selected_set = set(selected_ids)

    by_page: dict[int, list[pymupdf.Rect]] = {}
    match_id = 0
    for page_num, page in enumerate(doc):
        for rect in page.search_for(old_text):
            if match_id in selected_set:
                by_page.setdefault(page_num, []).append(rect)
            match_id += 1

    replaced = []
    skipped = []

    try:
        for page_num, rects in by_page.items():
            page = doc[page_num]
            for rect in rects:
                fontsize = fit_fontsize(font, new_text, rect.width)
                width_needed = font.text_length(new_text, fontsize)

                if width_needed > rect.width:
                    skipped.append({
                        "page": page_num,
                        "bbox": [round(rect.x0, 2), round(rect.y0, 2),
                                 round(rect.x1, 2), round(rect.y1, 2)],
                        "reason": f"не помещается даже при {MIN_FONTSIZE} pt",
                        "needed_pt": round(width_needed, 2),
                        "available_pt": round(rect.width, 2),
                    })
                    continue

                page.add_redact_annot(
                    rect,
                    text=new_text,
                    fontname=FONT_NAME,
                    fontsize=fontsize,
                    fill=FILL_COLOR,
                )
                replaced.append({
                    "page": page_num,
                    "bbox": [round(rect.x0, 2), round(rect.y0, 2),
                             round(rect.x1, 2), round(rect.y1, 2)],
                    "fontsize": fontsize,
                })

        for page_num in by_page:
            doc[page_num].apply_redactions()

        doc.save(output_path)
    finally:
        doc.close()

    return {
        "output_path": output_path,
        "replaced_count": len(replaced),
        "skipped_count": len(skipped),
        "replaced": replaced,
        "skipped": skipped,
    }