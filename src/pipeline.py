import json
import math
from pathlib import Path
import cv2

# Running context which updates iteratively
# Store Jsons at each step and use that as context

from .preprocess import preprocess_for_gemini
from .layout_detect import load_model, crop_and_save_figures, extract_text_blocks
from .ocr_gemini import ocr_image_to_text

import time

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _split_bbox_by_max(bbox: list[int], max_w: int, max_h: int) -> list[list[int]]: #From Layout Parser
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= max_w and h <= max_h:
        return [bbox]

    cols = max(1, math.ceil(w / max_w))
    rows = max(1, math.ceil(h / max_h))
    step_w = w / cols
    step_h = h / rows

    chunks = []
    for r in range(rows):
        for c in range(cols):
            cx1 = int(x1 + c * step_w)
            cy1 = int(y1 + r * step_h)
            cx2 = int(x1 + (c + 1) * step_w)
            cy2 = int(y1 + (r + 1) * step_h)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            chunks.append([cx1, cy1, cx2, cy2])
    return chunks

def _split_bbox_quadrants(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    return [
        [x1, y1, mid_x, mid_y],
        [mid_x, y1, x2, mid_y],
        [x1, mid_y, mid_x, y2],
        [mid_x, mid_y, x2, y2],
    ]

def _best_grid(num_tiles: int, w: int, h: int) -> tuple[int, int]:
    """Pick rows/cols for `num_tiles` that best match the page aspect."""
    aspect = w / h if h else 1
    best = (1, num_tiles)
    best_delta = float("inf")
    for rows in range(1, num_tiles + 1):
        if num_tiles % rows != 0:
            continue
        cols = num_tiles // rows
        delta = abs((cols / rows) - aspect)
        if delta < best_delta:
            best_delta = delta
            best = (rows, cols)
    return best

def _uniform_bboxes(img_shape: tuple[int, int], num_tiles: int) -> list[list[int]]:
    """Split the full image into `num_tiles` uniform, non-overlapping boxes."""
    h, w = img_shape[:2]
    rows, cols = _best_grid(num_tiles, w, h)
    tile_w = math.ceil(w / cols)
    tile_h = math.ceil(h / rows)
    bboxes: list[list[int]] = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * tile_w
            y1 = r * tile_h
            x2 = min(w, (c + 1) * tile_w)
            y2 = min(h, (r + 1) * tile_h)
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append([x1, y1, x2, y2])
    return bboxes

def _merge_outputs(outputs: list[dict]) -> dict:
    merged = {
        "metadata": {
            "publication": None,
            "date": None,
            "page_number": None,
            "edition_or_section_label": None,
        },
        "sections": {},
        "reading_order": [],
    }

    for payload in outputs:
        if not isinstance(payload, dict):
            continue
        meta = payload.get("metadata", {})
        for key in merged["metadata"]:
            if merged["metadata"][key] is None and meta.get(key) is not None:
                merged["metadata"][key] = meta.get(key)

        for section_name, section in payload.get("sections", {}).items():
            if section_name not in merged["sections"]:
                merged["sections"][section_name] = {
                    "section_label": section.get("section_label"),
                    "layout": section.get("layout"),
                    "items": [],
                    "unreadable_remainder": bool(section.get("unreadable_remainder")),
                    "unreadable_reason": section.get("unreadable_reason"),
                }
            merged["sections"][section_name]["items"].extend(section.get("items", []))
            if section.get("unreadable_remainder"):
                merged["sections"][section_name]["unreadable_remainder"] = True
                if section.get("unreadable_reason"):
                    merged["sections"][section_name]["unreadable_reason"] = section.get("unreadable_reason")

        for section_name in payload.get("reading_order", []):
            if section_name not in merged["reading_order"]:
                merged["reading_order"].append(section_name)

    if not merged["reading_order"] and merged["sections"]:
        merged["reading_order"] = list(merged["sections"].keys())

    return merged

def run(
    raw_dir="data/raw",
    prep_dir="data/preprocessed",
    maps_dir="data/maps",
    out_dir="data/out_json",
    tiles_dir="data/tiles",
    model="gemini-2.5-flash-lite",
    use_tiling=True,
    use_layout_parser=False,
    tile_max_dim=1800,
    max_split_depth=2,
    uniform_tile_count=24,
):
    raw_dir = Path(raw_dir)
    prep_dir = Path(prep_dir); prep_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = Path(maps_dir); maps_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = Path(tiles_dir); tiles_dir.mkdir(parents=True, exist_ok=True)

    lp_model = load_model() if use_layout_parser else None

    for p in sorted(raw_dir.glob("*")):
        log(f"Processing {p.name}") 
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            continue

        img = cv2.imread(str(p))
        if img is None:
            continue

        # 1) preprocess (Gemini-friendly grayscale PNG)
        log("Preprocessing")
        prep = preprocess_for_gemini(img)
        prep_path = prep_dir / f"{p.stem}_prep.png"
        cv2.imwrite(str(prep_path), prep)
        log("Preprocessing done")

        layout = None
        figures = []
        if use_layout_parser:
            log("Detecting layout")
            layout = lp_model.detect(img)
            log("Layout detection done")

            log("Cropping figures/maps")
            figures = crop_and_save_figures(p, maps_dir, layout)
            log(f"Found {len(figures)} figures")

        # 3) OCR (tiles)
        log("Performing structured extraction")
        outputs = []
        if use_tiling:
            prep_img = cv2.imread(str(prep_path), cv2.IMREAD_GRAYSCALE)
            if prep_img is None:
                log("Preprocessed image missing; skipping tiling")
            else:
                tile_idx = 0
                if use_layout_parser and layout is not None:
                    text_blocks = extract_text_blocks(layout, img.shape, pad=12)
                    log(f"Tiling into {len(text_blocks)} text regions (layout-driven)")
                    for block in text_blocks:
                        base_bbox = block["bbox"]
                        for bbox in _split_bbox_by_max(base_bbox, tile_max_dim, tile_max_dim):
                            queue = [(bbox, 0)]
                            while queue:
                                current_bbox, depth = queue.pop(0)
                                x1, y1, x2, y2 = current_bbox
                                crop = prep_img[y1:y2, x1:x2]
                                if crop is None or crop.size == 0:
                                    continue
                                tile_path = tiles_dir / f"{p.stem}_tile_{tile_idx:03d}.png"
                                tile_idx += 1
                                cv2.imwrite(str(tile_path), crop)
                                try:
                                    outputs.append(
                                        ocr_image_to_text(
                                            model=model,
                                            image_path=tile_path,
                                            max_output_tokens=10000,
                                            max_dim=None,
                                        )
                                    )
                                except ValueError as exc:
                                    log(f"Tile JSON error; split retry depth {depth}: {tile_path.name}")
                                    if depth < max_split_depth and (x2 - x1) > 200 and (y2 - y1) > 200:
                                        queue.extend((b, depth + 1) for b in _split_bbox_quadrants(current_bbox))
                                    else:
                                        log(f"Skipping tile after JSON error: {tile_path.name} ({exc})")
                else:
                    bboxes = _uniform_bboxes(prep_img.shape, uniform_tile_count)
                    log(f"Tiling into {len(bboxes)} uniform regions (no layout parser)")
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        crop = prep_img[y1:y2, x1:x2]
                        if crop is None or crop.size == 0:
                            continue
                        tile_path = tiles_dir / f"{p.stem}_tile_{tile_idx:03d}.png"
                        tile_idx += 1
                        cv2.imwrite(str(tile_path), crop)
                        try:
                            outputs.append(
                                ocr_image_to_text(
                                    model=model,
                                    image_path=tile_path,
                                    max_output_tokens=10000,
                                    max_dim=None,
                                )
                            )
                        except ValueError as exc:
                            log(f"Skipping tile after JSON error: {tile_path.name} ({exc})")
        if not outputs:
            log("Falling back to full-page OCR")
            outputs = [ocr_image_to_text(model=model, image_path=prep_path)]

        newspaper_data = _merge_outputs(outputs)
        log("Extraction done")

        # 2. Save DIRECTLY to file (bypassing the page_json wrapper)
        out_path = out_dir / f"{p.stem}.json"

        out_path.write_text(
        json.dumps(newspaper_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
        )

        print(f"Wrote {out_path}")

if __name__ == "__main__":
    run()

    #Add kick off after 5 mins
