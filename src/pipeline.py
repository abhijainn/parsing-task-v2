import json
from pathlib import Path
import cv2

from .preprocess import preprocess_for_gemini
from .layout_detect import load_model, crop_and_save_figures
from .ocr_gemini import ocr_image_to_text

import time

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(
    raw_dir="data/raw",
    prep_dir="data/preprocessed",
    maps_dir="data/maps",
    out_dir="data/out_json",
    model="gemini-2.5-flash-lite",
):
    raw_dir = Path(raw_dir)
    prep_dir = Path(prep_dir); prep_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = Path(maps_dir); maps_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    lp_model = load_model()

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

        # 2) layout detection on ORIGINAL (better for figures/maps)
        
        log("Detecting layout")
        layout = lp_model.detect(img)
        log("Layout detection done")

        # 3) crop figures/maps
        log("Cropping figures/maps")
        figures = crop_and_save_figures(p, maps_dir, layout)
        log(f"Found {len(figures)} figures")

        # 4) OCR the FULL preprocessed page (no tiles)
        log("Performing structured extraction on full page")

        # Ensure this function returns a Python DICTIONARY (via json.loads inside the function)
        # If it currently returns a string, parse it here: newspaper_data = json.loads(ocr_result)
        newspaper_data = ocr_image_to_text(model=model, image_path=prep_path)

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
