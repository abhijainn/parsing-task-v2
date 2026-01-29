import cv2
import numpy as np
from pathlib import Path

def preprocess_for_gemini(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simple preprocessing:
    - grayscale
    - median blur
    - CLAHE contrast normalization
    Returns 8-bit grayscale.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def main(in_dir="data/raw", out_dir="data/preprocessed"):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(in_dir.glob("*")):
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue

        out = preprocess_for_gemini(img)
        out_path = out_dir / f"{p.stem}_prep.png"
        cv2.imwrite(str(out_path), out)

if __name__ == "__main__":
    main()
