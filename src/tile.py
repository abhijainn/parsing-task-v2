import cv2
from pathlib import Path

def tile_image(
    img_path: Path,
    out_dir: Path,
    tile_w: int = 1400,
    tile_h: int = 1800,
    overlap: int = 140,
) -> list[dict]:
    """
    Gemini-friendly tiling:
    - tiles are roughly constant pixel size (better than rows/cols)
    - overlap prevents missing words at boundaries
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    H, W = img.shape[:2]
    tiles = []
    idx = 0

    y = 0
    while y < H:
        x = 0
        y1 = max(0, y - overlap)
        y2 = min(H, y + tile_h + overlap)
        while x < W:
            x1 = max(0, x - overlap)
            x2 = min(W, x + tile_w + overlap)

            crop = img[y1:y2, x1:x2]
            out_path = out_dir / f"{img_path.stem}_tile_{idx:03d}.png"
            cv2.imwrite(str(out_path), crop)

            tiles.append({"path": str(out_path), "bbox": [x1, y1, x2, y2], "order": idx})
            idx += 1
            x += tile_w

        y += tile_h

    return tiles
