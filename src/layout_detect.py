from pathlib import Path
import cv2
import layoutparser as lp

LP_MODEL_DIR = "/opt/layoutparser_models/PubLayNet/faster_rcnn_R_50_FPN_3x"
CONFIG = f"{LP_MODEL_DIR}/config.yml"
WEIGHTS = f"{LP_MODEL_DIR}/model_final.pth"

def load_model():
    return lp.Detectron2LayoutModel(
        config_path=CONFIG,
        model_path=WEIGHTS,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"},
    )

def crop_and_save_figures(page_path: Path, out_dir: Path, layout) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(page_path))
    h, w = img.shape[:2]

    figures = []
    fig_idx = 0
    for block in layout:
        if block.type != "Figure":
            continue
        x1, y1, x2, y2 = map(int, block.coordinates)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        path = out_dir / f"{page_path.stem}_figure_{fig_idx:02d}.png"
        cv2.imwrite(str(path), crop)
        figures.append({
            "type": "Figure",
            "bbox": [x1, y1, x2, y2],
            "path": str(path),
            "score": float(block.score) if hasattr(block, "score") else None,
        })
        fig_idx += 1
    return figures
