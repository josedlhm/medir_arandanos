#!/usr/bin/env python3
# eval_berries_rmse_paperstyle.py

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# --- paper-style preprocessing (steps 3–6, mm) ---
from utils.pre_processing_paper import preprocess_berry_pointcloud_mm

# --- measurement utils (uses your inner-ellipsoid major diameter) ---
from utils.measure import inner_ellipsoid_major_diameter_mm

# ============================
# PAPER-STYLE PREPROCESS PARAMS (mm)
# tuned for blueberries (14–24mm) @ ~40cm ZED Mini X
# Following paper: bilateral filter, depth discontinuity filter, radial outlier removal, median distance filter
# ============================
BIL_D = 5                      # moderate smoothing to reduce noise
BIL_SIGMA_COLOR_MM = 6.0       # balanced - smooth noise but preserve berry shape
BIL_SIGMA_SPACE_PX = 5.0       # moderate spatial smoothing

DISC_THR_MM = 12.0             # balanced - remove obvious discontinuities but keep berry edges
DISC_KSIZE = 3

ROR_RADIUS_MM = 6.0            # larger radius - berries are 14-24mm, so 6mm is ~25-40% of diameter
ROR_MIN_NEIGHBORS = 3         # very low - only remove truly isolated noise points

MED_K = 6.0                    # very permissive - keep edge points that define berry size

MIN_PTS = 50
BORDER_MARGIN_PX = 2
MIN_VALID_FRAMES = 1
GT_COL_FALLBACK = "caliber_mm"
# ============================

# ---- Dataset / Model Config (berries) ----
ROOT = Path("/Volumes/USBDATA/berry_dataset_enero")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "./weights/arandanos_medidas.pth"
CONFIG  = "./weights/arandanos_medidas.yaml"
fx, fy, cx, cy = (1272.44, 1272.67, 920.062, 618.949)  # intrinsics (px; depth in mm)
MASKS_OUTPUT = ROOT / "masks_visualization"

# ---- Detectron2 predictor ----
def load_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = getattr(
            cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", 0.5
        )
    cfg.INPUT.FORMAT = "BGR"
    return DefaultPredictor(cfg)

predictor = load_predictor()

# ---- Utilities ----
def pick_mask(outputs, class_id_keep=None):
    inst = outputs["instances"].to("cpu")
    if len(inst) == 0:
        return None
    if class_id_keep is not None:
        keep = (inst.pred_classes.numpy() == class_id_keep)
        if not keep.any():
            return None
        masks = inst.pred_masks[keep].numpy()
    else:
        masks = inst.pred_masks.numpy()
    areas = masks.sum(axis=(1, 2))
    idx = int(np.argmax(areas))
    return (masks[idx].astype(np.uint8) * 255)

def mask_touches_border(mask_u8: np.ndarray, margin_px: int = 1) -> bool:
    if mask_u8 is None:
        return True
    h, w = mask_u8.shape[:2]
    m = max(1, int(margin_px))
    return (mask_u8[:m, :].any() or mask_u8[h-m:, :].any()
            or mask_u8[:, :m].any() or mask_u8[:, w-m:].any())

def save_mask_image(img_bgr: np.ndarray, mask: np.ndarray, sample_id: str, frame_name: str, pred_diameter: float):
    MASKS_OUTPUT.mkdir(parents=True, exist_ok=True)

    mask_colored = np.zeros_like(img_bgr)
    mask_colored[:, :, 1] = mask  # green
    overlay = cv2.addWeighted(img_bgr, 0.6, mask_colored, 0.4, 0)

    h, w = overlay.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        overlay = cv2.resize(overlay, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    text = f"Sample {sample_id}, d={pred_diameter:.2f}mm"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    output_path = MASKS_OUTPUT / f"sample_{sample_id}_{frame_name}.png"
    cv2.imwrite(str(output_path), overlay)

def per_sample_prediction(sample_row, save_mask=False):
    sid = str(int(sample_row["sample_id"]))
    img_dir   = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")
    if not img_dir.exists() or not depth_dir.exists():
        return np.nan, 0

    vals = []
    first_valid_frame = True

    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        dpth_path = depth_dir / f"{name}.npy"
        if not dpth_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        depth_mm = np.load(dpth_path).astype(np.float32)
        # treat invalid depth as NaN (paper pipelines assume invalids are removed)
        depth_mm[~np.isfinite(depth_mm)] = np.nan
        depth_mm[depth_mm <= 0] = np.nan

        outputs = predictor(img_bgr)
        raw_mask = pick_mask(outputs)
        if raw_mask is None:
            continue

        if mask_touches_border(raw_mask, margin_px=BORDER_MARGIN_PX):
            continue

        # Paper-style preprocessing (as described in paper)
        pts, dbg = preprocess_berry_pointcloud_mm(
            depth_mm, raw_mask, fx, fy, cx, cy,
            bilateral_d=BIL_D,
            bilateral_sigma_color_mm=BIL_SIGMA_COLOR_MM,
            bilateral_sigma_space_px=BIL_SIGMA_SPACE_PX,
            disc_thr_mm=DISC_THR_MM,
            disc_ksize=DISC_KSIZE,
            ror_radius_mm=ROR_RADIUS_MM,
            ror_min_neighbors=ROR_MIN_NEIGHBORS,
            med_k=MED_K,
            min_pts=MIN_PTS,
        )
        if dbg.get("reason") != "ok":
            continue

        # measure (inner-ellipsoid major diameter)
        try:
            d_mm = inner_ellipsoid_major_diameter_mm(pts)
        except Exception:
            d_mm = np.nan

        if np.isfinite(d_mm):
            vals.append(float(d_mm))
            if save_mask and first_valid_frame:
                save_mask_image(img_bgr, raw_mask, sid, name, float(d_mm))
                first_valid_frame = False

    if len(vals) < MIN_VALID_FRAMES:
        return np.nan, len(vals)

    return float(np.median(vals)), len(vals)

def rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    e = y_pred - y_true
    return float(np.sqrt(np.mean(e**2)))

def mape(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    pct_err = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
    return float(np.mean(pct_err))

def main():
    meta = pd.read_csv(META_CSV)
    gt_col = "gt_caliber_mm" if "gt_caliber_mm" in meta.columns else GT_COL_FALLBACK

    rows = [
        row for _, row in meta.iterrows()
        if (SAMPLES / str(int(row["sample_id"])) / "images").exists()
        and (SAMPLES / str(int(row["sample_id"])) / "depth").exists()
    ]
    if not rows:
        raise RuntimeError("No samples with images/ and depth/ found on disk.")

    records = []
    for row in rows:
        sid = int(row["sample_id"])
        gt  = float(row[gt_col])
        pred, nframes = per_sample_prediction(row, save_mask=True)
        status = "ok" if np.isfinite(pred) else "skip"
        print(f"[{status}] sample {sid}: frames_used={nframes}, pred={pred if np.isfinite(pred) else 'NaN'}")
        if np.isfinite(pred):
            records.append({"sample_id": sid, "gt_caliber_mm": gt, "pred_mm": pred})

    if not records:
        print("No valid predictions after guards.")
        return

    df = pd.DataFrame(records)
    df["err_mm"] = df["pred_mm"] - df["gt_caliber_mm"]
    df["pct_err"] = (np.abs(df["err_mm"]) / df["gt_caliber_mm"]) * 100

    overall_rmse = rmse(df["pred_mm"].values, df["gt_caliber_mm"].values)
    overall_mape = mape(df["pred_mm"].values, df["gt_caliber_mm"].values)

    print("\nPaper-style preprocessing (mm):")
    print(f"  bilateral: d={BIL_D}, sigmaColor={BIL_SIGMA_COLOR_MM}mm, sigmaSpace={BIL_SIGMA_SPACE_PX}px")
    print(f"  discontinuity: thr={DISC_THR_MM}mm, ksize={DISC_KSIZE}")
    print(f"  radius outlier: radius={ROR_RADIUS_MM}mm, min_neighbors={ROR_MIN_NEIGHBORS}")
    print(f"  median dist: k={MED_K}")
    print(f"  border_margin_px={BORDER_MARGIN_PX}, min_pts={MIN_PTS}")
    print(f"Samples evaluated: {len(df)}")
    print(f"RMSE (mm): {overall_rmse:.2f}")
    print(f"Median Abs Error (mm): {np.median(np.abs(df['err_mm'])):.2f}")
    print(f"Mean Abs Error (mm): {np.mean(np.abs(df['err_mm'])):.2f}")
    print(f"Mean Abs Percentage Error (%): {overall_mape:.2f}")
    print(f"Median Abs Percentage Error (%): {np.median(df['pct_err']):.2f}")

    print("\n--- Diagnostic Statistics ---")
    print(f"GT diameter range: {df['gt_caliber_mm'].min():.2f} - {df['gt_caliber_mm'].max():.2f} mm (mean: {df['gt_caliber_mm'].mean():.2f})")
    print(f"Pred diameter range: {df['pred_mm'].min():.2f} - {df['pred_mm'].max():.2f} mm (mean: {df['pred_mm'].mean():.2f})")
    print(f"Bias (pred - gt mean): {df['pred_mm'].mean() - df['gt_caliber_mm'].mean():.2f} mm")
    print(f"  → {'UNDERESTIMATING' if df['pred_mm'].mean() < df['gt_caliber_mm'].mean() else 'OVERESTIMATING'}")
    print(f"\nCorrelation coefficient: {np.corrcoef(df['pred_mm'], df['gt_caliber_mm'])[0,1]:.3f}")

    out_csv = ROOT / "berries_inner_major_paperstyle_rmse.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Saved per-sample results to: {out_csv}")

if __name__ == "__main__":
    main()
