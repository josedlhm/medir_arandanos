#!/usr/bin/env python3
# eval_berries_rmse_fixed.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# --- preprocessing utils ---
from utils.pre_processing_original import (
    depth_mask_to_points_mm,
    keep_near_core_depth_mm,   # core-depth band-pass
)

# --- measurement utils (uses your inner-ellipsoid major diameter) ---
from utils.measure import inner_ellipsoid_major_diameter_mm

# ============================
# FIXED PARAMS (from your berries tuning)
# ============================
ERODE_PX  = 5             # slightly reduced to reduce underestimation while still filtering edge noise
TRIM_FRAC = 0.12          # slightly reduced to keep more depth data for better core estimate
BAND_MM   = 7.0           # slightly increased to include a bit more surface area (berries up to 24mm)
BORDER_MARGIN_PX = 2     # skip detections within this many px of the image edge
MIN_PTS = 50             # minimum 3D points after filtering to accept a frame
MIN_VALID_FRAMES = 1     # per-sample minimum frames to keep the sample
GT_COL_FALLBACK = "caliber_mm"  # if gt_caliber_mm not present
# ============================

# ---- Dataset / Model Config (berries) ----
ROOT = Path("/Volumes/USBDATA/berry_dataset_enero")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "./weights/arandanos_medidas.pth"
CONFIG  = "./weights/arandanos_medidas.yaml"
fx, fy, cx, cy = (1272.44, 1272.67, 920.062, 618.949)  # intrinsics (px; depth in mm)
MASKS_OUTPUT = ROOT / "masks_visualization"  # directory for saved mask images

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
    """
    Return uint8 {0,255} mask for the target instance (largest or class-filtered).
    """
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
    """True if any positive pixel lies within 'margin_px' of image border."""
    if mask_u8 is None:
        return True
    h, w = mask_u8.shape[:2]
    m = max(1, margin_px)
    return (mask_u8[:m, :].any() or mask_u8[h-m:, :].any()
            or mask_u8[:, :m].any() or mask_u8[:, w-m:].any())

def save_mask_image(img_bgr: np.ndarray, mask: np.ndarray, sample_id: str, frame_name: str, pred_diameter: float):
    """Save a visualization of the mask overlaid on the original image."""
    # Create output directory if it doesn't exist
    MASKS_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Convert mask to 3-channel for overlay
    mask_colored = np.zeros_like(img_bgr)
    mask_colored[:, :, 1] = mask  # Green channel
    
    # Create overlay (50% transparency)
    overlay = cv2.addWeighted(img_bgr, 0.6, mask_colored, 0.4, 0)
    
    # Resize if too large (max width 800px for reasonable file size)
    h, w = overlay.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Add text with sample ID and predicted diameter
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Sample {sample_id}, d={pred_diameter:.2f}mm"
    cv2.putText(overlay, text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (10, 30), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Save image
    output_path = MASKS_OUTPUT / f"sample_{sample_id}_{frame_name}.png"
    cv2.imwrite(str(output_path), overlay)

def per_sample_prediction(sample_row, save_mask=False):
    """Return median predicted diameter (mm) for a sample, or np.nan if invalid."""
    sid = str(int(sample_row["sample_id"]))
    img_dir  = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")
    if not img_dir.exists() or not depth_dir.exists():
        return np.nan, 0  # no frames

    vals = []
    first_valid_frame = True  # Save mask for the first valid measurement
    
    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        dpth_path = depth_dir / f"{name}.npy"
        if not dpth_path.exists():
            continue

        img_bgr  = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        depth_mm = np.load(dpth_path)

        outputs  = predictor(img_bgr)
        raw_mask = pick_mask(outputs)
        if raw_mask is None:
            continue

        # Guard: skip detections touching the image border
        if mask_touches_border(raw_mask, margin_px=BORDER_MARGIN_PX):
            continue

        inlier = (raw_mask > 0)
        pts = depth_mask_to_points_mm(depth_mm, inlier, fx, fy, cx, cy)  # (N,3) mm
        if pts.shape[0] < MIN_PTS:
            continue

        # fixed core-depth band-pass
        pts = keep_near_core_depth_mm(
            pts, depth_mm, raw_mask,
            erode_px=ERODE_PX, trim=TRIM_FRAC, band_mm=BAND_MM
        )
        if pts.shape[0] < MIN_PTS:
            continue

        # Additional outlier removal: remove points far from centroid
        if pts.shape[0] > MIN_PTS:
            centroid = pts.mean(axis=0)
            distances = np.linalg.norm(pts - centroid, axis=1)
            # Keep points within 2.5 MAD of median distance
            med_dist = np.median(distances)
            mad_dist = np.median(np.abs(distances - med_dist)) + 1e-6
            keep_mask = distances <= (med_dist + 2.5 * mad_dist)
            pts = pts[keep_mask]
            
        if pts.shape[0] < MIN_PTS:
            continue

        # measure (inner-ellipsoid major diameter)
        try:
            d_mm = inner_ellipsoid_major_diameter_mm(pts)
        except Exception:
            d_mm = np.nan

        if np.isfinite(d_mm):
            vals.append(d_mm)
            
            # Save mask visualization for first valid measurement
            if save_mask and first_valid_frame:
                save_mask_image(img_bgr, raw_mask, sid, name, d_mm)
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
    """Mean Absolute Percentage Error (%)"""
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.nan
    pct_err = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100
    return float(np.mean(pct_err))

# ---- Main ----
def main():
    meta = pd.read_csv(META_CSV)
    gt_col = "gt_caliber_mm" if "gt_caliber_mm" in meta.columns else GT_COL_FALLBACK

    # collect samples that exist on disk
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
            records.append({"sample_id": sid, "gt_caliber_mm": gt, "pred_mm": pred, "frames_used": nframes})

    if not records:
        print("No valid predictions after guards.")
        return

    df = pd.DataFrame(records)
    df["err_mm"] = df["pred_mm"] - df["gt_caliber_mm"]
    df["pct_err"] = (np.abs(df["err_mm"]) / df["gt_caliber_mm"]) * 100
    overall_rmse = rmse(df["pred_mm"].values, df["gt_caliber_mm"].values)
    overall_mape = mape(df["pred_mm"].values, df["gt_caliber_mm"].values)

    print(f"\nFixed preprocessing:"
          f" erode_px={ERODE_PX}, trim={TRIM_FRAC}, band_mm={BAND_MM}, border_margin_px={BORDER_MARGIN_PX}")
    print(f"Samples evaluated: {len(df)}")
    print(f"RMSE (mm): {overall_rmse:.2f}")
    print(f"Median Abs Error (mm): {np.median(np.abs(df['err_mm'])):.2f}")
    print(f"Mean Abs Error (mm): {np.mean(np.abs(df['err_mm'])):.2f}")
    print(f"Mean Abs Percentage Error (%): {overall_mape:.2f}")
    print(f"Median Abs Percentage Error (%): {np.median(df['pct_err']):.2f}")
    
    # Diagnostic statistics
    print(f"\n--- Diagnostic Statistics ---")
    print(f"GT diameter range: {df['gt_caliber_mm'].min():.2f} - {df['gt_caliber_mm'].max():.2f} mm (mean: {df['gt_caliber_mm'].mean():.2f})")
    print(f"Pred diameter range: {df['pred_mm'].min():.2f} - {df['pred_mm'].max():.2f} mm (mean: {df['pred_mm'].mean():.2f})")
    print(f"Bias (pred - gt mean): {df['pred_mm'].mean() - df['gt_caliber_mm'].mean():.2f} mm")
    print(f"  → {'UNDERESTIMATING' if df['pred_mm'].mean() < df['gt_caliber_mm'].mean() else 'OVERESTIMATING'}")
    print(f"\nCorrelation coefficient: {np.corrcoef(df['pred_mm'], df['gt_caliber_mm'])[0,1]:.3f}")

    # Save per-sample table
    out_csv = ROOT / "berries_inner_major_fixed_rmse.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Saved per-sample results to: {out_csv}")

if __name__ == "__main__":
    main()
