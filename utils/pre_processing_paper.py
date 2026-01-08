# utils/pre_processing_paper.py
from __future__ import annotations

import numpy as np
import cv2

try:
    import open3d as o3d
except Exception as e:
    o3d = None


def bilateral_depth_mm(depth_mm: np.ndarray, d: int = 7, sigma_color_mm: float = 4.0, sigma_space_px: float = 7.0):
    """
    Bilateral filter on depth (mm). Smooths depth noise while preserving edges.
    depth_mm: float/uint depth in mm, shape (H,W). Invalids are <=0 or non-finite.
    Returns float32 depth (mm) with invalids as NaN.
    """
    depth = depth_mm.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    filled = depth.copy()
    fill_val = float(np.median(depth[valid])) if np.any(valid) else 0.0
    filled[~valid] = fill_val

    out = cv2.bilateralFilter(
        filled,
        d=int(d),
        sigmaColor=float(sigma_color_mm),   # mm
        sigmaSpace=float(sigma_space_px),   # px
    )
    out[~valid] = np.nan
    return out


def depth_discontinuity_good_pixels_mm(depth_mm: np.ndarray, thr_mm: float = 6.0, ksize: int = 3):
    """
    Depth discontinuity filter (mm): marks pixels 'good' if local depth range is small.
    Pixels where local (max-min) >= thr_mm are considered unreliable (edge/occlusion artifacts).
    Returns boolean mask (H,W) of good pixels.
    """
    depth = depth_mm.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    d = depth.copy()
    d[~valid] = 0.0

    kernel = np.ones((int(ksize), int(ksize)), np.uint8)
    local_min = cv2.erode(d, kernel)
    local_max = cv2.dilate(d, kernel)

    local_range = local_max - local_min
    good = valid & (local_range < float(thr_mm))
    return good


def depth_mask_to_points_mm(depth_mm: np.ndarray, mask_bool: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """
    Backproject masked depth to 3D points (mm). Output shape (N,3) in mm.
    """
    ys, xs = np.where(mask_bool)
    z = depth_mm[ys, xs].astype(np.float32)
    ok = np.isfinite(z) & (z > 0)

    xs = xs[ok].astype(np.float32)
    ys = ys[ok].astype(np.float32)
    z  = z[ok]

    x = (xs - float(cx)) * z / float(fx)
    y = (ys - float(cy)) * z / float(fy)

    return np.stack([x, y, z], axis=1)


def radius_outlier_removal_mm(points_xyz_mm: np.ndarray, radius_mm: float = 3.0, min_neighbors: int = 15):
    """
    Radial outlier removal in mm using Open3D.
    Removes points with fewer than min_neighbors within radius_mm.
    """
    if o3d is None:
        raise ImportError("open3d is required for radius outlier removal. pip install open3d")

    if points_xyz_mm.shape[0] == 0:
        return points_xyz_mm, np.array([], dtype=int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz_mm.astype(np.float64))
    pcd2, ind = pcd.remove_radius_outlier(nb_points=int(min_neighbors), radius=float(radius_mm))
    return np.asarray(pcd2.points), np.asarray(ind, dtype=int)


def median_distance_filter_mm(points_xyz_mm: np.ndarray, k: float = 3.0):
    """
    Median distance filter (mm): remove points far from the main cluster using MAD.
    Keeps points with distance-to-centroid within med + k*sigma (sigma from MAD).
    """
    pts = points_xyz_mm
    if pts.shape[0] == 0:
        return pts, np.array([], dtype=int)

    c = pts.mean(axis=0)
    r = np.linalg.norm(pts - c, axis=1)  # mm

    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-6
    sigma = 1.4826 * mad

    keep = r <= (med + float(k) * sigma)
    return pts[keep], np.where(keep)[0]


def preprocess_berry_pointcloud_mm(
    depth_mm: np.ndarray,
    raw_mask_u8: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    *,
    bilateral_d: int = 7,
    bilateral_sigma_color_mm: float = 4.0,
    bilateral_sigma_space_px: float = 7.0,
    disc_thr_mm: float = 6.0,
    disc_ksize: int = 3,
    ror_radius_mm: float = 3.0,
    ror_min_neighbors: int = 15,
    med_k: float = 3.0,
    min_pts: int = 50,
):
    """
    Paper-style preprocessing (as described in paper):
      1) bilateral filter on depth
      2) depth discontinuity filter
      3) backproject to 3D
      4) radius outlier removal
      5) median distance filter

    Returns (points_xyz_mm, debug_dict)
    """
    if raw_mask_u8 is None:
        return np.empty((0, 3), np.float32), {"reason": "no_mask"}

    dbg: dict = {"reason": None}

    # Ensure float depth
    depth0 = depth_mm.astype(np.float32)
    depth0[~np.isfinite(depth0)] = np.nan
    depth0[depth0 <= 0] = np.nan

    # 3) Bilateral
    depth_b = bilateral_depth_mm(
        depth0,
        d=bilateral_d,
        sigma_color_mm=bilateral_sigma_color_mm,
        sigma_space_px=bilateral_sigma_space_px,
    )
    dbg["n_valid_depth_before_disc"] = int(np.isfinite(depth_b).sum())

    # 2) Discontinuity
    good = depth_discontinuity_good_pixels_mm(depth_b, thr_mm=disc_thr_mm, ksize=disc_ksize)
    depth_c = depth_b.copy()
    depth_c[~good] = np.nan
    dbg["n_good_after_disc"] = int(good.sum())

    # Use mask directly (no erosion - not in paper)
    mask_bool = (raw_mask_u8 > 0).astype(bool)
    if not mask_bool.any():
        dbg["reason"] = "mask_empty"
        return np.empty((0, 3), np.float32), dbg

    # How many valid depth pixels survive inside mask after disc?
    dbg["n_valid_depth_in_mask_after_disc"] = int((mask_bool & np.isfinite(depth_c) & (depth_c > 0)).sum())

    # 3) Backproject (mm)
    pts = depth_mask_to_points_mm(depth_c, mask_bool, fx, fy, cx, cy)
    if pts.shape[0] < int(min_pts):
        dbg["reason"] = "too_few_points_after_backproject"
        dbg["n_after_backproject"] = int(pts.shape[0])
        return np.empty((0, 3), np.float32), dbg
    dbg["n_after_backproject"] = int(pts.shape[0])

    # 4) Radius outlier removal
    pts, _ = radius_outlier_removal_mm(pts, radius_mm=ror_radius_mm, min_neighbors=ror_min_neighbors)
    if pts.shape[0] < int(min_pts):
        dbg["reason"] = "too_few_points_after_ror"
        dbg["n_after_ror"] = int(pts.shape[0])
        return np.empty((0, 3), np.float32), dbg
    dbg["n_after_ror"] = int(pts.shape[0])

    # 5) Median distance filter
    pts, _ = median_distance_filter_mm(pts, k=med_k)
    if pts.shape[0] < int(min_pts):
        dbg["reason"] = "too_few_points_after_median"
        dbg["n_after_median"] = int(pts.shape[0])
        return np.empty((0, 3), np.float32), dbg
    dbg["n_after_median"] = int(pts.shape[0])

    dbg["reason"] = "ok"
    dbg["n_final"] = int(pts.shape[0])
    return pts.astype(np.float32), dbg
