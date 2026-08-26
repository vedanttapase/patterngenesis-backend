"""
PatternGenesis - Restoration Module
Given a damaged/incomplete geometric design, this module:
  1. Locates the missing/damaged region (user-supplied mask, or an automatic
     low-structure-density heuristic).
  2. Detects the symmetry of the SURVIVING geometry (rotational/reflection),
     using geometry_core's real symmetry detectors on extracted contour
     points from the undamaged area.
  3. Reconstructs the missing pixels by copying content from the
     symmetry-equivalent region elsewhere in the same image (e.g. if the
     design has 4-fold rotational symmetry, the missing wedge is filled from
     the corresponding wedge 90/180/270 degrees away).

This is explicitly a MATHEMATICALLY INFERRED reconstruction, built from the
image's own detected symmetry -- never claimed as historically verified.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from .geometry_core import (
    detect_rotational_symmetry, detect_reflection_symmetry, rotate, reflect,
)
from .vision import preprocess, extract_edges, extract_contours


def auto_detect_missing_region(img: np.ndarray, block: int = 24,
                                density_threshold: float = 0.015) -> np.ndarray:
    """
    Heuristic: split the image into blocks, compute edge-pixel density per
    block. Blocks with near-zero structure surrounded by high-structure
    neighbours are flagged as 'missing/blank'. Returns a binary mask
    (255 = missing) the same size as the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = extract_edges(gray)
    h, w = edges.shape
    mask = np.zeros((h, w), np.uint8)

    rows = h // block
    cols = w // block
    density = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            patch = edges[i * block:(i + 1) * block, j * block:(j + 1) * block]
            density[i, j] = np.count_nonzero(patch) / (block * block)

    global_mean = density.mean()
    for i in range(rows):
        for j in range(cols):
            neighbours = density[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if density[i, j] < density_threshold and neighbours.max() > global_mean:
                mask[i * block:(i + 1) * block, j * block:(j + 1) * block] = 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def _surviving_points(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = extract_edges(gray)
    edges[mask > 0] = 0  # ignore the damaged area itself
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return np.zeros((0, 2))
    pts = np.column_stack([xs, ys]).astype(float)
    if len(pts) > 2000:
        idx = np.random.default_rng(0).choice(len(pts), 2000, replace=False)
        pts = pts[idx]
    return pts


def reconstruct(img: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Returns a dict with all the views required by the spec:
      missing_only (image with only the reconstructed pixels, rest transparent-ish black)
      reconstructed (full composited image)
      symmetry_info (what math was used)
    """
    h, w = img.shape[:2]
    points = _surviving_points(img, mask)
    rot = detect_rotational_symmetry(points, max_order=8, threshold=0.55)
    refl = detect_reflection_symmetry(points, threshold=0.55)

    center = (w / 2.0, h / 2.0)
    candidates = []
    if rot["order"] > 1:
        for k in range(1, rot["order"]):
            candidates.append(("rotation", 360.0 * k / rot["order"]))
    for axis in refl.get("axes", []):
        candidates.append(("reflection", axis["axis_angle_deg"]))

    reconstructed = img.copy()
    missing_only = np.zeros_like(img)
    ys, xs = np.where(mask > 0)
    filled = np.zeros((h, w), dtype=bool)
    method_used = "none"

    if len(xs) > 0 and candidates:
        pts = np.column_stack([xs, ys]).astype(float)
        for kind, param in candidates:
            if kind == "rotation":
                src = rotate(pts, -param, center=center)
            else:
                src = reflect(pts, param, center=center)
            sx = np.clip(src[:, 0], 0, w - 1).astype(int)
            sy = np.clip(src[:, 1], 0, h - 1).astype(int)
            valid = mask[sy, sx] == 0  # source pixel must itself be undamaged
            valid &= ~filled[ys, xs]
            take_y, take_x = ys[valid], xs[valid]
            take_sy, take_sx = sy[valid], sx[valid]
            reconstructed[take_y, take_x] = img[take_sy, take_sx]
            missing_only[take_y, take_x] = img[take_sy, take_sx]
            filled[ys, xs] |= valid
            method_used = "symmetry_copy"
            if filled[ys, xs].all():
                break

    # anything still unfilled: fall back to simple directional inpainting
    remaining = mask.copy()
    remaining[filled] = 0
    if remaining.any():
        reconstructed = cv2.inpaint(reconstructed, remaining, 5, cv2.INPAINT_TELEA)
        if method_used == "none":
            method_used = "inpaint_fallback"
        else:
            method_used += "+inpaint_fallback"

    return {
        "symmetry_used": {"rotational": rot, "reflection": refl},
        "candidates_tried": candidates,
        "method": method_used,
        "coverage_from_symmetry": float(filled[ys, xs].mean()) if len(xs) else 0.0,
        "reconstructed_img": reconstructed,
        "missing_only_img": missing_only,
        "mask": mask,
    }


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("PNG encode failed")
    return buf.tobytes()
