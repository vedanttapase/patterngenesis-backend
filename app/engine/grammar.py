"""Deterministic, image-derived design grammar inference.

This module describes measurable contour structure. It deliberately does not
infer cultural attribution or claim that a geometric family proves origin.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import cv2
import numpy as np


def infer_design_grammar(contours: Iterable[Iterable], geometry_dna: dict | None = None) -> dict[str, Any]:
    """Summarize contour primitives and repeated measurable motif families."""
    records: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    for contour in contours:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        cv_contour = points.reshape(-1, 1, 2)
        perimeter = float(cv2.arcLength(cv_contour, True))
        area = float(cv2.contourArea(cv_contour))
        if perimeter <= 1e-6:
            continue
        approx = cv2.approxPolyDP(cv_contour, 0.018 * perimeter, True).reshape(-1, 2)
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        lengths = lengths[lengths > 1e-6]
        length_cv = float(lengths.std() / lengths.mean()) if len(lengths) else 1.0
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        width, height = float(x_max - x_min), float(y_max - y_min)
        aspect = width / height if height > 1e-6 else 1.0

        if circularity >= 0.82:
            family = "circle-like contour"
            primitive = "circle_or_arc"
        elif 3 <= len(approx) <= 12 and length_cv <= 0.22:
            family = f"approximately regular {len(approx)}-sided polygon"
            primitive = "regular_polygon"
        elif len(approx) <= 10:
            family = "polygonal contour"
            primitive = "polygon"
        else:
            family = "freeform closed contour"
            primitive = "closed_curve"

        family_counts[family] += 1
        primitive_counts[primitive] += 1
        records.append({
            "family": family,
            "primitive": primitive,
            "vertex_count": int(len(approx)),
            "area_px2": round(area, 2),
            "perimeter_px": round(perimeter, 2),
            "circularity": round(max(0.0, min(1.0, circularity)), 4),
            "aspect_ratio": round(aspect, 4),
            "side_length_cv": round(length_cv, 4),
        })

    rules: list[str] = []
    if not records:
        rules.append("No closed contours with measurable area were extracted; improve contrast or crop the artwork.")
    else:
        rules.append(f"Extracted {len(records)} measurable closed contour(s) from the source image.")
        for family, count in family_counts.most_common(5):
            rules.append(
                f"Repeated motif family: {family} appears {count} time(s)."
                if count > 1 else f"Primitive family present: {family}."
            )
        orders = (geometry_dna or {}).get("rotational_order")
        score = (geometry_dna or {}).get("rotational_score")
        if isinstance(orders, int) and orders > 1 and isinstance(score, (int, float)) and score >= 0.8:
            rules.append(f"Point-set analysis suggests {orders}-fold rotational symmetry (score {score:.2f}); verify visually.")
        else:
            rules.append("No high-confidence rotational symmetry rule was established from the extracted point set.")
    return {
        "method": "opencv_contour_heuristics",
        "confidence": "heuristic",
        "contour_count": len(records),
        "primitive_counts": dict(primitive_counts),
        "motif_families": [{"family": family, "count": count} for family, count in family_counts.most_common()],
        "contours": records[:250],
        "rules": rules,
        "limitations": [
            "Contour families are geometric heuristics, not proof of cultural origin or historical meaning.",
            "Raster noise, overlapping strokes, perspective, and low contrast can change contour counts.",
        ],
    }
