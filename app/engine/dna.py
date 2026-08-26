"""
PatternGenesis - Geometry DNA
A structured, comparable signature of a design's mathematical structure,
independent of pixel similarity. Two designs with the same DNA fields close
together are geometrically related even if they look visually different
(different color, stroke width, resolution).
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict
from .geometry_core import (
    GeometryGraph, detect_rotational_symmetry, detect_reflection_symmetry,
    dihedral_order,
)


def compute_geometry_dna(graph: GeometryGraph, extra: Dict = None) -> Dict:
    points = graph.node_points()
    rot = detect_rotational_symmetry(points) if len(points) >= 3 else {"order": 1, "score": 0.0, "type": "none"}
    refl = detect_reflection_symmetry(points) if len(points) >= 3 else {"axes": [], "type": "none"}
    point_group = dihedral_order(rot, refl)

    n_nodes = graph.graph.number_of_nodes()
    n_edges = graph.graph.number_of_edges()
    components = graph.connected_components()

    if len(points) > 0:
        centroid = points.mean(axis=0)
        radii = np.linalg.norm(points - centroid, axis=1)
        proportions = {
            "mean_radius": round(float(radii.mean()), 3),
            "radius_std": round(float(radii.std()), 3),
            "bbox_aspect": round(float(_bbox_aspect(points)), 3),
        }
    else:
        proportions = {"mean_radius": 0.0, "radius_std": 0.0, "bbox_aspect": 1.0}

    dna = {
        "symmetry_type": point_group,
        "rotational_order": rot["order"],
        "rotational_score": rot["score"],
        "reflection_axes": refl["axes"],
        "num_primary_nodes": n_nodes,
        "num_edges": n_edges,
        "connected_components": components,
        "avg_degree": round((2 * n_edges / n_nodes), 3) if n_nodes else 0.0,
        "proportions": proportions,
        "complexity_index": round(_complexity_index(n_nodes, n_edges, rot["order"]), 3),
    }
    if extra:
        dna["extra"] = extra
    return dna


def _bbox_aspect(points: np.ndarray) -> float:
    w = points[:, 0].max() - points[:, 0].min()
    h = points[:, 1].max() - points[:, 1].min()
    if h == 0:
        return 1.0
    return w / h


def _complexity_index(n_nodes: int, n_edges: int, symmetry_order: int) -> float:
    """Rough scalar summarizing structural richness relative to how much
    of it is 'explained' by symmetry (a highly symmetric design with many
    nodes is still 'simple' because one fundamental domain repeats)."""
    if n_nodes == 0:
        return 0.0
    raw = math.log1p(n_nodes + n_edges)
    return raw / max(1, symmetry_order) ** 0.5


def compare_dna(a: Dict, b: Dict) -> Dict:
    """Structural distance between two Geometry DNA signatures (0 = identical
    structure, larger = more different). Compares symmetry order, node/edge
    counts and proportions -- NOT pixels."""
    def num(x, default=0.0):
        return float(x) if isinstance(x, (int, float)) else default

    sym_diff = abs(num(a.get("rotational_order")) - num(b.get("rotational_order")))
    node_diff = abs(num(a.get("num_primary_nodes")) - num(b.get("num_primary_nodes")))
    edge_diff = abs(num(a.get("num_edges")) - num(b.get("num_edges")))
    complexity_diff = abs(num(a.get("complexity_index")) - num(b.get("complexity_index")))

    pa = a.get("proportions", {})
    pb = b.get("proportions", {})
    aspect_diff = abs(num(pa.get("bbox_aspect"), 1.0) - num(pb.get("bbox_aspect"), 1.0))

    # weighted normalized distance
    distance = (
        0.35 * min(sym_diff / 4.0, 1.0)
        + 0.20 * min(node_diff / 50.0, 1.0)
        + 0.20 * min(edge_diff / 50.0, 1.0)
        + 0.15 * min(complexity_diff / 3.0, 1.0)
        + 0.10 * min(aspect_diff / 2.0, 1.0)
    )
    similarity = round(1.0 - distance, 4)
    return {
        "similarity": max(0.0, similarity),
        "distance": round(distance, 4),
        "same_symmetry_type": a.get("symmetry_type") == b.get("symmetry_type"),
        "details": {
            "rotational_order_diff": sym_diff,
            "node_count_diff": node_diff,
            "edge_count_diff": edge_diff,
            "complexity_diff": round(complexity_diff, 3),
        },
    }
