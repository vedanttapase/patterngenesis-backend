"""
PatternGenesis - Core Geometry Engine
Deterministic, explicit mathematical geometry: transformations, symmetry
detection, graph representation. No neural nets here on purpose -- this
module is the "ground truth" math layer that everything else builds on.
"""
from __future__ import annotations
import math
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Transformations (explicit 2D affine math -- no black boxes)
# --------------------------------------------------------------------------

def translate(points: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = points.copy().astype(float)
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def rotate(points: np.ndarray, angle_deg: float, center: Point = (0, 0)) -> np.ndarray:
    a = math.radians(angle_deg)
    cx, cy = center
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    shifted = points - np.array([cx, cy])
    rotated = shifted @ R.T
    return rotated + np.array([cx, cy])


def reflect(points: np.ndarray, axis_angle_deg: float, center: Point = (0, 0)) -> np.ndarray:
    """Reflect points across a line through `center` at angle axis_angle_deg."""
    a = math.radians(axis_angle_deg)
    cx, cy = center
    # Reflection matrix across a line through origin at angle a
    M = np.array([
        [math.cos(2 * a), math.sin(2 * a)],
        [math.sin(2 * a), -math.cos(2 * a)],
    ])
    shifted = points - np.array([cx, cy])
    reflected = shifted @ M.T
    return reflected + np.array([cx, cy])


def scale(points: np.ndarray, sx: float, sy: Optional[float] = None, center: Point = (0, 0)) -> np.ndarray:
    if sy is None:
        sy = sx
    cx, cy = center
    shifted = points - np.array([cx, cy])
    shifted[:, 0] *= sx
    shifted[:, 1] *= sy
    return shifted + np.array([cx, cy])


# --------------------------------------------------------------------------
# Symmetry detection (point-set based -- works on extracted node/contour sets)
# --------------------------------------------------------------------------

def _centroid(points: np.ndarray) -> Point:
    return (float(points[:, 0].mean()), float(points[:, 1].mean()))


def _normalized_point_set(points: np.ndarray, center: Point) -> np.ndarray:
    """Center points; used for chamfer-style matching."""
    return points - np.array(center)


def _chamfer_similarity(a: np.ndarray, b: np.ndarray, tol_scale: float) -> float:
    """
    Symmetric nearest-neighbour distance between two point sets, normalized
    into a 0..1 similarity score (1 = perfect match). Uses a simple KD-tree
    free O(n*m) approach which is fine at the node-count scale (<2000) that
    this engine deals with.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    # KD-trees avoid allocating an O(n*m) pairwise-distance tensor for
    # dense image-derived point clouds.
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)
    d_ba, _ = tree_a.query(b, k=1)
    mean_err = (float(d_ab.mean()) + float(d_ba.mean())) / 2.0
    sim = math.exp(-mean_err / max(tol_scale, 1e-6))
    return float(sim)


def detect_rotational_symmetry(points: np.ndarray, max_order: int = 12,
                                threshold: float = 0.80) -> Dict:
    """
    Tests rotational symmetry orders 2..max_order by rotating the point set
    about its centroid and measuring overlap with the original set.
    Returns the highest-scoring order above `threshold`.
    """
    if len(points) < 3:
        return {"order": 1, "score": 0.0, "type": "none"}

    center = _centroid(points)
    norm = _normalized_point_set(points, center)
    span = np.linalg.norm(norm, axis=1).max() or 1.0
    tol_scale = span * 0.06  # 6% of radius tolerance

    best = {"order": 1, "score": 0.0}
    for n in range(2, max_order + 1):
        angle = 360.0 / n
        rotated = rotate(norm, angle, center=(0, 0))
        sim = _chamfer_similarity(norm, rotated, tol_scale)
        if sim > best["score"]:
            best = {"order": n, "score": sim}

    if best["score"] >= threshold:
        return {"order": best["order"], "score": round(best["score"], 4), "type": "rotational"}
    return {"order": 1, "score": round(best["score"], 4), "type": "none"}


def detect_reflection_symmetry(points: np.ndarray, angle_steps: int = 36,
                                threshold: float = 0.80) -> Dict:
    """
    Sweeps candidate reflection axes (through the centroid) from 0-180deg
    and returns the best-matching axis, if any exceeds `threshold`.
    """
    if len(points) < 3:
        return {"axes": [], "type": "none"}

    center = _centroid(points)
    norm = _normalized_point_set(points, center)
    span = np.linalg.norm(norm, axis=1).max() or 1.0
    tol_scale = span * 0.06

    found = []
    for i in range(angle_steps):
        angle = i * (180.0 / angle_steps)
        reflected = reflect(norm, angle, center=(0, 0))
        sim = _chamfer_similarity(norm, reflected, tol_scale)
        if sim >= threshold:
            found.append({"axis_angle_deg": round(angle, 2), "score": round(sim, 4)})

    # de-duplicate axes that are within 5 degrees of each other, keep best
    found.sort(key=lambda x: -x["score"])
    deduped = []
    for f in found:
        if all(abs(f["axis_angle_deg"] - g["axis_angle_deg"]) > 5 for g in deduped):
            deduped.append(f)

    return {"axes": deduped, "type": "reflection" if deduped else "none"}


def dihedral_order(rot: Dict, refl: Dict) -> str:
    """Combine rotational + reflection results into a point-group label."""
    n = rot.get("order", 1)
    has_refl = len(refl.get("axes", [])) > 0
    if n <= 1 and not has_refl:
        return "C1 (asymmetric)"
    if has_refl:
        return f"D{n} (dihedral, order {2 * n})"
    return f"C{n} (cyclic, order {n})"


# --------------------------------------------------------------------------
# Graph representation
# --------------------------------------------------------------------------

@dataclass
class GeometryGraph:
    """Node/edge topology representation of an extracted or generated design."""
    graph: nx.Graph = field(default_factory=nx.Graph)

    def add_node(self, idx: int, x: float, y: float, kind: str = "point"):
        self.graph.add_node(idx, x=float(x), y=float(y), kind=kind)

    def add_edge(self, a: int, b: int, kind: str = "line", **attrs):
        self.graph.add_edge(a, b, kind=kind, **attrs)

    def node_points(self) -> np.ndarray:
        if self.graph.number_of_nodes() == 0:
            return np.zeros((0, 2))
        return np.array([[d["x"], d["y"]] for _, d in self.graph.nodes(data=True)])

    def connected_components(self) -> int:
        return nx.number_connected_components(self.graph)

    def to_json(self) -> Dict:
        nodes = [{"id": int(n), "x": d["x"], "y": d["y"], "kind": d.get("kind", "point")}
                 for n, d in self.graph.nodes(data=True)]
        edges = [{"source": int(u), "target": int(v), "kind": d.get("kind", "line")}
                 for u, v, d in self.graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def from_json(data: Dict) -> "GeometryGraph":
        g = GeometryGraph()
        for n in data.get("nodes", []):
            g.add_node(n["id"], n["x"], n["y"], n.get("kind", "point"))
        for e in data.get("edges", []):
            g.add_edge(e["source"], e["target"], e.get("kind", "line"))
        return g


def polygon_angles(points: List[Point]) -> List[float]:
    """Interior angles (degrees) of a closed polygon, in order."""
    n = len(points)
    angles = []
    for i in range(n):
        p0 = np.array(points[i - 1])
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % n])
        v1, v2 = p0 - p1, p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cos_a = max(-1.0, min(1.0, cos_a))
        angles.append(round(math.degrees(math.acos(cos_a)), 2))
    return angles


def polygon_side_lengths(points: List[Point]) -> List[float]:
    n = len(points)
    out = []
    for i in range(n):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % n])
        out.append(round(float(np.linalg.norm(p2 - p1)), 3))
    return out
