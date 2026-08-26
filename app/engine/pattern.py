"""
PatternGenesis - General Pattern Generator
Not Kolam-specific. Produces radially/linearly symmetric geometric motifs
(the kind found in jali lattices, temple ceiling medallions, rose windows,
tile borders) from explicit numeric parameters: fold count, ring count,
motif shape, and repetition rule. This is the module that demonstrates the
engine generalizes past the Kolam validation domain.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
from .geometry_core import GeometryGraph, rotate
import numpy as np


def _polygon_points(cx: float, cy: float, r: float, n: int, rot_deg: float = 0.0) -> List[Tuple[float, float]]:
    pts = []
    for k in range(n):
        a = math.radians(rot_deg + 360.0 * k / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def generate_radial_pattern(nfold: int = 8, rings: int = 3, motif: str = "polygon",
                             motif_sides: int = 4, base_radius: float = 40.0,
                             ring_gap: float = 35.0, size: float = 420.0) -> Dict:
    """
    Builds an n-fold rotationally symmetric pattern: `rings` concentric bands
    of a repeated motif, each band containing `nfold` copies of the motif
    rotated evenly around the center (the literal generator of the C_n / D_n
    symmetry group requested by the user, e.g. "8-fold rotational jali").
    """
    nfold = max(2, min(nfold, 24))
    rings = max(1, min(rings, 8))
    cx = cy = size / 2.0

    graph = GeometryGraph()
    node_lookup = {}
    next_id = [0]

    def node_for(pt):
        key = (round(pt[0], 2), round(pt[1], 2))
        if key not in node_lookup:
            node_lookup[key] = next_id[0]
            graph.add_node(next_id[0], pt[0], pt[1])
            next_id[0] += 1
        return node_lookup[key]

    shapes = []  # list of list-of-points (closed polygons) for SVG
    for ring in range(rings):
        r = base_radius + ring * ring_gap
        motif_r = ring_gap * 0.42
        for k in range(nfold):
            a = 360.0 * k / nfold
            mx = cx + r * math.cos(math.radians(a))
            my = cy + r * math.sin(math.radians(a))
            if motif == "polygon":
                pts = _polygon_points(mx, my, motif_r, motif_sides, rot_deg=a)
            elif motif == "star":
                pts = _star_points(mx, my, motif_r, motif_sides, rot_deg=a)
            else:
                pts = _polygon_points(mx, my, motif_r, 3, rot_deg=a)
            shapes.append(pts)
            ids = [node_for(p) for p in pts]
            for i in range(len(ids)):
                graph.add_edge(ids[i], ids[(i + 1) % len(ids)], kind="polygon")

    # connecting radial spokes between rings (typical jali lattice bracing)
    if rings > 1:
        for k in range(nfold):
            a = math.radians(360.0 * k / nfold)
            prev = None
            for ring in range(rings):
                r = base_radius + ring * ring_gap
                pt = (cx + r * math.cos(a), cy + r * math.sin(a))
                nid = node_for(pt)
                if prev is not None:
                    graph.add_edge(prev, nid, kind="spoke")
                prev = nid

    return {
        "nfold": nfold, "rings": rings, "motif": motif, "motif_sides": motif_sides,
        "size": size, "center": (cx, cy),
        "shapes": shapes,
        "graph": graph.to_json(),
    }


def _star_points(cx, cy, r, points, rot_deg=0.0):
    pts = []
    for k in range(points * 2):
        rad = r if k % 2 == 0 else r * 0.45
        a = math.radians(rot_deg + 360.0 * k / (points * 2))
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return pts


def radial_pattern_to_svg(data: Dict, stroke="#1a1a2e", fill="none", stroke_width=1.8) -> str:
    size = data["size"]
    cx, cy = data["center"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
             f'width="{size}" height="{size}">']
    for shape in data["shapes"]:
        d = "M " + " L ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in shape) + " Z"
        parts.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
    for edge in data["graph"]["edges"]:
        if edge["kind"] == "spoke":
            nodes = {n["id"]: n for n in data["graph"]["nodes"]}
            a, b = nodes[edge["source"]], nodes[edge["target"]]
            parts.append(f'<line x1="{a["x"]:.2f}" y1="{a["y"]:.2f}" x2="{b["x"]:.2f}" y2="{b["y"]:.2f}" '
                          f'stroke="{stroke}" stroke-width="{stroke_width*0.6}" stroke-dasharray="2,3"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{stroke}"/>')
    parts.append('</svg>')
    return "\n".join(parts)


def generate_linear_border(unit_count: int = 10, unit_width: float = 40.0,
                            unit_height: float = 60.0, motif_sides: int = 6) -> Dict:
    """Linear repetition pattern (temple/tile borders): one motif translated
    along a line, i.e. the frieze-group generator p1/pm depending on motif."""
    graph = GeometryGraph()
    shapes = []
    node_lookup = {}
    next_id = [0]

    def node_for(pt):
        key = (round(pt[0], 2), round(pt[1], 2))
        if key not in node_lookup:
            node_lookup[key] = next_id[0]
            graph.add_node(next_id[0], pt[0], pt[1])
            next_id[0] += 1
        return node_lookup[key]

    for i in range(unit_count):
        cx = unit_width * (i + 0.5)
        cy = unit_height / 2
        pts = _polygon_points(cx, cy, min(unit_width, unit_height) * 0.38, motif_sides,
                               rot_deg=0 if i % 2 == 0 else 180.0 / motif_sides)
        shapes.append(pts)
        ids = [node_for(p) for p in pts]
        for k in range(len(ids)):
            graph.add_edge(ids[k], ids[(k + 1) % len(ids)], kind="polygon")

    return {
        "unit_count": unit_count, "unit_width": unit_width, "unit_height": unit_height,
        "width": unit_count * unit_width, "height": unit_height,
        "shapes": shapes, "graph": graph.to_json(),
    }


def linear_border_to_svg(data: Dict, stroke="#1a1a2e", stroke_width=1.8) -> str:
    w, h = data["width"], data["height"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for shape in data["shapes"]:
        d = "M " + " L ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in shape) + " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}"/>')
    parts.append('</svg>')
    return "\n".join(parts)
