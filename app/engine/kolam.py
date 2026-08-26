"""
PatternGenesis - Kolam Module (SIH validation domain)
Real deterministic generation algorithm (not a diffusion/image model):

  1. Lay out a pulli (dot) grid.
  2. Each grid cell is filled with one of two "Truchet" quarter-arc tiles
     (a well-documented basis for algorithmic kolam/rangoli weaving).
  3. Tile choice is decided only inside one fundamental domain; the domain
     is then reflected/rotated across the grid according to the requested
     symmetry order, so the *global* pattern is provably symmetric by
     construction, not by accident.
  4. The result is emitted as SVG arcs plus a node/edge graph (loops =
     connected components with cycles), so it plugs into the same
     GeometryGraph / Geometry DNA / symmetry-detection machinery used for
     analyzing uploaded images.
"""
from __future__ import annotations
import random
import math
from typing import Dict, List, Tuple
from .geometry_core import GeometryGraph

# Two tile variants: each connects the 4 side-midpoints of a unit cell with
# two arcs. Variant 0 connects (top-left pair), Variant 1 connects the other
# diagonal pair. Alternating/mirroring these is what produces the classic
# continuous interlocking kolam loops.
TILE_ARCS = {
    0: [("top", "left"), ("bottom", "right")],
    1: [("top", "right"), ("bottom", "left")],
}

SIDE_MIDPOINT = {
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
}


def _fundamental_domain_size(rows: int, cols: int, symmetry: str) -> Tuple[int, int]:
    if symmetry == "4fold":
        return math.ceil(rows / 2), math.ceil(cols / 2)
    if symmetry == "2fold":
        return math.ceil(rows / 2), cols
    return rows, cols  # 'none' -> full random field


def _mirror_index(i: int, n: int) -> int:
    return n - 1 - i


def build_tile_grid(rows: int, cols: int, symmetry: str, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    grid = [[0] * cols for _ in range(rows)]
    fr, fc = _fundamental_domain_size(rows, cols, symmetry)

    for i in range(fr):
        for j in range(fc):
            grid[i][j] = rng.randint(0, 1)

    if symmetry == "4fold":
        for i in range(fr):
            for j in range(fc):
                v = grid[i][j]
                mi, mj = _mirror_index(i, rows), _mirror_index(j, cols)
                if mj < cols:
                    grid[i][mj] = v  # mirror horizontally
                if mi < rows:
                    grid[mi][j] = v  # mirror vertically
                if mi < rows and mj < cols:
                    grid[mi][mj] = v  # mirror both (rotational completion)
    elif symmetry == "2fold":
        for i in range(fr):
            for j in range(cols):
                v = grid[i][j]
                mi = _mirror_index(i, rows)
                if mi < rows:
                    grid[mi][j] = v
    return grid


def generate_kolam(rows: int = 7, cols: int = 7, symmetry: str = "4fold",
                    seed: int = 42, cell_size: float = 40.0) -> Dict:
    """
    Returns dict with: dot_grid points, svg path data (list of arc/line
    primitives), and a GeometryGraph built from the arc endpoints so the
    same symmetry-detection + Geometry DNA code path used for uploaded
    images also applies to generated kolams.
    """
    rows = max(2, min(rows, 25))
    cols = max(2, min(cols, 25))
    tile_grid = build_tile_grid(rows, cols, symmetry, seed)

    dots = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            dots.append((j * cell_size, i * cell_size))

    svg_arcs = []
    graph = GeometryGraph()
    node_lookup: Dict[Tuple[float, float], int] = {}
    next_id = [0]

    def node_for(pt: Tuple[float, float]) -> int:
        key = (round(pt[0], 2), round(pt[1], 2))
        if key not in node_lookup:
            node_lookup[key] = next_id[0]
            graph.add_node(next_id[0], pt[0], pt[1])
            next_id[0] += 1
        return node_lookup[key]

    for i in range(rows):
        for j in range(cols):
            variant = tile_grid[i][j]
            ox, oy = j * cell_size, i * cell_size
            for (side_a, side_b) in TILE_ARCS[variant]:
                pa = SIDE_MIDPOINT[side_a]
                pb = SIDE_MIDPOINT[side_b]
                p1 = (ox + pa[0] * cell_size, oy + pa[1] * cell_size)
                p2 = (ox + pb[0] * cell_size, oy + pb[1] * cell_size)
                # the "corner" the arc bulges toward, for a proper quarter-circle
                corner = _shared_corner(side_a, side_b)
                cx, cy = ox + corner[0] * cell_size, oy + corner[1] * cell_size
                r = cell_size / 2.0
                svg_arcs.append({
                    "type": "arc", "p1": p1, "p2": p2, "center": (cx, cy),
                    "radius": r,
                })
                n1, n2 = node_for(p1), node_for(p2)
                graph.add_edge(n1, n2, kind="arc")

    width = cols * cell_size
    height = rows * cell_size
    return {
        "rows": rows, "cols": cols, "symmetry": symmetry, "seed": seed,
        "cell_size": cell_size,
        "width": width, "height": height,
        "dot_grid": dots,
        "arcs": svg_arcs,
        "graph": graph.to_json(),
    }


def _shared_corner(side_a: str, side_b: str) -> Tuple[float, float]:
    corners = {
        frozenset(["top", "left"]): (0.0, 0.0),
        frozenset(["top", "right"]): (1.0, 0.0),
        frozenset(["bottom", "left"]): (0.0, 1.0),
        frozenset(["bottom", "right"]): (1.0, 1.0),
    }
    return corners[frozenset([side_a, side_b])]


def kolam_to_svg(data: Dict, stroke: str = "#1a1a2e", stroke_width: float = 2.5,
                  show_dots: bool = True) -> str:
    w, h = data["width"], data["height"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
             f'width="{w}" height="{h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="transparent"/>')
    if show_dots:
        for (x, y) in data["dot_grid"]:
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.2" fill="#c94b4b"/>')
    for arc in data["arcs"]:
        p1, p2, r = arc["p1"], arc["p2"], arc["radius"]
        # large-arc-flag=0, sweep-flag chosen for a quarter circle bulge
        d = f'M {p1[0]:.2f} {p1[1]:.2f} A {r:.2f} {r:.2f} 0 0 1 {p2[0]:.2f} {p2[1]:.2f}'
        parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" '
                      f'stroke-width="{stroke_width}" stroke-linecap="round"/>')
    parts.append('</svg>')
    return "\n".join(parts)


def count_loops(graph_json: Dict) -> int:
    """Loops = independent cycles in the arc graph (first Betti number)."""
    import networkx as nx
    g = nx.Graph()
    for n in graph_json["nodes"]:
        g.add_node(n["id"])
    for e in graph_json["edges"]:
        g.add_edge(e["source"], e["target"])
    if g.number_of_nodes() == 0:
        return 0
    components = nx.number_connected_components(g)
    return g.number_of_edges() - g.number_of_nodes() + components
