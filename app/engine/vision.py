"""
PatternGenesis - Vision Layer
IMAGE -> GEOMETRY. Real OpenCV processing: edge/contour extraction, dot/node
detection (Hough circles + Harris corners), skeletonization, and conversion
of raw pixel structure into the GeometryGraph used by the rest of the engine.
No generative image models here -- this stage produces vector/graph data,
not another picture.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .geometry_core import GeometryGraph


def load_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data")
    return img


def preprocess(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def extract_edges(img_gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return edges


def extract_contours(edges: np.ndarray, epsilon_ratio: float = 0.01) -> List[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    simplified = []
    for c in contours:
        if cv2.contourArea(c) < 6 and cv2.arcLength(c, True) < 15:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon_ratio * peri, True)
        simplified.append(approx.reshape(-1, 2))
    return simplified


def detect_dot_grid(img_gray: np.ndarray) -> np.ndarray:
    """Detect circular dots (common substrate of Kolam/rangoli designs) via
    Hough Circle Transform. Returns Nx2 array of (x, y) centers."""
    blurred = cv2.medianBlur(img_gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
        param1=80, param2=25, minRadius=2, maxRadius=25,
    )
    if circles is None:
        return np.zeros((0, 2))
    circles = np.round(circles[0, :]).astype(int)
    return circles[:, :2].astype(float)


def detect_corners(img_gray: np.ndarray, max_corners: int = 300) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        img_gray, maxCorners=max_corners, qualityLevel=0.02, minDistance=8
    )
    if corners is None:
        return np.zeros((0, 2))
    return corners.reshape(-1, 2)


def skeletonize(binary_img: np.ndarray) -> np.ndarray:
    """Zhang-Suen style skeletonization via iterative morphological thinning."""
    img = binary_img.copy()
    img[img > 0] = 1
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img.copy() * 255
    done = False
    iterations = 0
    while not done and iterations < 200:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()
        done = cv2.countNonZero(temp) == 0
        iterations += 1
    return skel


def contours_to_polylines(contours: List[np.ndarray]) -> List[List[Tuple[float, float]]]:
    return [[(float(p[0]), float(p[1])) for p in c] for c in contours]


def build_geometry_graph_from_contours(contours: List[np.ndarray],
                                        merge_radius: float = 6.0) -> GeometryGraph:
    """
    Converts a set of polyline contours into a node/edge topology graph,
    merging vertices that are within `merge_radius` px of each other so that
    shared intersections in the original artwork collapse to single nodes
    (this is what turns disconnected pixel contours into real topology).
    """
    g = GeometryGraph()
    all_points: List[Tuple[float, float]] = []
    point_owner: List[Tuple[int, int]] = []  # (contour_idx, point_idx)
    for ci, c in enumerate(contours):
        for pi, p in enumerate(c):
            all_points.append((float(p[0]), float(p[1])))
            point_owner.append((ci, pi))

    if not all_points:
        return g

    pts = np.array(all_points)
    n = len(pts)
    node_id_of = -np.ones(n, dtype=int)
    next_id = 0

    # naive spatial merge (fine at typical contour vertex counts < 3000)
    for i in range(n):
        if node_id_of[i] != -1:
            continue
        node_id_of[i] = next_id
        dists = np.linalg.norm(pts - pts[i], axis=1)
        close = np.where((dists < merge_radius) & (node_id_of == -1))[0]
        node_id_of[close] = next_id
        next_id += 1

    for nid in range(next_id):
        member_idx = np.where(node_id_of == nid)[0]
        centroid = pts[member_idx].mean(axis=0)
        g.add_node(nid, centroid[0], centroid[1])

    offset = 0
    for c in contours:
        count = len(c)
        node_seq = [int(nid) for nid in node_id_of[offset:offset + count]]
        offset += count
        for k in range(len(node_seq) - 1):
            a, b = node_seq[k], node_seq[k + 1]
            if a != b:
                g.add_edge(a, b, kind="contour")
        # OpenCV contours are closed curves; preserve the closing edge.
        if len(node_seq) > 2 and node_seq[-1] != node_seq[0]:
            g.add_edge(node_seq[-1], node_seq[0], kind="contour")
    return g


def analyze_image(img_bytes: bytes) -> Dict:
    """
    Full IMAGE -> GEOMETRY pipeline. Returns raw structured data (contours,
    dot grid, corners, graph) that downstream modules (geometry_core, dna)
    turn into symmetry + mathematics.
    """
    img = preprocess(load_image_bytes(img_bytes))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = extract_edges(gray)
    contours = extract_contours(edges)
    dots = detect_dot_grid(gray)
    corners = detect_corners(gray)
    graph = build_geometry_graph_from_contours(contours)

    h, w = img.shape[:2]
    return {
        "image_size": {"width": int(w), "height": int(h)},
        "contours": contours_to_polylines(contours),
        "num_contours": len(contours),
        "dot_grid": dots.tolist(),
        "num_dots": int(len(dots)),
        "corners": corners.tolist(),
        "num_corners": int(len(corners)),
        "graph": graph.to_json(),
    }
