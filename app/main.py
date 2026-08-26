"""
PatternGenesis Backend
Endpoints implementing: ANALYZE, GENERATE (kolam + general pattern),
RESTORE, GEOMETRY DNA / COMPARE, AI-assisted parameter EDIT.
"""
from __future__ import annotations
import base64
import io
import json
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import vision, geometry_core, dna as dna_mod, kolam as kolam_mod
from .engine import pattern as pattern_mod, restore as restore_mod, ai_edit

app = FastAPI(title="PatternGenesis Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _decode_data_url(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "PatternGenesis", "version": "0.1.0"}


# --------------------------------------------------------------------------
# ANALYZE: image -> geometry -> mathematics
# --------------------------------------------------------------------------

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        result = vision.analyze_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    graph = geometry_core.GeometryGraph.from_json(result["graph"])
    dna = dna_mod.compute_geometry_dna(graph, extra={
        "num_contours": result["num_contours"],
        "num_dots": result["num_dots"],
        "num_corners": result["num_corners"],
    })

    return JSONResponse({
        "image_size": result["image_size"],
        "contours": result["contours"],
        "dot_grid": result["dot_grid"],
        "corners": result["corners"],
        "graph": result["graph"],
        "geometry_dna": dna,
    })


# --------------------------------------------------------------------------
# GENERATE: kolam
# --------------------------------------------------------------------------

class KolamParams(BaseModel):
    rows: int = 7
    cols: int = 7
    symmetry: str = "4fold"   # 4fold | 2fold | none
    seed: int = 42
    cell_size: float = 40.0


@app.post("/api/generate/kolam")
def generate_kolam(params: KolamParams):
    data = kolam_mod.generate_kolam(params.rows, params.cols, params.symmetry,
                                     params.seed, params.cell_size)
    svg = kolam_mod.kolam_to_svg(data)
    graph = geometry_core.GeometryGraph.from_json(data["graph"])
    dna = dna_mod.compute_geometry_dna(graph, extra={"loops": kolam_mod.count_loops(data["graph"])})
    return {"params": params.dict(), "svg": svg, "graph": data["graph"],
            "dot_grid": data["dot_grid"], "geometry_dna": dna,
            "width": data["width"], "height": data["height"]}


# --------------------------------------------------------------------------
# GENERATE: general radial / linear pattern (jali, borders, façade motifs)
# --------------------------------------------------------------------------

class RadialParams(BaseModel):
    nfold: int = 8
    rings: int = 3
    motif: str = "polygon"     # polygon | star
    motif_sides: int = 4
    base_radius: float = 40.0
    ring_gap: float = 35.0
    size: float = 420.0


@app.post("/api/generate/pattern")
def generate_pattern(params: RadialParams):
    data = pattern_mod.generate_radial_pattern(
        params.nfold, params.rings, params.motif, params.motif_sides,
        params.base_radius, params.ring_gap, params.size,
    )
    svg = pattern_mod.radial_pattern_to_svg(data)
    graph = geometry_core.GeometryGraph.from_json(data["graph"])
    dna = dna_mod.compute_geometry_dna(graph)
    return {"params": params.dict(), "svg": svg, "graph": data["graph"],
            "geometry_dna": dna, "size": data["size"]}


class BorderParams(BaseModel):
    unit_count: int = 10
    unit_width: float = 40.0
    unit_height: float = 60.0
    motif_sides: int = 6


@app.post("/api/generate/border")
def generate_border(params: BorderParams):
    data = pattern_mod.generate_linear_border(
        params.unit_count, params.unit_width, params.unit_height, params.motif_sides,
    )
    svg = pattern_mod.linear_border_to_svg(data)
    graph = geometry_core.GeometryGraph.from_json(data["graph"])
    dna = dna_mod.compute_geometry_dna(graph)
    return {"params": params.dict(), "svg": svg, "graph": data["graph"], "geometry_dna": dna}


# --------------------------------------------------------------------------
# RESTORE: damaged / missing geometry reconstruction
# --------------------------------------------------------------------------

@app.post("/api/restore")
async def restore_endpoint(file: UploadFile = File(...), mask: Optional[str] = Form(None)):
    """
    `mask` (optional): a data-URL PNG painted by the user on the frontend
    canvas (white = missing region). If omitted, an automatic low-structure
    heuristic detects candidate missing regions.
    """
    data = await file.read()
    img = vision.preprocess(vision.load_image_bytes(data))

    if mask:
        mask_img = _decode_data_url(mask)
        if mask_img.ndim == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]))
        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    else:
        mask_bin = restore_mod.auto_detect_missing_region(img)

    result = restore_mod.reconstruct(img, mask_bin)

    return {
        "original_image": _b64_png(img),
        "mask_image": _b64_png(mask_bin),
        "missing_only_image": _b64_png(result["missing_only_img"]),
        "reconstructed_image": _b64_png(result["reconstructed_img"]),
        "symmetry_used": result["symmetry_used"],
        "method": result["method"],
        "coverage_from_symmetry": result["coverage_from_symmetry"],
        "disclaimer": ("This is a MATHEMATICALLY INFERRED reconstruction derived from "
                       "detected symmetry in the surviving geometry. It is NOT a claim "
                       "of historical accuracy."),
    }


# --------------------------------------------------------------------------
# GEOMETRY DNA compare
# --------------------------------------------------------------------------

class CompareRequest(BaseModel):
    dna_a: dict
    dna_b: dict


@app.post("/api/compare")
def compare(req: CompareRequest):
    return dna_mod.compare_dna(req.dna_a, req.dna_b)


# --------------------------------------------------------------------------
# AI-assisted parameter edit (rule-based instruction -> geometry op)
# --------------------------------------------------------------------------

class EditRequest(BaseModel):
    instruction: str
    current_params: dict
    generator: str = "kolam"   # kolam | pattern


@app.post("/api/edit/ai")
def ai_edit_endpoint(req: EditRequest):
    parsed = ai_edit.parse_instruction(req.instruction, req.current_params)
    updated_params = ai_edit.apply_operation(req.current_params, parsed)

    regenerated = None
    if parsed["op"] != "complete_missing":
        if req.generator == "kolam":
            kp = KolamParams(**{k: updated_params.get(k, getattr(KolamParams(), k))
                                 for k in KolamParams().dict().keys()})
            data = kolam_mod.generate_kolam(kp.rows, kp.cols, kp.symmetry, kp.seed, kp.cell_size)
            svg = kolam_mod.kolam_to_svg(data)
            regenerated = {"svg": svg, "params": kp.dict(), "graph": data["graph"]}
        else:
            rp = RadialParams(**{k: updated_params.get(k, getattr(RadialParams(), k))
                                  for k in RadialParams().dict().keys()})
            data = pattern_mod.generate_radial_pattern(rp.nfold, rp.rings, rp.motif,
                                                        rp.motif_sides, rp.base_radius,
                                                        rp.ring_gap, rp.size)
            svg = pattern_mod.radial_pattern_to_svg(data)
            regenerated = {"svg": svg, "params": rp.dict(), "graph": data["graph"]}

    return {
        "op": parsed["op"],
        "changes": parsed["changes"],
        "explanation": parsed["explanation"],
        "updated_params": updated_params,
        "regenerated": regenerated,
    }


# Serve the frontend as static files (single-page app)
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
