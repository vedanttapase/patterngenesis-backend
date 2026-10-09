"""
PatternGenesis Backend
Endpoints: image analysis, pattern generation, restoration, Geometry DNA
comparison, and deterministic parameter editing.
"""
from __future__ import annotations

import base64
import binascii
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from .engine import vision, geometry_core, dna as dna_mod, kolam as kolam_mod
from .engine import pattern as pattern_mod, restore as restore_mod, ai_edit

MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = FastAPI(title="PatternGenesis Engine", version="0.1.1")

# Keep permissive CORS for the current public frontend. For production, set
# explicit trusted origins using an environment variable/deployment setting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image output.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_data_url(data_url: str) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise HTTPException(status_code=400, detail="Mask must be a valid image data URL.")
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        raise HTTPException(status_code=400, detail="Mask must be a base64-encoded image data URL.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Mask contains invalid base64 data.")
    if not raw:
        raise HTTPException(status_code=400, detail="Mask image is empty.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode mask image.")
    return img


async def _read_image_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MiB upload limit.")
    return data


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "PatternGenesis", "version": app.version}


# ANALYZE: image -> geometry -> mathematics
@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    data = await _read_image_upload(file)
    try:
        result = vision.analyze_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    graph = geometry_core.GeometryGraph.from_json(result["graph"])
    dna = dna_mod.compute_geometry_dna(graph, extra={
        "num_contours": result["num_contours"],
        "num_dots": result["num_dots"],
        "num_corners": result["num_corners"],
    })
    return {
        "image_size": result["image_size"], "contours": result["contours"],
        "dot_grid": result["dot_grid"], "corners": result["corners"],
        "graph": result["graph"], "geometry_dna": dna,
    }


# GENERATE: kolam
class KolamParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: int = Field(default=7, ge=2, le=25)
    cols: int = Field(default=7, ge=2, le=25)
    symmetry: Literal["4fold", "2fold", "none"] = "4fold"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    cell_size: float = Field(default=40.0, gt=0, le=500)


@app.post("/api/generate/kolam")
def generate_kolam(params: KolamParams):
    data = kolam_mod.generate_kolam(
        params.rows, params.cols, params.symmetry, params.seed, params.cell_size
    )
    svg = kolam_mod.kolam_to_svg(data)
    graph = geometry_core.GeometryGraph.from_json(data["graph"])
    dna = dna_mod.compute_geometry_dna(
        graph, extra={"loops": kolam_mod.count_loops(data["graph"])}
    )
    return {
        "params": params.model_dump(), "svg": svg, "graph": data["graph"],
        "dot_grid": data["dot_grid"], "geometry_dna": dna,
        "width": data["width"], "height": data["height"],
    }


# GENERATE: general radial pattern
class RadialParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nfold: int = Field(default=8, ge=2, le=24)
    rings: int = Field(default=3, ge=1, le=8)
    motif: Literal["polygon", "star"] = "polygon"
    motif_sides: int = Field(default=4, ge=3, le=16)
    base_radius: float = Field(default=40.0, gt=0, le=500)
    ring_gap: float = Field(default=35.0, gt=0, le=500)
    size: float = Field(default=420.0, gt=0, le=2000)


@app.post("/api/generate/pattern")
def generate_pattern(params: RadialParams):
    data = pattern_mod.generate_radial_pattern(
        params.nfold, params.rings, params.motif, params.motif_sides,
        params.base_radius, params.ring_gap, params.size,
    )
    svg = pattern_mod.radial_pattern_to_svg(data)
    graph = geometry_core.GeometryGraph.from_json(data["graph"])
    dna = dna_mod.compute_geometry_dna(graph)
    return {"params": params.model_dump(), "svg": svg, "graph": data["graph"],
            "geometry_dna": dna, "size": data["size"]}


class BorderParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_count: int = Field(default=10, ge=1, le=100)
    unit_width: float = Field(default=40.0, gt=0, le=500)
    unit_height: float = Field(default=60.0, gt=0, le=500)
    motif_sides: int = Field(default=6, ge=3, le=16)


@app.post("/api/generate/border")
def generate_border(params: BorderParams):
    data = pattern_mod.generate_linear_border(
        params.unit_count, params.unit_width, params.unit_height, params.motif_sides,
    )
    svg = pattern_mod.linear_border_to_svg(data)
    graph = geometry_core.GeometryGraph.from_json(data["graph"])
    dna = dna_mod.compute_geometry_dna(graph)
    return {"params": params.model_dump(), "svg": svg, "graph": data["graph"],
            "geometry_dna": dna}


# RESTORE: damaged / missing geometry reconstruction
@app.post("/api/restore")
async def restore_endpoint(file: UploadFile = File(...), mask: Optional[str] = Form(None)):
    """Mask is an optional base64 image data URL; white pixels mark missing areas."""
    data = await _read_image_upload(file)
    try:
        img = vision.preprocess(vision.load_image_bytes(data))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if mask:
        mask_img = _decode_data_url(mask)
        if mask_img.ndim == 3:
            if mask_img.shape[2] == 4:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2GRAY)
            else:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.resize(
            mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        _, mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    else:
        mask_bin = restore_mod.auto_detect_missing_region(img)

    result = restore_mod.reconstruct(img, mask_bin)
    return {
        "original_image": _b64_png(img), "mask_image": _b64_png(mask_bin),
        "missing_only_image": _b64_png(result["missing_only_img"]),
        "reconstructed_image": _b64_png(result["reconstructed_img"]),
        "symmetry_used": result["symmetry_used"], "method": result["method"],
        "coverage_from_symmetry": result["coverage_from_symmetry"],
        "disclaimer": (
            "This is a mathematically inferred reconstruction derived from detected "
            "symmetry in surviving geometry. It is not a claim of historical accuracy."
        ),
    }


# GEOMETRY DNA compare
class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dna_a: dict
    dna_b: dict


@app.post("/api/compare")
def compare(req: CompareRequest):
    return dna_mod.compare_dna(req.dna_a, req.dna_b)


# AI-assisted parameter edit
class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=1, max_length=500)
    current_params: dict
    generator: Literal["kolam", "pattern"] = "kolam"


@app.post("/api/edit/ai")
def ai_edit_endpoint(req: EditRequest):
    parsed = ai_edit.parse_instruction(req.instruction, req.current_params)
    updated_params = ai_edit.apply_operation(req.current_params, parsed)
    regenerated = None

    if parsed["op"] != "complete_missing":
        if req.generator == "kolam":
            defaults = KolamParams()
            allowed = defaults.model_fields.keys()
            kp = KolamParams(**{
                k: updated_params.get(k, getattr(defaults, k)) for k in allowed
            })
            data = kolam_mod.generate_kolam(
                kp.rows, kp.cols, kp.symmetry, kp.seed, kp.cell_size
            )
            regenerated = {
                "svg": kolam_mod.kolam_to_svg(data), "params": kp.model_dump(),
                "graph": data["graph"],
            }
        else:
            defaults = RadialParams()
            allowed = defaults.model_fields.keys()
            rp = RadialParams(**{
                k: updated_params.get(k, getattr(defaults, k)) for k in allowed
            })
            data = pattern_mod.generate_radial_pattern(
                rp.nfold, rp.rings, rp.motif, rp.motif_sides,
                rp.base_radius, rp.ring_gap, rp.size,
            )
            regenerated = {
                "svg": pattern_mod.radial_pattern_to_svg(data), "params": rp.model_dump(),
                "graph": data["graph"],
            }

    return {
        "op": parsed["op"], "changes": parsed["changes"],
        "explanation": parsed["explanation"], "updated_params": updated_params,
        "regenerated": regenerated,
    }
