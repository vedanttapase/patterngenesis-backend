"""
PatternGenesis - deterministic natural-language parameter editing.

Instructions are mapped to explicit operations. Unsupported requests stay
unchanged instead of pretending that an unsupported feature was applied.
"""
from __future__ import annotations

import re
from typing import Dict, Any

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
SET_TO_RE = re.compile(r"\b(?:set|make|change|switch)\b.*?\bto\s+(-?\d+(?:\.\d+)?)")


def _first_number(text: str, default=None):
    match = NUM_RE.search(text)
    return float(match.group()) if match else default


def _target_number(text: str):
    match = SET_TO_RE.search(text)
    return float(match.group(1)) if match else None


def parse_instruction(instruction: str, current_params: Dict) -> Dict:
    """Translate a supported instruction into an auditable parameter change."""
    text = str(instruction or "").lower().strip()
    changes: Dict[str, Any] = {}
    op = "unknown"
    explanation = "No confident mapping found; parameters unchanged."

    if not text:
        return {"op": op, "changes": changes, "explanation": explanation}

    if any(k in text for k in ("missing", "damaged", "complete the", "reconstruct", "fill in")):
        return {
            "op": "complete_missing",
            "changes": {},
            "explanation": "Restoration requires an image and optional mask via /api/restore.",
        }

    if any(k in text for k in ("symmetric", "symmetry", "fold")):
        n = _first_number(text)
        if n is not None:
            order = int(n)
            if order < 2:
                return {"op": op, "changes": {}, "explanation": "Symmetry order must be at least 2; parameters unchanged."}
            if "nfold" in current_params or "rings" in current_params or "motif" in current_params:
                changes["nfold"] = min(order, 24)
                op = "set_symmetry"
                explanation = f"Set rotational repetition to {changes['nfold']}-fold."
            elif "symmetry" in current_params:
                if order in (2, 4):
                    changes["symmetry"] = f"{order}fold"
                    op = "set_symmetry"
                    explanation = f"Set grid symmetry to {order}-fold."
                else:
                    return {
                        "op": "unknown",
                        "changes": {},
                        "explanation": "The grid generator supports only 2-fold, 4-fold, or no symmetry; parameters unchanged.",
                    }
        elif "make" in text and "symmetric" in text:
            if "symmetry" in current_params:
                changes["symmetry"] = "4fold"
            else:
                changes["nfold"] = min(24, max(2, int(current_params.get("nfold", 8))))
            op = "set_symmetry"
            explanation = "Applied the generator's supported symmetric mode."
        return {"op": op, "changes": changes, "explanation": explanation}

    if "ring" in text:
        target = _target_number(text)
        n = _first_number(text)
        base = int(current_params.get("rings", 3))
        if target is not None:
            value = int(target)
        else:
            delta = int(n) if n is not None else 1
            if any(k in text for k in ("decrease", "fewer", "less", "reduce", "remove")):
                delta = -abs(delta)
            elif any(k in text for k in ("increase", "more", "add")):
                delta = abs(delta)
            value = base + delta
        changes["rings"] = max(1, min(8, value))
        op = "change_rings"
        explanation = f"Changed ring count from {base} to {changes['rings']}."
        return {"op": op, "changes": changes, "explanation": explanation}

    if any(k in text for k in ("repetition", "repeat", "columns", "rows")):
        n = _first_number(text)
        increase = any(k in text for k in ("increase", "more", "add"))
        decrease = any(k in text for k in ("decrease", "fewer", "less", "reduce", "remove"))
        base_rows = int(current_params.get("rows", 7))
        base_cols = int(current_params.get("cols", 7))
        base_nfold = int(current_params.get("nfold", 8))
        delta = int(n) if n is not None else 2
        if decrease:
            delta = -abs(delta)
        elif increase or n is not None:
            delta = abs(delta)
        if "nfold" in current_params or "rings" in current_params or "motif" in current_params:
            changes["nfold"] = max(2, min(24, base_nfold + delta))
        else:
            changes["rows"] = max(2, min(25, base_rows + delta))
            changes["cols"] = max(2, min(25, base_cols + delta))
        op = "increase_repetitions" if delta > 0 else "decrease_repetitions"
        explanation = f"Adjusted repetition count by {delta:+d}."
        return {"op": op, "changes": changes, "explanation": explanation}

    if "star" in text:
        changes["motif"] = "star"
        op = "change_motif"
        explanation = "Switched motif to star."
        return {"op": op, "changes": changes, "explanation": explanation}

    if any(k in text for k in ("polygon", "hexagon", "square", "triangle")):
        changes["motif"] = "polygon"
        if "hexagon" in text:
            changes["motif_sides"] = 6
        elif "square" in text:
            changes["motif_sides"] = 4
        elif "triangle" in text:
            changes["motif_sides"] = 3
        op = "change_motif"
        explanation = "Switched motif shape/side-count."
        return {"op": op, "changes": changes, "explanation": explanation}

    if any(k in text for k in ("another", "regenerate", "different version", "new version")):
        if any(k in current_params for k in ("nfold", "rings", "motif")):
            current_sides = int(current_params.get("motif_sides", 6))
            next_sides = 3 if current_sides >= 16 else max(3, current_sides + 1)
            return {"op": "regenerate_seed", "changes": {"motif_sides": next_sides}, "explanation": f"Generated a new deterministic variation using a {next_sides}-sided motif."}
        return {"op": "regenerate_seed", "changes": {"seed": int(current_params.get("seed", 42)) + 1}, "explanation": "Regenerated with the next deterministic seed."}

    if any(k in text for k in ("height", "extrude", "taller", "shorter")):
        # This backend does not yet generate 3D geometry. Do not report a
        # height change as if it affected the rendered 2D output.
        return {
            "op": "unknown",
            "changes": {},
            "explanation": "3D extrusion is not implemented by this backend; parameters unchanged.",
        }

    return {"op": op, "changes": changes, "explanation": explanation}


def apply_operation(current_params: Dict, result: Dict) -> Dict:
    updated = dict(current_params)
    updated.update(result.get("changes", {}))
    return updated
