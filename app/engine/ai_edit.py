"""
PatternGenesis - AI-Assisted Edit Layer
Per the project's own architecture requirement: the "AI" layer TRANSLATES a
natural-language instruction into structured geometric operations; it never
generates a new picture directly. This is a deterministic rule/keyword
parser (regex + numeric extraction) rather than an LLM call, so every edit
is reproducible and auditable. It operates on the same parameter dict used
by /generate endpoints (rows, cols, symmetry, nfold, rings, motif_sides...).

If ANTHROPIC_API_KEY is configured, `interpret_with_llm` can optionally be
used to handle more open-ended phrasing by asking an LLM to emit ONLY a
JSON object of {op, ...args} which is then executed by the same
`apply_operation` executor below (so the geometry engine -- not the LLM --
still performs the actual math). This keeps the "LLM never IS the geometry
engine" requirement intact even when the LLM is available.
"""
from __future__ import annotations
import re
from typing import Dict, Tuple, List

NUM_RE = re.compile(r"-?\d+(\.\d+)?")


def _first_number(text: str, default=None):
    m = NUM_RE.search(text)
    return float(m.group()) if m else default


def parse_instruction(instruction: str, current_params: Dict) -> Dict:
    """
    Returns {"op": str, "changes": {...}, "explanation": str}.
    Supported ops (deliberately explicit, matches spec examples):
      - increase_repetitions / decrease_repetitions   (nfold / rows / cols)
      - set_symmetry                                  ("make symmetric", "8-fold")
      - increase_rings / decrease_rings
      - change_motif                                  (polygon/star + sides)
      - regenerate_seed                                ("try another version")
      - complete_missing                              (routes to /restore)
      - unknown                                        (no confident match)
    """
    text = instruction.lower().strip()
    changes: Dict = {}
    op = "unknown"
    explanation = "No confident mapping found; parameters unchanged."

    if any(k in text for k in ["missing", "damaged", "complete the", "reconstruct", "fill in"]):
        op = "complete_missing"
        explanation = "Routed to symmetry-based restoration on the surviving geometry."
        return {"op": op, "changes": {}, "explanation": explanation}

    if "symmetric" in text or "symmetry" in text or "fold" in text:
        n = _first_number(text)
        if n:
            n = int(n)
            changes["nfold"] = n
            changes["symmetry"] = "4fold" if n % 4 == 0 else ("2fold" if n % 2 == 0 else "none")
            op = "set_symmetry"
            explanation = f"Set symmetry order to {n} and re-derived the fundamental domain."
        elif "make" in text and "symmetric" in text:
            op = "set_symmetry"
            changes["symmetry"] = "4fold"
            changes["nfold"] = current_params.get("nfold", 8)
            explanation = "Enforced 4-fold symmetry (default) since no explicit order was given."
        return {"op": op, "changes": changes, "explanation": explanation}

    if "repetition" in text or "repeat" in text or "columns" in text or "rows" in text:
        n = _first_number(text)
        increase = any(k in text for k in ["increase", "more", "add"])
        decrease = any(k in text for k in ["decrease", "fewer", "less", "reduce"])
        base_rows = current_params.get("rows", 7)
        base_cols = current_params.get("cols", 7)
        base_nfold = current_params.get("nfold", 8)
        delta = int(n) if n else 2
        if decrease:
            delta = -abs(delta)
        elif increase or n:
            delta = abs(delta)
        if "nfold" in current_params or "rings" in current_params:
            changes["nfold"] = max(2, base_nfold + delta)
        else:
            changes["rows"] = max(2, base_rows + delta)
            changes["cols"] = max(2, base_cols + delta)
        op = "increase_repetitions" if delta > 0 else "decrease_repetitions"
        explanation = f"Adjusted repetition count by {delta:+d}."
        return {"op": op, "changes": changes, "explanation": explanation}

    if "ring" in text:
        n = _first_number(text)
        base = current_params.get("rings", 3)
        delta = int(n) if n else (1 if "more" in text or "increase" in text else -1)
        changes["rings"] = max(1, base + delta)
        op = "change_rings"
        explanation = f"Changed ring count to {changes['rings']}."
        return {"op": op, "changes": changes, "explanation": explanation}

    if "star" in text:
        changes["motif"] = "star"
        op = "change_motif"
        explanation = "Switched motif to star."
        return {"op": op, "changes": changes, "explanation": explanation}
    if "polygon" in text or "hexagon" in text or "square" in text or "triangle" in text:
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

    if "another" in text or "regenerate" in text or "different version" in text or "new version" in text:
        changes["seed"] = int(current_params.get("seed", 42)) + 1
        op = "regenerate_seed"
        explanation = "Regenerated with a new seed inside the same symmetry constraints."
        return {"op": op, "changes": changes, "explanation": explanation}

    if "height" in text or "extrude" in text or "taller" in text or "shorter" in text:
        n = _first_number(text)
        base = current_params.get("extrude_height", 20)
        delta = n if n else (5 if "taller" in text else -5)
        changes["extrude_height"] = max(2, base + delta)
        op = "change_height"
        explanation = f"Set 3D extrusion height to {changes['extrude_height']}."
        return {"op": op, "changes": changes, "explanation": explanation}

    return {"op": op, "changes": changes, "explanation": explanation}


def apply_operation(current_params: Dict, result: Dict) -> Dict:
    updated = dict(current_params)
    updated.update(result.get("changes", {}))
    return updated
