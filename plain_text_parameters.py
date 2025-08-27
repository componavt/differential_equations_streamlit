# plain_text_parameters.py
# Utility functions to convert parameters <-> plain text
# All comments are in English by request.

from typing import Dict, Any

# Preferred output order for stable copy/paste strings
_ORDER = [
    "t_number",
    "t_end",
    "alpha",
    "K",
    "b",
    "gamma1",
    "gamma2",
    "initial_radius",
    "num_points",
    "circle_start",
    "circle_end",
]


def _fmt_value(v: Any) -> str:
    """Format numbers concisely for plain text output."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    try:
        fv = float(v)
        # Use general format and strip trailing zeros
        s = ("%g" % fv)
        return s
    except Exception:
        return str(v)


# --- Convert dictionary to plain text ---
def parameters_to_text(params: Dict[str, Any]) -> str:
    # Emit keys in the preferred order and then any extras
    parts = []
    used = set()
    for k in _ORDER:
        if k in params:
            parts.append(f"{k}={_fmt_value(params[k])}")
            used.add(k)
    for k, v in params.items():
        if k not in used:
            parts.append(f"{k}={_fmt_value(v)}")
    return "; ".join(parts)


# --- Parse plain text to dictionary ---
def text_to_parameters(text: str) -> Dict[str, Any]:
    """
    Parse strings like "t_number=100; t_end=1.0; alpha=0.001" into a dict.
    Whitespace and newlines are ignored, both ';' and newlines separate pairs.
    Values are auto-cast to int where possible, else float, else raw string.
    """
    result: Dict[str, Any] = {}
    if not isinstance(text, str):
        return result
    # Normalize separators
    raw = text.replace("\n", ";")
    for chunk in raw.split(";"):
        if "=" not in chunk:
            continue
        key, val = chunk.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # Remove a trailing degree sign if someone pasted it (e.g., "90°")
        if val.endswith("°"):
            val = val[:-1]
        # Try to cast to int first (including negatives), then float
        try:
            if val.lower().startswith("0x"):
                # hex int support (rare)
                result[key] = int(val, 16)
            else:
                # int cast (handles +/-, spaces trimmed)
                iv = int(val)
                result[key] = iv
                continue
        except Exception:
            pass
        try:
            fv = float(val)
            result[key] = fv
            continue
        except Exception:
            pass
        # Fallback to raw string
        result[key] = val
    return result
