# plain_text_parameters.py
# Utility functions to convert parameters <-> plain text

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
    "center_x",
    "center_y",
    "enabled_idx",
]


def _fmt_value(v: Any) -> str:
    """Format numbers concisely for plain text output."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    try:
        fv = float(v)
        s = ("%g" % fv)
        return s
    except Exception:
        return str(v)


# --- Convert dictionary to plain text ---
def parameters_to_text(params: Dict[str, Any]) -> str:
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
    raw = text.replace("\n", ";")
    for chunk in raw.split(";"):
        if "=" not in chunk:
            continue
        key, val = chunk.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if val.endswith("°"):
            val = val[:-1]
        try:
            if val.lower().startswith("0x"):
                result[key] = int(val, 16)
            else:
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
        result[key] = val
    return result
