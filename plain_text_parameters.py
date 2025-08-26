# plain_text_parameters.py
# Utility functions to convert parameters <-> plain text

# --- Convert dictionary to plain text ---
def parameters_to_text(params: dict) -> str:
    # Join all key=value pairs separated by "; "
    return "; ".join([f"{k}={v}" for k, v in params.items()])

# --- Parse plain text to dictionary ---
def text_to_parameters(text: str) -> dict:
    result = {}
    try:
        pairs = text.replace("\n", ";").split(";")
        for p in pairs:
            if "=" in p:
                key, val = p.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key:
                    # Try to cast to int or float
                    if val.isdigit():
                        result[key] = int(val)
                    else:
                        try:
                            result[key] = float(val)
                        except ValueError:
                            result[key] = val
    except Exception:
        return {}
    return result
