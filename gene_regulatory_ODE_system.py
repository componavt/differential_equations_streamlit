import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plain_text_parameters import parameters_to_text, text_to_parameters

# --------------------------------------------------
# Simple UI: sliders + a plain-text box + two buttons
# - No automatic synchronization.
# - Two buttons control copying in both directions:
#     1) Apply text -> sliders (parse the text and set sliders)
#     2) Read sliders -> text (read current slider values and fill the text box)
# Implementation note: Streamlit requires session state to programmatically
# change widget values; we only use st.session_state inside button callbacks
# (no continuous syncing, no auto-overwrites).
# --------------------------------------------------

# Default parameter values
DEFAULTS = {
    "alpha": 0.001,
    "K": 1.0,
    "b": 1.0,
    "gamma1": 1.0,
    "gamma2": 1.0,
    "initial_radius": 0.01,
    "num_points": 12,
    "t_number": 100,
    "t_end": 1.0,
    # store pair for the sector slider under this key
    "circle_start_end": (0, 360),
}

# Initialize missing session-state keys with defaults (safe to do before widgets are created)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Sidebar: parameter sliders ---
st.sidebar.header("Simulation Settings")

# Time / solver resolution
t_number = st.sidebar.slider("t_number", min_value=10, max_value=1000, step=10, value=int(st.session_state["t_number"]), key="t_number")
# end time options (keeps previous style)
t_end_vals = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time t_end", options=t_end_vals, value=float(st.session_state["t_end"]), key="t_end")

# Parameters
alpha = st.sidebar.number_input("alpha (1/\u03B1 exponent)", min_value=1e-9, max_value=10.0, value=float(st.session_state["alpha"]), format="%g", key="alpha")
K = st.sidebar.slider("K", min_value=0.1, max_value=5.0, step=0.1, value=float(st.session_state["K"]), key="K")
b = st.sidebar.number_input("b", min_value=0.0, max_value=10.0, value=float(st.session_state["b"]), format="%g", key="b")

# Gammas
gamma1 = st.sidebar.number_input("gamma1", min_value=0.0, max_value=10.0, value=float(st.session_state["gamma1"]), format="%g", key="gamma1")
gamma2 = st.sidebar.number_input("gamma2", min_value=0.0, max_value=10.0, value=float(st.session_state["gamma2"]), format="%g", key="gamma2")

# Initial conditions parameters
initial_radius = st.sidebar.number_input("Initial radius (R)", min_value=0.0, max_value=10.0, value=float(st.session_state["initial_radius"]), format="%g", key="initial_radius")
num_points = st.sidebar.slider("Number of trajectories", min_value=3, max_value=50, step=1, value=int(st.session_state["num_points"]), key="num_points")

# Sector on circle (degrees) - two-knob slider
circle_start_end = st.sidebar.slider("Sector on circle (degrees)", 0, 360, tuple(map(int, st.session_state["circle_start_end"])), step=1, key="circle_start_end")

# --- Plain text area and the two buttons ---
# Prepare a textual representation of the current slider values (used for the initial value)
def collect_params_from_widgets():
    # Read current widget values from st.session_state (these keys are the widget keys)
    cs, ce = st.session_state.get("circle_start_end", (0, 360))
    params = {
        "t_number": int(st.session_state.get("t_number", DEFAULTS["t_number"])),
        "t_end": float(st.session_state.get("t_end", DEFAULTS["t_end"])),
        "alpha": float(st.session_state.get("alpha", DEFAULTS["alpha"])),
        "K": float(st.session_state.get("K", DEFAULTS["K"])),
        "b": float(st.session_state.get("b", DEFAULTS["b"])),
        "gamma1": float(st.session_state.get("gamma1", DEFAULTS["gamma1"])),
        "gamma2": float(st.session_state.get("gamma2", DEFAULTS["gamma2"])),
        "initial_radius": float(st.session_state.get("initial_radius", DEFAULTS["initial_radius"])),
        "num_points": int(st.session_state.get("num_points", DEFAULTS["num_points"])),
        "circle_start": int(cs),
        "circle_end": int(ce),
    }
    return params

# Initialize the text area with the current widget values if not present
if "params_text" not in st.session_state:
    st.session_state["params_text"] = parameters_to_text(collect_params_from_widgets())

st.sidebar.markdown("**Parameters (plain text)**")
params_text = st.sidebar.text_area("Edit parameters here:", value=st.session_state["params_text"], height=180, key="params_text")

# Callback: parse text and apply to widgets (text -> sliders)
def apply_text_to_sliders():
    txt = st.session_state.get("params_text", "")
    parsed = text_to_parameters(txt)
    if not parsed:
        # nothing parsed; do nothing
        return
    # For each recognized parameter, set the corresponding widget session_state key
    mapping_int = {"t_number", "num_points", "circle_start", "circle_end"}
    mapping_float = {"t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius"}

    # Apply integer keys
    for key in mapping_int:
        if key in parsed:
            try:
                if key in {"circle_start", "circle_end"}:
                    # We'll set the pair at the end
                    continue
                st.session_state[key] = int(parsed[key])
            except Exception:
                pass
    # Apply float keys
    for key in mapping_float:
        if key in parsed:
            try:
                st.session_state[key] = float(parsed[key])
            except Exception:
                pass
    # Handle circle pair specially: preserve unspecified part
    cs_old, ce_old = st.session_state.get("circle_start_end", (0, 360))
    cs = int(parsed.get("circle_start", cs_old))
    ce = int(parsed.get("circle_end", ce_old))
    st.session_state["circle_start_end"] = (cs, ce)
    # Update the stored text to a normalized representation (so the text area shows canonical form)
    st.session_state["params_text"] = parameters_to_text(collect_params_from_widgets())

# Callback: read widget values and put them into the text area (sliders -> text)
def read_sliders_to_text():
    params = collect_params_from_widgets()
    st.session_state["params_text"] = parameters_to_text(params)

# Buttons (note: callbacks update st.session_state before rerun)
col_apply, col_read = st.sidebar.columns(2)
col_apply.button("Apply text → sliders", on_click=apply_text_to_sliders)
col_read.button("Read sliders → text", on_click=read_sliders_to_text)

# --- Main application: compute and plot using current widget values ---
# Read the current (possibly updated) values from widget state
alpha_val = float(st.session_state["alpha"])
K_val = float(st.session_state["K"])
b_val = float(st.session_state["b"])
g1_val = float(st.session_state["gamma1"])
g2_val = float(st.session_state["gamma2"])
R_val = float(st.session_state["initial_radius"])
num_pts = int(st.session_state["num_points"])
tn = int(st.session_state["t_number"])
te = float(st.session_state["t_end"])
cs_val, ce_val = st.session_state.get("circle_start_end", (0, 360))

# Robust RHS with overflow handling
def rhs(t, state):
    x, y = state
    n = 1.0 / alpha_val
    # approximate step-function behavior for very large exponents
    if n > 1000:
        frac_x = K_val if x > b_val else 0.0
        frac_y = K_val if y > b_val else 0.0
    else:
        x_pos = max(x, 0.0)
        y_pos = max(y, 0.0)
        # use numpy power with checks to avoid runtime warnings producing NaNs
        try:
            pow_b = np.power(b_val, n)
            pow_x = np.power(x_pos, n)
            pow_y = np.power(y_pos, n)
            frac_x = (K_val * pow_x) / (pow_b + pow_x) if np.isfinite(pow_x) else (K_val if x > b_val else 0.0)
            frac_y = (K_val * pow_y) / (pow_b + pow_y) if np.isfinite(pow_y) else (K_val if y > b_val else 0.0)
        except Exception:
            frac_x = K_val if x > b_val else 0.0
            frac_y = K_val if y > b_val else 0.0
    dxdt = frac_x - g1_val * x
    dydt = frac_y - g2_val * y
    return [dxdt, dydt]

# Build angle array for sector, support wrap-around
span = (ce_val - cs_val) % 360
if np.isclose(span, 0.0):
    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
else:
    cs = cs_val % 360
    ce = ce_val % 360
    if ce >= cs:
        degs = np.linspace(cs, ce, num_pts, endpoint=False)
    else:
        span2 = (ce + 360) - cs
        degs = (cs + np.linspace(0, span2, num_pts, endpoint=False)) % 360
    angles = np.deg2rad(degs)

initial_conditions = [(b_val + R_val * np.cos(a), b_val + R_val * np.sin(a)) for a in angles]

# Integrate and plot
t_eval = np.linspace(0, te, tn)
fig, ax = plt.subplots(figsize=(8, 6))
styles = ['-', '--', '-.', ':']
colors = plt.cm.tab20.colors
for idx, (x0, y0) in enumerate(initial_conditions):
    try:
        sol = solve_ivp(rhs, (0, te), (x0, y0), method='DOP853', t_eval=t_eval)
        x, y = sol.y
    except Exception as exc:
        st.warning(f"Solver error for trajectory {idx}: {exc}")
        continue
    style = styles[idx % len(styles)]
    color = colors[idx % len(colors)]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.2)
    ax.plot(x[0], y[0], 'o', color=color, markersize=4)
    ax.plot(x[-1], y[-1], 'x', color=color, markersize=5)

ax.set_title(f"Gene regulatory trajectories — t_end={te}, pts={tn}")
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.grid(True)
st.pyplot(fig)

# Show textual report of current parameters
st.markdown("**Parameters currently used:**")
st.text(parameters_to_text(collect_params_from_widgets()))

# Footer
st.markdown("---")
st.markdown("**System of ODEs (safe):**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")
st.markdown("- Robust handling of overflow: np.isfinite checks prevent NaNs/Infs.")
