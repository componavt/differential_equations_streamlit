import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plain_text_parameters import parameters_to_text, text_to_parameters

# --- Sidebar controls ---
st.sidebar.header("Simulation Settings")

# Solver and resolution
# NOTE: store interactive widget values in session_state so we can update them programmatically
if "t_method" not in st.session_state:
    st.session_state.t_method = "DOP853"
if "t_number" not in st.session_state:
    st.session_state.t_number = 100
if "t_end" not in st.session_state:
    st.session_state.t_end = 1.0

# Default parameter values (used to initialize session state)
defaults = {
    "alpha": 0.001,
    "K": 1.0,
    "b": 1.0,
    "gamma1": 1.0,
    "gamma2": 1.0,
    "initial_radius": 0.01,
    "num_points": 12,
    "circle_start": 0.0,
    "circle_end": 360.0,
}

# Initialize any missing session_state entries from defaults
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# t_number and t_end widgets (use session_state keys so Apply can update them)
t_number = st.sidebar.slider("t_number", min_value=10, max_value=1000, step=10, value=st.session_state.t_number, key="t_number")
# t_end choices
t_end_vals = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time t_end", options=t_end_vals, value=st.session_state.t_end, key="t_end")
# compute t_eval from session_state values
# ensure we always use the widget values (session_state updated automatically by Streamlit)
st.session_state.t_number = int(st.session_state.t_number)
st.session_state.t_end = float(st.session_state.t_end)
t_eval = np.linspace(0, st.session_state.t_end, st.session_state.t_number)

# Parameters grouping
col1, col2, col3 = st.sidebar.columns(3)
# Alpha slider
alpha_opts = list(np.round(np.arange(0.1, 1.1, 0.1), 3)) + [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
alpha_opts = sorted(set(alpha_opts), reverse=True)
alpha = col1.select_slider("Alpha", options=alpha_opts, index=alpha_opts.index(st.session_state["alpha"]), key="alpha")
# K
K = col2.slider("K", min_value=0.1, max_value=5.0, step=0.1, value=st.session_state["K"], key="K")
# b
b_values = [0.9, 0.99, 0.999, 0.9999, 1.0, 1.0001, 1.001, 1.01, 1.1]
b_values += list(np.round(np.arange(0, 3.1, 0.1), 3))
# select_slider can't accept index if the value isn't exactly in the list; ensure value present
b_default = st.session_state["b"] if st.session_state["b"] in sorted(set(b_values)) else 1.0
b = col3.select_slider("b", options=sorted(set(b_values)), value=b_default, key="b")

# Gamma sliders with fine/coarse steps
g1, g2 = st.sidebar.columns(2)
gamma_range = [round(v, 3) for v in list(np.arange(0, 1.01, 0.01)) + list(np.arange(1.1, 3.1, 0.1))]
gamma1 = g1.select_slider("Gamma 1", options=gamma_range, value=st.session_state["gamma1"], key="gamma1")
gamma2 = g2.select_slider("Gamma 2", options=gamma_range, value=st.session_state["gamma2"], key="gamma2")

# Initial conditions parameters
rcol, ncol = st.sidebar.columns(2)
initial_radius = rcol.select_slider("Initial radius", options=[0.001] + list(np.round(np.arange(0.01, 0.11, 0.01), 3)) + [0.2, 0.3], value=st.session_state["initial_radius"], key="initial_radius")
num_points = ncol.slider("Number of trajectories", min_value=3, max_value=50, step=1, value=st.session_state["num_points"], key="num_points")

# Sector on circle (degrees) - two-knob slider
circle_start_end = st.sidebar.slider("Sector on circle (degrees)", 0, 360, (int(st.session_state["circle_start"]), int(st.session_state["circle_end"])), step=1, key="circle_start_end")
# Unpack
circle_start, circle_end = circle_start_end
# Keep session state in sync
st.session_state["circle_start"] = float(circle_start)
st.session_state["circle_end"] = float(circle_end)

# Collect current parameters into dictionary (from session_state to guarantee consistency)
params = {
    "t_number": int(st.session_state["t_number"]),
    "t_end": float(st.session_state["t_end"]),
    "alpha": float(st.session_state["alpha"]),
    "K": float(st.session_state["K"]),
    "b": float(st.session_state["b"]),
    "gamma1": float(st.session_state["gamma1"]),
    "gamma2": float(st.session_state["gamma2"]),
    "initial_radius": float(st.session_state["initial_radius"]),
    "num_points": int(st.session_state["num_points"]),
    "circle_start": float(st.session_state["circle_start"]),
    "circle_end": float(st.session_state["circle_end"]),
}

# Convert to plain text using helper
curr_text = parameters_to_text(params)

# Use session state to keep track whether the text area was last synced from sliders
if "param_text" not in st.session_state:
    st.session_state.param_text = curr_text
if "param_text_synced" not in st.session_state:
    st.session_state.param_text_synced = curr_text

# If the user hasn't edited the text area (i.e. param_text equals last synced value), auto-update it
# This implements requirement (5): when sliders change, text area auto-updates unless the user is editing
if st.session_state.param_text == st.session_state.param_text_synced:
    st.session_state.param_text = curr_text
    st.session_state.param_text_synced = curr_text

# Text area to show and edit parameters
param_text_input = st.sidebar.text_area("Parameters (plain text)", value=st.session_state.param_text, height=150, key="param_text_widget")
# keep a copy in session state for detection of edits
st.session_state.param_text = param_text_input

# Button to parse text and update sliders/session_state
if st.sidebar.button("Apply parameters from text"):
    parsed_params = text_to_parameters(param_text_input)
    if parsed_params:
        # Update all recognized parameters in session_state so widgets reflect new values on rerun
        for key in ["t_number", "t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius", "num_points", "circle_start", "circle_end"]:
            if key in parsed_params:
                try:
                    # cast to appropriate type
                    if key in ["t_number", "num_points"]:
                        st.session_state[key] = int(parsed_params[key])
                    else:
                        st.session_state[key] = float(parsed_params[key])
                except Exception:
                    # ignore bad casts and keep previous value
                    pass
        # After updating session_state values, recompute text synced value so text_area won't be immediately overwritten
        params_after = {k: st.session_state[k] for k in params}
        new_text = parameters_to_text(params_after)
        st.session_state.param_text = new_text
        st.session_state.param_text_synced = new_text
        # also recompute t_eval
        st.session_state.t_number = int(st.session_state.t_number)
        st.session_state.t_end = float(st.session_state.t_end)
        t_eval = np.linspace(0, st.session_state.t_end, st.session_state.t_number)
        # Rerun will occur automatically and widgets will show updated values

# --- Robust RHS with overflow handling ---
# Note: these functions read parameter values from st.session_state so they always use the latest

def get_rhs_safe(t, state):
    x, y = state
    alpha = st.session_state["alpha"]
    K = st.session_state["K"]
    b = st.session_state["b"]
    gamma1 = st.session_state["gamma1"]
    gamma2 = st.session_state["gamma2"]
    n = 1 / alpha
    # approximate step-function behavior for very large exponents
    if n > 1000:
        frac_x = K if x > b else 0.0
        frac_y = K if y > b else 0.0
    else:
        x_pos = max(x, 0.0)
        y_pos = max(y, 0.0)
        try:
            pow_x = x_pos**n
            pow_b = b**n
            frac_x = (K * pow_x) / (pow_b + pow_x)
        except (OverflowError, FloatingPointError):
            frac_x = K if x > b else 0.0
        try:
            pow_y = y_pos**n
            frac_y = (K * pow_y) / (pow_b + pow_y)
        except (OverflowError, FloatingPointError):
            frac_y = K if y > b else 0.0
    dxdt = frac_x - gamma1 * x
    dydt = frac_y - gamma2 * y
    return [dxdt, dydt]

# --- Compute initial points on circle (sector) centered at (b,b) ---
# Convert sector to radians and handle wrap-around if circle_end < circle_start
circle_start = float(st.session_state["circle_start"])
circle_end = float(st.session_state["circle_end"])
num_points = int(st.session_state["num_points"])
initial_radius = float(st.session_state["initial_radius"])
b = float(st.session_state["b"])

# Build angle array over the requested sector. If the sector covers the full circle, distribute evenly.
if (circle_end - circle_start) % 360 == 0:
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
else:
    # normalize to [0,360)
    cs = circle_start % 360
    ce = circle_end % 360
    if ce >= cs:
        degs = np.linspace(cs, ce, num_points, endpoint=False)
    else:
        # wrap-around e.g., start=300, end=30 -> go from 300..360 and 0..30
        span = (ce + 360) - cs
        degs = (cs + np.linspace(0, span, num_points, endpoint=False)) % 360
    angles = np.deg2rad(degs)

initial_conditions = [(b + initial_radius * np.cos(a), b + initial_radius * np.sin(a)) for a in angles]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
title = f"Safe DOP853, t_end={st.session_state['t_end']}, pts={st.session_state['t_number']}"
ax.set_title(title)
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.grid(True)

styles = ['-', '--', '-.', ':']
colors = plt.cm.tab20.colors

for idx, (x0, y0) in enumerate(initial_conditions):
    try:
        sol = solve_ivp(
            fun=get_rhs_safe,
            t_span=(0, st.session_state['t_end']),
            y0=[x0, y0],
            method=st.session_state['t_method'],
            t_eval=np.linspace(0, st.session_state['t_end'], st.session_state['t_number'])
        )
        x, y = sol.y
    except Exception as e:
        # In case solver fails, skip this trajectory but keep the app running
        st.warning(f"Solver error for trajectory {idx}: {e}")
        continue
    style = styles[idx % len(styles)]
    color = colors[idx % len(colors)]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.5)
    ax.plot(x[0], y[0], 'o', color=color, markersize=4)
    ax.plot(x[-1], y[-1], 'x', color=color, markersize=5)
    step = max(1, len(x)//10)
    for j in range(step, len(x), step):
        ax.annotate('', xy=(x[j], y[j]), xytext=(x[j-step], y[j-step]), arrowprops=dict(arrowstyle='->', color=color, lw=1))
    ax.text(x[-1], y[-1], f"({x0:.3f},{y0:.3f})", fontsize=8)

st.pyplot(fig)

# --- Parameters report ---
st.markdown("**Parameters used in this run:**")
st.text(parameters_to_text(params))

# --- Explanations below ---
st.markdown("---")
st.markdown("**System of ODEs (safe):**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")
st.markdown("- Robust step-function approx when exponent is huge to avoid overflow." )

# Footer title
st.markdown("---")
st.markdown("*safe_calculation: stabilized gene ODE explorer*")
