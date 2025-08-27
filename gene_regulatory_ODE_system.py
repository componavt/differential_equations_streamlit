import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plain_text_parameters import parameters_to_text, text_to_parameters

# --- Sidebar controls ---
st.sidebar.header("Simulation Settings")

# Initialize default session state values if missing
defaults = {
    "t_method": "DOP853",
    "t_number": 100,
    "t_end": 1.0,
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
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Text area for plain-text parameters (user can edit this)
params_initial = {k: st.session_state[k] for k in ["t_number", "t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius", "num_points", "circle_start", "circle_end"]}
curr_text = parameters_to_text(params_initial)
if "param_text" not in st.session_state:
    st.session_state.param_text = curr_text
if "param_text_synced" not in st.session_state:
    st.session_state.param_text_synced = curr_text

# --- Widgets (sliders/selectors) ---
# t_number and t_end
# Do NOT assign to st.session_state[...] for keys that are widget keys after widget creation.
# Instead, read their values and use local variables for computations.
t_number = st.sidebar.slider("t_number", min_value=10, max_value=1000, step=10, value=int(st.session_state["t_number"]), key="t_number")

t_end_vals = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time t_end", options=t_end_vals, value=float(st.session_state["t_end"]), key="t_end")

# Parameters grouping
col1, col2, col3 = st.sidebar.columns(3)
alpha_opts = list(np.round(np.arange(0.1, 1.1, 0.1), 3)) + [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
alpha_opts = sorted(set(alpha_opts), reverse=True)
alpha_default = st.session_state["alpha"] if st.session_state["alpha"] in alpha_opts else min(alpha_opts, key=lambda x: abs(x - st.session_state["alpha"]))
alpha = col1.select_slider("Alpha", options=alpha_opts, value=alpha_default, key="alpha")

K = col2.slider("K", min_value=0.1, max_value=5.0, step=0.1, value=float(st.session_state["K"]), key="K")

b_values = [0.9, 0.99, 0.999, 0.9999, 1.0, 1.0001, 1.001, 1.01, 1.1]
b_values += list(np.round(np.arange(0, 3.1, 0.1), 3))
_b_options = sorted(set(b_values))
b_default = st.session_state["b"] if st.session_state["b"] in _b_options else min(_b_options, key=lambda x: abs(x - st.session_state["b"]))
b = col3.select_slider("b", options=_b_options, value=b_default, key="b")

# Gamma sliders
g1, g2 = st.sidebar.columns(2)
gamma_range = [round(v, 3) for v in list(np.arange(0, 1.01, 0.01)) + list(np.arange(1.1, 3.1, 0.1))]
gamma1 = g1.select_slider("Gamma 1", options=gamma_range, value=float(st.session_state["gamma1"]), key="gamma1")
gamma2 = g2.select_slider("Gamma 2", options=gamma_range, value=float(st.session_state["gamma2"]), key="gamma2")

# Initial conditions parameters
rcol, ncol = st.sidebar.columns(2)
initial_radius = rcol.select_slider("Initial radius", options=[0.001] + list(np.round(np.arange(0.01, 0.11, 0.01), 3)) + [0.2, 0.3], value=float(st.session_state["initial_radius"]), key="initial_radius")
num_points = ncol.slider("Number of trajectories", min_value=3, max_value=50, step=1, value=int(st.session_state["num_points"]), key="num_points")

# Sector on circle (degrees)
circle_start_end = st.sidebar.slider("Sector on circle (degrees)", 0, 360, (int(st.session_state["circle_start"]), int(st.session_state["circle_end"])), step=1, key="circle_start_end")
circle_start, circle_end = circle_start_end
st.session_state["circle_start"] = float(circle_start)
st.session_state["circle_end"] = float(circle_end)

# --- Plain text area ---
if st.session_state.get("param_text") == st.session_state.get("param_text_synced"):
    params_now = {
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
    st.session_state.param_text = parameters_to_text(params_now)
    st.session_state.param_text_synced = st.session_state.param_text

param_text_input = st.sidebar.text_area("Parameters (plain text)", value=st.session_state.param_text, height=150, key="param_text_widget")

# --- Apply callback ---
def _apply_from_text():
    txt = st.session_state.get("param_text_widget", "")
    parsed = text_to_parameters(txt)
    if not parsed:
        return
    int_keys = {"t_number", "num_points"}
    float_keys = {"t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius", "circle_start", "circle_end"}
    for key in int_keys:
        if key in parsed:
            try:
                st.session_state[key] = int(parsed[key])
            except Exception:
                pass
    for key in float_keys:
        if key in parsed:
            try:
                st.session_state[key] = float(parsed[key])
            except Exception:
                pass
    params_after = {k: st.session_state[k] for k in ["t_number", "t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius", "num_points", "circle_start", "circle_end"]}
    new_text = parameters_to_text(params_after)
    st.session_state["param_text"] = new_text
    st.session_state["param_text_synced"] = new_text

st.sidebar.button("Apply parameters from text", on_click=_apply_from_text)

# --- RHS ---
def get_rhs_safe(t, state):
    x, y = state
    alpha = float(st.session_state["alpha"])
    K = float(st.session_state["K"])
    b = float(st.session_state["b"])
    gamma1 = float(st.session_state["gamma1"])
    gamma2 = float(st.session_state["gamma2"])
    n = 1.0 / alpha
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
t_number_local = int(st.session_state["t_number"])
t_end_local = float(st.session_state["t_end"])
num_points_local = int(st.session_state["num_points"])
initial_radius_local = float(st.session_state["initial_radius"])
b_local = float(st.session_state["b"])
circle_start_local = float(st.session_state["circle_start"])
circle_end_local = float(st.session_state["circle_end"])

span_deg = (circle_end_local - circle_start_local) % 360.0
if np.isclose(span_deg, 0.0):
    angles = np.linspace(0, 2 * np.pi, num_points_local, endpoint=False)
else:
    cs = circle_start_local % 360.0
    ce = circle_end_local % 360.0
    if ce >= cs:
        degs = np.linspace(cs, ce, num_points_local, endpoint=False)
    else:
        span = (ce + 360.0) - cs
        degs = (cs + np.linspace(0.0, span, num_points_local, endpoint=False)) % 360.0
    angles = np.deg2rad(degs)

initial_conditions = [(b_local + initial_radius_local * np.cos(a), b_local + initial_radius_local * np.sin(a)) for a in angles]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
title = f"Safe DOP853, t_end={t_end_local}, pts={t_number_local}"
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
            t_span=(0, t_end_local),
            y0=[x0, y0],
            method=st.session_state.get('t_method', 'DOP853'),
            t_eval=np.linspace(0, t_end_local, t_number_local)
        )
        x, y = sol.y
    except Exception as e:
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
params_report = {
    "t_number": t_number_local,
    "t_end": t_end_local,
    "alpha": float(st.session_state["alpha"]),
    "K": float(st.session_state["K"]),
    "b": float(st.session_state["b"]),
    "gamma1": float(st.session_state["gamma1"]),
    "gamma2": float(st.session_state["gamma2"]),
    "initial_radius": initial_radius_local,
    "num_points": num_points_local,
    "circle_start": circle_start_local,
    "circle_end": circle_end_local,
}
st.text(parameters_to_text(params_report))

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

# Footer
st.markdown("---")
st.markdown("*safe_calculation: stabilized gene ODE explorer*")
