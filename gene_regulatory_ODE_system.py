# gene_regulatory_ODE_system6.py (updated with fixes)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from plain_text_parameters import parameters_to_text, text_to_parameters
from scipy.integrate import solve_ivp

# --- Default values ---
default_params = {
    "alpha": 0.5,
    "K": 1.0,
    "b": 2.0,
    "x0": 0.5,
    "y0": 0.5,
    "t_number": 200,
    "t_end": 10.0,
    "circle_start": 0.0,
    "circle_end": 360.0,
    "gamma1": 1.0,
    "gamma2": 1.0,
}

# --- Session state init ---
for key, value in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Text area with plain text parameters ---
if "param_text_widget" not in st.session_state:
    st.session_state.param_text_widget = parameters_to_text(st.session_state)

st.session_state.param_text_widget = st.text_area(
    "Parameters (editable):",
    value=st.session_state.param_text_widget,
    height=200,
    key="param_text_widget",
)

# --- Apply from text callback ---
def _apply_from_text():
    parsed = text_to_parameters(st.session_state.param_text_widget, default_params)
    for key, val in parsed.items():
        st.session_state[key] = val

st.sidebar.button("Apply parameters from text", on_click=_apply_from_text)

# --- Sliders ---
st.session_state["alpha"] = st.sidebar.slider("alpha", 0.01, 2.0, float(st.session_state["alpha"]), 0.01)
st.session_state["K"] = st.sidebar.slider("K", 0.1, 5.0, float(st.session_state["K"]), 0.1)
st.session_state["b"] = st.sidebar.slider("b", 0.1, 5.0, float(st.session_state["b"]), 0.1)
st.session_state["x0"] = st.sidebar.slider("x0", 0.0, 2.0, float(st.session_state["x0"]), 0.01)
st.session_state["y0"] = st.sidebar.slider("y0", 0.0, 2.0, float(st.session_state["y0"]), 0.01)
st.session_state["t_number"] = st.sidebar.slider("t_number", 50, 1000, int(st.session_state["t_number"]), 10)
st.session_state["t_end"] = st.sidebar.slider("t_end", 1.0, 50.0, float(st.session_state["t_end"]), 0.5)
st.session_state["gamma1"] = st.sidebar.slider("gamma1", 0.1, 5.0, float(st.session_state["gamma1"]), 0.1)
st.session_state["gamma2"] = st.sidebar.slider("gamma2", 0.1, 5.0, float(st.session_state["gamma2"]), 0.1)

circle_start, circle_end = st.sidebar.slider(
    "Initial condition sector (degrees)",
    0, 360,
    (int(st.session_state["circle_start"]), int(st.session_state["circle_end"])),
    1,
)
st.session_state["circle_start"] = circle_start
st.session_state["circle_end"] = circle_end

# --- Keep text synced if not edited manually ---
current_text = parameters_to_text(st.session_state)
if current_text != st.session_state.param_text_widget:
    st.session_state.param_text_widget = current_text

# --- ODE system ---
def gene_regulatory_2d(t, z, alpha, K, b, gamma1, gamma2):
    x, y = z
    n = 1.0 / alpha
    x_pos, y_pos = max(x, 0), max(y, 0)
    try:
        pow_b = np.power(b, n)
        pow_x = np.power(x_pos, n, dtype=np.float64)
        pow_y = np.power(y_pos, n, dtype=np.float64)
        frac_x = (K * pow_x) / (pow_b + pow_x) if np.isfinite(pow_x) else (K if x_pos > b else 0)
        frac_y = (K * pow_y) / (pow_b + pow_y) if np.isfinite(pow_y) else (K if y_pos > b else 0)
    except OverflowError:
        frac_x = K if x_pos > b else 0
        frac_y = K if y_pos > b else 0
    dxdt = frac_x - gamma1 * x
    dydt = frac_y - gamma2 * y
    return [dxdt, dydt]

# --- Simulation setup ---
alpha = st.session_state["alpha"]
K = st.session_state["K"]
b = st.session_state["b"]
gamma1 = st.session_state["gamma1"]
gamma2 = st.session_state["gamma2"]
t_number_local = int(st.session_state["t_number"])
t_end_local = float(st.session_state["t_end"])

# Time vector
t_eval = np.linspace(0, t_end_local, t_number_local)

# Initial conditions in sector
start_deg, end_deg = st.session_state["circle_start"], st.session_state["circle_end"]
if end_deg < start_deg:
    degs = np.linspace(start_deg, end_deg + 360, 50)
else:
    degs = np.linspace(start_deg, end_deg, 50)
angles = np.deg2rad(degs)

r0 = st.session_state["x0"]
ics = [(np.cos(a) * r0, np.sin(a) * r0) for a in angles]

# --- Plot trajectories ---
fig, ax = plt.subplots()
for x0, y0 in ics:
    try:
        sol = solve_ivp(
            gene_regulatory_2d, [0, t_end_local], [x0, y0],
            args=(alpha, K, b, gamma1, gamma2), t_eval=t_eval
        )
        ax.plot(sol.y[0], sol.y[1], lw=1)
    except Exception as e:
        st.warning(f"Solver failed for initial condition ({x0:.2f}, {y0:.2f}): {e}")

ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.set_title("Gene regulatory system trajectories in sector")
st.pyplot(fig)

# --- Explanations below ---
st.markdown("---")
st.markdown("**System of ODEs (safe):**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")
st.markdown("- Robust handling of overflow: np.isfinite checks prevent NaNs/Infs.")
