import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Try to import the plain-text helpers from either module name (flexible for canvas filenames)
try:
    from plain_text_parameters7 import parameters_to_text, text_to_parameters
except Exception:
    from plain_text_parameters import parameters_to_text, text_to_parameters

# --------------------------------------------------
# gene_regulatory_ODE_system
# - Adds optional per-trajectory metrics (FTLE estimate, amplitude) and
#   a UI to select which trajectories to display.
# - Two buttons for copying parameters: text -> widgets and widgets -> text.
# --------------------------------------------------

# Default parameter values (used as widget defaults and fallbacks)
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
    # store pair for the sector slider
    "circle_start_end": (0, 360),
}

# --- Sidebar: widgets ---
st.sidebar.header("Simulation Settings")

# Time / solver resolution
# Provide explicit defaults from DEFAULTS, but do NOT pre-assign st.session_state keys.
t_number = st.sidebar.slider("t_number", min_value=10, max_value=1000, step=10,
                             value=int(DEFAULTS["t_number"]), key="t_number")

# end time options (keeps previous style)
t_end_vals = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time t_end", options=t_end_vals,
                                  value=float(DEFAULTS["t_end"]), key="t_end")

# Parameters
alpha = st.sidebar.number_input("alpha (1/α exponent)", min_value=1e-9, max_value=10.0,
                                value=float(DEFAULTS["alpha"]), format="%g", key="alpha")
K = st.sidebar.slider("K", min_value=0.1, max_value=5.0, step=0.1,
                       value=float(DEFAULTS["K"]), key="K")
b = st.sidebar.number_input("b", min_value=0.0, max_value=10.0,
                             value=float(DEFAULTS["b"]), format="%g", key="b")

# Gammas
gamma1 = st.sidebar.number_input("gamma1", min_value=0.0, max_value=10.0,
                                 value=float(DEFAULTS["gamma1"]), format="%g", key="gamma1")
gamma2 = st.sidebar.number_input("gamma2", min_value=0.0, max_value=10.0,
                                 value=float(DEFAULTS["gamma2"]), format="%g", key="gamma2")

# Initial conditions parameters
initial_radius = st.sidebar.number_input("Initial radius (R)", min_value=0.0, max_value=10.0,
                                         value=float(DEFAULTS["initial_radius"]), format="%g", key="initial_radius")
num_points = st.sidebar.slider("Number of trajectories", min_value=3, max_value=50, step=1,
                               value=int(DEFAULTS["num_points"]), key="num_points")

# Sector on circle (degrees) - two-knob slider
circle_start_end = st.sidebar.slider("Sector on circle (degrees)", 0, 360,
                                    tuple(map(int, DEFAULTS["circle_start_end"])), step=1,
                                    key="circle_start_end")

# --- Plain text area and the two buttons ---
# collect current widget values (read from session_state if present, else DEFAULTS)
def collect_params_from_widgets():
    cs, ce = st.session_state.get("circle_start_end", DEFAULTS["circle_start_end"])
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

# Initialize the text area value in session_state (safe: this key is not owned by a widget before we create it)
if "params_text" not in st.session_state:
    st.session_state["params_text"] = parameters_to_text(collect_params_from_widgets())

st.sidebar.markdown("**Parameters (plain text)**")
params_text = st.sidebar.text_area("Edit parameters here:", value=st.session_state["params_text"], height=180, key="params_text")

# Callback: parse text and apply to widgets (text -> sliders)
def apply_text_to_sliders():
    txt = st.session_state.get("params_text", "")
    parsed = text_to_parameters(txt)
    if not parsed:
        return
    mapping_int = {"t_number", "num_points", "circle_start", "circle_end"}
    mapping_float = {"t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius"}

    # Apply integer keys (except circle endpoints handled below)
    for key in mapping_int:
        if key in parsed and key not in {"circle_start", "circle_end"}:
            try:
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
    # Handle circle pair specially
    cs_old, ce_old = st.session_state.get("circle_start_end", DEFAULTS["circle_start_end"])
    cs = int(parsed.get("circle_start", cs_old))
    ce = int(parsed.get("circle_end", ce_old))
    st.session_state["circle_start_end"] = (cs, ce)
    # Update canonical text representation
    st.session_state["params_text"] = parameters_to_text(collect_params_from_widgets())

# Callback: read widget values and put them into the text area (sliders -> text)
def read_sliders_to_text():
    st.session_state["params_text"] = parameters_to_text(collect_params_from_widgets())

# Buttons (callbacks update session_state before rerun)
col_apply, col_read = st.sidebar.columns(2)
col_apply.button("Apply text → sliders", on_click=apply_text_to_sliders)
col_read.button("Read sliders → text", on_click=read_sliders_to_text)

# --- Option: compute per-trajectory metrics (may be slow) ---
compute_metrics = st.sidebar.checkbox("Compute per-trajectory metrics (FTLE, amplitude)", value=False)

# --- Build angle array for sector, support wrap-around ---
cs_val, ce_val = st.session_state.get("circle_start_end", DEFAULTS["circle_start_end"])
num_pts = int(st.session_state.get("num_points", DEFAULTS["num_points"]))
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

# --- Prepare solver settings ---
alpha_val = float(st.session_state.get("alpha", DEFAULTS["alpha"]))
K_val = float(st.session_state.get("K", DEFAULTS["K"]))
b_val = float(st.session_state.get("b", DEFAULTS["b"]))
g1_val = float(st.session_state.get("gamma1", DEFAULTS["gamma1"]))
g2_val = float(st.session_state.get("gamma2", DEFAULTS["gamma2"]))
R_val = float(st.session_state.get("initial_radius", DEFAULTS["initial_radius"]))

tn = int(st.session_state.get("t_number", DEFAULTS["t_number"]))
te = float(st.session_state.get("t_end", DEFAULTS["t_end"]))

# Right-hand side (reads constants from local variables for speed)
def rhs(t, state):
    x, y = state
    n = 1.0 / alpha_val
    if n > 1000:
        frac_x = K_val if x > b_val else 0.0
        frac_y = K_val if y > b_val else 0.0
    else:
        x_pos = max(x, 0.0)
        y_pos = max(y, 0.0)
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

# --- Integrate and (optionally) compute metrics ---
initial_conditions = [(b_val + R_val * np.cos(a), b_val + R_val * np.sin(a)) for a in angles]

# Prepare containers
solutions = []  # list of (x, y) arrays
metrics = []    # list of dicts with ftle, amp, final_dist

# Time vector for integration
t_eval = np.linspace(0, te, tn)

# We'll compute perturbed trajectories only if compute_metrics is True
for idx, (x0, y0) in enumerate(initial_conditions):
    try:
        sol = solve_ivp(rhs, (0, te), (x0, y0), method='DOP853', t_eval=t_eval)
        if not sol.success:
            st.warning(f"Solver failed for initial condition {idx}")
            continue
        x, y = sol.y
    except Exception as exc:
        st.warning(f"Solver error for trajectory {idx}: {exc}")
        continue
    solutions.append((x, y))

    # default metrics
    ftle = np.nan
    amp = float(np.max(np.sqrt(x * x + y * y)) - np.min(np.sqrt(x * x + y * y)))
    final_d = np.nan

    if compute_metrics:
        # small relative perturbation
        eps = 1e-6 * (1.0 + abs(x0) + abs(y0))
        xp0 = x0 + eps
        yp0 = y0 + eps * 0.5
        try:
            sol_p = solve_ivp(rhs, (0, te), (xp0, yp0), method='DOP853', t_eval=t_eval)
            if sol_p.success:
                xp, yp = sol_p.y
                dist = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
                # avoid zeros
                dist = np.where(dist <= 0, 1e-12, dist)
                final_d = float(dist[-1])
                # estimate FTLE by linear regression on ln(dist) vs t over the middle 50% of the trajectory
                npoints = len(t_eval)
                s_idx = max(1, int(0.25 * npoints))
                e_idx = max(s_idx + 1, int(0.75 * npoints))
                try:
                    ln_d = np.log(dist[s_idx:e_idx])
                    t_slice = t_eval[s_idx:e_idx]
                    if len(ln_d) >= 2 and np.isfinite(ln_d).all():
                        slope = np.polyfit(t_slice, ln_d, 1)[0]
                        ftle = float(slope)
                except Exception:
                    ftle = np.nan
        except Exception:
            ftle = np.nan
    metrics.append({"idx": idx, "ftle": ftle, "amp": amp, "final_dist": final_d})

# Build labels for UI (showing small summary per trajectory)
labels = []
for m in metrics:
    ft = m.get("ftle")
    amp = m.get("amp")
    if np.isfinite(ft):
        lbl = f"{m['idx']}: FTLE={ft:.4g}, amp={amp:.4g}"
    else:
        lbl = f"{m['idx']}: FTLE=nan, amp={amp:.4g}"
    labels.append(lbl)

# Selection UI: allow user to toggle which trajectories to display
# For many trajectories a multiselect is preferable to creating dozens of checkboxes.
selected = st.sidebar.multiselect("Select trajectories to display", options=labels, default=labels)
selected_idx = {int(s.split(":", 1)[0]) for s in selected}

# --- Plot only selected trajectories ---
fig, ax = plt.subplots(figsize=(8, 6))
styles = ['-', '--', '-.', ':']
colors = plt.cm.tab20.colors
for m, (x, y) in enumerate(solutions):
    if m not in selected_idx:
        continue
    style = styles[m % len(styles)]
    color = colors[m % len(colors)]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.2)
    ax.plot(x[0], y[0], 'o', color=color, markersize=4)
    ax.plot(x[-1], y[-1], 'x', color=color, markersize=5)

ax.set_title(f"Gene regulatory trajectories — t_end={te}, t_points={tn}")
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.grid(True)
st.pyplot(fig)

# Show metrics table if computed
if compute_metrics:
    df = pd.DataFrame(metrics)
    st.markdown("**Per-trajectory metrics (FTLE estimate, amplitude, final distance)**")
    st.dataframe(df)
else:
    st.markdown("(Per-trajectory metrics disabled — enable 'Compute per-trajectory metrics' in the sidebar to compute FTLE and amplitude.)")

# Show textual report of current parameters
st.markdown("**Parameters currently used:**")
st.text(parameters_to_text(collect_params_from_widgets()))

# Footer and notes
st.markdown("---")
st.markdown("**Notes on metrics and UI design choices:**")
st.markdown("- FTLE is estimated by integrating a slightly perturbed initial condition and fitting the slope of ln(distance) vs time over the central portion of the trajectory. It is a finite-time, approximate measure and may be noisy.")
st.markdown("- For many trajectories, using `st.sidebar.multiselect` is more scalable than creating dozens of individual checkboxes. If you prefer individual toggles, we can generate a compact grid of checkboxes, but it may slow UI rendering.")
st.markdown("- This per-trajectory analysis is an intermediate step toward building training data / diagnostics for neural solvers (e.g., label high-FTLE trajectories as 'sensitive', use metrics as features).")

st.markdown("---")
st.markdown("**System of ODEs (safe):**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")
st.markdown("- Robust handling of overflow: np.isfinite checks prevent NaNs/Infs.")
