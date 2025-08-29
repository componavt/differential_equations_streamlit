import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

from plain_text_parameters import parameters_to_text, text_to_parameters

# --------------------------------------------------
# gene_regulatory_ODE_system91
# - Sorts checkboxes by FTLE (descending).
# - Fixes Apply/Read buttons functionality.
# - Saves enabled checkboxes state into text field and restores from it.
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
    "circle_start_end": (0, 360),
}

# --- Sidebar: widgets ---
st.sidebar.header("Simulation Settings")

# Time / solver resolution
t_number = st.sidebar.slider("t_number", min_value=10, max_value=1000, step=10,
                             value=int(DEFAULTS["t_number"]))

t_end_vals = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time t_end", options=t_end_vals,
                                  value=float(DEFAULTS["t_end"]))

alpha = st.sidebar.number_input("alpha (1/α exponent)", min_value=1e-9, max_value=10.0,
                                value=float(DEFAULTS["alpha"]), format="%g")
K = st.sidebar.slider("K", min_value=0.1, max_value=5.0, step=0.1,
                       value=float(DEFAULTS["K"]))
b = st.sidebar.number_input("b", min_value=0.0, max_value=10.0,
                             value=float(DEFAULTS["b"]), format="%g")

gamma1 = st.sidebar.number_input("gamma1", min_value=0.0, max_value=10.0,
                                 value=float(DEFAULTS["gamma1"]), format="%g")
gamma2 = st.sidebar.number_input("gamma2", min_value=0.0, max_value=10.0,
                                 value=float(DEFAULTS["gamma2"]), format="%g")

initial_radius = st.sidebar.number_input("Initial radius (R)", min_value=0.0, max_value=10.0,
                                         value=float(DEFAULTS["initial_radius"]), format="%g")
num_points = st.sidebar.slider("Number of trajectories", min_value=3, max_value=50, step=1,
                               value=int(DEFAULTS["num_points"]))

circle_start_end = st.sidebar.slider("Sector on circle (degrees)", 0, 360,
                                    tuple(map(int, DEFAULTS["circle_start_end"])), step=1)

# --- Plain text area and two buttons ---
def collect_params_from_widgets():
    cs, ce = circle_start_end
    params = {
        "t_number": int(t_number),
        "t_end": float(t_end),
        "alpha": float(alpha),
        "K": float(K),
        "b": float(b),
        "gamma1": float(gamma1),
        "gamma2": float(gamma2),
        "initial_radius": float(initial_radius),
        "num_points": int(num_points),
        "circle_start": int(cs),
        "circle_end": int(ce),
    }
    # Save only enabled checkboxes if available
    if "enabled_checkboxes" in st.session_state:
        enabled = st.session_state.enabled_checkboxes
        if enabled:
            params["enabled_idx"] = ",".join(map(str, enabled))
    return params

if "params_text" not in st.session_state:
    st.session_state.params_text = parameters_to_text(collect_params_from_widgets())

st.sidebar.markdown("**Parameters (plain text)**")
params_text = st.sidebar.text_area("Edit parameters here:", value=st.session_state.params_text, height=220, key="params_text")

# Callback: parse text and apply to widgets
def apply_text_to_sliders():
    parsed = text_to_parameters(st.session_state.params_text)
    if not parsed:
        return
    for key in parsed:
        if key in DEFAULTS:
            try:
                if key in {"t_number", "num_points", "circle_start", "circle_end"}:
                    st.session_state[key] = int(parsed[key])
                else:
                    st.session_state[key] = float(parsed[key])
            except Exception:
                pass
    if "circle_start" in parsed or "circle_end" in parsed:
        cs = int(parsed.get("circle_start", circle_start_end[0]))
        ce = int(parsed.get("circle_end", circle_start_end[1]))
        st.session_state.circle_start_end = (cs, ce)
    if "enabled_idx" in parsed:
        enabled_idx = [int(x) for x in str(parsed["enabled_idx"]).split(",") if x.strip().isdigit()]
        st.session_state.enabled_checkboxes = enabled_idx
    st.session_state.params_text = parameters_to_text(collect_params_from_widgets())

# Callback: read widget values and put into text area
def read_sliders_to_text():
    st.session_state.params_text = parameters_to_text(collect_params_from_widgets())

col_apply, col_read = st.sidebar.columns(2)
col_apply.button("Apply text → sliders", on_click=apply_text_to_sliders)
col_read.button("Read sliders → text", on_click=read_sliders_to_text)

# --- Compute per-trajectory metrics (optional) ---
compute_metrics = st.sidebar.checkbox("Compute per-trajectory metrics (FTLE, amplitude)", value=False)

# --- Build angle array ---
cs_val, ce_val = circle_start_end
span = (ce_val - cs_val) % 360
if np.isclose(span, 0.0):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
else:
    cs = cs_val % 360
    ce = ce_val % 360
    if ce >= cs:
        degs = np.linspace(cs, ce, num_points, endpoint=False)
    else:
        span2 = (ce + 360) - cs
        degs = (cs + np.linspace(0, span2, num_points, endpoint=False)) % 360
    angles = np.deg2rad(degs)

# --- Prepare solver settings ---
alpha_val = float(alpha)
K_val = float(K)
b_val = float(b)
g1_val = float(gamma1)
g2_val = float(gamma2)
R_val = float(initial_radius)

tn = int(t_number)
te = float(t_end)

def rhs(t, state):
    x, y = state
    n = 1.0 / alpha_val
    if n > 1000:
        frac_x = K_val if x > b_val else 0.0
        frac_y = K_val if y > b_val else 0.0
    else:
        x_pos, y_pos = max(x, 0.0), max(y, 0.0)
        try:
            pow_b = np.power(b_val, n)
            pow_x = np.power(x_pos, n)
            pow_y = np.power(y_pos, n)
            frac_x = (K_val * pow_x) / (pow_b + pow_x) if np.isfinite(pow_x) else (K_val if x > b_val else 0.0)
            frac_y = (K_val * pow_y) / (pow_b + pow_y) if np.isfinite(pow_y) else (K_val if y > b_val else 0.0)
        except Exception:
            frac_x, frac_y = (K_val if x > b_val else 0.0), (K_val if y > b_val else 0.0)
    return [frac_x - g1_val * x, frac_y - g2_val * y]

initial_conditions = [(b_val + R_val * np.cos(a), b_val + R_val * np.sin(a)) for a in angles]
solutions, metrics = [], []
t_eval = np.linspace(0, te, tn)

for idx, (x0, y0) in enumerate(initial_conditions):
    try:
        sol = solve_ivp(rhs, (0, te), (x0, y0), method='DOP853', t_eval=t_eval)
        if not sol.success:
            continue
        x, y = sol.y
    except Exception:
        continue
    solutions.append((x, y))
    amp = float(np.max(np.sqrt(x * x + y * y)) - np.min(np.sqrt(x * x + y * y)))
    ftle, final_d = np.nan, np.nan
    if compute_metrics:
        eps = 1e-6 * (1.0 + abs(x0) + abs(y0))
        xp0, yp0 = x0 + eps, y0 + 0.5 * eps
        try:
            sol_p = solve_ivp(rhs, (0, te), (xp0, yp0), method='DOP853', t_eval=t_eval)
            if sol_p.success:
                xp, yp = sol_p.y
                dist = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
                dist = np.where(dist <= 0, 1e-12, dist)
                final_d = float(dist[-1])
                s_idx, e_idx = int(0.25 * len(t_eval)), int(0.75 * len(t_eval))
                if e_idx > s_idx + 1:
                    ln_d, t_slice = np.log(dist[s_idx:e_idx]), t_eval[s_idx:e_idx]
                    if np.isfinite(ln_d).all():
                        ftle = float(np.polyfit(t_slice, ln_d, 1)[0])
        except Exception:
            pass
    metrics.append({"idx": idx, "ftle": ftle, "amp": amp, "final_dist": final_d})

df_metrics = pd.DataFrame(metrics)

# --- Selection UI (checkbox list, sorted by FTLE descending) ---
selected_idx = []
st.sidebar.markdown("**Select trajectories to display (sorted by FTLE)**")

if not df_metrics.empty:
    df_sorted = df_metrics.sort_values(by="ftle", ascending=False, na_position="last")
    enabled_set = set(st.session_state.get("enabled_checkboxes", []))
    new_enabled = []
    for m, row in df_sorted.iterrows():
        label = f"{row['idx']}: FTLE={row['ftle']:.4g}, amp={row['amp']:.4g}" if np.isfinite(row['ftle']) else f"{row['idx']}: FTLE=nan, amp={row['amp']:.4g}"
        default_val = row["idx"] in enabled_set if enabled_set else True
        if st.sidebar.checkbox(label, value=default_val, key=f"sel_{row['idx']}"):
            selected_idx.append(int(row['idx']))
            new_enabled.append(int(row['idx']))
    st.session_state.enabled_checkboxes = new_enabled

# --- Plot trajectories ---
fig, ax = plt.subplots(figsize=(8, 6))
styles, colors = ['-', '--', '-.', ':'], plt.cm.tab20.colors
for m, (x, y) in enumerate(solutions):
    if m not in selected_idx:
        continue
    style, color = styles[m % len(styles)], colors[m % len(colors)]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.2)
    ax.plot(x[0], y[0], 'o', color=color, markersize=4)
    ax.plot(x[-1], y[-1], 'x', color=color, markersize=6)
    ax.text(x[-1] + 0.01, y[-1] + 0.01, f"{m}", fontsize=8, color=color)

ax.set_title(f"Gene regulatory trajectories — t_end={te}, t_points={tn}")
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.grid(True)
st.pyplot(fig)

# --- Show metrics table (without redundant index) ---
st.markdown("**Per-trajectory metrics (FTLE estimate, amplitude, final distance)**")
st.dataframe(df_metrics.reset_index(drop=True))

st.markdown("**Parameters currently used:**")
st.text(parameters_to_text(collect_params_from_widgets()))

st.markdown("---")
st.markdown("- Each curve is annotated with its `idx` at the final point.")
st.markdown("- Table shows metrics without redundant pandas index.")
st.markdown("- Checkbox list sorted by FTLE descending, saves only enabled indices.")

st.markdown("---")
st.markdown("**System of ODEs (safe):**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")
