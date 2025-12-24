import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

from plain_text_parameters import parameters_to_text, text_to_parameters

# --------------------------------------------------
# gene_regulatory_ODE_system (patched v2)
# - same as patched but STRAIGHT_DIST removed
# - Robust curvature statistics (clip extreme radii)
# - Path length computed and shown in metrics
# - FTLE diagnostics include R^2 of ln(dist) fit and clipping
# - Added additional metrics: max_kappa, frac_high_curv, curv p10/p90, curv finite count
# - Anomaly score computed (combines FTLE, path_len, max_kappa, penalizes low R^2)
# - Display and CSV export rounded to 3 decimal places
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

alpha = st.sidebar.number_input("alpha (1/alpha exponent)", min_value=1e-9, max_value=10.0,
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

# Add controllable center of the initial circle
center_x = st.sidebar.number_input("Center X (circle center)", value=float(b), format="%g")
center_y = st.sidebar.number_input("Center Y (circle center)", value=float(b), format="%g")

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
        "center_x": float(center_x),
        "center_y": float(center_y),
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

    int_keys = {"t_number", "num_points", "circle_start", "circle_end"}
    float_keys = {"t_end", "alpha", "K", "b", "gamma1", "gamma2", "initial_radius", "center_x", "center_y"}

    for key, val in parsed.items():
        if key in int_keys:
            try:
                st.session_state[key] = int(val)
            except Exception:
                pass
        elif key in float_keys:
            try:
                st.session_state[key] = float(val)
            except Exception:
                pass
        elif key == "enabled_idx":
            enabled_idx = [int(x) for x in str(val).split(",") if x.strip().isdigit()]
            npts = int(parsed.get("num_points", st.session_state.get("num_points", num_points)))
            enabled_idx = [i for i in enabled_idx if 0 <= i < npts]
            st.session_state.enabled_checkboxes = enabled_idx

    if "circle_start" in parsed or "circle_end" in parsed:
        cs = int(parsed.get("circle_start", circle_start_end[0]))
        ce = int(parsed.get("circle_end", circle_start_end[1]))
        st.session_state.circle_start_end = (cs, ce)

    if "num_points" in parsed:
        npts = int(parsed["num_points"])
        enabled = st.session_state.get("enabled_checkboxes", [])
        st.session_state.enabled_checkboxes = [i for i in enabled if 0 <= i < npts]

    st.session_state.params_text = parameters_to_text(collect_params_from_widgets())

# Callback: read widget values and put into text area
def read_sliders_to_text():
    st.session_state.params_text = parameters_to_text(collect_params_from_widgets())

col_apply, col_read = st.sidebar.columns(2)
col_apply.button("Apply text → sliders", on_click=apply_text_to_sliders)
col_read.button("Read sliders → text", on_click=read_sliders_to_text)

# --- Compute per-trajectory metrics (optional) ---
compute_metrics = st.sidebar.checkbox("Compute per-trajectory metrics (FTLE, amplitude, Hurst, curvature, path)", value=False)

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

# --- Utility: Hurst exponent (R/S method) ---
def hurst_rs(ts):
    x = np.array(ts, dtype=float)
    N = len(x)
    if N < 20:
        return np.nan
    x = x - np.mean(x)
    Y = np.cumsum(x)
    R = np.zeros(N)
    S = np.zeros(N)
    for n in range(10, N // 2 + 1):
        seg = x[:n]
        Yseg = Y[:n]
        Rn = np.max(Yseg) - np.min(Yseg)
        Sn = np.std(seg, ddof=0)
        if Sn > 0:
            R[n - 1] = Rn
            S[n - 1] = Sn
    valid = (S > 0) & (R > 0)
    if np.sum(valid) < 3:
        return np.nan
    rs = R[valid] / S[valid]
    ns = np.arange(1, N + 1)[valid]
    try:
        H = np.polyfit(np.log(ns), np.log(rs), 1)[0]
    except Exception:
        H = np.nan
    return float(H)

# --- Utility: robust curvature/radius statistics for a parametric curve (x(t), y(t)) ---
def curvature_radius_stats(x, y, t, max_radius=1e6, clip_inf=True):
    x_t = np.gradient(x, t)
    y_t = np.gradient(y, t)
    x_tt = np.gradient(x_t, t)
    y_tt = np.gradient(y_t, t)
    denom = (x_t ** 2 + y_t ** 2) ** 1.5
    num = np.abs(x_t * y_tt - y_t * x_tt)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(denom > 0, num / denom, np.nan)
    radius = np.where(np.isfinite(kappa) & (kappa != 0), 1.0 / kappa, np.nan)
    if clip_inf:
        radius = np.where(radius > max_radius, np.nan, radius)
    finite = np.isfinite(radius)
    stats = {
        "count_total": len(radius),
        "count_finite": int(np.sum(finite)),
        "frac_finite": float(np.sum(finite) / len(radius)),
        "mean": float(np.nanmean(radius)) if np.isfinite(np.nanmean(radius)) else np.nan,
        "median": float(np.nanmedian(radius)) if np.isfinite(np.nanmedian(radius)) else np.nan,
        "p10": float(np.nanpercentile(radius, 10)) if np.isfinite(np.nanpercentile(radius, 10)) else np.nan,
        "p90": float(np.nanpercentile(radius, 90)) if np.isfinite(np.nanpercentile(radius, 90)) else np.nan,
        "std": float(np.nanstd(radius)) if np.isfinite(np.nanstd(radius)) else np.nan,
        "radius_array": radius,
        "kappa_array": (1.0 / radius)  # may contain inf/nan for radius==0
    }
    return stats

solutions, metrics = [], []
t_eval = np.linspace(0, te, tn)

# iterate initial conditions and compute solutions & metrics
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

    ftle, final_d, ftle_r2 = np.nan, np.nan, np.nan
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
                    d_slice = dist[s_idx:e_idx]
                    t_slice = t_eval[s_idx:e_idx]
                    d_slice = np.clip(d_slice, 1e-12, None)
                    ln_d = np.log(d_slice)
                    # linear fit and r2 diagnostics
                    slope, intercept = np.polyfit(t_slice, ln_d, 1)
                    ftle = float(slope)
                    resid = ln_d - (slope * t_slice + intercept)
                    ss_res = np.sum(resid ** 2)
                    ss_tot = np.sum((ln_d - np.mean(ln_d)) ** 2)
                    ftle_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            pass

    # Hurst: compute for x and y and take mean
    hx = hurst_rs(x)
    hy = hurst_rs(y)
    hurst_val = np.nanmean([hx, hy])

    # curvature radius stats
    cr_stats = curvature_radius_stats(x, y, t_eval)
    curv_mean = cr_stats["mean"]
    curv_median = cr_stats["median"]
    curv_std = cr_stats["std"]

    # path length
    dx = np.diff(x)
    dy = np.diff(y)
    seg_lengths = np.sqrt(dx * dx + dy * dy)
    path_len = float(np.sum(seg_lengths))

    # maximum curvature (kappa) and fraction of high curvature points
    kappa_arr = cr_stats.get("kappa_array")
    # kappa_array may include inf/nan; handle robustly
    kappa_vals = np.array(kappa_arr)
    kappa_vals = kappa_vals[np.isfinite(kappa_vals)] if kappa_vals is not None else np.array([])
    max_kappa = float(np.nanmax(kappa_vals)) if kappa_vals.size > 0 else np.nan
    # fraction of points with radius < 10 -> kappa > 0.1 (example threshold)
    frac_high_curv = float(np.sum(kappa_vals > 0.1) / len(t_eval)) if kappa_vals.size > 0 else np.nan

    metrics.append({
        "idx": idx,
        "ftle": ftle,
        "ftle_r2": ftle_r2,
        "amp": amp,
        "final_dist": final_d,
        "hurst": hurst_val,
        "curv_radius_mean": curv_mean,
        "curv_radius_median": curv_median,
        "curv_radius_std": curv_std,
        "curv_p10": cr_stats["p10"],
        "curv_p90": cr_stats["p90"],
        "curv_count_finite": cr_stats["count_finite"],
        "initial_x": float(x0),
        "initial_y": float(y0),
        "path_len": path_len,
        "max_kappa": max_kappa,
        "frac_high_curv": frac_high_curv,
    })

# Compute local z-score of curvature median versus nearest neighbours (fallback without sklearn available)
try:
    from sklearn.neighbors import NearestNeighbors
    use_sklearn = True
except Exception:
    use_sklearn = False

if metrics:
    arr_init = np.array([[m["initial_x"], m["initial_y"]] for m in metrics])
    rad_meds = np.array([m["curv_radius_median"] for m in metrics])
    local_z = np.full(len(metrics), np.nan)
    if len(metrics) > 1:
        nbrs_k = min(5, len(metrics) - 1)
        if use_sklearn:
            nbrs = NearestNeighbors(n_neighbors=nbrs_k + 1).fit(arr_init)
            distances, indices = nbrs.kneighbors(arr_init)
            for i in range(len(metrics)):
                neigh_idx = indices[i, 1:]
                neigh_vals = rad_meds[neigh_idx]
                neigh_vals = neigh_vals[np.isfinite(neigh_vals)]
                if not np.isfinite(rad_meds[i]) or len(neigh_vals) < 1:
                    local_z[i] = np.nan
                else:
                    mu = np.mean(neigh_vals)
                    sigma = np.std(neigh_vals)
                    local_z[i] = (rad_meds[i] - mu) / sigma if sigma != 0 else np.nan
        else:
            for i in range(len(metrics)):
                dists = np.linalg.norm(arr_init - arr_init[i : i + 1], axis=1)
                order = np.argsort(dists)
                neigh_idx = order[1 : 1 + nbrs_k]
                neigh_vals = rad_meds[neigh_idx]
                neigh_vals = neigh_vals[np.isfinite(neigh_vals)]
                if not np.isfinite(rad_meds[i]) or len(neigh_vals) < 1:
                    local_z[i] = np.nan
                else:
                    mu = np.mean(neigh_vals)
                    sigma = np.std(neigh_vals)
                    local_z[i] = (rad_meds[i] - mu) / sigma if sigma != 0 else np.nan
    for i in range(len(metrics)):
        metrics[i]["curv_radius_local_zscore"] = float(local_z[i]) if np.isfinite(local_z[i]) else np.nan

# Build dataframe
df_metrics = pd.DataFrame(metrics)

# --- Compute anomaly score (combine multiple indicators) ---
# Prepare columns for scoring: ftle (higher), path_len (higher), max_kappa (higher), ftle_r2 (higher means reliable)
# We'll compute robust z-scores (subtract median, divide by IQR) to avoid influence of outliers

def robust_z(arr):
    arr = np.array(arr, dtype=float)
    finite = np.isfinite(arr)
    out = np.full_like(arr, np.nan)
    if np.sum(finite) == 0:
        return out
    median = np.nanmedian(arr[finite])
    q1 = np.nanpercentile(arr[finite], 25)
    q3 = np.nanpercentile(arr[finite], 75)
    iqr = q3 - q1 if q3 - q1 != 0 else 1.0
    out[finite] = (arr[finite] - median) / iqr
    return out

if not df_metrics.empty:
    ftle_z = robust_z(df_metrics['ftle'].values)
    path_z = robust_z(df_metrics['path_len'].values)
    kappa_z = robust_z(df_metrics['max_kappa'].values)
    r2_z = robust_z(df_metrics['ftle_r2'].fillna(0).values)
    # score = ftle_z + path_z + kappa_z - r2_z (penalize low r2 by subtracting its z)
    score_arr = ftle_z + path_z + kappa_z - r2_z
    df_metrics['anomaly_score'] = score_arr

# --- Selection UI (checkbox list, sorted by anomaly_score desc) ---
selected_idx = []
st.sidebar.markdown("**Select trajectories to display (sorted by anomaly score)**")

if not df_metrics.empty:
    df_sorted = df_metrics.sort_values(by="anomaly_score", ascending=False, na_position="last")
    enabled_raw = st.session_state.get("enabled_checkboxes", [])
    st.session_state.enabled_checkboxes = [i for i in enabled_raw if 0 <= i < len(solutions)]
    enabled_set = set(st.session_state.get("enabled_checkboxes", []))
    new_enabled = []
    for m, row in df_sorted.iterrows():
        label = f"{int(row['idx'])}: score={row.get('anomaly_score', np.nan):.3g}, FTLE={row.get('ftle', np.nan):.3g}"
        default_val = (row["idx"] in enabled_set) if enabled_set else True
        if st.sidebar.checkbox(label, value=default_val, key=f"sel_{int(row['idx'])}"):
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

# --- Show metrics table (rounded) ---
st.markdown("**Per-trajectory metrics (rounded to 3 decimals)**")
st.dataframe(df_metrics.reset_index(drop=True).round(3))

# CSV export with rounding
if not df_metrics.empty:
    csv_name = "export_metrics_rounded.csv"
    df_metrics.to_csv(csv_name, index=False, float_format="%.3f")
    st.download_button("Download metrics CSV (rounded)", data=open(csv_name, 'rb'), file_name=csv_name)

st.markdown("**Parameters currently used:**")
st.text(parameters_to_text(collect_params_from_widgets()))

# Import the documentation function
from documentation import display_ode_documentation

# Display the ODE system documentation
display_ode_documentation()
