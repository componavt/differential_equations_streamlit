import numpy as np
from scipy.integrate import solve_ivp


def compute_ftle_metrics(rhs, x0, y0, te, t_eval, x, y):
    """
    Computes FTLE (Finite-Time Lyapunov Exponent) and related metrics.
    
    Args:
        rhs: Right-hand side function of the ODE system
        x0, y0: Initial conditions
        te: End time
        t_eval: Time points array
        x, y: Solution arrays from the main trajectory
    
    Returns:
        tuple: (ftle, final_d, ftle_r2) or (np.nan, np.nan, np.nan) if computation fails
    """
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
                return ftle, final_d, ftle_r2
        # Return NaN values if computation was unsuccessful
        return np.nan, np.nan, np.nan
    except Exception:
        # Return NaN values in case of exception
        return np.nan, np.nan, np.nan


def hurst_rs(ts):
    """
    Compute the Hurst exponent using the Rescaled Range (R/S) method.
    
    Args:
        ts: Time series data
        
    Returns:
        float: Hurst exponent or np.nan if computation fails
    """
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


def curvature_radius_stats(x, y, t, max_radius=1e6, clip_inf=True):
    """
    Compute robust curvature/radius statistics for a parametric curve (x(t), y(t)).
    
    Args:
        x, y: Coordinates of the curve
        t: Parameter values
        max_radius: Maximum radius to consider (values above are clipped)
        clip_inf: Whether to clip infinite/very large radii
        
    Returns:
        dict: Dictionary containing various curvature statistics
    """
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


def compute_path_length(x, y):
    """
    Compute the total path length of a curve (x(t), y(t)).
    
    Args:
        x, y: Coordinates of the curve
        
    Returns:
        float: Total path length
    """
    dx = np.diff(x)
    dy = np.diff(y)
    seg_lengths = np.sqrt(dx * dx + dy * dy)
    return float(np.sum(seg_lengths))


# Constants for metrics computation
EPSILON = 1e-12
FTLE_START_FRAC = 0.25
FTLE_END_FRAC = 0.75
HURST_MIN_SIZE = 20
CURVATURE_RADIUS_MAX = 1e6

def compute_anomaly_score(ftle, path_len, max_kappa, ftle_r2, hurst=None):
    """
    Compute an anomaly score combining multiple indicators.
    
    Args:
        ftle: Finite-Time Lyapunov Exponent
        path_len: Path length
        max_kappa: Maximum curvature
        ftle_r2: R^2 of FTLE fit
        hurst: Hurst exponent (optional)
        
    Returns:
        float: Anomaly score
    """
    # Normalize inputs using robust z-scores (using median and IQR)
    def robust_z_single(value, median, iqr):
        if iqr == 0:
            return 0.0
        return (value - median) / iqr
    
    # In a real implementation, we'd compute medians and IQRs from a dataset
    # For now, we'll use placeholder normalization factors
    ftle_norm = ftle  # Would be normalized in practice
    path_norm = path_len  # Would be normalized in practice
    kappa_norm = max_kappa  # Would be normalized in practice
    r2_norm = ftle_r2 # Would be normalized in practice
    
    # Basic anomaly score combining multiple indicators
    score = ftle_norm + path_norm + kappa_norm
    
    # Penalize low reliability (low r2)
    if not np.isnan(ftle_r2):
        score -= r2_norm
    
    # Include Hurst exponent if provided
    if hurst is not None and not np.isnan(hurst):
        score += hurst  # Adjust weight as needed
    
    return score