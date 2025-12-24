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