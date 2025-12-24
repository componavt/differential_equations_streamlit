import streamlit as st

def display_ode_documentation():
    """
    Displays the ODE system documentation and equations in the Streamlit interface.
    This includes annotations about the curves, metrics, FTLE diagnostics, and the
    mathematical formulation of the gene regulatory ODE system.
    """
    st.markdown("---")
    st.markdown("- Each curve is annotated with its `idx` at the final point.")
    st.markdown("- Table shows robust curvature statistics (median, p10, p90) and path length.")
    st.markdown("- FTLE diagnostics include R^2 of ln(dist) fit (ftle_r2). Anomaly score combines FTLE, path length, max curvature and R^2 reliability.")
    
    st.markdown("---")
    st.markdown("**Column Descriptions:**")
    st.markdown("- idx - Index of the trajectory, starting from 0")
    st.markdown("- ftle - Finite-Time Lyapunov Exponent, computed as the slope of linear fit of ln(d(t)) vs t on a central window (25%–75% of t_eval) after clipping d(t) to a minimum (1e-12)")
    st.markdown("- ftle_r2 - Coefficient of determination (R²) of the linear fit for FTLE, indicating reliability of the ftle estimate")
    st.markdown("- amp - Amplitude (max−min of radial distance sqrt(x²+y²))")
    st.markdown("- final_dist - Final distance between the original trajectory and its companion trajectory with tiny perturbation")
    st.markdown("- hurst - Hurst exponent using the rescaled range (R/S) method calculated for x(t) and y(t) and averaged")
    st.markdown("- curv_radius_mean - Mean curvature radius computed from 1/κ(t) where κ(t) is the curvature")
    st.markdown("- curv_radius_median - Median curvature radius computed from 1/κ(t) where κ(t) is the curvature")
    st.markdown("- curv_radius_std - Standard deviation of curvature radius")
    st.markdown("- curv_p10 - 10th percentile of curvature radius")
    st.markdown("- curv_p90 - 90th percentile of curvature radius")
    st.markdown("- curv_count_finite - Number of finite radius samples")
    st.markdown("- initial_x - X coordinate of the initial point")
    st.markdown("- initial_y - Y coordinate of the initial point")
    st.markdown("- path_len - Total arclength computed as sum of Euclidean distances between consecutive points along the trajectory")
    st.markdown("- max_kappa - Maximum finite curvature value, useful to detect sharp bends")
    st.markdown("- frac_high_curv - Fraction of time points with κ(t) above a threshold (default κ > 0.1, i.e., radius < 10), measuring density of sharp bends along the trajectory")
    st.markdown("- curv_radius_local_zscore - Local z-score of curve median curvature radius relative to nearest neighbors in initial condition space")
    st.markdown("- anomaly_score - Aggregated score combining robust z-scores (IQR-based) of ftle, path_len, max_kappa, and ftle_r2 (ftle + path_len + max_kappa − ftle_r2)")

    st.markdown("---")
    st.markdown("**System of ODEs (safe):**")
    st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")