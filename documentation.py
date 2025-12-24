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
    st.markdown("**System of ODEs (safe):**")
    st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")