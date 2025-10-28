import streamlit as st

def print_system_of_odes():
    st.markdown("---")
    st.markdown("**System of ODEs (safe new):**")
  
    st.latex(r"""
    \begin{cases}
    \frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
    \frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
    \end{cases}
    """)
  
