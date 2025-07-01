"""
Gene Regulatory ODE System – Streamlit App

Explore numerical solutions to a system of ODEs modeling gene regulation
using different solvers and parameter settings.

🔗 Live App: https://neuraldiffur.streamlit.app/
📦 Source code: https://github.com/componavt/differential_equations_streamlit

Author: Andrew Krizhanovsky
"""

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load saved data
@st.cache_data
def load_data(path="data/001_silver.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

data = load_data()

# Extract all unique parameter values
alpha_list = sorted(set(d['alpha'] for d in data))
gamma1_list = sorted(set(d['gamma1'] for d in data))
gamma2_list = sorted(set(d['gamma2'] for d in data))
methods = sorted(set(d['method'] for d in data))

st.title("ODE Solution Visualization")
st.markdown("Select parameters to display all solutions.")

# Sliders for Gamma1 and Gamma2 in two columns
col1, col2 = st.columns(2)
with col1:
    gamma1 = st.slider("Gamma 1 (γ1)", min_value=min(gamma1_list), max_value=max(gamma1_list), 
                      value=gamma1_list[0], step=gamma1_list[1] - gamma1_list[0])
with col2:
    gamma2 = st.slider("Gamma 2 (γ2)", min_value=min(gamma2_list), max_value=max(gamma2_list), 
                      value=gamma2_list[0], step=gamma2_list[1] - gamma2_list[0])

# Prepare layout: 2 rows x 2 columns
cols = st.columns(2)

# Color map for alpha values
cmap = cm.get_cmap("viridis", len(alpha_list))
alpha_to_color = {a: cmap(i) for i, a in enumerate(alpha_list)}

for idx, method in enumerate(methods):
    row = idx // 2
    col = idx % 2
    with cols[col]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Method: {method}", fontsize=11)

        # Filter solutions for selected gamma1, gamma2, and method, varying alpha
        filtered = [
            d for d in data
            if np.isclose(d['gamma1'], gamma1)
            and np.isclose(d['gamma2'], gamma2)
            and d['method'] == method
        ]

        for d in filtered:
            alpha = d['alpha']
            x = d['x']
            y = d['y']
            color = alpha_to_color[alpha]

            # Plot trajectory with color-coded alpha
            ax.plot(x, y, label=f"α={alpha:.0e}", color=color, linewidth=1.5)

            # Arrows along trajectory (sparser sampling)
            skip = max(3, len(x) // 30)
            x_skip = x[::skip]
            y_skip = y[::skip]
            if len(x_skip) >= 2:
                dx = np.gradient(x_skip)
                dy = np.gradient(y_skip)
                ax.quiver(x_skip, y_skip, dx, dy, angles='xy', scale_units='xy', scale=2.5, width=0.003, color=color, alpha=0.5)

            # Label mid-trajectory with alpha
            mid = len(x) // 2
            ax.text(x[mid], y[mid], f"α={alpha:.0e}", fontsize=7, alpha=0.6, color=color)

            # Start and end point markers
            ax.plot(x[0], y[0], marker='o', color=color, markersize=4, label=None)
            ax.plot(x[-1], y[-1], marker='x', color=color, markersize=4, label=None)

        ax.set_xlabel("x(t)", fontsize=9)
        ax.set_ylabel("y(t)", fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0.5, 2.0)
        ax.set_ylim(0.5, 2.0)
        ax.legend(fontsize=7, loc='upper right')
        st.pyplot(fig)
