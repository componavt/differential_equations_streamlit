import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

st.title("ODE Solution Visualization: 4 Integration Methods")
st.markdown("Select parameters to display all solutions for each method.")

# User selects shared parameters
gamma1 = st.selectbox("Gamma 1 (γ₁)", gamma1_list)
gamma2 = st.selectbox("Gamma 2 (γ₂)", gamma2_list)

# Prepare layout for 4 integration methods in one row
cols = st.columns(4)

for idx, method in enumerate(methods):
    with cols[idx]:
        fig, ax = plt.subplots()
        ax.set_title(method, fontsize=10)

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

            # Plot trajectory
            ax.plot(x, y, label=f"α={alpha:.0e}", linewidth=1.2)

            # Arrows along trajectory (sparser sampling)
            skip = max(3, len(x) // 30)
            x_skip = x[::skip]
            y_skip = y[::skip]
            if len(x_skip) >= 2:
                dx = np.gradient(x_skip)
                dy = np.gradient(y_skip)
                ax.quiver(x_skip, y_skip, dx, dy, angles='xy', scale_units='xy', scale=2.5, width=0.003, alpha=0.5)

            # Label mid-trajectory
            mid = len(x) // 2
            ax.text(x[mid], y[mid], f"α={alpha:.0e}", fontsize=7, alpha=0.6)

        ax.set_xlabel("x(t)", fontsize=8)
        ax.set_ylabel("y(t)", fontsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.5, 2.0)
        ax.set_ylim(0.5, 2.0)
        ax.legend(fontsize=6, loc='best')
        st.pyplot(fig)
