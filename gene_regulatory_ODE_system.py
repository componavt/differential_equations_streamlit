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

st.title("ODE Solution Visualization")
st.markdown("Select parameters to display one of the solutions.")

# User selects parameters
alpha = st.selectbox("Alpha", alpha_list, format_func=lambda a: f"{a:.0e}")
gamma1 = st.selectbox("Gamma 1 (γ₁)", gamma1_list)
gamma2 = st.selectbox("Gamma 2 (γ₂)", gamma2_list)
method = st.selectbox("Integration Method", methods)

# Filter the dataset based on selected parameters
filtered = [
    d for d in data
    if np.isclose(d['alpha'], alpha)
    and np.isclose(d['gamma1'], gamma1)
    and np.isclose(d['gamma2'], gamma2)
    and d['method'] == method
]

# Plot the result
if filtered:
    d = filtered[0]
    t = d['t']
    x = d['x']
    y = d['y']
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label="x(t), y(t)")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("y(t)")
    ax.set_title(f"alpha={alpha:.0e}, γ₁={gamma1}, γ₂={gamma2}, method={method}")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("No solution found for the selected parameters.")
