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

# Integration method - horizontal radio buttons
method = st.radio("Integration Method", methods, horizontal=True)

# Sliders for Alpha, Gamma1, Gamma2 in three vertical sliders placed horizontally
col1, col2, col3 = st.columns(3)

with col1:
    alpha = st.slider("Alpha", min_value=min(alpha_list), max_value=max(alpha_list), 
                     value=alpha_list[0], step=alpha_list[1] - alpha_list[0], format="%0.0e")

with col2:
    gamma1 = st.slider("Gamma 1 (\u03b31)", min_value=min(gamma1_list), max_value=max(gamma1_list), 
                      value=gamma1_list[0], step=gamma1_list[1] - gamma1_list[0])

with col3:
    gamma2 = st.slider("Gamma 2 (\u03b32)", min_value=min(gamma2_list), max_value=max(gamma2_list), 
                      value=gamma2_list[0], step=gamma2_list[1] - gamma2_list[0])

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
    ax.set_title(f"alpha={alpha:.0e}, γ1={gamma1}, γ2={gamma2}, method={method}")
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("No solution found for the selected parameters.")
