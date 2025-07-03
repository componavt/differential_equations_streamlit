import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Sidebar controls
st.sidebar.header("Simulation Settings")
# Sliders for parameters a, b, k, m
a = st.sidebar.slider("Prey growth rate a", min_value=0.0, max_value=2.0, step=0.1, value=0.2)
b = st.sidebar.slider("Predation rate b", min_value=0.0, max_value=2.0, step=0.1, value=0.2)
k = st.sidebar.slider("Predator efficiency k", min_value=0.0, max_value=2.0, step=0.1, value=0.5)
m = st.sidebar.slider("Predator death rate m", min_value=0.0, max_value=2.0, step=0.1, value=0.1)
# End time slider from 1 to 50
t_end = st.sidebar.slider("End time (t_end)", min_value=1, max_value=50, step=1, value=35)

# Number of evaluation points and time array
N = 500
t_eval = np.linspace(0, t_end, N)

# Solver choice
method = "DOP853"

# Initial conditions for prey (x0) and predator y0
x0_values = np.arange(1.0, 2.01, 0.1)
y0 = 1.0

# Compute trajectories without caching (to reflect t_end changes)
def compute_trajectories(params):
    a_, b_, k_, m_ = params
    trajs = []
    for x0 in x0_values:
        def lv_system(t, z):
            x, y = z
            return [x * (a_ - b_ * y), y * (k_ * b_ * x - m_)]
        sol = solve_ivp(lv_system, (0, t_end), [x0, y0], method=method, t_eval=t_eval)
        trajs.append(sol.y)
    return trajs

trajectories = compute_trajectories((a, b, k, m))((a, b, k, m))

# Plot static phase trajectories
fig, ax = plt.subplots(figsize=(8, 6))
title = f"Lotka–Volterra (a={a}, b={b}, k={k}, m={m}; t_end={t_end})"
ax.set_title(title, fontsize=14)
ax.set_xlabel("Prey population x(t)")
ax.set_ylabel("Predator population y(t)")
ax.grid(True, linestyle='--', linewidth=0.5)

for idx, (x, y) in enumerate(trajectories):
    ax.plot(x, y, linewidth=1.0)
    ax.plot(x[0], y[0], 'o', color='black', markersize=4)
    ax.plot(x[-1], y[-1], 'x', color='red', markersize=4)
    ax.text(x[-1], y[-1], f"x0={x0_values[idx]:.1f}", fontsize=8)

st.pyplot(fig)

# Mathematical formulation and notes on main page
st.markdown("---")
st.markdown("**Lotka–Volterra system of equations:**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = x \left(a - b y\right),\\
\frac{dy}{dt} = y \left(k b x - m\right),
\end{cases}
""")

# Notes
st.markdown("---")
st.markdown("""
**Notes:**  
- Start point ●, end point ×.  
- Initial x₀ from 1.0 to 2.0 step 0.1; y₀ = 1.0.  
- Solver: DOP853, points N = 500.
""")

# App title at bottom
st.markdown("---")
st.markdown("*Lotka–Volterra Model Explorer*")
