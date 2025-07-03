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

# Animation slider: time t
time_slider = st.sidebar.slider("Current time t", min_value=0.0, max_value=float(t_end), step=float(t_end)/499, value=0.0)
frame = int(time_slider / t_end * (499))

# Number of evaluation points and time array
N = 500
t_eval = np.linspace(0, t_end, N)

# Solver choice
method = "DOP853"

# Initial conditions for prey (x0) and predator y0
x0_values = np.arange(1.0, 2.01, 0.1)
y0 = 1.0

# Precompute trajectories once (cacheable)
@st.cache_data
def compute_trajectories():
    trajs = []
    for x0 in x0_values:
        def lv_system(t, z):
            x, y = z
            return [x * (a - b * y), y * (k * b * x - m)]
        sol = solve_ivp(lv_system, (0, t_end), [x0, y0], method=method, t_eval=t_eval)
        trajs.append(sol.y)
    return trajs

trajectories = compute_trajectories()

# Plot static trajectories and dynamic marker
fig, ax = plt.subplots(figsize=(8, 6))
title = f"Lotka–Volterra (a={a}, b={b}, k={k}, m={m}; t_end={t_end})"
ax.set_title(title, fontsize=14)
ax.set_xlabel("Prey population x(t)")
ax.set_ylabel("Predator population y(t)")
ax.grid(True, linestyle='--', linewidth=0.5)

for idx, (x, y) in enumerate(trajectories):
    ax.plot(x, y, linewidth=1.0)
    # dynamic marker
    ax.plot(x[frame], y[frame], 'o', markersize=6, color='red')
    # label initial condition at end
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
- The red marker shows the state at the selected time t.  
- Initial x₀ from 1.0 to 2.0 step 0.1; y₀ = 1.0.  
- Solver: DOP853, points N = 500.
""")

# App title at bottom
st.markdown("---")
st.markdown("*Lotka–Volterra Model Explorer*")
