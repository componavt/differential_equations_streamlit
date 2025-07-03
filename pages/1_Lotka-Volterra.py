import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import streamlit.components.v1 as components

# Sidebar controls
st.sidebar.header("Simulation Settings")
# Sliders for parameters a, b, k, m
a = st.sidebar.slider("Prey growth rate a", min_value=0.0, max_value=2.0, step=0.1, value=0.2)
b = st.sidebar.slider("Predation rate b", min_value=0.0, max_value=2.0, step=0.1, value=0.2)
k = st.sidebar.slider("Predator efficiency k", min_value=0.0, max_value=2.0, step=0.1, value=0.5)
m = st.sidebar.slider("Predator death rate m", min_value=0.0, max_value=2.0, step=0.1, value=0.1)
# End time slider from 1 to 50 with step 1
t_end = st.sidebar.slider("End time (t_end)", min_value=1, max_value=50, step=1, value=35)

# Number of evaluation points and time array
N = 500
t_eval = np.linspace(0, t_end, N)

# Solver choice
method = "DOP853"

# Initial conditions for prey (x0) and fixed predator initial y0
x0_values = np.arange(1.0, 2.01, 0.1)
y0 = 1.0

# Set up plot
fig, ax = plt.subplots(figsize=(8, 6))
title = f"({a}, {b}, {k}, {m}); t_end={t_end}"
ax.set_title(f"Lotka–Volterra {title}", fontsize=14)
ax.set_xlabel("Prey population x(t)", fontsize=12)
ax.set_ylabel("Predator population y(t)", fontsize=12)
ax.grid(True, linestyle='--', linewidth=0.5)

# Precompute trajectories and create static curves and markers
trajectories = []
markers = []
for x0 in x0_values:
    def lv_system(t, z):
        x, y = z
        dxdt = x * (a - b * y)
        dydt = y * (k * b * x - m)
        return [dxdt, dydt]

    sol = solve_ivp(fun=lv_system, t_span=(0, t_end), y0=[x0, y0], method=method, t_eval=t_eval)
    x, y = sol.y
    ax.plot(x, y, linewidth=1.0)
    marker, = ax.plot([], [], 'o', markersize=6)
    trajectories.append((x, y))
    markers.append(marker)

# Animation function
def update(frame):
    for (x, y), marker in zip(trajectories, markers):
        marker.set_data(x[frame], y[frame])
    return markers

ani = animation.FuncAnimation(fig, update, frames=N, blit=True, interval=50)

# Display animation as HTML
html = ani.to_jshtml()
components.html(html, height=600)

# Mathematical formulation and notes
st.markdown("---")
st.markdown("**Lotka–Volterra system of equations:**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = x \left(a - b y\right),\\
\frac{dy}{dt} = y \left(k b x - m\right),
\end{cases}
""")
st.markdown(f"""
**Notes:**
- Start point ●, end point continuously moves over time.
- Initial x₀ from 1.0 to 2.0 step 0.1; y₀ = 1.0.
- Solver: {method}, points N = {N}.
""")

# App title at bottom
st.markdown("---")
st.markdown("*Lotka–Volterra Model Explorer*")
