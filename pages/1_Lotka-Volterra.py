import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Streamlit app for Lotka–Volterra equations
st.title("Lotka–Volterra Model Explorer")

# Fixed Lotka–Volterra parameters
a = 0.2  # Prey growth rate
b = 0.2  # Predation rate
k = 0.5  # Predator efficiency factor
m = 0.1  # Predator death rate

# Sidebar controls
st.sidebar.header("Simulation Settings")
# End time slider from 1 to 50 with step 1
t_end = st.sidebar.slider("End time (t_end)", min_value=1, max_value=50, step=1, value=10)

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
ax.set_title(f"Lotka–Volterra Phase Portrait (t_end = {t_end})", fontsize=14)
ax.set_xlabel("Prey population x(t)", fontsize=12)
ax.set_ylabel("Predator population y(t)", fontsize=12)
ax.grid(True, linestyle='--', linewidth=0.5)

# Solve and plot trajectories for each initial x0
for x0 in x0_values:
    # Define the Lotka–Volterra system
    def lv_system(t, z):
        x, y = z
        dxdt = x * (a - b * y)
        dydt = y * (k * b * x - m)
        return [dxdt, dydt]

    # Solve ODE
    sol = solve_ivp(
        fun=lv_system,
        t_span=(0, t_end),
        y0=[x0, y0],
        method=method,
        t_eval=t_eval
    )
    x, y = sol.y

    # Plot trajectory curve
    ax.plot(x, y, linewidth=1.5)
    # Mark start and end points
    ax.plot(x[0], y[0], marker='o', color='black', markersize=4)
    ax.plot(x[-1], y[-1], marker='x', color='red', markersize=4)
    # Label curve at end point
    ax.text(x[-1], y[-1], f"x0={x0:.1f}", fontsize=8)

# Display plot in Streamlit
st.pyplot(fig)

# Display the mathematical system and parameter values
st.markdown("---")
st.markdown("**Lotka–Volterra system of equations:**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = x \left(a - b y\right),\\
\frac{dy}{dt} = y \left(k b x - m\right),
\end{cases}
""")
st.write(f"Parameters: a = {a}, b = {b}, k = {k}, m = {m}")

# Notes in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Notes:**")
st.sidebar.markdown("- The start point of each trajectory is ● and the end point is ×.")
st.sidebar.markdown("- Initial prey x₀ varies from 1.0 to 2.0 with step 0.1; predator y₀ = 1.0.")
st.sidebar.markdown("- Solver: DOP853, points N = 500.")
