import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Sidebar controls ---
st.sidebar.header("Simulation Settings")

# ODE solver parameters
method = "DOP853"
N = 500  # number of points

# Time range slider: 0–1 step 0.1, then 1–5 step 0.5
t_end_values = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time (t_end)", options=t_end_values, value=1.0)
t_eval = np.linspace(0, t_end, N)

# Gene-regulatory parameters as sliders
gamma1 = st.sidebar.slider("Gamma 1", min_value=0.0, max_value=5.0, step=0.1, value=1.0)
gamma2 = st.sidebar.slider("Gamma 2", min_value=0.0, max_value=5.0, step=0.1, value=1.0)
K = st.sidebar.slider("K", min_value=0.1, max_value=5.0, step=0.1, value=1.0)
alpha = st.sidebar.selectbox("Alpha", options=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], index=2)

# Initial circle parameters
num_points = st.sidebar.slider("Number of trajectories", min_value=3, max_value=50, step=1, value=12)
initial_radius = st.sidebar.select_slider(
    "Initial radius",
    options=[0.001] + list(np.round(np.arange(0.01, 0.11, 0.01), 2)) + [0.2, 0.3]
)

# --- Define RHS from notebook ---
def get_rhs(t, state):
    x, y = state
    b = 1.0
    dxdt = (K * x**(1 / alpha)) / (b**(1 / alpha) + x**(1 / alpha)) - gamma1 * x
    dydt = (K * y**(1 / alpha)) / (b**(1 / alpha) + y**(1 / alpha)) - gamma2 * y
    return [dxdt, dydt]

# --- Compute initial points on circle ---
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
initial_conditions = [(initial_radius * np.cos(a), initial_radius * np.sin(a)) for a in angles]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
title = f"DOP853, t_end={t_end}, K={K}, α={alpha}, γ1={gamma1}, γ2={gamma2}, R={initial_radius}, n={num_points}"
ax.set_title(title)
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.grid(True)

styles = ['-', '--', '-.', ':']
colors = plt.cm.tab20.colors

for idx, (x0, y0) in enumerate(initial_conditions):
    sol = solve_ivp(
        fun=get_rhs,
        t_span=(0, t_end),
        y0=[x0, y0],
        method=method,
        t_eval=t_eval
    )
    x, y = sol.y
    style = styles[idx % len(styles)]
    color = colors[idx % len(colors)]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.5)
    # start and end markers
    ax.plot(x[0], y[0], marker='o', color=color, markersize=4)
    ax.plot(x[-1], y[-1], marker='x', color=color, markersize=5)
    # add direction arrows along curve
    step = len(x) // 10
    for j in range(step, len(x), step):
        dx = x[j] - x[j - step]
        dy = y[j] - y[j - step]
        ax.annotate('', xy=(x[j], y[j]), xytext=(x[j - step], y[j - step]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1))
    # label at end
    ax.text(x[-1], y[-1], f"({x0:.3f},{y0:.3f})", fontsize=8)

st.pyplot(fig)

# --- Explanations below ---
st.markdown("---")
st.markdown("**System of ODEs:**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} \;\-\; \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} \;\-\; \gamma_2\,y.
\end{cases}
""")
st.markdown("""
- Solver: DOP853, N = 500 points.
- Initial conditions on circle of radius R.
- Parameters gamma1, gamma2, K, and alpha selectable.
- Start point ●, end point ×. Arrows indicate direction of motion.
""")

# Footer title
st.markdown("---")
st.markdown("*DOP853gene: merged gene regulatory ODE solver and plot*")
