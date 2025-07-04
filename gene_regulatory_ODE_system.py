import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Sidebar controls ---
st.sidebar.header("Simulation Settings")

# Solver and resolution
t_method = "DOP853"
t_number = st.sidebar.slider("t_number", min_value=10, max_value=1000, step=10, value=100)
t_end_vals = list(np.round(np.arange(0, 1.01, 0.1), 2)) + list(np.round(np.arange(1.5, 5.1, 0.5), 2))
t_end = st.sidebar.select_slider("End time t_end", options=t_end_vals, value=1.0)
t_eval = np.linspace(0, t_end, t_number)

# Parameters grouping
col1, col2, col3 = st.sidebar.columns(3)
# Alpha slider
alpha_opts = list(np.round(np.arange(0.1, 1.1, 0.1), 3)) + [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
alpha_opts = sorted(set(alpha_opts), reverse=True)
alpha = col1.select_slider("Alpha", options=alpha_opts, value=0.001)
# K
K = col2.slider("K", min_value=0.1, max_value=5.0, step=0.1, value=1.0)
# b
b_values = [0.9, 0.99, 0.999, 0.9999, 1.0, 1.0001, 1.001, 1.01, 1.1]
b_values += list(np.round(np.arange(0, 3.1, 0.1), 3))
b = col3.select_slider("b", options=sorted(set(b_values)), value=1.0)

# Gamma sliders with fine/coarse steps
g1, g2 = st.sidebar.columns(2)
gamma_range = [round(v, 3) for v in list(np.arange(0, 1.01, 0.01)) + list(np.arange(1.1, 3.1, 0.1))]
gamma1 = g1.select_slider("Gamma 1", options=gamma_range, value=1.0)
gamma2 = g2.select_slider("Gamma 2", options=gamma_range, value=1.0)

# Initial conditions parameters
rcol, ncol = st.sidebar.columns(2)
initial_radius = rcol.select_slider("Initial radius", options=[0.001] + list(np.round(np.arange(0.01, 0.11, 0.01), 3)) + [0.2, 0.3])
num_points = ncol.slider("Number of trajectories", min_value=3, max_value=50, step=1, value=12)

# --- Safe RHS definition with small epsilon to avoid invalid power ---
eps = 1e-8

def get_rhs_safe(t, state):
    x, y = state
    # ensure positivity for fractional power
    x_pos = np.maximum(x, eps)
    y_pos = np.maximum(y, eps)
    dxdt = (K * x_pos**(1 / alpha)) / (b**(1 / alpha) + x_pos**(1 / alpha)) - gamma1 * x
    dydt = (K * y_pos**(1 / alpha)) / (b**(1 / alpha) + y_pos**(1 / alpha)) - gamma2 * y
    return [dxdt, dydt]

# --- Compute initial points on circle centered at (b,b) ---
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
initial_conditions = [(b + initial_radius * np.cos(a), b + initial_radius * np.sin(a)) for a in angles]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))
title = f"Safe DOP853, t_end={t_end}, pts={t_number}"
ax.set_title(title)
ax.set_xlabel("x(t)")
ax.set_ylabel("y(t)")
ax.grid(True)

styles = ['-', '--', '-.', ':']
colors = plt.cm.tab20.colors

for idx, (x0, y0) in enumerate(initial_conditions):
    sol = solve_ivp(
        fun=get_rhs_safe,
        t_span=(0, t_end),
        y0=[x0, y0],
        method=t_method,
        t_eval=t_eval
    )
    x, y = sol.y
    style = styles[idx % len(styles)]
    color = colors[idx % len(colors)]
    ax.plot(x, y, linestyle=style, color=color, linewidth=1.5)
    ax.plot(x[0], y[0], 'o', color=color, markersize=4)
    ax.plot(x[-1], y[-1], 'x', color=color, markersize=5)
    step = max(1, len(x)//10)
    for j in range(step, len(x), step):
        ax.annotate('', xy=(x[j], y[j]), xytext=(x[j-step], y[j-step]), arrowprops=dict(arrowstyle='->', color=color, lw=1))
    ax.text(x[-1], y[-1], f"({x0:.3f},{y0:.3f})", fontsize=8)

st.pyplot(fig)

# --- Parameters report ---
st.markdown("**Parameters used in this run:**")
st.text(f"DOP853, t_end={t_end}, K={K}, α={alpha}, b={b}, γ1={gamma1}, γ2={gamma2}, R={initial_radius}, n={num_points}, pts={t_number}")

# --- Explanations below ---
st.markdown("---")
st.markdown("**System of ODEs (safe):**")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
""")
st.markdown("- Numeric safeguard: replaced x,y<ε with ε before power operations to avoid NaNs.")

# Footer title
st.markdown("---")
st.markdown("*safe_calculation: stabilized gene ODE explorer*")
