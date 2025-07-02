import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# Load saved DataFrame from .pkl file
@st.cache_data
def load_data(path="data/009_solutions_gene_DOP853_points33K.pkl"):
    return pd.read_pickle(path)

df = load_data()

# Extract all unique parameter values
alpha_list = sorted(df['alpha'].unique())
gamma1_list = sorted(df['gamma1'].unique())
gamma2_list = sorted(df['gamma2'].unique())

# Sliders and explanation after the plot
st.markdown("---")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} \;-\; \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} \;-\; \gamma_2\,y.
\end{cases}
""")

st.markdown("Select parameters to display the solution using the DOP853 method.")

# Selection widgets
col1, col2, col3 = st.columns(3)
with col1:
    alpha = st.selectbox("Alpha (\u03b1)", options=alpha_list, format_func=lambda a: f"{a:.0e}")
with col2:
    gamma1 = st.slider("\u03b31 (Gamma 1)", min_value=min(gamma1_list), max_value=max(gamma1_list), 
                      value=gamma1_list[0], step=gamma1_list[1] - gamma1_list[0])
with col3:
    gamma2 = st.slider("\u03b32 (Gamma 2)", min_value=min(gamma2_list), max_value=max(gamma2_list), 
                      value=gamma2_list[0], step=gamma2_list[1] - gamma2_list[0])

# Filter DataFrame for selected parameters
filtered_df = df[(df['alpha'] == alpha) & (df['gamma1'] == gamma1) & (df['gamma2'] == gamma2)]

# Plot only for DOP853
method = "DOP853"
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title(f"Method: {method}, α={alpha:.0e}, γ1={gamma1:.1f}, γ2={gamma2:.1f}", fontsize=11)
ax.set_xlabel("x(t)", fontsize=9)
ax.set_ylabel("y(t)", fontsize=9)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(labelsize=8)
ax.set_xlim(0.5, 2.0)
ax.set_ylim(0.5, 2.0)

for _, row in filtered_df.iterrows():
    x0 = row['x0']
    y0 = row['y0']
    t = row['t']
    solution = row['solutions']

    if method in solution:
        x, y = solution[method]
        label_text = f"x₀={x0:.3f}, y₀={y0:.3f}"
        ax.plot(x, y, label=label_text, linewidth=1.5)
        skip = max(3, len(x) // 30)
        x_skip = x[::skip]
        y_skip = y[::skip]
        if len(x_skip) >= 2:
            dx = np.gradient(x_skip)
            dy = np.gradient(y_skip)
            ax.quiver(x_skip, y_skip, dx, dy, angles='xy', scale_units='xy', scale=2.5, width=0.003, alpha=0.5)
        ax.plot(x[0], y[0], marker='o', color='black', markersize=4)
        ax.plot(x[-1], y[-1], marker='x', color='red', markersize=4)

ax.legend(fontsize=7, loc='best')
st.pyplot(fig)

# Display note
st.markdown("**Note:** The start point of the trajectory is marked with a circle (\u25cf), and the end point with a cross (\u00d7).")
