import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load saved data from .npz file (structured arrays)
@st.cache_data
def load_data(path="data/003_solutions_gene_ODU_unique.npz"):
    npzfile = np.load(path, allow_pickle=True)
    data = []
    for i in range(len(npzfile['alpha'])):
        data.append({
            'alpha': npzfile['alpha'][i],
            'gamma1': npzfile['gamma1'][i],
            'gamma2': npzfile['gamma2'][i],
            'method': npzfile['method'][i],
            'x0': npzfile['x0'][i],
            'y0': npzfile['y0'][i],
            't': npzfile['t'][i],
            'x': npzfile['x'][i],
            'y': npzfile['y'][i],
        })
    return data

data = load_data()

# Extract all unique parameter values
alpha_list = sorted(set(d['alpha'] for d in data))
gamma1_list = sorted(set(d['gamma1'] for d in data))
gamma2_list = sorted(set(d['gamma2'] for d in data))
methods = sorted(set(d['method'] for d in data))

# Sliders and explanation after the plots
st.markdown("---")
st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} \;-\; \gamma_1\,x,\\[6pt]
\frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} \;-\; \gamma_2\,y.
\end{cases}
""")

st.markdown("Select parameters to display the solution for each method.")

# Selection widgets
col1, col2, col3 = st.columns(3)
with col1:
    alpha = st.selectbox("Alpha (α)", options=alpha_list, format_func=lambda a: f"{a:.0e}")
with col2:
    gamma1 = st.slider("Gamma 1 (γ1)", min_value=min(gamma1_list), max_value=max(gamma1_list), 
                      value=gamma1_list[0], step=gamma1_list[1] - gamma1_list[0])
with col3:
    gamma2 = st.slider("Gamma 2 (γ2)", min_value=min(gamma2_list), max_value=max(gamma2_list), 
                      value=gamma2_list[0], step=gamma2_list[1] - gamma2_list[0])

# Display updated plots for selected alpha
figs_axes = []
cols = st.columns(2)
for idx, method in enumerate(methods):
    col = cols[idx % 2]
    with col:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"Method: {method}, α={alpha:.0e}", fontsize=11)
        ax.set_xlabel("x(t)", fontsize=9)
        ax.set_ylabel("y(t)", fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0.5, 2.0)
        ax.set_ylim(0.5, 2.0)

        filtered = [
            d for d in data
            if np.isclose(d['gamma1'], gamma1)
            and np.isclose(d['gamma2'], gamma2)
            and np.isclose(d['alpha'], alpha)
            and d['method'] == method
        ]

        for d in filtered:
            x = d['x']
            y = d['y']
            ax.plot(x, y, label=f"α={alpha:.0e}", linewidth=1.5)
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
st.markdown("**Note:** The start point of the trajectory is marked with a circle (●), and the end point with a cross (×).")
