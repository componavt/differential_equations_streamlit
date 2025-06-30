# Differential Equations Streamlit App

This Streamlit web application visualizes solutions to a gene regulatory system of differential equations. The numerical solutions are precomputed and stored in a `.pkl` file and loaded by the app.

## 🌐 Purpose

The app is designed for researchers to explore the behavior of gene regulatory dynamics under varying parameters such as:

- **α (alpha)**: sensitivity parameter
- **γ₁, γ₂ (gamma1, gamma2)**: degradation or feedback coefficients
- **Initial conditions**: variations near equilibrium
- **Solver methods**: RK45, Radau, DOP853, Symplectic, etc.
