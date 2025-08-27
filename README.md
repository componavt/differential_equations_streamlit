# Differential Equations Streamlit App

This Streamlit web application visualizes solutions to a gene regulatory system of differential equations. The numerical solutions are precomputed and stored in a `.pkl` file and loaded by the app.

## 🌐 Purpose

The app is designed for researchers to explore the behavior of gene regulatory dynamics under varying parameters such as:

- **α (alpha)**: sensitivity parameter
- **γ₁, γ₂ (gamma1, gamma2)**: degradation or feedback coefficients
- **Initial conditions**: variations near equilibrium
- **Solver methods**: RK45, Radau, DOP853, Symplectic, etc.

# Gene Regulatory ODE System – Streamlit App  

Explore numerical solutions to a system of ODEs modeling gene regulation using different solvers and parameter settings.  

🔗 Live Demo [neuraldiffur.streamlit.app](https://neuraldiffur.streamlit.app/) for source code [differential_equations_streamlit](https://github.com/componavt/differential_equations_streamlit).

## 📁 Project Structure

gene_regulatory_ODE_system.py - Main file
└── plain_text_parameters.py - Parameters in plain text (copy / paste)

- **gene_regulatory_ODE_system.py** - Main application file with ODE solver and visualization
- **└── plain_text_parameters.py** - Utility module for parameter serialization/deserialization

## 🎯 Parameter Set Example

Use this parameter combination to observe specific dynamic behavior:

t_number=100; t_end=0.4; alpha=0.001; K=1.0; b=0.999; gamma1=0.73; gamma2=1.1; initial_radius=0.04; num_points=12
