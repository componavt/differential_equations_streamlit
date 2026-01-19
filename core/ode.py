import numpy as np


def gene_regulatory_rhs(alpha_val, K_val, b_val, g1_val, g2_val):
    """
    Create the right-hand side function for the gene regulatory ODE system.
    
    The system is:
    dx/dt = (K * x^(1/alpha))/(b^(1/alpha) + x^(1/alpha)) - gamma1 * x
    dy/dt = (K * y^(1/alpha))/(b^(1/alpha) + y^(1/alpha)) - gamma2 * y
    """
    def rhs(t, state):
        x, y = state
        n = 1.0 / alpha_val
        if n > 1000:
            # Handle very large n (approaching infinity) case
            frac_x = K_val if x > b_val else 0.0
            frac_y = K_val if y > b_val else 0.0
        else:
            x_pos, y_pos = max(x, 0.0), max(y, 0.0)
            try:
                pow_b = np.power(b_val, n)
                pow_x = np.power(x_pos, n)
                pow_y = np.power(y_pos, n)
                frac_x = (K_val * pow_x) / (pow_b + pow_x) if np.isfinite(pow_x) else (K_val if x > b_val else 0.0)
                frac_y = (K_val * pow_y) / (pow_b + pow_y) if np.isfinite(pow_y) else (K_val if y > b_val else 0.0)
            except Exception:
                frac_x, frac_y = (K_val if x > b_val else 0.0), (K_val if y > b_val else 0.0)
        return [frac_x - g1_val * x, frac_y - g2_val * y]
    
    return rhs


def lotka_volterra_rhs(a, b, k, m):
    """
    Create the right-hand side function for the Lotka-Volterra ODE system.
    
    The system is:
    dx/dt = x * (a - b * y)
    dy/dt = y * (k * b * x - m)
    """
    def rhs(t, state):
        x, y = state
        return [x * (a - b * y), y * (k * b * x - m)]
    
    return rhs