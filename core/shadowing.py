import numpy as np


# Constants for shadowing analysis
SHADOWING_EPSILON_THRESHOLD = 1e-3


def compute_epsilon_t(distances):
    """
    Compute epsilon(t) as the running maximum of distances.
    
    Args:
        distances: Array of distances between two trajectories at each time point
        
    Returns:
        Array representing epsilon(t) - the running maximum of distances
    """
    return np.maximum.accumulate(distances)


def find_shadowing_breakdown_time(epsilon_t, t_eval, epsilon_threshold=SHADOWING_EPSILON_THRESHOLD):
    """
    Find the shadowing breakdown time t* where epsilon(t) exceeds a threshold.
    
    Args:
        epsilon_t: Array of epsilon(t) values (running maximum of distances)
        t_eval: Time points corresponding to epsilon(t) values
        epsilon_threshold: Threshold above which shadowing is considered broken
        
    Returns:
        tuple: (shadowing_time, shadowing_length, shadowing_ratio)
               shadowing_time: Time t* where epsilon(t) > epsilon_threshold (or None if never exceeded)
               shadowing_length: Length of time where epsilon(t) <= epsilon_threshold
               shadowing_ratio: Ratio of valid shadowing time to total time
    """
    if len(epsilon_t) != len(t_eval):
        raise ValueError("epsilon_t and t_eval must have the same length")
    
    # Find the first time where epsilon(t) exceeds the threshold
    exceed_indices = np.where(epsilon_t > epsilon_threshold)[0]
    
    if len(exceed_indices) == 0:
        # If threshold is never exceeded, shadowing holds for the entire duration
        shadowing_time = None  # Indicates no breakdown occurred
        shadowing_length = t_eval[-1] - t_eval[0]
        shadowing_ratio = 1.0
    else:
        # Take the first occurrence where threshold is exceeded
        first_exceed_idx = exceed_indices[0]
        shadowing_time = t_eval[first_exceed_idx]
        shadowing_length = t_eval[first_exceed_idx] - t_eval[0]
        shadowing_ratio = shadowing_length / (t_eval[-1] - t_eval[0])
    
    return shadowing_time, shadowing_length, shadowing_ratio


def compute_shadowing_diagnostics(dop_solution, nn_solution, t_eval, epsilon_threshold=SHADOWING_EPSILON_THRESHOLD):
    """
    Compute comprehensive shadowing diagnostics by comparing two solutions.
    
    Args:
        dop_solution: Dictionary with 'x' and 'y' arrays from DOP853 solver
        nn_solution: Dictionary with 'x' and 'y' arrays from neural network solver
        t_eval: Time points array
        epsilon_threshold: Threshold for shadowing breakdown
        
    Returns:
        dict: Dictionary containing shadowing diagnostics
    """
    # Calculate distance between DOP853 and NN solutions
    dist = np.sqrt((dop_solution['x'] - nn_solution['x'])**2 + (dop_solution['y'] - nn_solution['y'])**2)
    
    # Calculate epsilon(t) as the running maximum (sup norm)
    epsilon_t = compute_epsilon_t(dist)
    
    # Calculate shadowing breakdown diagnostics
    shadowing_time, shadowing_length, shadowing_ratio = find_shadowing_breakdown_time(
        epsilon_t, t_eval, epsilon_threshold
    )
    
    return {
        'epsilon_t': epsilon_t,
        'distances': dist,
        'shadowing_time': shadowing_time,
        'shadowing_length': shadowing_length,
        'shadowing_ratio': shadowing_ratio,
        'epsilon_threshold': epsilon_threshold,
        'has_breakdown': shadowing_time is not None
    }