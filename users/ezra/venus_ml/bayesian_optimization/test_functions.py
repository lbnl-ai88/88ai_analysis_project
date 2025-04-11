import re

import numpy as np

def generate_shifted_noisy_gaussian(dim=2, alpha=5.0, shift=None, noise=0.01,
                                    lower=-1.0, upper=1.0, random_state=None):
    """Generates a shifted Gaussian function and its bounds."""
    rng = np.random.default_rng(random_state)
    pbounds = {f"x{i}": (lower, upper) for i in range(dim)}
    if shift is None:
        shift = rng.uniform(lower, upper, size=dim)
    else:
        shift = np.asarray(shift)

    def func(**kwargs):
        x = np.array([kwargs[f"x{i}"] for i in range(dim)])
        val = np.exp(-alpha * np.sum((x - shift)**2))
        if noise > 0:
            val += rng.normal(0, noise)
        return val
    # Add a name attribute for easier filename generation
    func.__name__ = f"shifted_gaussian_d{dim}_a{alpha}_n{noise}"
    return func, pbounds

def generate_styblinski_tang(dim=3, lower=-5.0, upper=5.0):
    """Generates the negative Styblinski–Tang function and its bounds."""
    def styblinski_tang(**kwargs):
        x = np.array([kwargs[f"x{i}"] for i in range(dim)])
        return -0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
    pbounds = {f"x{i}": (lower, upper) for i in range(dim)}
    # Add a name attribute
    styblinski_tang.__name__ = f"styblinski_tang_d{dim}"
    return styblinski_tang, pbounds

def generate_narrow_peak_gaussian(
    dim=2,
    alpha=100.0,        # Controls sharpness (larger = narrower peak)
    amplitude=1.0,      # Height of the peak at its center
    shift=None,         # Location of the peak center [dim] array
    noise=0.0,          # Std dev of Gaussian noise added
    lower=-1.0,         # Lower bound for search space dimensions
    upper=1.0,          # Upper bound for search space dimensions
    random_state=None   # Seed for reproducibility (shift and noise)
):
    """
    Generates a potentially sharp Gaussian function and its bounds.

    The function is defined as:
        f(x) = amplitude * exp(-alpha * ||x - shift||^2) + N(0, noise^2)

    A larger `alpha` value leads to a significantly narrower peak, meaning
    the function value drops off very quickly as you move away from the 'shift'.

    Args:
      dim (int): Dimensionality of the input space.
      alpha (float): Controls the "sharpness" or "narrowness" of the Gaussian peak.
                     Larger values (e.g., 50, 100, 500+) create narrower peaks
                     with faster drop-offs.
      amplitude (float): The maximum value of the deterministic part of the
                         function, reached at the `shift` location.
      shift (array-like, optional): The center coordinates of the Gaussian peak.
                                    If None, chosen randomly within [lower, upper]^dim.
      noise (float): Standard deviation of Gaussian noise added to the function value.
      lower (float): Lower bound for each dimension's search space.
      upper (float): Upper bound for each dimension's search space.
      random_state (int or None): Seed for reproducible random shift (if shift=None)
                                  and noise generation.

    Returns:
      - func: A function f(**kwargs) usable by bayes_opt.
      - pbounds: A dict of param_name -> (lower_bound, upper_bound).
    """
    rng = np.random.default_rng(random_state)

    # Define the parameter bounds for each dimension
    pbounds = {f"x{i}": (lower, upper) for i in range(dim)}

    # If no shift provided, pick a random shift in [lower, upper]^dim
    if shift is None:
        shift_arr = rng.uniform(lower, upper, size=dim)
        # Store the generated shift for potential reference/debugging if needed
        # print(f"Generated random shift: {shift_arr}")
    else:
        shift_arr = np.asarray(shift)
        if shift_arr.shape != (dim,):
             raise ValueError(f"Provided shift has shape {shift_arr.shape}, expected ({dim},)")
        # Ensure the provided shift is within bounds (optional check)
        if np.any(shift_arr < lower) or np.any(shift_arr > upper):
             print(f"Warning: Provided shift {shift_arr} is outside bounds [{lower}, {upper}].")


    def narrow_peak_gaussian(**kwargs):
        # Construct the input vector from keyword arguments.
        # Ensure order matches dimension index (0, 1, 2, ...)
        x = np.array([kwargs[f"x{i}"] for i in range(dim)])

        # Calculate squared Euclidean distance from the peak center
        distance_sq = np.sum((x - shift_arr)**2)

        # Deterministic Gaussian "spike"
        val = amplitude * np.exp(-alpha * distance_sq)

        # Add noise
        if noise > 0:
            val += rng.normal(0, noise)
        return val

    # Add a descriptive name for easier identification (e.g., in filenames)
    # Sanitize alpha value for filename (replace '.')
    alpha_str = str(alpha).replace('.', 'p')
    func_name = f"narrowGauss_d{dim}_a{alpha_str}_amp{amplitude:.1f}_n{noise:.2f}"
    func_name = re.sub(r'[^\w.\-]+', '', func_name) # Basic sanitize
    narrow_peak_gaussian.__name__ = func_name

    return narrow_peak_gaussian, pbounds
