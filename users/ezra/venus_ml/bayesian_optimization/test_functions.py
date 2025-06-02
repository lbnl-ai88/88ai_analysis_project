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

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# --- Assume BayesOptimizer class and other generators exist ---

def generate_multi_peak_gaussian(
    dim=2,
    n_peaks=5,          # Total number of peaks (including the global one)
    amplitude_global=1.0, # Height of the global peak
    amplitude_local=0.6,  # Height of the local peaks
    alpha_global=50.0,    # Sharpness of the global peak (larger = narrower)
    alpha_local=30.0,     # Sharpness of the local peaks
    shifts=None,          # List/array of peak center locations [(n_peaks, dim)]
                          # If None, generated randomly. First shift is global.
    min_peak_distance=0.5,# Minimum distance desired between generated peaks (for warning)
    noise=0.0,            # Std dev of Gaussian noise added
    lower=-2.0,           # Lower bound for search space dimensions
    upper=2.0,            # Upper bound for search space dimensions
    random_state=None     # Seed for reproducibility (shifts and noise)
):
    """
    Generates a function landscape composed of multiple Gaussian peaks,
    with one designated global maximum.

    The function is a sum of Gaussians:
        f(x) = sum_{i=1}^{n_peaks} amplitude_i * exp(-alpha_i * ||x - shift_i||^2) + N(0, noise^2)
    where one peak (index 0 by default if shifts are random) has amplitude_global
    and alpha_global, while others have amplitude_local and alpha_local.

    Args:
      dim (int): Dimensionality of the input space.
      n_peaks (int): Total number of Gaussian peaks.
      amplitude_global (float): Amplitude (height) of the single global maximum peak.
      amplitude_local (float): Amplitude (height) of the other local maximum peaks.
                                Should be less than amplitude_global.
      alpha_global (float): Sharpness parameter for the global peak.
      alpha_local (float): Sharpness parameter for the local peaks.
      shifts (array-like, optional): A list or array of shape (n_peaks, dim)
          specifying the exact center coordinates for each peak. The first
          entry (index 0) is treated as the global peak. If None, shifts
          are generated randomly within the bounds.
      min_peak_distance (float): If generating random shifts, a warning is printed
          if any two generated peaks are closer than this distance.
      noise (float): Standard deviation of Gaussian noise added.
      lower (float): Lower bound for each dimension's search space.
      upper (float): Upper bound for each dimension's search space.
      random_state (int or None): Seed for reproducible random shifts (if shifts=None)
                                  and noise generation.

    Returns:
      - func: A function f(**kwargs) usable by bayes_opt.
      - pbounds: A dict of param_name -> (lower_bound, upper_bound).
      - peak_info: A dictionary containing the actual shifts, amplitudes, and alphas used.
                   {'shifts': array(n_peaks, dim), 'amplitudes': array(n_peaks), 'alphas': array(n_peaks)}
    """
    if amplitude_local >= amplitude_global:
        print("Warning: amplitude_local should be less than amplitude_global for a unique global max.")

    rng = np.random.default_rng(random_state)

    # Define parameter bounds
    pbounds = {f"x{i}": (lower, upper) for i in range(dim)}

    # Determine peak locations
    if shifts is not None:
        shifts_arr = np.asarray(shifts)
        if shifts_arr.shape != (n_peaks, dim):
            raise ValueError(f"Provided shifts have shape {shifts_arr.shape}, expected ({n_peaks}, {dim})")
        # Ensure provided shifts are within bounds (optional check)
        if np.any(shifts_arr < lower) or np.any(shifts_arr > upper):
             print(f"Warning: Some provided shifts are outside bounds [{lower}, {upper}].")
    else:
        # Generate random shifts
        shifts_arr = rng.uniform(lower, upper, size=(n_peaks, dim))
        # Check minimum distance between generated peaks
        if n_peaks > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(shifts_arr)
            min_dist_found = np.min(distances) if distances.size > 0 else np.inf
            if min_dist_found < min_peak_distance:
                print(f"Warning: Minimum distance between generated peaks ({min_dist_found:.3f}) "
                      f"is less than the desired minimum ({min_peak_distance:.3f}). Peaks might be clustered.")

    # Assign amplitudes and alphas (global peak is index 0)
    amplitudes_arr = np.full(n_peaks, amplitude_local)
    alphas_arr = np.full(n_peaks, alpha_local)
    amplitudes_arr[0] = amplitude_global
    alphas_arr[0] = alpha_global

    peak_info = {
        'shifts': shifts_arr,
        'amplitudes': amplitudes_arr,
        'alphas': alphas_arr,
        'global_peak_index': 0 # By convention
    }

    def multi_peak_gaussian(**kwargs):
        x = np.array([kwargs[f"x{i}"] for i in range(dim)])
        total_value = 0.0
        for i in range(n_peaks):
            distance_sq = np.sum((x - shifts_arr[i])**2)
            peak_val = amplitudes_arr[i] * np.exp(-alphas_arr[i] * distance_sq)
            total_value += peak_val

        if noise > 0:
            total_value += rng.normal(0, noise)
        return total_value

    # Add descriptive name
    amp_g_str = f"{amplitude_global:.1f}".replace('.', 'p')
    amp_l_str = f"{amplitude_local:.1f}".replace('.', 'p')
    alpha_g_str = f"{alpha_global:.1f}".replace('.', 'p')
    alpha_l_str = f"{alpha_local:.1f}".replace('.', 'p')
    func_name = f"multiPeakGauss_d{dim}_p{n_peaks}_ag{amp_g_str}_al{amp_l_str}_alphag{alpha_g_str}_alphal{alpha_l_str}_n{noise:.2f}"
    func_name = re.sub(r'[^\w.\-]+', '', func_name) # Basic sanitize
    multi_peak_gaussian.__name__ = func_name

    return multi_peak_gaussian, pbounds, peak_info
