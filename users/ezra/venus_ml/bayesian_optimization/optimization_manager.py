import numpy as np
import matplotlib.pyplot as plt
import os
import re  # For sanitizing filenames
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement, UpperConfidenceBound  # Import specific acquisitions
from tqdm.notebook import tqdm  # Import tqdm for notebook progress bars
import copy # To deep copy acquisition function if needed

# --- Assume generator functions are defined elsewhere ---
# generate_styblinski_tang, generate_shifted_noisy_gaussian, etc.

class BayesOptimizer:
    """
    A class to perform Bayesian Optimization using the bayes_opt package,
    including early stopping, single runs, acquisition parameter sweeps,
    plotting, saving, and progress bars (tqdm). Allows providing a known true maximum.
    """

    def __init__(self, func, pbounds,
                 known_true_max=None, # New parameter
                 early_stopping_threshold=0.01, early_stopping_patience=5,
                 random_state=None, verbose=True, plot_dir=None):
        """
        Initializes the Bayesian Optimizer setup.

        Args:
            func (callable): The objective function to maximize.
            pbounds (dict): Parameter bounds {'param': (low, high), ...}.
            known_true_max (float, optional): If the true maximum value of the
                function is known, provide it here to bypass random sampling
                approximation and use it for comparisons/early stopping. Defaults to None.
            early_stopping_threshold (float): Stop if % diff from comparison max
                is below this threshold. Defaults to 0.01 (1%).
            early_stopping_patience (int): Number of consecutive iterations
                meeting the threshold to trigger stopping. Defaults to 5.
            random_state (int, optional): Seed for reproducibility of runs.
            verbose (bool): If True, enables print statements and progress bars.
                            Defaults to True.
            plot_dir (str, optional): Directory to save plots in. If None, defaults
                                      to 'toy_bayes'. Defaults to None.
        """
        self.plot_dir = plot_dir if plot_dir is not None else "toy_bayes"
        self.func = func
        self.pbounds = pbounds
        self.dim = len(pbounds)
        self.known_true_max = known_true_max # Store if provided
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose

        # --- Results Storage ---
        self.iteration_data = {}
        self.best_params_history = []
        self.best_value_history = []
        self.current_best_value = -np.inf
        self.current_best_params = None
        self.stopped_early = False
        # This will hold either the known_true_max or the sampled approx max
        self.comparison_max = known_true_max
        self.sampled_approx_max = None # Store sampled value separately if calculated
        self.total_iterations = 0
        self.last_run_acquisition_fn = None
        self.last_run_n_random = 0
        self.last_run_n_acq_max = 0
        self._last_optimizer_instance = None

    # --- (_sanitize_filename, _get_acq_info, _generate_filename, _ensure_dir_exists unchanged) ---
    def _sanitize_filename(self, name):
        """Removes potentially problematic characters for filenames."""
        name = str(name); name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w.\-]+', '', name); name = name.strip('._-')
        if not name or name in ['.', '..']: return "plot"
        return name

    def _get_acq_info(self, acq_fn):
        """Extracts name and key parameter from acquisition function."""
        if acq_fn is None: return "N/A", ""
        name = type(acq_fn).__name__; param_str = ""
        if hasattr(acq_fn, 'xi'): param_str = f"xi{acq_fn.xi:.2f}".replace('.', 'p')
        elif hasattr(acq_fn, 'kappa'): param_str = f"kappa{acq_fn.kappa:.2f}".replace('.', 'p')
        return name, param_str

    def _generate_filename(self, base_name=None, prefix="bayesopt", directory=None):
        """Generates a descriptive filename within the specified directory."""
        target_directory = directory if directory is not None else self.plot_dir
        if base_name:
            sanitized_base = self._sanitize_filename(base_name)
            filename = f"{sanitized_base}.png"
        else:
            try: func_name = self.func.__name__
            except AttributeError: func_name = "unknown_func"
            acq_name, acq_param = self._get_acq_info(self.last_run_acquisition_fn)
            parts = [self._sanitize_filename(prefix), self._sanitize_filename(func_name),
                     f"d{self.dim}", self._sanitize_filename(acq_name),
                     self._sanitize_filename(acq_param), f"r{self.last_run_n_random}",
                     f"a{self.last_run_n_acq_max}", f"i{self.total_iterations}",
                     f"seed{self.random_state}" if self.random_state is not None else "noseed"]
            parts = [p for p in parts if p]
            filename = "_".join(parts) + ".png"
        full_path = os.path.join(target_directory, filename)
        return full_path

    def _ensure_dir_exists(self, filename):
        """Checks if the directory for the filename exists, creates if not."""
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                if self.verbose: print(f"Created directory: {directory}")
            except OSError as e:
                if self.verbose: print(f"Error creating directory {directory}: {e}")
                return False
        return True

    def _approximate_function_max(self, n_samples=10_000, seed=None):
        """Approximates the true maximum via random sampling. Sets self.sampled_approx_max."""
        if self.verbose: print(f"Approximating true maximum using {n_samples} random samples...")
        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        max_val = -np.inf
        for _ in range(n_samples):
            random_point = {k: rng.uniform(low, high) for k, (low, high) in self.pbounds.items()}
            try:
                val = self.func(**random_point)
                if val > max_val: max_val = val
            except Exception as e:
                if self.verbose: print(f"Warning: Error evaluating function during approximation: {e}")
                continue
        if np.isinf(max_val):
            if self.verbose: print("Warning: Approximation of true maximum failed or yielded -inf.")
            self.sampled_approx_max = None # Indicate failure
        else:
            self.sampled_approx_max = max_val # Store the sampled value
            if self.verbose: print(f"Sampled Approx Max: {self.sampled_approx_max:.6f}")
        # Do not overwrite self.comparison_max here if it was set by known_true_max

    def _run_single_optimization(self, acquisition_fn, n_random, n_acq_max, run_seed):
        """Internal method to run one optimization sequence with progress bars."""
        local_iteration_data = {}
        local_best_params_history = []
        local_best_value_history = []
        local_current_best_value = -np.inf
        local_current_best_params = None
        local_stopped_early = False
        local_total_iterations = 0

        optimizer = BayesianOptimization(
            f=None, acquisition_function=acquisition_fn, pbounds=self.pbounds,
            verbose=0, random_state=run_seed
        )
        rng_run = np.random.default_rng(run_seed)

        # --- Random Burn-in Phase ---
        random_iterator = range(n_random)
        if self.verbose: random_iterator = tqdm(random_iterator, desc="Random Exploration", leave=False)
        for i in random_iterator:
            random_point = {k: rng_run.uniform(low, high) for k, (low, high) in self.pbounds.items()}
            try: y_val = self.func(**random_point)
            except Exception as e:
                 if self.verbose: print(f"\nWarning: Error evaluating function at random point {i}: {e}")
                 continue
            optimizer.register(params=random_point, target=y_val)
            if y_val > local_current_best_value: local_current_best_value, local_current_best_params = y_val, random_point
            try:
                gp = optimizer._gp; X_pred = np.array([list(random_point.values())])
                mu, sigma = gp.predict(X_pred, return_std=True); mu, sigma = np.atleast_1d(mu)[0], np.atleast_1d(sigma)[0]
            except (AttributeError, Exception): mu, sigma = None, None
            local_iteration_data[local_total_iterations] = {'x': list(random_point.values()), 'y': y_val, 'mu': mu, 'sigma': sigma, 'random': True}
            local_best_value_history.append(local_current_best_value); local_best_params_history.append(local_current_best_params)
            local_total_iterations += 1

        # --- Acquisition-driven Phase ---
        consecutive_threshold_met = 0
        acq_iterator = range(n_acq_max)
        if self.verbose: acq_iterator = tqdm(acq_iterator, desc="Acquisition Phase ", leave=False)
        for j in acq_iterator:
            try: next_point = optimizer.suggest()
            except IndexError:
                 if self.verbose: print(f"\nWarning: IndexError during suggestion (iteration {local_total_iterations}). Stopping acquisition.")
                 break
            except Exception as e:
                 if self.verbose: print(f"\nError during suggestion (iteration {local_total_iterations}): {e}")
                 break
            try: y_val = self.func(**next_point)
            except Exception as e:
                 if self.verbose: print(f"\nWarning: Error evaluating function at suggested point {j}: {e}")
                 continue
            optimizer.register(params=next_point, target=y_val)
            if y_val > local_current_best_value: local_current_best_value, local_current_best_params = y_val, next_point
            try:
                gp = optimizer._gp; X_pred = np.array([list(next_point.values())])
                mu, sigma = gp.predict(X_pred, return_std=True); mu, sigma = np.atleast_1d(mu)[0], np.atleast_1d(sigma)[0]
            except Exception as e:
                 if self.verbose: print(f"\nWarning: GP prediction failed at iteration {local_total_iterations}: {e}")
                 mu, sigma = None, None
            local_iteration_data[local_total_iterations] = {'x': list(next_point.values()), 'y': y_val, 'mu': mu, 'sigma': sigma, 'random': False}
            local_best_value_history.append(local_current_best_value); local_best_params_history.append(local_current_best_params)
            local_total_iterations += 1

            # --- Early Stopping Check (using self.comparison_max) ---
            # Ensure comparison_max is valid before checking
            if self.comparison_max is not None and not np.isinf(self.comparison_max):
                denom = abs(self.comparison_max)
                diff_percentage = abs(self.comparison_max - local_current_best_value) / denom if denom > 1e-9 else abs(self.comparison_max - local_current_best_value)
                if diff_percentage < self.early_stopping_threshold: consecutive_threshold_met += 1
                else: consecutive_threshold_met = 0
                if consecutive_threshold_met >= self.early_stopping_patience:
                    local_stopped_early = True
                    if self.verbose and isinstance(acq_iterator, tqdm): acq_iterator.set_description("Acquisition Phase (Stopped Early)", refresh=True)
                    break
            # --- End Early Stopping Check ---

        return (local_iteration_data, local_best_value_history, local_best_params_history,
                local_stopped_early, local_total_iterations, optimizer)


    def run_experiment(self, acquisition_fn, n_random=5, n_acq_max=50,
                       approx_max_samples=10_000, plot_summary=True, verbose=None,
                       save_plot=False, save_filename=None):
        """Runs a single Bayesian optimization experiment with progress bars."""
        effective_verbose = verbose if verbose is not None else self.verbose
        original_verbose = self.verbose; self.verbose = effective_verbose

        # --- Reset state ---
        self.iteration_data = {}; self.best_params_history = []; self.best_value_history = []
        self.current_best_value = -np.inf; self.current_best_params = None
        self.stopped_early = False; self.total_iterations = 0
        self.last_run_acquisition_fn = acquisition_fn
        self.last_run_n_random = n_random; self.last_run_n_acq_max = n_acq_max
        self._last_optimizer_instance = None
        self.sampled_approx_max = None # Reset sampled max for this run

        # --- Set Comparison Maximum ---
        if self.comparison_max is None: # Only approximate if not provided via known_true_max
            self._approximate_function_max(n_samples=approx_max_samples, seed=self.random_state)
            # If sampling failed, we cannot proceed with comparison/early stopping
            if self.sampled_approx_max is None:
                 if effective_verbose: print("Approximation failed. Cannot proceed with comparison/early stopping.")
                 # Decide how to handle: maybe disable early stopping? For now, stop run.
                 self.verbose = original_verbose; return {}
            else:
                 # Use the sampled value for comparison in this run
                 self.comparison_max = self.sampled_approx_max
        elif effective_verbose:
             # comparison_max was already set from known_true_max
             print(f"Using known true maximum for comparison: {self.comparison_max:.6f}")

        # --- Run Optimization ---
        if effective_verbose:
            print(f"\nStarting single optimization run: {n_random} random, max {n_acq_max} acquisition steps.")
            acq_name, acq_param = self._get_acq_info(acquisition_fn)
            print(f"Using Acquisition Function: {acq_name} ({acq_param})")

        run_results = self._run_single_optimization(
            acquisition_fn=acquisition_fn, n_random=n_random,
            n_acq_max=n_acq_max, run_seed=self.random_state
        )
        (self.iteration_data, self.best_value_history, self.best_params_history,
         self.stopped_early, self.total_iterations, self._last_optimizer_instance) = run_results

        if self.best_value_history: self.current_best_value, self.current_best_params = self.best_value_history[-1], self.best_params_history[-1]
        else: self.current_best_value, self.current_best_params = -np.inf, None

        # --- Final Report ---
        if effective_verbose:
            print("\n--- Single Run Finished ---")
            status = f"stopped early after {self.total_iterations}" if self.stopped_early else f"finished after {self.total_iterations}"
            print(f"Optimization {status} iterations.")
            if self.current_best_params is not None:
                print(f"Best value found (Observed): {self.current_best_value:.6f}")
                if self._last_optimizer_instance and hasattr(self._last_optimizer_instance, 'max') and self._last_optimizer_instance.max:
                     gp_max_val = self._last_optimizer_instance.max.get('target', 'N/A')
                     gp_max_val_str = f"{gp_max_val:.6f}" if isinstance(gp_max_val, (float, np.float_)) else str(gp_max_val)
                     print(f"Best value found (GP Max):   {gp_max_val_str}")
                print(f"Best parameters found: {self.current_best_params}")
                # Report difference based on the comparison max used (known or sampled)
                comparison_type = "Known True Max" if self.known_true_max is not None else "Sampled Approx Max"
                if self.comparison_max is not None:
                    final_diff = abs(self.comparison_max - self.current_best_value)
                    final_diff_perc = (final_diff / abs(self.comparison_max) * 100 if abs(self.comparison_max) > 1e-9 else final_diff * 100)
                    print(f"Difference from {comparison_type} ({self.comparison_max:.6f}): {final_diff:.6f} ({final_diff_perc:.4f}%)")
                else:
                    print("Comparison maximum not available for difference calculation.")
            else: print("No valid points evaluated.")
            print("-" * 27)

        # --- Plotting ---
        if plot_summary and self.iteration_data:
            self.plot_summary(save_plot=save_plot, save_filename=save_filename)

        # --- Reset comparison_max if it came from sampling ---
        if self.known_true_max is None:
            self.comparison_max = None

        self.verbose = original_verbose # Restore original verbosity
        return self.iteration_data

    # --- Static Aggregation Methods (_aggregate_iteration_data, _aggregate_best_history unchanged) ---
    @staticmethod
    def _aggregate_iteration_data(data_list, verbose=True):
        """Averages iteration data across multiple runs."""
        if not data_list: return {}
        valid_data_list = [d for d in data_list if d]
        if not valid_data_list:
             if verbose: print("Warning: All runs in data_list are empty or None. Aggregation failed.")
             return {}
        if len(valid_data_list) < len(data_list) and verbose:
             print(f"Warning: {len(data_list) - len(valid_data_list)} runs were empty/None. Aggregating over {len(valid_data_list)} runs.")
        data_list = valid_data_list
        reference_iterations = sorted(data_list[0].keys())
        num_iterations_ref = len(reference_iterations)
        num_runs = len(data_list)
        aggregated = {}; min_len = num_iterations_ref; max_len = num_iterations_ref
        for i, d in enumerate(data_list):
            run_len = len(d); min_len = min(min_len, run_len); max_len = max(max_len, run_len)
            if run_len != num_iterations_ref and verbose: print(f"Warning: Run {i} has {run_len} iterations, expected {num_iterations_ref}.")
        if min_len != max_len and verbose:
             print(f"Aggregating up to iteration {min_len - 1} (minimum run length).")
             reference_iterations = sorted(list(range(min_len)))
        dim = len(list(data_list[0][0]['x']))
        for i in reference_iterations:
            xs, ys, mus, sigmas, random_flags = [], [], [], [], []
            for run_idx, d in enumerate(data_list):
                if i in d:
                    data_point = d[i]; xs.append(np.array(data_point['x'])); ys.append(data_point['y'])
                    mus.append(data_point['mu'] if data_point['mu'] is not None else np.nan)
                    sigmas.append(data_point['sigma'] if data_point['sigma'] is not None else np.nan)
                    random_flags.append(data_point['random'])
                else: xs.append(np.full(dim, np.nan)); ys.append(np.nan); mus.append(np.nan); sigmas.append(np.nan); random_flags.append(np.nan)
            valid_xs = [x for x in xs if not np.isnan(x).all()]
            xs_avg = np.nanmean(np.stack(valid_xs, axis=0), axis=0).tolist() if valid_xs else np.full(dim, np.nan).tolist()
            aggregated[i] = {'x': xs_avg, 'y': np.nanmean(ys), 'mu': np.nanmean(mus), 'sigma': np.nanmean(sigmas), 'random': np.nanmean(random_flags) > 0.5}
        return aggregated

    @staticmethod
    def _aggregate_best_history(history_list):
        """Averages the best-value-so-far history across runs."""
        valid_histories = [h for h in history_list if h];
        if not valid_histories: return []
        min_len = min(len(h) for h in valid_histories);
        if min_len == 0: return []
        stacked_histories = np.array([h[:min_len] for h in valid_histories])
        avg_history = np.nanmean(stacked_histories, axis=0)
        return avg_history.tolist()

    # --- run_acquisition_sweep (only needs minor change for comparison_max handling) ---
    def run_acquisition_sweep(self, acq_param_dict, acquisition_fn_constructor,
                              n_random=5, n_acq_max=50, num_runs=3,
                              approx_max_samples=10_000,
                              title="Acquisition Parameter Sweep", verbose=None,
                              save_plot=False, save_filename=None):
        effective_verbose = verbose if verbose is not None else self.verbose
        original_verbose = self.verbose; self.verbose = effective_verbose
        if len(acq_param_dict) != 1: raise ValueError("acq_param_dict must contain exactly one key.")
        param_name, param_values = list(acq_param_dict.items())[0]
        num_params = len(param_values)

        # --- Set Comparison Maximum (Once for the sweep) ---
        # Store original comparison_max in case it was known
        original_comparison_max = self.comparison_max
        if self.comparison_max is None: # Only approximate if not provided
            self._approximate_function_max(n_samples=approx_max_samples, seed=self.random_state)
            if self.sampled_approx_max is None:
                 if effective_verbose: print("Approximation failed. Cannot proceed with sweep.")
                 self.verbose = original_verbose; return {}
            else:
                 self.comparison_max = self.sampled_approx_max # Use sampled for the sweep
        elif effective_verbose:
             print(f"Using known true maximum for comparison: {self.comparison_max:.6f}")

        # --- Proceed with sweep logic ---
        fig, axs = plt.subplots(num_params, 2, figsize=(16, 6 * num_params), squeeze=False)
        sweep_results = {}
        for i, param_value in enumerate(param_values):
            if effective_verbose: print(f"\n--- Running Sweep: {param_name} = {param_value} ({num_runs} runs) ---")
            run_iteration_data_list, run_best_value_history_list = [], []
            run_iterator = range(num_runs)
            if effective_verbose: run_iterator = tqdm(run_iterator, desc=f"Runs ({param_name}={param_value})", leave=False)
            for r in run_iterator:
                run_seed = self.random_state + r if self.random_state is not None else None
                try: acq_fn = acquisition_fn_constructor(param_value)
                except Exception as e:
                    if effective_verbose: print(f"\nError creating acquisition function: {e}. Skipping runs.")
                    run_iteration_data_list.append(None); run_best_value_history_list.append(None)
                    continue
                # Ignore optimizer instance from tuple
                run_results_tuple = self._run_single_optimization(acq_fn, n_random, n_acq_max, run_seed)
                run_iteration_data_list.append(run_results_tuple[0]); run_best_value_history_list.append(run_results_tuple[1])
            if effective_verbose: print(f"Aggregating results for {param_name} = {param_value}...")
            agg_data = self._aggregate_iteration_data(run_iteration_data_list, verbose=effective_verbose)
            agg_best_history = self._aggregate_best_history(run_best_value_history_list)
            sweep_results[param_value] = {'agg_iteration_data': agg_data, 'agg_best_value_history': agg_best_history}
            if not agg_data:
                 if effective_verbose: print(f"No aggregated data to plot for {param_name}={param_value}.")
                 axs[i, 0].set_title(f"{param_name}={param_value} (No Data)"); axs[i, 1].set_title(f"{param_name}={param_value} (No Data)")
                 axs[i, 0].axis('off'); axs[i, 1].axis('off')
                 continue
            # Pass the comparison_max determined for the sweep to plotting
            self._plot_feature_evolution_data(agg_data, axs[i, 0], title_suffix=f"({param_name}={param_value}, Avg over {num_runs} runs)")
            self._plot_performance_vs_max_data(agg_best_history, self.comparison_max, axs[i, 1], title_suffix=f"({param_name}={param_value}, Avg over {num_runs} runs)")

        # --- Final Plot Adjustments and Saving ---
        fig.suptitle(title, fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        if save_plot:
            if save_filename is None:
                try: func_name = self.func.__name__
                except AttributeError: func_name = "unknown_func"
                param_vals_str = "-".join(self._sanitize_filename(str(pv)) for pv in param_values)
                base_fname = f"sweep_{func_name}_d{self.dim}_{param_name}_{param_vals_str}_r{n_random}_a{n_acq_max}_runs{num_runs}"
                full_filename = self._generate_filename(base_name=base_fname, directory=self.plot_dir)
            else: full_filename = self._generate_filename(base_name=save_filename, directory=self.plot_dir)
            if self._ensure_dir_exists(full_filename):
                try:
                    fig.savefig(full_filename, bbox_inches='tight', dpi=150)
                    if effective_verbose: print(f"Sweep plot saved to {full_filename}")
                except Exception as e:
                    if effective_verbose: print(f"Error saving sweep plot to {full_filename}: {e}")
            else:
                 if effective_verbose: print(f"Could not save plot, directory creation failed for {full_filename}")
        plt.show()

        # --- Restore original comparison_max if it was known ---
        self.comparison_max = original_comparison_max
        # Reset sampled max just in case
        self.sampled_approx_max = None

        self.verbose = original_verbose
        return sweep_results


    # --- Plotting Methods ---
    # (_plot_feature_evolution_data, plot_feature_evolution unchanged)
    def _plot_feature_evolution_data(self, iteration_data, ax, title_suffix=""):
        """Plots feature evolution using provided data and axes."""
        if not iteration_data: return
        all_iterations = sorted(iteration_data.keys());
        if not all_iterations: return
        X = np.array([iteration_data[i]['x'] for i in all_iterations])
        if np.isnan(X).all():
            if self.verbose: ax.text(0.5, 0.5, 'No valid feature data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Feature Evolution {title_suffix}"); return
        n_iters, dim = X.shape
        with np.errstate(invalid='ignore'): mins, maxs = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        ranges = maxs - mins; ranges[ranges < 1e-9] = 1.0; X_scaled = (X - mins) / ranges
        switch_index = -1
        for i in all_iterations:
            if 'random' in iteration_data[i] and not iteration_data[i]['random']: switch_index = i; break
        for d in range(dim): ax.scatter(all_iterations, X_scaled[:, d], label=f"x{d}", s=10, alpha=0.7)
        if switch_index != -1 and switch_index > 0: ax.axvline(switch_index - 0.5, color='black', linestyle='--', label="Switch")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Scaled Feature Value"); ax.set_title(f"Feature Evolution {title_suffix}")
        ax.legend(fontsize='small', loc='best'); ax.grid(True, axis='y', linestyle=':')
        ax.set_ylim(-0.1, 1.1)

    def plot_feature_evolution(self, ax=None):
        """Plots feature evolution for the last run stored in `self`."""
        if not self.iteration_data:
            if self.verbose: print("No iteration data available from the last run.")
            return
        if ax is None: fig, ax = plt.subplots(figsize=(10, 5)); show_plot = True
        else: show_plot = False
        self._plot_feature_evolution_data(self.iteration_data, ax, title_suffix="(Last Run)")
        if show_plot: plt.tight_layout(); plt.show()

    # Modify performance plot to use comparison_max
    def _plot_performance_vs_max_data(self, best_value_history, comparison_max, ax, title_suffix=""):
        """Plots performance vs the comparison maximum (known or sampled)."""
        if not best_value_history or comparison_max is None or np.isinf(comparison_max):
            if self.verbose: ax.text(0.5, 0.5, 'Performance data unavailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Performance vs Comparison Max {title_suffix}"); return
        all_iterations = range(len(best_value_history)); perc_diffs = []
        denom = abs(comparison_max); use_abs_diff = denom < 1e-9
        for best_val_so_far in best_value_history:
            if np.isnan(best_val_so_far): perc_diffs.append(np.nan); continue
            if use_abs_diff: diff = abs(comparison_max - best_val_so_far); perc = diff * 100
            else: diff = comparison_max - best_val_so_far; perc = max(0, diff) / denom * 100
            perc_diffs.append(perc)
        ax.plot(all_iterations, perc_diffs, marker='.', linestyle='-', color='red')
        ax.set_xlabel("Iteration"); ax.set_ylabel("Difference from Comparison Max (%)")
        # Update title to use the actual comparison value
        ax.set_title(f"Performance vs Comparison Max ({comparison_max:.4f}) {title_suffix}")
        ax.set_ylim(-0.1, 101)
        ax.grid(True, linestyle=':')

    def plot_performance_vs_max(self, ax=None):
        """Plots performance vs comparison max for the last run stored in `self`."""
        if not self.best_value_history or self.comparison_max is None:
            if self.verbose: print("Performance data not available from the last run.")
            return
        if ax is None: fig, ax = plt.subplots(figsize=(10, 5)); show_plot = True
        else: show_plot = False
        # Pass self.comparison_max
        self._plot_performance_vs_max_data(self.best_value_history, self.comparison_max, ax, title_suffix="(Last Run)")
        if show_plot: plt.tight_layout(); plt.show()


    def plot_summary(self, save_plot=False, save_filename=None):
        """
        Generates a 1x2 summary plot for the last single run, including max value comparison.
        """
        if not self.iteration_data:
            if self.verbose: print("No data available from the last run to generate summary plot.")
            return

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        self._plot_feature_evolution_data(self.iteration_data, axs[0], title_suffix="(Last Run)")
        # Pass self.comparison_max to the plotting helper
        self._plot_performance_vs_max_data(self.best_value_history, self.comparison_max, axs[1],
                                           title_suffix="(Last Run)")

        # --- Add Max Value Comparison Text ---
        gp_max_val_str = "N/A"
        if self._last_optimizer_instance and hasattr(self._last_optimizer_instance, 'max') and self._last_optimizer_instance.max:
             gp_max_val = self._last_optimizer_instance.max.get('target', 'N/A')
             if isinstance(gp_max_val, (float, np.float_)): gp_max_val_str = f"{gp_max_val:.4f}"

        # Determine how the comparison max was obtained for labeling
        if self.known_true_max is not None:
            comparison_label = "Known True Max:"
            comparison_val_str = f"{self.comparison_max:.4f}" if self.comparison_max is not None else "N/A"
            # Don't display sampled max if known max was used
            sampled_max_line = ""
        elif self.sampled_approx_max is not None: # Sampled value exists and was used
            comparison_label = "Sampled Approx Max:"
            comparison_val_str = f"{self.comparison_max:.4f}" # comparison_max holds the sampled value here
            sampled_max_line = f"{comparison_label:<20} {comparison_val_str}\n"
        else: # Should not happen if run completed, but handle defensively
            comparison_label = "Comparison Max:"
            comparison_val_str = "N/A"
            sampled_max_line = f"{comparison_label:<20} {comparison_val_str}\n"

        observed_max_str = f"{self.current_best_value:.4f}" if self.current_best_value > -np.inf else "N/A"

        # Construct text, only including sampled line if relevant
        max_info_text = ""
        if self.known_true_max is not None:
             max_info_text += f"{comparison_label:<20} {comparison_val_str}\n"
        elif self.sampled_approx_max is not None:
             max_info_text += sampled_max_line # Already includes label and value

        max_info_text += (f"{'Observed Best Value:':<20} {observed_max_str}\n"
                          f"{'GP Estimated Max:':<20} {gp_max_val_str}")

        plt.figtext(0.5, 0.97, max_info_text, ha="center", va="top", fontsize=9,
                    bbox={"facecolor":"white", "alpha":0.7, "pad":3},
                    family='monospace') # Use monospace for alignment
        # --- End Text Addition ---

        acq_name, acq_param = self._get_acq_info(self.last_run_acquisition_fn)
        fig.suptitle(f"Bayesian Optimization Summary ({self.total_iterations} Iterations, Acq: {acq_name} {acq_param})",
                     fontsize=16, y=1.03) # Slightly raise title
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout

        if save_plot:
            full_filename = self._generate_filename(
                base_name=save_filename, prefix="bayesopt_summary", directory=self.plot_dir
            )
            if self._ensure_dir_exists(full_filename):
                try:
                    fig.savefig(full_filename, bbox_inches='tight', dpi=150)
                    if self.verbose: print(f"Summary plot saved to {full_filename}")
                except Exception as e:
                    if self.verbose: print(f"Error saving summary plot to {full_filename}: {e}")
            else:
                 if self.verbose: print(f"Could not save plot, directory creation failed for {full_filename}")

        plt.show()
