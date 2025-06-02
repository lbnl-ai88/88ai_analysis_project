import numpy as np
import matplotlib.pyplot as plt
import os
import re  # For sanitizing filenames
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement, UpperConfidenceBound  # Import specific acquisitions
from tqdm.notebook import tqdm  # Import tqdm for notebook progress bars
import copy # To deep copy acquisition function if needed
import math # For ceil

# --- Assume generator functions are defined elsewhere ---
# generate_styblinski_tang, generate_shifted_noisy_gaussian, etc.

class BayesOptimizer:
    """
    A class to perform Bayesian Optimization using the bayes_opt package,
    including early stopping, single runs, acquisition parameter sweeps,
    plotting, saving, and progress bars (tqdm). Allows providing a known true maximum.
    """

    def __init__(self, func, pbounds,
                 known_true_max=None,
                 early_stopping_threshold=0.01, early_stopping_patience=5,
                 random_state=None, verbose=True, plot_dir=None):
        """
        Initializes the Bayesian Optimizer setup.
        (Constructor code remains the same as your provided version)
        """
        self.plot_dir = plot_dir if plot_dir is not None else "toy_bayes"
        self.func = func
        self.pbounds = pbounds
        self.dim = len(pbounds)
        self.known_true_max = known_true_max
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose
        self.iteration_data = {}
        self.best_params_history = []
        self.best_value_history = []
        self.current_best_value = -np.inf
        self.current_best_params = None
        self.stopped_early = False
        self.comparison_max = known_true_max
        self.sampled_approx_max = None
        self.total_iterations = 0
        self.last_run_acquisition_fn = None
        self.last_run_n_random = 0
        self.last_run_n_acq_max = 0
        self._last_optimizer_instance = None
        self._color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- (_sanitize_filename, _get_acq_info, _generate_filename, _ensure_dir_exists unchanged) ---
    def _sanitize_filename(self, name):
        name = str(name); name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w.\-]+', '', name); name = name.strip('._-')
        if not name or name in ['.', '..']: return "plot"
        return name

    def _get_acq_info(self, acq_fn):
        if acq_fn is None: return "N/A", ""
        name = type(acq_fn).__name__; param_str = ""
        if hasattr(acq_fn, 'xi'): param_str = f"xi{acq_fn.xi:.2f}".replace('.', 'p')
        elif hasattr(acq_fn, 'kappa'): param_str = f"kappa{acq_fn.kappa:.2f}".replace('.', 'p')
        return name, param_str

    def _generate_filename(self, base_name=None, prefix="bayesopt", directory=None):
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
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                if self.verbose: print(f"Created directory: {directory}")
            except OSError as e:
                if self.verbose: print(f"Error creating directory {directory}: {e}")
                return False
        return True

    # --- (_approximate_function_max, _run_single_optimization, run_experiment unchanged) ---
    def _approximate_function_max(self, n_samples=10_000, seed=None):
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
            self.sampled_approx_max = None
        else:
            self.sampled_approx_max = max_val
            if self.verbose: print(f"Sampled Approx Max: {self.sampled_approx_max:.6f}")

    def _run_single_optimization(self, acquisition_fn, n_random, n_acq_max, run_seed):
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
            if self.comparison_max is not None and not np.isinf(self.comparison_max):
                denom = abs(self.comparison_max); diff_percentage = abs(self.comparison_max - local_current_best_value) / denom if denom > 1e-9 else abs(self.comparison_max - local_current_best_value)
                if diff_percentage < self.early_stopping_threshold: consecutive_threshold_met += 1
                else: consecutive_threshold_met = 0
                if consecutive_threshold_met >= self.early_stopping_patience:
                    local_stopped_early = True
                    if self.verbose and isinstance(acq_iterator, tqdm): acq_iterator.set_description("Acquisition Phase (Stopped Early)", refresh=True)
                    break
        return (local_iteration_data, local_best_value_history, local_best_params_history,
                local_stopped_early, local_total_iterations, optimizer)

    def run_experiment(self, acquisition_fn, n_random=5, n_acq_max=50,
                       approx_max_samples=10_000, plot_summary=True, verbose=None,
                       save_plot=False, save_filename=None):
        """Runs a single Bayesian optimization experiment with progress bars."""
        effective_verbose = verbose if verbose is not None else self.verbose
        original_verbose = self.verbose; self.verbose = effective_verbose
        self.iteration_data = {}; self.best_params_history = []; self.best_value_history = []
        self.current_best_value = -np.inf; self.current_best_params = None
        self.stopped_early = False; self.total_iterations = 0
        self.last_run_acquisition_fn = acquisition_fn
        self.last_run_n_random = n_random; self.last_run_n_acq_max = n_acq_max
        self._last_optimizer_instance = None; self.sampled_approx_max = None
        if self.comparison_max is None:
            self._approximate_function_max(n_samples=approx_max_samples, seed=self.random_state)
            if self.sampled_approx_max is None:
                 if effective_verbose: print("Approximation failed. Cannot proceed.")
                 self.verbose = original_verbose; return {}
            else: self.comparison_max = self.sampled_approx_max
        elif effective_verbose: print(f"Using known true maximum for comparison: {self.comparison_max:.6f}")
        if effective_verbose:
            print(f"\nStarting single optimization run: {n_random} random, max {n_acq_max} acquisition steps.")
            acq_name, acq_param = self._get_acq_info(acquisition_fn)
            print(f"Using Acquisition Function: {acq_name} ({acq_param})")
        run_results = self._run_single_optimization(acquisition_fn, n_random, n_acq_max, self.random_state)
        (self.iteration_data, self.best_value_history, self.best_params_history,
         self.stopped_early, self.total_iterations, self._last_optimizer_instance) = run_results
        if self.best_value_history: self.current_best_value, self.current_best_params = self.best_value_history[-1], self.best_params_history[-1]
        else: self.current_best_value, self.current_best_params = -np.inf, None
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
                comparison_type = "Known True Max" if self.known_true_max is not None else "Sampled Approx Max"
                if self.comparison_max is not None:
                    final_diff = abs(self.comparison_max - self.current_best_value)
                    final_diff_perc = (final_diff / abs(self.comparison_max) * 100 if abs(self.comparison_max) > 1e-9 else final_diff * 100)
                    print(f"Difference from {comparison_type} ({self.comparison_max:.6f}): {final_diff:.6f} ({final_diff_perc:.4f}%)")
                else: print("Comparison maximum not available.")
            else: print("No valid points evaluated.")
            print("-" * 27)
        if plot_summary and self.iteration_data:
            self.plot_summary(save_plot=save_plot, save_filename=save_filename)
        if self.known_true_max is None: self.comparison_max = None
        self.verbose = original_verbose
        return self.iteration_data

    # --- Static Aggregation Methods (_aggregate_iteration_data, _aggregate_best_history unchanged) ---
    @staticmethod
    def _aggregate_iteration_data(data_list, verbose=True):
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
        valid_histories = [h for h in history_list if h];
        if not valid_histories: return []
        min_len = min(len(h) for h in valid_histories);
        if min_len == 0: return []
        stacked_histories = np.array([h[:min_len] for h in valid_histories])
        avg_history = np.nanmean(stacked_histories, axis=0)
        return avg_history.tolist()

    # --- run_acquisition_sweep (Modified Plotting Section) ---
    def run_acquisition_sweep(self, acq_param_dict, acquisition_fn_constructor,
                              n_random=5, n_acq_max=50, num_runs=3,
                              approx_max_samples=10_000,
                              title="Acquisition Parameter Sweep", verbose=None,
                              save_plot=False, save_filename=None, plot_cols=3): # Added plot_cols
        effective_verbose = verbose if verbose is not None else self.verbose
        original_verbose = self.verbose; self.verbose = effective_verbose
        if len(acq_param_dict) != 1: raise ValueError("acq_param_dict must contain exactly one key.")
        param_name, param_values = list(acq_param_dict.items())[0]
        num_params = len(param_values)

        original_comparison_max = self.comparison_max
        if self.comparison_max is None:
            self._approximate_function_max(n_samples=approx_max_samples, seed=self.random_state)
            if self.sampled_approx_max is None:
                 if effective_verbose: print("Approximation failed. Cannot proceed with sweep.")
                 self.verbose = original_verbose; return {}
            else: self.comparison_max = self.sampled_approx_max
        elif effective_verbose: print(f"Using known true maximum for comparison: {self.comparison_max:.6f}")

        # --- Setup Plot Grid ---
        n_plots_per_param = self.dim + 1 # dim feature plots + 1 performance plot
        n_cols = min(plot_cols, n_plots_per_param)
        n_rows_per_param = math.ceil(n_plots_per_param / n_cols)
        total_rows = num_params * n_rows_per_param

        fig, axs = plt.subplots(total_rows, n_cols,
                                figsize=(6 * n_cols, 4.5 * total_rows), # Adjusted figsize
                                squeeze=False)
        sweep_results = {}

        # --- Main Sweep Loop ---
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
                run_results_tuple = self._run_single_optimization(acq_fn, n_random, n_acq_max, run_seed)
                run_iteration_data_list.append(run_results_tuple[0]); run_best_value_history_list.append(run_results_tuple[1])

            # --- Aggregation ---
            if effective_verbose: print(f"Aggregating results for {param_name} = {param_value}...")
            agg_data = self._aggregate_iteration_data(run_iteration_data_list, verbose=effective_verbose)
            agg_best_history = self._aggregate_best_history(run_best_value_history_list)
            sweep_results[param_value] = {'agg_iteration_data': agg_data, 'agg_best_value_history': agg_best_history}

            # --- Plotting for this Parameter Value ---
            if not agg_data:
                 if effective_verbose: print(f"No aggregated data to plot for {param_name}={param_value}.")
                 # Optionally turn off axes for this parameter's rows
                 start_row_idx = i * n_rows_per_param
                 for row_offset in range(n_rows_per_param):
                     for col_idx in range(n_cols):
                         axs[start_row_idx + row_offset, col_idx].axis('off')
                 continue

            # Plot averaged feature evolution and performance
            title_suffix = f"({param_name}={param_value}, Avg over {num_runs} runs)"
            start_row_idx = i * n_rows_per_param # Starting row for this parameter value

            for plot_idx in range(n_plots_per_param):
                current_row = start_row_idx + (plot_idx // n_cols)
                current_col = plot_idx % n_cols
                ax = axs[current_row, current_col]

                if plot_idx < self.dim: # Feature plot
                    self._plot_averaged_feature_evolution(
                        agg_data=agg_data,
                        feature_index=plot_idx,
                        ax=ax,
                        title_suffix=title_suffix
                    )
                else: # Performance plot (last plot in the sequence)
                    self._plot_performance_vs_max_data(
                        best_value_history=agg_best_history,
                        comparison_max=self.comparison_max,
                        ax=ax,
                        title_suffix=title_suffix
                    )

            # Add a Row Label / Separator (using text on the first plot of the row)
            row_label = f"{param_name} = {param_value}"
            axs[start_row_idx, 0].text(-0.1, 1.15, row_label, transform=axs[start_row_idx, 0].transAxes,
                                       fontsize=14, fontweight='bold', ha='left', va='top')

            # Turn off any unused axes in the last row for this parameter
            last_plot_idx = n_plots_per_param - 1
            last_row_idx = start_row_idx + (last_plot_idx // n_cols)
            plots_in_last_row = (last_plot_idx % n_cols) + 1
            for unused_col in range(plots_in_last_row, n_cols):
                 axs[last_row_idx, unused_col].axis('off')


        # --- Final Plot Adjustments and Saving ---
        fig.suptitle(title, fontsize=18, y=1.0) # Adjust y if needed
        plt.tight_layout(rect=[0, 0.02, 1, 0.97]) # Adjust top/bottom margins
        # fig.subplots_adjust(hspace=0.4, wspace=0.3) # Optional manual spacing

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

        self.comparison_max = original_comparison_max
        self.sampled_approx_max = None
        self.verbose = original_verbose
        return sweep_results


    # --- Plotting Methods ---

    def _plot_single_feature_convergence(self, iteration_data, feature_index, optimal_value, ax, title_suffix=""):
        """Plots the evolution of a single feature against its optimal value."""
        if not iteration_data: return
        all_iterations = sorted(iteration_data.keys());
        if not all_iterations: return
        feature_values = [iteration_data[i]['x'][feature_index] for i in all_iterations]
        feature_name = f"x{feature_index}"
        color = self._color_cycle[feature_index % len(self._color_cycle)]
        switch_iter = -1
        for i in all_iterations:
            if not iteration_data[i]['random']: switch_iter = i; break
        ax.scatter(all_iterations, feature_values, s=10, alpha=0.6, label="Sampled Values", c=color)
        if optimal_value is not None: ax.axhline(optimal_value, color='red', linestyle='--', linewidth=2, label=f"Optimal ({optimal_value:.3f})")
        if switch_iter != -1 and switch_iter > 0: ax.axvline(switch_iter - 0.5, color='black', linestyle=':', linewidth=1.5, label="Switch")
        ax.set_xlabel("Iteration"); ax.set_ylabel(f"{feature_name} Value")
        ax.set_title(f"Convergence of {feature_name} {title_suffix}")
        ax.legend(fontsize='small', loc='best'); ax.grid(True, linestyle=':')
        if feature_name in self.pbounds:
             lower_b, upper_b = self.pbounds[feature_name]; margin = (upper_b - lower_b) * 0.05
             ax.set_ylim(lower_b - margin, upper_b + margin)

    def _plot_averaged_feature_evolution(self, agg_data, feature_index, ax, title_suffix=""):
        """Plots the averaged evolution of a single feature."""
        if not agg_data: return
        all_iterations = sorted(agg_data.keys())
        if not all_iterations: return

        # Extract averaged values, handling potential NaNs
        feature_values = []
        valid_iterations = []
        for i in all_iterations:
            val = agg_data[i]['x'][feature_index]
            if not np.isnan(val):
                feature_values.append(val)
                valid_iterations.append(i)

        if not valid_iterations: # No valid data points for this feature
             if self.verbose: ax.text(0.5, 0.5, 'No valid avg data', ha='center', va='center', transform=ax.transAxes)
             ax.set_title(f"Avg Convergence of x{feature_index} {title_suffix}"); return

        feature_name = f"x{feature_index}"
        color = self._color_cycle[feature_index % len(self._color_cycle)]

        # Find switch point based on aggregated random flag
        switch_iter = -1
        for i in all_iterations: # Check all original iterations for the flag
            if i in agg_data and not agg_data[i]['random']:
                switch_iter = i
                break

        # Plot averaged sampled values
        ax.scatter(valid_iterations, feature_values, s=10, alpha=0.6, label="Avg Sampled Values", c=color)

        # Plot switch line
        if switch_iter != -1 and switch_iter > 0:
            ax.axvline(switch_iter - 0.5, color='black', linestyle=':', linewidth=1.5, label="Switch")

        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Avg Feature {feature_name} Value")
        ax.set_title(f"Avg Convergence of {feature_name} {title_suffix}")
        ax.legend(fontsize='small', loc='best')
        ax.grid(True, linestyle=':')
        if feature_name in self.pbounds:
             lower_b, upper_b = self.pbounds[feature_name]; margin = (upper_b - lower_b) * 0.05
             ax.set_ylim(lower_b - margin, upper_b + margin)


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
        ax.set_title(f"Performance vs Comparison Max ({comparison_max:.4f}) {title_suffix}")
        ax.set_ylim(-1, 101)
        ax.grid(True, linestyle=':')

    def plot_summary(self, save_plot=False, save_filename=None, plot_cols=3):
        """
        Generates a summary plot for the last single run. Shows individual feature
        convergence plots (with optimal line) and the performance vs max plot.
        """
        if not self.iteration_data:
            if self.verbose: print("No data available from the last run to generate summary plot.")
            return
        if not self._last_optimizer_instance or not hasattr(self._last_optimizer_instance, 'max') or not self._last_optimizer_instance.max:
             if self.verbose: print("Optimizer instance or its maximum not found. Cannot plot optimal feature values.")
             optimal_params = None
        else:
             optimal_params = self._last_optimizer_instance.max.get('params', None)
             if optimal_params is None and self.verbose: print("Warning: Could not retrieve optimal parameters found by the optimizer.")

        num_feature_plots = self.dim; num_perf_plots = 1
        total_plots = num_feature_plots + num_perf_plots
        n_cols = min(plot_cols, total_plots)
        n_rows = math.ceil(total_plots / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

        for d in range(self.dim):
            row = d // n_cols; col = d % n_cols; ax = axs[row, col]
            optimal_val_d = None
            if optimal_params:
                try: optimal_val_d = optimal_params[f'x{d}']
                except KeyError:
                     if self.verbose: print(f"Warning: Optimal parameter 'x{d}' not found.")
            # Call the single feature plotter (which includes optimal line)
            self._plot_single_feature_convergence(
                iteration_data=self.iteration_data, feature_index=d,
                optimal_value=optimal_val_d, ax=ax, title_suffix="(Last Run)"
            )

        perf_plot_idx = self.dim; row = perf_plot_idx // n_cols; col = perf_plot_idx % n_cols
        ax = axs[row, col]
        self._plot_performance_vs_max_data(
            best_value_history=self.best_value_history, comparison_max=self.comparison_max,
            ax=ax, title_suffix="(Last Run)"
        )

        for i in range(total_plots, n_rows * n_cols):
            row = i // n_cols; col = i % n_cols; axs[row, col].axis('off')

        gp_max_val_str = "N/A"
        if self._last_optimizer_instance and hasattr(self._last_optimizer_instance, 'max') and self._last_optimizer_instance.max:
             gp_max_val = self._last_optimizer_instance.max.get('target', 'N/A')
             if isinstance(gp_max_val, (float, np.float_)): gp_max_val_str = f"{gp_max_val:.4f}"
        if self.known_true_max is not None:
            comparison_label = "Known True Max:"; comparison_val_str = f"{self.comparison_max:.4f}" if self.comparison_max is not None else "N/A"; sampled_max_line = ""
        elif self.sampled_approx_max is not None:
            comparison_label = "Sampled Approx Max:"; comparison_val_str = f"{self.comparison_max:.4f}"; sampled_max_line = f"{comparison_label:<20} {comparison_val_str}\n"
        else: comparison_label = "Comparison Max:"; comparison_val_str = "N/A"; sampled_max_line = f"{comparison_label:<20} {comparison_val_str}\n"
        observed_max_str = f"{self.current_best_value:.4f}" if self.current_best_value > -np.inf else "N/A"
        max_info_text = ""
        if self.known_true_max is not None: max_info_text += f"{comparison_label:<20} {comparison_val_str}\n"
        elif self.sampled_approx_max is not None: max_info_text += sampled_max_line
        max_info_text += (f"{'Observed Best Value:':<20} {observed_max_str}\n"
                          f"{'GP Estimated Max:':<20} {gp_max_val_str}")
        plt.figtext(0.5, 0.98, max_info_text, ha="center", va="top", fontsize=9,
                    bbox={"facecolor":"white", "alpha":0.7, "pad":3}, family='monospace')

        acq_name, acq_param = self._get_acq_info(self.last_run_acquisition_fn)
        fig.suptitle(f"Bayesian Optimization Summary ({self.total_iterations} Iterations, Acq: {acq_name} {acq_param})",
                     fontsize=16, y=1.0)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust top margin

        if save_plot:
            full_filename = self._generate_filename(base_name=save_filename, prefix="bayesopt_summary", directory=self.plot_dir)
            if self._ensure_dir_exists(full_filename):
                try:
                    fig.savefig(full_filename, bbox_inches='tight', dpi=150)
                    if self.verbose: print(f"Summary plot saved to {full_filename}")
                except Exception as e:
                    if self.verbose: print(f"Error saving summary plot to {full_filename}: {e}")
            else:
                 if self.verbose: print(f"Could not save plot, directory creation failed for {full_filename}")
        plt.show()
