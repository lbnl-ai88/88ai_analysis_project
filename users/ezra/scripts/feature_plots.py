#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from users.ezra.venus_ml import VenusDataset
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


def split_by_set(dataset, feature):
    grouped_data = dataset.inputs.groupby('run_id')[feature].apply(lambda x: np.array(x)).to_dict()
    return grouped_data


def get_box_plot_data(grouped_data):
    set_names = list(grouped_data.keys())

    min_values = []
    q1_values = []
    median_values = []
    q3_values = []
    max_values = []

    for values in grouped_data.values():
        values = np.array(values, dtype=np.float64)  # Ensure values are of float type
        min_values.append(np.min(values))
        q1_values.append(np.percentile(values, 25))
        median_values.append(np.median(values))
        q3_values.append(np.percentile(values, 75))
        max_values.append(np.max(values))

    return set_names, min_values, q1_values, median_values, q3_values, max_values


def get_plot_data(grouped_data):
    set_names = list(grouped_data.keys())
    means = [values.mean() for values in list(grouped_data.values())]
    std = [values.std() for values in list(grouped_data.values())]
    mins = [values.min() for values in grouped_data.values()]
    maxs = [values.max() for values in grouped_data.values()]
    lower_errors = np.subtract(means, mins)
    upper_errors = np.subtract(maxs, means)
    errors = [lower_errors, upper_errors]

    return set_names, means, std, errors


def plot_box_plots(set_names, min_values, q1_values, median_values, q3_values, max_values, feature):
    plt.figure()

    data = []
    for i in range(len(set_names)):
        data.append([min_values[i], q1_values[i], median_values[i], q3_values[i], max_values[i]])

    plt.boxplot(data, positions=range(len(set_names)), patch_artist=True)
    plt.xticks(ticks=range(len(set_names)), labels=set_names, rotation=90)
    plt.xlabel('Set Names')
    plt.ylabel(f'{feature} Values')
    plt.title(f'Box Plot for {feature}')
    plt.ticklabel_format(useOffset=False, style='plain', axis='y')
    plt.tight_layout()

    if not os.path.exists('box_plots'):
        os.makedirs('box_plots')

    plt.savefig(f'box_plots/{feature}.png')
    plt.close()


def plot_set_densities(dataset, grouped_data, feature):
    # Create a directory for the feature if it does not exist
    feature_dir = f'set_densities/{feature}'
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # Iterate through each set in the grouped data
    for set_name, values in grouped_data.items():
        values = np.array(values, dtype=np.float64)  # Ensure values are of float type

        plt.figure()
        sns.kdeplot(values, fill=True, warn_singular=False)

        # Calculate the mean and standard deviations
        mean = np.mean(values)
        std_dev = np.std(values)

        # Plot the mean line
        plt.axvline(x=mean, color='red', linestyle='--', linewidth=2, label='Mean')

        # Plot the 1 std and 2 std lines
        plt.axvline(x=mean + std_dev, color='blue', linestyle=':', linewidth=1, label='+1 Std Dev')
        plt.axvline(x=mean - std_dev, color='blue', linestyle=':', linewidth=1, label='-1 Std Dev')
        plt.axvline(x=mean + 2 * std_dev, color='blue', linestyle=':', linewidth=1, label='+2 Std Dev')
        plt.axvline(x=mean - 2 * std_dev, color='blue', linestyle=':', linewidth=1, label='-2 Std Dev')

        plt.xlabel(f'{feature} Values')
        plt.ylabel('Density')
        plt.title(f'Density Plot for {feature} in {set_name}')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        plt.savefig(f'{feature_dir}/{set_name}.png')
        plt.close()


def plot_error_bars(set_names, means, std, errors, feature):
    plt.figure()

    lower_errors = np.abs(errors[0])
    upper_errors = np.abs(errors[1])

    yerr = [lower_errors, upper_errors]
    plt.errorbar(x=set_names, y=means, yerr=yerr, fmt='o', ecolor='grey', capsize=8, label='Data Points')

    # Calculate the overall mean and standard deviations
    overall_mean = np.mean(means)
    overall_std = np.mean(std)

    # Plot the overall mean line
    plt.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, label='Overall Mean')

    # Plot the 1 std and 2 std lines
    plt.axhline(y=overall_mean + overall_std, color='blue', linestyle=':', linewidth=1, label='+1 Std Dev')
    plt.axhline(y=overall_mean - overall_std, color='blue', linestyle=':', linewidth=1, label='-1 Std Dev')

    plt.xlabel('Set Names')
    plt.ylabel(f'{feature} Values')
    plt.title(f'Error Bar Plot for {feature}')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    if not os.path.exists('error_bars'):
        os.makedirs('error_bars')

    plt.savefig(f'error_bars/{feature}.png')
    plt.close()


def plot_densities(dataset, feature):
    data = dataset.inputs[feature].dropna()  # Drop NaN values if any

    # Calculate the overall mean and standard deviations
    mean = np.mean(data)
    std_dev = np.std(data)

    plt.figure()
    sns.kdeplot(data, fill=True, warn_singular=False)

    # Plot the mean line
    plt.axvline(x=mean, color='red', linestyle='--', linewidth=2, label='Mean')

    # Plot the 1 std and 2 std lines
    plt.axvline(x=mean + std_dev, color='blue', linestyle=':', linewidth=1, label='+1 Std Dev')
    plt.axvline(x=mean - std_dev, color='blue', linestyle=':', linewidth=1, label='-1 Std Dev')
    plt.axvline(x=mean + 2 * std_dev, color='blue', linestyle=':', linewidth=1, label='+2 Std Dev')
    plt.axvline(x=mean - 2 * std_dev, color='blue', linestyle=':', linewidth=1, label='-2 Std Dev')

    plt.xlabel(f'{feature} Values')
    plt.ylabel('Density')
    plt.title(f'Density Plot for {feature}')
    plt.legend()
    plt.tight_layout()

    if not os.path.exists('densities'):
        os.makedirs('densities')

    plt.savefig(f'densities/{feature}.png')
    plt.close()


def get_features(names, mean=True, std=False):
    out_list = []
    for name in names:
        if mean:
            out_list.append(name + "_mean")
        if std:
            out_list.append(name + "_std")
    return out_list


def main(dataset_config):
    features = dataset_config['input_columns']
    dataset = VenusDataset(**dataset_config)
    features.remove('run_id')
    for feature in features:
        grouped_data = split_by_set(dataset, feature)
        set_names, means, std, errors = get_plot_data(grouped_data)
        plot_error_bars(set_names, means, std, errors, feature)
        set_names, min_values, q1_values, median_values, q3_values, max_values = get_box_plot_data(grouped_data)
        plot_box_plots(set_names, min_values, q1_values, median_values, q3_values, max_values, feature)
        plot_densities(dataset, feature)
        plot_set_densities(dataset, grouped_data, feature)


if __name__ == "__main__":
    dataset_config = {
        "file_path": "../data/raw_watch_data.parquet",
        "input_columns": ['m_over_q', 'fcv1_i', 'gas_balzer_1', 'gas_balzer_2', 'gas_balzer_5',
                          'gas_balzer_6', 'gas_balzer_7', 'inj_i', 'ext_i', 'mid_i', 'sext_i', 'inj_v', 'ext_v', 'mid_v', 'sext_v',
                          'inj_ps_v', 'ext_ps_v', 'mid_ps_v', 'sext_ps_v', 'inj_mbar', 'ext_mbar', 'bias_v', 'bias_i',
                          'k18_fw', 'k18_ref', 'g28_fw', 'glaser_1', 'batman_i', 'batman_field', 'x_ray_source',
                          'x_ray_exit', 'extraction_v', 'extraction_i', 'puller_v', 'puller_i', 'puller_raw_gap',
                          'bl_mig2_torr', 'robin_i', 'ht_oven_v', 'ht_oven_i', 'lt_oven_1_sp', 'lt_oven_2_sp',
                          'lt_oven_1_temp', 'lt_oven_2_temp', 'LHe_psi', 'LHe_level_percent', 'cryo_vac_torr',
                          'four_k_heater_power', 'four_k_cold_mass', 'four_k_cryo_e', 'four_k_cryo_w', 'four_k_cryo_ne',
                          'four_k_cryo_nw', 'four_k_heat_cond', 'four_k_i_feedthrough', 'four_k_heater_k',
                          'fifty_k_cond_bar', 'fifty_k_cond_bar_ne', 'fifty_k_cond_bar_nw', 'fifty_k_shield_bot',
                          'bottom_ln_vessel', 'seventy_k_cond_bar', 'gas_balzer_2_set',] + ['run_id'],
        "output_columns": [],
    }
    main(dataset_config)
