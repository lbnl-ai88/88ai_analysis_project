import argparse
import copy
import csv
import os
import re
import sys

import numpy as np
import pandas as pd

import models



def cli_parser():
    models = ["knn", "tree"]

    # TODO stop using argparser and replace with something else
    # TODO weird bug when we do --fcv1_i it thinks we did --fcv1_in
    parser = argparse.ArgumentParser(description="Test arbitrary models")

    parser.add_argument("--parquet_files", required=True, nargs="+", help="Parquet files")
    parser.add_argument("--predict_columns", default=["fcv1_i"], nargs="+", help="Column to predict")

    parser.add_argument("--dummy", help="dummy variable so argparser knows when to take model argument")

    subparsers = parser.add_subparsers(title="Model selector", dest="model", required=True, help="Model to use")

    # Create subparsers so we can handle different cli
    tree_parser = subparsers.add_parser("tree", help="Options for decision tree")
    tree_parser.add_argument("--importance", action="store_true", help="Print the importance of each of the features")

    tree_parser.add_argument("--n_estimators", required=True, type=int, help="Specify the number of estimators to use")
    tree_parser.add_argument("--max_depth", required=True, type=int, help="Specify the maximum depth of the tree")
    tree_parser.add_argument("--max_leaves", required=True, type=int, help="Specify the maximum number of leaf nodes to use, grows the tree in a best-first fashion")
    tree_parser.add_argument("--max_bin", required=True, type=int, help="Specify the maximum number of bins to use for histogram-based tree methods")
    tree_parser.add_argument("--grow_policy", required=True, choices=["depthwise", "lossguide"], help="Specify the tree growing policy, 0: favor splitting near root, 1: favor splitting at nodes with highest loss change")
    tree_parser.add_argument("--learning_rate", required=True, type=float, help="Specify the boosting's learning rate")
    tree_parser.add_argument("--objective", required=True, choices=["reg:squarederror", "reg:squaredlogerror", "reg:pseudohubererror", "reg:absoluteerror"], help="Specify the learning task and corresponding learning objective")
    tree_parser.add_argument("--tree_method", required=True, choices=["exact", "hist"], help="Specify the tree construction algorithm to use")
    tree_parser.add_argument("--reg_alpha", required=True, type=float, help="Specify the L1 regularization term on weights")
    tree_parser.add_argument("--reg_lambda", required=True, type=float, help="Specify the L2 regularization term on weights")
    tree_parser.add_argument("--subsample", required=True, type=float, help="Specify the fraction of random selection of rows")
    tree_parser.add_argument("--colsample_bynode", required=True, type=float, help="Specify the fraction of columns to randomly sample at each node split")
    tree_parser.add_argument("--colsample_bytree", required=True, type=float, help="Specify the fraction of columns to randomly sample at the creation of each tree")
    tree_parser.add_argument("--num_boost_round", required=True, type=int, help="Specify the number of boosting iterations")
    tree_parser.add_argument("--num_parallel_tree", required=True, type=int, help="Specify the number of trees to train in parallel")

    knn_parser = subparsers.add_parser("knn", help="Options for k nearest neighbor regression")
    knn_parser.add_argument("--num_neighbors", required=True, type=int, help="Specify the number of neighbors to use")
    knn_parser.add_argument("--to_normalize", required=True, type=int, help="Specify the whether to normalize the input data (0: False, 1: True)")

    cheat_parser = subparsers.add_parser("cheat", help="Options for cheat model")

    # parse the known args
    tmp_args, _ = parser.parse_known_args()

    # check to make sure we have enough files
    if 2 > len(tmp_args.parquet_files):
        raise(ValueError("parquet_files argument must have at least 2 values"))

    # get the file columns
    column_names = list(filter(lambda x: x not in tmp_args.predict_columns, pd.read_parquet(tmp_args.parquet_files[0], engine="fastparquet").columns)) # TODO clean up

    for column in column_names:
        knn_parser.add_argument(f"--{column}", default=0, type=float, help=f"Specify the weight value for {column} (default: 0)")

    args = parser.parse_args()

    pretty_args = copy.deepcopy(args)
    delattr(pretty_args, "parquet_files")
    delattr(pretty_args, "predict_columns")
    delattr(pretty_args, "model")
    delattr(pretty_args, "dummy")

    if "tree" == args.model:
        delattr(pretty_args, "importance")

    if "knn" == args.model:
        weights = []
        for column in column_names:
            weights.append(getattr(args, column))
            delattr(args, column)

        args.weights = np.asarray(weights)

    return(args, pretty_args)



def read_parquets(parquet_files):
    # Read training data
    df_list = []

    for i, parquet_file in enumerate(parquet_files):
        df = pd.read_parquet(parquet_file, engine="fastparquet")
        df_list.append(df)

    return(df_list)



def process_dataframes(df_list, input_columns, predict_columns):
    # Separate x and y data
    x_df_list = []
    y_df_list = []

    for i, df in enumerate(df_list):
        x = df[input_columns].values
        y = df[predict_columns].values

        x_df_list.append(x)
        y_df_list.append(y)

    return(x_df_list, y_df_list)



def create_model(model_args):
    irrelevant_list = ["parquet_files", "model", "predict_columns", "importance", "dummy"] # TODO?
    model_type = model_args.model
    if model_type == "knn":
        relevant_args = {k: v for k, v in vars(model_args).items() if k not in irrelevant_list}
        return models.KNN_Regressor(**relevant_args)
    elif model_type == "tree":
        relevant_args = {k: v for k, v in vars(model_args).items() if k not in irrelevant_list}
        return models.Tree_Regressor(**relevant_args)
    elif model_type == "cheat":
        relevant_args = {k: v for k, v in vars(model_args).items() if k not in irrelevant_list}
        return models.Cheat_Regressor(**relevant_args)
    else:
        raise ValueError("invalid model type")



def shorten_parquet_files(parquet_files):
    shortened = []

    for filename in parquet_files:
        base_name = os.path.basename(filename)
        run_label = re.search(r"watch_data_(.*?)_processed", filename).group(1) # TODO
        shortened.append(run_label)

    return "_".join(shortened)



def save_data(filename, args, label_list, mse_list, mae_list, mape_list, n_list):
    # Check if file already contains headers
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0

    # Save the data to CSV
    with open(filename, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write headers only if file is empty
        if not file_exists:
            headers = list(vars(args).keys()) + ["Label", "MSE", "MAE", "MAPE", "N"] * len(label_list)
            writer.writerow(headers)

        row = list(vars(args).values())

        for i in range(len(label_list)):
            row.extend([label_list[i], mse_list[i], mae_list[i], mape_list[i], n_list[i]])

        writer.writerow(row)



def generate_parameter_string(args):
    param_string = ""

    for arg, value in vars(args).items():
        arg_str = f"{arg}:{value};"

        if arg == "parquet_files":
            arg_str = f"{arg}:{shorten_parquet_files(value)};"

        param_string += arg_str

    return param_string



def has_matching_parameters(filename, pretty_values):
    if not os.path.isfile(filename):
        return False

    str_pretty_values = list(map(str, pretty_values))
    length = len(str_pretty_values)

    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) > 0 and row[:length] == str_pretty_values:
                return True
    return False



if __name__ == "__main__":
    save_dir = "./results/"

    args, pretty_args = cli_parser()

    predict_columns = args.predict_columns

    csv_filename = save_dir + shorten_parquet_files(args.parquet_files) + "_model_" + args.model + "_cols_" + "_".join(args.predict_columns) + ".csv"


    if not has_matching_parameters(csv_filename, list(vars(pretty_args).values())):
        df_list = read_parquets(args.parquet_files)

        # Get columns
        column_names = list(df_list[0].columns)
        input_columns = list(filter(lambda x: x not in predict_columns, column_names))

        x_df_list, y_df_list = process_dataframes(df_list, input_columns, predict_columns)

        model = create_model(args)
        label_list, mse_list, mae_list, mape_list, n_list = model.k_fold_cross_validation(x_df_list, y_df_list, args.parquet_files)

        def weighted_average(val_list, count_list):
            total = 0
            total_count = 0

            for i, _ in enumerate(val_list):
                total += val_list[i] * count_list[i]
                total_count += count_list[i]

            return(total/total_count)

        # Aggregate statistics
        label_list = ["agg"] + label_list
        mse_list = [weighted_average(mse_list, n_list)] + mse_list
        mae_list = [weighted_average(mae_list, n_list)] + mae_list
        mape_list = [weighted_average(mape_list, n_list)] + mape_list
        n_list = [sum(n_list)] + n_list

        save_data(csv_filename, pretty_args, label_list, mse_list, mae_list, mape_list, n_list)

        if "tree" == args.model and args.importance:
            print(model.feature_importances_)

    else:
        print("skipping")
