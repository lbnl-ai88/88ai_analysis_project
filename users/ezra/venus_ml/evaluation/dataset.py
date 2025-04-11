from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DatasetTester:
    """
    Class to evaluate various scikit-learn regressor models on multiple datasets with specified pre-processing techniques.
    It tracks training and validation errors, allowing comparisons between models and datasets.
    """

    def __init__(self, model_class, datasets_dict, run_ids, validation_size=0.2):
        """
        Initialize the evaluator with datasets and model configurations.

        Args:
            model_class: The scikit-learn model class to instantiate for each dataset.
            datasets_dict: A dictionary where keys are dataset names and values are VenusDataset instances.
            run_ids: Run IDs to select subsets from each dataset.
            validation_size: Proportion of the dataset used for validation.
            epochs: Number of epochs for training (not used in scikit-learn, kept for compatibility).
        """
        self.model_class = model_class
        self.datasets_dict = datasets_dict  # Dictionary with names and VenusDataset instances
        self.run_ids = run_ids
        self.validation_size = validation_size

        self.results = {}

    def fit_model(self, model, train_inputs, train_outputs, val_inputs, val_outputs):
        """
        Fits a scikit-learn model and evaluates on the validation dataset.
        Tracks training and validation errors.

        Args:
            model: The scikit-learn model instance to train.
            train_inputs: Training input data (NumPy array).
            train_outputs: Training target data (NumPy array).
            val_inputs: Validation input data (NumPy array).
            val_outputs: Validation target data (NumPy array).

        Returns:
            A dictionary containing training and validation errors.
        """
        # Train the model using the training data
        model.fit(train_inputs, train_outputs)

        # Predict on the training set and calculate error
        train_predictions = model.predict(train_inputs)
        train_error = mean_squared_error(train_outputs, train_predictions)

        # Predict on the validation set and calculate error
        val_predictions = model.predict(val_inputs)
        val_error = mean_squared_error(val_outputs, val_predictions)

        print(f"Training MSE: {train_error:.4f}, Validation MSE: {val_error:.4f}")

        return {"train_errors": [train_error], "val_errors": [val_error]}

    def evaluate(self):
        """
        Evaluates each model on the provided datasets and stores the results for comparison.
        """
        for dataset_name, dataset in self.datasets_dict.items():
            # Get training and validation data as NumPy arrays
            train_dataset, val_dataset = dataset.generate_splits(run_ids=self.run_ids,
                                                                 validation_size=self.validation_size)

            # Convert to NumPy arrays
            train_inputs, train_outputs = train_dataset.to_numpy()
            val_inputs, val_outputs = val_dataset.to_numpy()

            # Instantiate the model
            model = self.model_class()

            print(f"Evaluating on {dataset_name}...")

            # Fit and evaluate the model
            results = self.fit_model(model, train_inputs, train_outputs, val_inputs, val_outputs)
            self.results[dataset_name] = results

    def plot_results(self):
        """
        Plots the training and validation errors for each dataset.
        """
        for dataset_name, result in self.results.items():
            train_errors = result["train_errors"]
            val_errors = result["val_errors"]
            epochs = range(1, len(train_errors) + 1)

            plt.plot(epochs, train_errors, label=f"{dataset_name} - Training Error")
            plt.plot(epochs, val_errors, label=f"{dataset_name} - Validation Error", linestyle="--")

        plt.xlabel('Epochs')
        plt.ylabel('Error (MSE)')
        plt.title('Training and Validation Error Comparison')
        plt.legend()
        plt.show()

    def compare_models(self):
        """
        Print final training/validation errors for comparison across datasets.
        """
        comparison_table = []
        for dataset_name, result in self.results.items():
            final_train_error = result["train_errors"][-1]
            final_val_error = result["val_errors"][-1]
            comparison_table.append([dataset_name, final_train_error, final_val_error])

        df = pd.DataFrame(comparison_table, columns=["Dataset", "Final Training Error", "Final Validation Error"])
        print(df)
        return df
