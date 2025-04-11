import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from typing import List, Optional, Union, Sequence, Tuple, Callable


def read_file(file_path: str):
    """
    Reads .csv or .parquet file into a pandas dataframe
    Args:
        file_path (str): path to file

    Returns:
        A pandas dataframe containing file contents
    Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file is not found at the given path.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Invalid file type: {file_path}. File must be a .csv or .parquet file")


def run_select(df: pd.DataFrame, run_selection: Union[int, float, Sequence[Union[int, float, str]]]) -> pd.DataFrame:
    """
    Select the desired runs for the dataset. Runs should be in column 'run_id'.
    Args:
        df (pd.DataFrame): Full dataset
        run_selection (Union[float, Sequence[float]]): desired runs

    Returns:
        A new dataframe with only the selected runs

    Raises:
        ValueError: if an invalid form of run selection is inputted
    """
    if type(run_selection) in [int, float, str]:
        mask = df["run_id"] == run_selection
    elif type(run_selection) in [list, tuple]:
        mask = df["run_id"].isin(run_selection)
    else:
        raise ValueError(f"Invalid run selection: {run_selection}. Must be int, float, str Sequence[Union[float, int, str]]")
    return df[mask]


class VenusDataset(Dataset):
    """
     Custom dataset that can read from .csv or .parquet files. Includes functionality to pass in a scaler and
     transforms to preprocess dataset. Stores scaling parameters in order to apply them to test/validation datasets

     If the sequence length is set to anything but the default 0, then this functions as a time-series style dataset, where
     a sequence of inputs leading up to a target output is returned from __getitem__ instead of all inputs and outputs
     at a single timestep. The __len__ method is also modified so there are no invalid accesses.

     Attributes:
        df : pandas dataframe containing dataset
        inputs : dataframe of input features
        outputs : dataframe of output features
        scale_function : function to scale data
        transforms : sequence of transforms applied to data
        sequence_length : the length of the sequence returned by __getitem__
    """

    def __init__(self,
                 file_path: str,
                 input_columns: List[str],
                 output_columns: List[str],
                 run_selection: Union[int, float, str, Sequence] = None,
                 scaler: Callable[[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]], Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series]]] = None,
                 transforms: Sequence[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 sequence_length: int = 0
                 ):
        """
        Initializes a dataset with specified run and input/output columns then scales and transforms it

        Args:
            file_path (str): path to .csv or .parquet file containing dataset
            input_columns (List[str]): list of input column names
            output_columns (List[str]): list of output column names
            run_selection (Union[int, float, Sequence[float]]): run or list of runs to use
            scaler (Callable[[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]],
                                  Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series]]]): function to scale data
            transforms (Sequence[Callable[[pd.DataFrame], pd.DataFrame]]): list of transform functions
            sequence_length (int): length of sequence to return
        """
        self.df = read_file(file_path)
        if run_selection:
            self.df = run_select(self.df, run_selection)
        if scaler:
            self.scale_function = scaler
            self.df, self.scale_params = scaler(self.df, None, None)
        if transforms:
            self.transforms = transforms
            for transform in transforms:
                self.df = transform(self.df)

        self.sequence_length = sequence_length
        self.inputs = self.df[[col for col in self.df.columns if col.startswith(tuple(input_columns))]].fillna(0)
        self.outputs = self.df[output_columns].fillna(0)

    def __len__(self) -> int:
        return len(self.inputs) - self.sequence_length

    def dataset_size(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.sequence_length:
            inputs = self.inputs.iloc[idx:idx + self.sequence_length]
            outputs = self.outputs.iloc[idx + self.sequence_length]
        else:
            inputs = self.inputs.iloc[idx]
            outputs = self.outputs.iloc[idx]

        # Convert inputs and outputs to numeric values
        inputs = pd.DataFrame(inputs).apply(pd.to_numeric, errors='coerce').fillna(0).values
        outputs = pd.DataFrame(outputs).apply(pd.to_numeric, errors='coerce').fillna(0).values

        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.tensor(outputs, dtype=torch.float32)
        inputs = inputs.squeeze()
        outputs = outputs.squeeze()

        return inputs, outputs

    def apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data scaling to another dataset using this dataset's parameters
        Args:
            df: test/validation dataset

        Returns:
            A scaled dataframe
        """
        param1, param2 = self.scale_params
        return self.scale_function(df, param1, param2)[0]

    def apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data transforms to another dataframe, intended for use on test/validation datasets
        Args:
            df: test/validation dataset

        Returns:
            A transformed dataframe
        """
        for transform in self.transforms:
            df = transform(df)
        return df

    def to_numpy(self):
        """
        Get numpy arrays containing the data for use with sci-kit learn models using .fit(X_data, y_data)
        Returns:
            A tuple containing numpy arrays for the inputs and outputs

        """
        return self.inputs.values, self.outputs.values

    def to_tensor(self):
        """
        Get torch tensors containing the data
        Returns:
            A tuple containing numpy arrays for the inputs and outputs

        """
        inputs, outputs = self.to_numpy()
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

    def get_runs(self, run_ids: Sequence[Union[float, int, str]]) -> dict:
        """
        Retrieves a dictionary of run_ids paired with corresponding data as pandas DataFrames
        Args:
            run_ids (Sequence[Union[float, int, str]]): Which runs to select

        Returns:
            A dictionary containing run_ids paired with corresponding DataFrame slices
        """
        runs_data = {}
        for run_id in run_ids:
            run_df = run_select(self.df, run_id)
            input_columns = [col for col in run_df.columns if col in self.inputs.columns]
            output_columns = [col for col in run_df.columns if col in self.outputs.columns]
            runs_data[run_id] = (run_df[input_columns], run_df[output_columns])
        return runs_data

    def generate_splits(self, run_ids: Sequence[Union[int, float, str]],
                        validation_size: Optional[float] = 0.2,
                        random_state: Optional[int] = None, shuffle=True) -> Tuple:
        """
        Retrieves runs data and splits into training and validation sets with the split occurring in each run
        individually, returning modified copies of the original dataset for training and validation.
        Args:
            original_dataset: An instance of VenusDataset to be copied and modified
            run_ids (Sequence): Which runs to select
            validation_size (float): Which percentage of data to make validation
            random_state (int): Optional parameter to set random state, defaults to None
            shuffle (bool): Whether to shuffle the run before splitting, defaults to True

        Returns:
            Tuple of VenusDataset objects containing (training_dataset, validation_dataset)
        """
        train_inputs_list, val_inputs_list, train_outputs_list, val_outputs_list = [], [], [], []
        runs_data = self.get_runs(run_ids)
        for run_id, (inputs, outputs) in runs_data.items():
            x_train, x_val, y_train, y_val = train_test_split(
                inputs,
                outputs,
                test_size=validation_size,
                random_state=random_state,
                shuffle=shuffle
            )
            train_inputs_list.append(x_train)
            val_inputs_list.append(x_val)
            train_outputs_list.append(y_train)
            val_outputs_list.append(y_val)

        # Combine all training and validation splits
        train_inputs = pd.concat(train_inputs_list, ignore_index=True)
        validation_inputs = pd.concat(val_inputs_list, ignore_index=True)
        train_outputs = pd.concat(train_outputs_list, ignore_index=True)
        validation_outputs = pd.concat(val_outputs_list, ignore_index=True)

        # Create deep copies of the original dataset and update inputs and outputs
        training_dataset = copy.deepcopy(self)
        training_dataset.inputs = train_inputs
        training_dataset.outputs = train_outputs

        validation_dataset = copy.deepcopy(self)
        validation_dataset.inputs = validation_inputs
        validation_dataset.outputs = validation_outputs

        return training_dataset, validation_dataset

    def get_data_loaders(self, run_ids: Sequence[Union[int, float, str]], batch_size: int = 32,
                         validation_size: Optional[float] = 0.2, random_state: Optional[int] = None, shuffle=True) -> (
            Tuple)[DataLoader, DataLoader]:
        """
        Returns DataLoaders for training and validation datasets.

        Args:
            run_ids (Sequence[Union[int, float, str]]): Which runs to select
            batch_size (int): Number of samples per batch to load
            validation_size (float): Which percentage of data to make validation
            random_state (int): Optional parameter to set random state, defaults to None
            shuffle (bool): Whether to shuffle the run before splitting, defaults to True

        Returns:
            Tuple of DataLoader objects containing (training_loader, validation_loader)
        """
        training_dataset, validation_dataset = self.generate_splits(run_ids, validation_size, random_state, shuffle)

        training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        return training_loader, validation_loader



