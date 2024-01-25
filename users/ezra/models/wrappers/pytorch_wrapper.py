from typing import Callable, Optional, Type, Tuple, Sequence

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from users.ezra.data import VenusDataset


# TODO Add a recurrent functionality to deal with hidden states
# TODO Add serialization for model parameters, save to local files


class PyTorchWrapper:
    """
        Wrapper class for PyTorch models to standardize training, prediction, and evaluation.

        Attributes:
            model (torch.nn.Module): The PyTorch model to be wrapped.
            criterion (Callable): The loss function used during training.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            device (str): The device to run the model on ('cpu', 'cuda' or 'mps').
            loss_history (list): The list of batch losses during training
    """

    def __init__(self, model: nn.Module, criterion: Callable[[torch.Tensor, torch.Tensor], torch.tensor],
                 optimizer: torch.optim.Optimizer, training_parameters: Sequence[int, int, str]):
        """
            Initializes the PyTorchModelWrapper instance.

            Args:
                model (nn.Module): The PyTorch model to be wrapped
                criterion (Callable[[torch.Tensor, torch.Tensor], float]): The loss function used during training.
                optimizer (torch.optim.Optimizer): The optimizer used for training the model.
                training_parameters (Tuple[int, int, str]): Epochs, Batch Size, Device
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs, self.batch_size, self.device = training_parameters
        self.loss_history = []

    def predict(self, data: torch.tensor) -> torch.Tensor:
        """
            Generates predictions from the model.

            Args:
                data (torch.Tensor): Input tensor for model

            Returns:
                torch.Tensor: The predictions generated by the model.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data)

        return predictions

    def evaluate(self, dataset: VenusDataset) -> float:
        """
            Evaluates the model on a test dataset.

            Args:
                dataset (VenusDataset): The dataset to evaluate the model on

            Returns:
                float: The calculated loss over the test dataset.
        """
        inputs, outputs = dataset.to_tensor()
        self.model.eval()
        predictions = self.predict(inputs)
        loss = self.criterion(predictions, outputs).item()
        return loss

    def train(self, dataset: VenusDataset):
        """
            Trains the model for a given number of epochs.

            Args:
                dataset (Dataset): The dataset containing the training data.
        """
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.loss_history.append(loss.item())

    def graph_loss(self):
        """
        Make a plot of the loss history during training.
        """
        batches = range(len(self.loss_history))
        plt.plot(batches, self.loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Batches")
        plt.ylabel("Loss")
        plt.show()
