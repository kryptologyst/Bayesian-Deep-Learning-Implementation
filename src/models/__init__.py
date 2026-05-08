"""Bayesian Neural Network models."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class BayesianLinear(PyroModule):
    """Bayesian Linear layer using Pyro."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_scale: float = 1.0,
        bias: bool = True,
    ):
        """Initialize Bayesian Linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_scale: Scale of the prior distribution
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Define weight prior
        self.weight = PyroSample(
            dist.Normal(0.0, prior_scale).expand([out_features, in_features]).to_event(2)
        )
        
        if bias:
            self.bias_param = PyroSample(
                dist.Normal(0.0, prior_scale).expand([out_features]).to_event(1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.bias:
            return F.linear(x, self.weight, self.bias_param)
        else:
            return F.linear(x, self.weight)


class BayesianNN(PyroModule):
    """Bayesian Neural Network using Pyro."""
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 400,
        output_size: int = 10,
        prior_scale: float = 1.0,
    ):
        """Initialize Bayesian Neural Network.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output (number of classes)
            prior_scale: Scale of prior distributions
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = BayesianLinear(input_size, hidden_size, prior_scale)
        self.fc2 = BayesianLinear(hidden_size, output_size, prior_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MCDropoutNN(nn.Module):
    """Monte Carlo Dropout Neural Network."""
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 400,
        output_size: int = 10,
        dropout_rate: float = 0.5,
    ):
        """Initialize MC Dropout Neural Network.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output (number of classes)
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass with optional dropout."""
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        if training:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for CIFAR-10."""
    
    def __init__(self, num_classes: int = 10):
        """Initialize Simple CNN.
        
        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(
    model_type: str,
    input_size: int = 784,
    hidden_size: int = 400,
    output_size: int = 10,
    **kwargs
) -> nn.Module:
    """Create a model based on type.
    
    Args:
        model_type: Type of model ("bayesian", "mc_dropout", "simple_cnn")
        input_size: Input feature size
        hidden_size: Hidden layer size
        output_size: Output size
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if model_type == "bayesian":
        return BayesianNN(input_size, hidden_size, output_size, **kwargs)
    elif model_type == "mc_dropout":
        return MCDropoutNN(input_size, hidden_size, output_size, **kwargs)
    elif model_type == "simple_cnn":
        return SimpleCNN(output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
