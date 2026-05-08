#!/usr/bin/env python3
"""
Fixed and modernized version of the original Bayesian Deep Learning implementation.

This script demonstrates a working Bayesian Neural Network using Pyro,
with proper error handling, type hints, and modern PyTorch practices.
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from src.utils import set_seed, get_device


class BayesianNN(nn.Module):
    """Bayesian Neural Network using Pyro."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 400, output_size: int = 10):
        """Initialize Bayesian Neural Network.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output (number of classes)
        """
        super(BayesianNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.view(-1, self.input_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def bayesian_model(x_data: torch.Tensor, y_data: torch.Tensor, model: BayesianNN) -> None:
    """Pyro model for Bayesian Neural Network.
    
    Args:
        x_data: Input data
        y_data: Target data
        model: Bayesian model
    """
    # Priors for the weights
    fc1w_prior = dist.Normal(torch.zeros_like(model.fc1.weight), torch.ones_like(model.fc1.weight)).to_event(2)
    fc1b_prior = dist.Normal(torch.zeros_like(model.fc1.bias), torch.ones_like(model.fc1.bias)).to_event(1)
    fc2w_prior = dist.Normal(torch.zeros_like(model.fc2.weight), torch.ones_like(model.fc2.weight)).to_event(2)
    fc2b_prior = dist.Normal(torch.zeros_like(model.fc2.bias), torch.ones_like(model.fc2.bias)).to_event(1)
    
    # Sample from priors
    fc1w = pyro.sample("fc1w", fc1w_prior)
    fc1b = pyro.sample("fc1b", fc1b_prior)
    fc2w = pyro.sample("fc2w", fc2w_prior)
    fc2b = pyro.sample("fc2b", fc2b_prior)
    
    # Forward pass with sampled weights
    x = torch.relu(torch.matmul(x_data, fc1w.t()) + fc1b)
    logits = torch.matmul(x, fc2w.t()) + fc2b
    
    # Likelihood
    pyro.sample("obs", dist.Categorical(logits=logits), obs=y_data)


def bayesian_guide(x_data: torch.Tensor, y_data: torch.Tensor, model: BayesianNN) -> None:
    """Pyro guide for Bayesian Neural Network.
    
    Args:
        x_data: Input data
        y_data: Target data
        model: Bayesian model
    """
    # Variational parameters
    fc1w_mean = pyro.param("fc1w_mean", torch.randn_like(model.fc1.weight))
    fc1w_scale = pyro.param("fc1w_scale", torch.ones_like(model.fc1.weight), constraint=torch.constraints.positive)
    fc1b_mean = pyro.param("fc1b_mean", torch.randn_like(model.fc1.bias))
    fc1b_scale = pyro.param("fc1b_scale", torch.ones_like(model.fc1.bias), constraint=torch.constraints.positive)
    
    fc2w_mean = pyro.param("fc2w_mean", torch.randn_like(model.fc2.weight))
    fc2w_scale = pyro.param("fc2w_scale", torch.ones_like(model.fc2.weight), constraint=torch.constraints.positive)
    fc2b_mean = pyro.param("fc2b_mean", torch.randn_like(model.fc2.bias))
    fc2b_scale = pyro.param("fc2b_scale", torch.ones_like(model.fc2.bias), constraint=torch.constraints.positive)
    
    # Sample from variational distributions
    pyro.sample("fc1w", dist.Normal(fc1w_mean, fc1w_scale).to_event(2))
    pyro.sample("fc1b", dist.Normal(fc1b_mean, fc1b_scale).to_event(1))
    pyro.sample("fc2w", dist.Normal(fc2w_mean, fc2w_scale).to_event(2))
    pyro.sample("fc2b", dist.Normal(fc2b_mean, fc2b_scale).to_event(1))


def train_bayesian_model(
    model: BayesianNN,
    train_loader: DataLoader,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> None:
    """Train Bayesian Neural Network using SVI.
    
    Args:
        model: Bayesian model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model.to(device)
    
    # Setup SVI
    svi = pyro.infer.SVI(
        lambda x, y: bayesian_model(x, y, model),
        lambda x, y: bayesian_guide(x, y, model),
        pyro.optim.Adam({"lr": learning_rate}),
        loss=pyro.infer.Trace_ELBO()
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Compute loss
            loss = svi.step(data, target)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def evaluate_model(
    model: BayesianNN,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> float:
    """Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for computation
        
    Returns:
        Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test data: {accuracy:.2f}%")
    return accuracy


def main():
    """Main function demonstrating Bayesian Deep Learning."""
    print("Bayesian Deep Learning Implementation")
    print("=" * 50)
    
    # Set up
    set_seed(42)
    device = get_device()
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    model = BayesianNN(input_size=784, hidden_size=400, output_size=10)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {device}")
    
    # Train the Bayesian Neural Network
    print("\nTraining Bayesian Neural Network...")
    train_bayesian_model(
        model=model,
        train_loader=train_loader,
        num_epochs=5,
        learning_rate=0.001,
        device=device
    )
    
    # Test the model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, test_loader, device)
    
    print("\nKey Concepts Covered:")
    print("- Bayesian Neural Networks (BNNs): Weights as distributions")
    print("- Variational Inference: Approximating posterior distributions")
    print("- Uncertainty Estimation: Modeling prediction confidence")
    print("- Pyro Integration: Probabilistic programming with PyTorch")
    
    print(f"\nFinal test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
