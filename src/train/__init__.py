"""Training utilities for Bayesian models."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from .models import BayesianNN, MCDropoutNN


def bayesian_model(x_data: torch.Tensor, y_data: torch.Tensor, model: BayesianNN) -> None:
    """Pyro model for Bayesian Neural Network.
    
    Args:
        x_data: Input data
        y_data: Target data
        model: Bayesian model
    """
    # Sample from the model
    logits = model(x_data)
    
    # Likelihood
    pyro.sample("obs", dist.Categorical(logits=logits), obs=y_data)


def bayesian_guide(x_data: torch.Tensor, y_data: torch.Tensor, model: BayesianNN) -> None:
    """Pyro guide for Bayesian Neural Network.
    
    Args:
        x_data: Input data
        y_data: Target data
        model: Bayesian model
    """
    # The guide is automatically defined by PyroSample in the model
    pass


def train_bayesian_model(
    model: BayesianNN,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Train Bayesian Neural Network using SVI.
    
    Args:
        model: Bayesian model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Dictionary with training metrics
    """
    model.to(device)
    
    # Setup SVI
    svi = SVI(
        lambda x, y: bayesian_model(x, y, model),
        lambda x, y: bayesian_guide(x, y, model),
        Adam({"lr": learning_rate}),
        loss=Trace_ELBO()
    )
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Compute loss
            loss = svi.step(data, target)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return {"losses": losses}


def train_deterministic_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Train deterministic model (MC Dropout or regular NN).
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Dictionary with training metrics
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return {
        "train_losses": train_losses,
        "test_accuracies": test_accuracies
    }


def predict_with_uncertainty(
    model: nn.Module,
    data_loader: DataLoader,
    num_samples: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict with uncertainty estimation using MC Dropout.
    
    Args:
        model: Trained model
        data_loader: Data loader for predictions
        num_samples: Number of Monte Carlo samples
        device: Device for computation
        
    Returns:
        Tuple of (predictions, uncertainties)
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            batch_predictions = []
            
            # Sample multiple predictions
            for _ in range(num_samples):
                if isinstance(model, MCDropoutNN):
                    output = model(data, training=True)  # Keep dropout on
                else:
                    output = model(data)
                
                probs = torch.softmax(output, dim=1)
                batch_predictions.append(probs)
            
            # Stack predictions
            batch_predictions = torch.stack(batch_predictions, dim=0)
            all_predictions.append(batch_predictions)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=1)
    
    # Compute mean and uncertainty
    mean_predictions = torch.mean(all_predictions, dim=0)
    uncertainty = torch.std(all_predictions, dim=0)
    
    return mean_predictions, uncertainty
