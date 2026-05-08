"""Tests for Bayesian Deep Learning models."""

import pytest
import torch
import numpy as np

from src.utils import set_seed, get_device, count_parameters
from src.data import MNISTDataset, CIFAR10Dataset, get_data_loaders
from src.models import BayesianNN, MCDropoutNN, SimpleCNN, create_model
from src.metrics import expected_calibration_error, maximum_calibration_error, brier_score


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        assert param_count == 55  # 10*5 + 5


class TestData:
    """Test data loading functionality."""
    
    def test_mnist_dataset(self):
        """Test MNIST dataset loading."""
        dataset = MNISTDataset(train=True)
        assert len(dataset) > 0
        
        # Test data loading
        data, target = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, int)
        assert data.shape == (1, 28, 28)
    
    def test_cifar10_dataset(self):
        """Test CIFAR-10 dataset loading."""
        dataset = CIFAR10Dataset(train=True)
        assert len(dataset) > 0
        
        # Test data loading
        data, target = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, int)
        assert data.shape == (3, 32, 32)
    
    def test_get_data_loaders(self):
        """Test data loader creation."""
        train_loader, test_loader = get_data_loaders(
            dataset_name="mnist",
            batch_size=32,
            num_workers=0  # Use 0 for testing
        )
        
        assert len(train_loader) > 0
        assert len(test_loader) > 0
        
        # Test batch loading
        for data, target in train_loader:
            assert data.shape[0] <= 32
            assert data.shape[1:] == (1, 28, 28)
            assert target.shape[0] == data.shape[0]
            break


class TestModels:
    """Test model implementations."""
    
    def test_bayesian_nn(self):
        """Test Bayesian Neural Network."""
        model = BayesianNN(input_size=784, hidden_size=400, output_size=10)
        
        # Test forward pass
        x = torch.randn(32, 784)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert isinstance(output, torch.Tensor)
    
    def test_mc_dropout_nn(self):
        """Test Monte Carlo Dropout Neural Network."""
        model = MCDropoutNN(input_size=784, hidden_size=400, output_size=10)
        
        # Test forward pass
        x = torch.randn(32, 784)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert isinstance(output, torch.Tensor)
    
    def test_simple_cnn(self):
        """Test Simple CNN."""
        model = SimpleCNN(num_classes=10)
        
        # Test forward pass
        x = torch.randn(32, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (32, 10)
        assert isinstance(output, torch.Tensor)
    
    def test_create_model(self):
        """Test model creation function."""
        # Test Bayesian model
        model = create_model("bayesian", input_size=784, output_size=10)
        assert isinstance(model, BayesianNN)
        
        # Test MC Dropout model
        model = create_model("mc_dropout", input_size=784, output_size=10)
        assert isinstance(model, MCDropoutNN)
        
        # Test Simple CNN
        model = create_model("simple_cnn", output_size=10)
        assert isinstance(model, SimpleCNN)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_model("invalid_model")


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_expected_calibration_error(self):
        """Test ECE calculation."""
        # Perfect calibration
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.9, 0.1])
        
        ece = expected_calibration_error(y_true, y_prob)
        assert ece >= 0
        assert isinstance(ece, float)
    
    def test_maximum_calibration_error(self):
        """Test MCE calculation."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.9, 0.1])
        
        mce = maximum_calibration_error(y_true, y_prob)
        assert mce >= 0
        assert isinstance(mce, float)
    
    def test_brier_score(self):
        """Test Brier Score calculation."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.9, 0.1])
        
        brier = brier_score(y_true, y_prob)
        assert brier >= 0
        assert isinstance(brier, float)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Create small dataset
        train_loader, test_loader = get_data_loaders(
            dataset_name="mnist",
            batch_size=32,
            num_workers=0
        )
        
        # Create model
        model = MCDropoutNN(input_size=784, hidden_size=100, output_size=10)
        
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(2):  # Just 2 epochs for testing
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:  # Limit batches for testing
                    break
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Test evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                if total >= 100:  # Limit for testing
                    break
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        assert accuracy > 0  # Should be better than random
        assert accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
