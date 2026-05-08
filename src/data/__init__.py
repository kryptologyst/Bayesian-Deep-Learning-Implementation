"""Data loading and preprocessing utilities."""

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """MNIST dataset wrapper with proper preprocessing."""
    
    def __init__(self, train: bool = True, transform: Optional[transforms.Compose] = None):
        """Initialize MNIST dataset.
        
        Args:
            train: Whether to load training or test data
            transform: Optional transforms to apply
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
            ])
        
        self.dataset = datasets.MNIST(
            root='./data', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper with proper preprocessing."""
    
    def __init__(self, train: bool = True, transform: Optional[transforms.Compose] = None):
        """Initialize CIFAR-10 dataset.
        
        Args:
            train: Whether to load training or test data
            transform: Optional transforms to apply
        """
        if transform is None:
            if train:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        
        self.dataset = datasets.CIFAR10(
            root='./data', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


def get_data_loaders(
    dataset_name: str = "mnist",
    batch_size: int = 128,
    num_workers: int = 4,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders for training and testing.
    
    Args:
        dataset_name: Name of dataset ("mnist" or "cifar10")
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_transform: Transforms for training data
        test_transform: Transforms for test data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_name.lower() == "mnist":
        train_dataset = MNISTDataset(train=True, transform=train_transform)
        test_dataset = MNISTDataset(train=False, transform=test_transform)
    elif dataset_name.lower() == "cifar10":
        train_dataset = CIFAR10Dataset(train=True, transform=train_transform)
        test_dataset = CIFAR10Dataset(train=False, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
