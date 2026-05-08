#!/usr/bin/env python3
"""Main training script for Bayesian Deep Learning models."""

import argparse
import os
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf

from src.utils import set_seed, get_device, save_checkpoint
from src.data import get_data_loaders
from src.models import create_model
from src.train import train_bayesian_model, train_deterministic_model, predict_with_uncertainty
from src.metrics import evaluate_uncertainty
from src.viz import plot_training_curves, plot_predictions_with_uncertainty


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Bayesian Deep Learning models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="bayesian", help="Model type")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "device": args.device
        })
    
    # Override with command line args
    config.model = args.model
    config.dataset = args.dataset
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.seed = args.seed
    
    # Set up
    set_seed(config.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    
    print(f"Training {config.model} model on {config.dataset}")
    print(f"Device: {device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Load data
    train_loader, test_loader = get_data_loaders(
        dataset_name=config.dataset,
        batch_size=config.batch_size
    )
    
    # Create model
    input_size = 784 if config.dataset == "mnist" else 32*32*3
    model = create_model(
        model_type=config.model,
        input_size=input_size,
        hidden_size=400,
        output_size=10
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    if config.model == "bayesian":
        metrics = train_bayesian_model(
            model=model,
            train_loader=train_loader,
            num_epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device
        )
    else:
        metrics = train_deterministic_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device
        )
    
    # Evaluate model
    model.eval()
    
    # Get predictions with uncertainty
    if config.model in ["bayesian", "mc_dropout"]:
        predictions, uncertainties = predict_with_uncertainty(
            model=model,
            data_loader=test_loader,
            num_samples=100,
            device=device
        )
        
        # Get true labels
        true_labels = []
        for _, labels in test_loader:
            true_labels.append(labels)
        true_labels = torch.cat(true_labels).to(device)
        
        # Evaluate uncertainty
        uncertainty_metrics = evaluate_uncertainty(
            y_true=true_labels,
            y_pred=predictions,
            y_uncertainty=uncertainties
        )
        
        print("\nUncertainty Metrics:")
        for metric, value in uncertainty_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Save results
    os.makedirs("assets", exist_ok=True)
    
    # Plot training curves
    if "losses" in metrics:
        fig = plot_training_curves(metrics["losses"])
        fig.savefig("assets/training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    if "train_losses" in metrics and "test_accuracies" in metrics:
        fig = plot_training_curves(metrics["train_losses"], metrics["test_accuracies"])
        fig.savefig("assets/training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # Save model
    save_checkpoint(
        model=model,
        optimizer=None,  # Will be None for Bayesian models
        epoch=config.epochs,
        loss=metrics.get("losses", [0])[-1] if "losses" in metrics else 0,
        filepath=f"assets/{config.model}_{config.dataset}_final.pth",
        metadata={"config": config, "metrics": metrics}
    )
    
    print(f"\nTraining completed! Results saved to assets/")


if __name__ == "__main__":
    main()
