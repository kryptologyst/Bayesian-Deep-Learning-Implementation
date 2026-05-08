#!/usr/bin/env python3
"""Evaluation script for Bayesian Deep Learning models."""

import argparse
import os
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.utils import set_seed, get_device, load_checkpoint
from src.data import get_data_loaders
from src.models import create_model
from src.train import predict_with_uncertainty
from src.metrics import evaluate_uncertainty, plot_calibration_curve, plot_uncertainty_distribution
from src.viz import plot_predictions_with_uncertainty, plot_uncertainty_heatmap


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Deep Learning models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of MC samples")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Set up
    set_seed(42)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    
    print(f"Evaluating model from {args.checkpoint}")
    print(f"Device: {device}")
    
    # Load data
    train_loader, test_loader = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    
    # Create model
    input_size = 784 if args.dataset == "mnist" else 32*32*3
    model = create_model(
        model_type="mc_dropout",  # Default to MC Dropout for evaluation
        input_size=input_size,
        hidden_size=400,
        output_size=10
    )
    
    # Load checkpoint
    epoch, loss, metadata = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    # Evaluate model
    model.eval()
    
    # Get predictions with uncertainty
    predictions, uncertainties = predict_with_uncertainty(
        model=model,
        data_loader=test_loader,
        num_samples=args.num_samples,
        device=device
    )
    
    # Get true labels
    true_labels = []
    test_images = []
    for data, labels in test_loader:
        true_labels.append(labels)
        test_images.append(data)
    true_labels = torch.cat(true_labels).to(device)
    test_images = torch.cat(test_images)
    
    # Evaluate uncertainty
    uncertainty_metrics = evaluate_uncertainty(
        y_true=true_labels,
        y_pred=predictions,
        y_uncertainty=uncertainties
    )
    
    print("\nUncertainty Metrics:")
    for metric, value in uncertainty_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    os.makedirs("assets", exist_ok=True)
    
    # Class names
    class_names = [str(i) for i in range(10)]
    
    # Plot calibration curve
    fig = plot_calibration_curve(
        y_true=true_labels.cpu().numpy(),
        y_prob=torch.max(predictions, dim=1)[0].cpu().numpy(),
        title="Calibration Curve"
    )
    fig.savefig("assets/calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot uncertainty distribution
    fig = plot_uncertainty_distribution(
        y_uncertainty=uncertainties.mean(dim=1).cpu().numpy(),
        y_true=true_labels.cpu().numpy(),
        y_pred=torch.argmax(predictions, dim=1).cpu().numpy(),
        title="Uncertainty Distribution"
    )
    fig.savefig("assets/uncertainty_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot predictions with uncertainty
    fig = plot_predictions_with_uncertainty(
        images=test_images[:16],
        predictions=predictions[:16],
        uncertainties=uncertainties[:16],
        true_labels=true_labels[:16],
        class_names=class_names,
        num_samples=16
    )
    fig.savefig("assets/predictions_with_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot uncertainty heatmap
    fig = plot_uncertainty_heatmap(
        predictions=predictions,
        uncertainties=uncertainties,
        true_labels=true_labels,
        class_names=class_names
    )
    fig.savefig("assets/uncertainty_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nEvaluation completed! Results saved to assets/")


if __name__ == "__main__":
    main()
