"""Visualization utilities for Bayesian models."""

from typing import List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def plot_training_curves(
    train_losses: List[float],
    test_accuracies: Optional[List[float]] = None,
    title: str = "Training Curves"
) -> Figure:
    """Plot training curves.
    
    Args:
        train_losses: Training losses
        test_accuracies: Optional test accuracies
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2 if test_accuracies else 1, figsize=(12, 4))
    
    if test_accuracies:
        axes[0].plot(train_losses)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(test_accuracies)
        axes[1].set_title("Test Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].grid(True, alpha=0.3)
    else:
        axes.plot(train_losses)
        axes.set_title("Training Loss")
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Loss")
        axes.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_predictions_with_uncertainty(
    images: torch.Tensor,
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_labels: torch.Tensor,
    class_names: List[str],
    num_samples: int = 16
) -> Figure:
    """Plot predictions with uncertainty visualization.
    
    Args:
        images: Input images
        predictions: Predicted probabilities
        uncertainties: Uncertainty estimates
        true_labels: True labels
        class_names: Class names
        num_samples: Number of samples to display
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # Display image
        if images[i].dim() == 3:  # CIFAR-10
            img = images[i].permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img)
        else:  # MNIST
            img = images[i].squeeze()
            ax.imshow(img, cmap='gray')
        
        # Get prediction info
        pred_class = torch.argmax(predictions[i]).item()
        true_class = true_labels[i].item()
        uncertainty = uncertainties[i].mean().item()
        
        # Color based on correctness
        color = 'green' if pred_class == true_class else 'red'
        
        ax.set_title(
            f"True: {class_names[true_class]}\n"
            f"Pred: {class_names[pred_class]}\n"
            f"Unc: {uncertainty:.3f}",
            color=color
        )
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Predictions with Uncertainty")
    plt.tight_layout()
    
    return fig


def plot_uncertainty_heatmap(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    true_labels: torch.Tensor,
    class_names: List[str]
) -> Figure:
    """Plot uncertainty heatmap by class.
    
    Args:
        predictions: Predicted probabilities
        uncertainties: Uncertainty estimates
        true_labels: True labels
        class_names: Class names
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    unc_np = uncertainties.cpu().numpy()
    true_np = true_labels.cpu().numpy()
    
    # Average uncertainty by true class
    avg_uncertainty_by_class = []
    for i in range(len(class_names)):
        mask = true_np == i
        if mask.sum() > 0:
            avg_unc = unc_np[mask].mean()
            avg_uncertainty_by_class.append(avg_unc)
        else:
            avg_uncertainty_by_class.append(0)
    
    # Plot uncertainty by true class
    bars1 = ax1.bar(class_names, avg_uncertainty_by_class)
    ax1.set_title("Average Uncertainty by True Class")
    ax1.set_ylabel("Average Uncertainty")
    ax1.tick_params(axis='x', rotation=45)
    
    # Color bars by uncertainty level
    colors = plt.cm.Reds(np.array(avg_uncertainty_by_class) / max(avg_uncertainty_by_class))
    for bar, color in zip(bars1, colors):
        bar.set_color(color)
    
    # Confusion matrix with uncertainty overlay
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_np, np.argmax(pred_np, axis=1))
    
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.figure.colorbar(im, ax=ax2)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, i] > thresh else "black")
    
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_yticklabels(class_names)
    
    plt.tight_layout()
    
    return fig


def plot_model_comparison(
    results: dict,
    metric: str = "accuracy",
    title: str = "Model Comparison"
) -> Figure:
    """Plot model comparison results.
    
    Args:
        results: Dictionary with model results
        metric: Metric to compare
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    bars = ax.bar(models, values)
    ax.set_title(title)
    ax.set_ylabel(metric.capitalize())
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    return fig
