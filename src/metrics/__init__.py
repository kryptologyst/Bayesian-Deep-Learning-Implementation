"""Evaluation metrics for Bayesian models."""

from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        MCE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier Score
    """
    return np.mean((y_prob - y_true) ** 2)


def evaluate_uncertainty(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_uncertainty: torch.Tensor,
    confidence_levels: List[float] = [0.95, 0.90, 0.80, 0.70]
) -> Dict[str, float]:
    """Evaluate uncertainty quality metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        y_uncertainty: Uncertainty estimates
        confidence_levels: Confidence levels to evaluate
        
    Returns:
        Dictionary of uncertainty metrics
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_uncertainty_np = y_uncertainty.cpu().numpy()
    
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true_np, np.argmax(y_pred_np, axis=1))
    metrics["ece"] = expected_calibration_error(y_true_np, np.max(y_pred_np, axis=1))
    metrics["mce"] = maximum_calibration_error(y_true_np, np.max(y_pred_np, axis=1))
    metrics["brier_score"] = brier_score(y_true_np, np.max(y_pred_np, axis=1))
    
    # Uncertainty-based metrics
    avg_uncertainty = np.mean(y_uncertainty_np)
    metrics["avg_uncertainty"] = avg_uncertainty
    
    # Confidence intervals
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        lower_bound = np.percentile(y_pred_np, (alpha/2) * 100, axis=1)
        upper_bound = np.percentile(y_pred_np, (1 - alpha/2) * 100, axis=1)
        
        # Check if true labels fall within confidence interval
        in_interval = (y_true_np >= lower_bound) & (y_true_np <= upper_bound)
        coverage = np.mean(in_interval)
        metrics[f"coverage_{conf_level}"] = coverage
    
    return metrics


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Calibration Curve",
    n_bins: int = 10
) -> plt.Figure:
    """Plot calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        n_bins: Number of bins
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_uncertainty_distribution(
    y_uncertainty: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Uncertainty Distribution"
) -> plt.Figure:
    """Plot uncertainty distribution by correctness.
    
    Args:
        y_uncertainty: Uncertainty estimates
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct = (y_true == y_pred)
    incorrect = ~correct
    
    ax.hist(y_uncertainty[correct], bins=30, alpha=0.7, label="Correct", density=True)
    ax.hist(y_uncertainty[incorrect], bins=30, alpha=0.7, label="Incorrect", density=True)
    
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
