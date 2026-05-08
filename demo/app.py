"""Streamlit demo for Bayesian Deep Learning models."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

from src.utils import set_seed, get_device
from src.data import get_data_loaders
from src.models import create_model
from src.train import predict_with_uncertainty
from src.metrics import evaluate_uncertainty, expected_calibration_error
from src.viz import plot_predictions_with_uncertainty, plot_uncertainty_heatmap


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Bayesian Deep Learning Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Bayesian Deep Learning Demo")
    st.markdown("""
    This demo showcases Bayesian Neural Networks and uncertainty estimation in deep learning.
    Upload an image or use the sample data to see how different models handle uncertainty.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["bayesian", "mc_dropout", "simple_cnn"],
        help="Choose the type of model to use"
    )
    
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["mnist", "cifar10"],
        help="Choose the dataset for evaluation"
    )
    
    num_samples = st.sidebar.slider(
        "MC Samples",
        min_value=10,
        max_value=200,
        value=100,
        help="Number of Monte Carlo samples for uncertainty estimation"
    )
    
    # Safety disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ⚠️ Safety Notice
    This is a research/educational demo only. 
    Not for production decisions or control systems.
    """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🖼️ Image Analysis", "📈 Uncertainty Metrics", "ℹ️ About"])
    
    with tab1:
        st.header("Model Comparison")
        
        if st.button("Run Model Comparison"):
            with st.spinner("Training and evaluating models..."):
                # Set up
                set_seed(42)
                device = get_device()
                
                # Load data
                train_loader, test_loader = get_data_loaders(
                    dataset_name=dataset,
                    batch_size=128
                )
                
                # Test different models
                models = ["bayesian", "mc_dropout", "simple_cnn"]
                results = {}
                
                for model_name in models:
                    st.write(f"Evaluating {model_name}...")
                    
                    # Create model
                    input_size = 784 if dataset == "mnist" else 32*32*3
                    model = create_model(
                        model_type=model_name,
                        input_size=input_size,
                        hidden_size=400,
                        output_size=10
                    )
                    
                    # Quick evaluation (using pre-trained weights if available)
                    model.eval()
                    model.to(device)
                    
                    # Get predictions
                    if model_name in ["bayesian", "mc_dropout"]:
                        predictions, uncertainties = predict_with_uncertainty(
                            model=model,
                            data_loader=test_loader,
                            num_samples=num_samples,
                            device=device
                        )
                        
                        # Get true labels
                        true_labels = []
                        for _, labels in test_loader:
                            true_labels.append(labels)
                        true_labels = torch.cat(true_labels).to(device)
                        
                        # Evaluate
                        metrics = evaluate_uncertainty(
                            y_true=true_labels,
                            y_pred=predictions,
                            y_uncertainty=uncertainties
                        )
                        
                        results[model_name] = metrics
                    else:
                        # Simple evaluation for deterministic models
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)
                                output = model(data)
                                _, predicted = torch.max(output.data, 1)
                                total += target.size(0)
                                correct += (predicted == target).sum().item()
                        
                        results[model_name] = {"accuracy": correct / total}
                
                # Display results
                st.subheader("Results")
                
                # Create comparison table
                comparison_data = []
                for model_name, metrics in results.items():
                    row = {"Model": model_name}
                    for metric, value in metrics.items():
                        row[metric.capitalize()] = f"{value:.4f}"
                    comparison_data.append(row)
                
                st.table(comparison_data)
                
                # Plot comparison
                if len(results) > 1:
                    fig = go.Figure()
                    
                    metrics_to_plot = ["accuracy", "ece", "avg_uncertainty"]
                    for metric in metrics_to_plot:
                        if all(metric in results[model] for model in results.keys()):
                            values = [results[model][metric] for model in results.keys()]
                            fig.add_trace(go.Bar(
                                name=metric.capitalize(),
                                x=list(results.keys()),
                                y=values
                            ))
                    
                    fig.update_layout(
                        title="Model Comparison",
                        xaxis_title="Model",
                        yaxis_title="Metric Value",
                        barmode="group"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Image Analysis")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to analyze with Bayesian models"
        )
        
        if uploaded_file is not None:
            # Load and preprocess image
            image = Image.open(uploaded_file)
            
            # Convert to grayscale if MNIST
            if dataset == "mnist":
                image = image.convert('L')
                image = image.resize((28, 28))
            else:
                image = image.resize((32, 32))
            
            # Convert to tensor
            image_array = np.array(image)
            if dataset == "mnist":
                image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            else:
                image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Display image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Input Image", use_column_width=True)
            
            with col2:
                st.subheader("Analysis")
                
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        # Set up
                        set_seed(42)
                        device = get_device()
                        
                        # Create model
                        input_size = 784 if dataset == "mnist" else 32*32*3
                        model = create_model(
                            model_type=model_type,
                            input_size=input_size,
                            hidden_size=400,
                            output_size=10
                        )
                        
                        model.eval()
                        model.to(device)
                        image_tensor = image_tensor.to(device)
                        
                        # Get predictions
                        if model_type in ["bayesian", "mc_dropout"]:
                            predictions = []
                            for _ in range(num_samples):
                                with torch.no_grad():
                                    output = model(image_tensor)
                                    probs = torch.softmax(output, dim=1)
                                    predictions.append(probs)
                            
                            predictions = torch.stack(predictions, dim=0)
                            mean_pred = torch.mean(predictions, dim=0)
                            uncertainty = torch.std(predictions, dim=0)
                            
                            # Display results
                            pred_class = torch.argmax(mean_pred, dim=1).item()
                            confidence = torch.max(mean_pred, dim=1)[0].item()
                            avg_uncertainty = torch.mean(uncertainty).item()
                            
                            class_names = [str(i) for i in range(10)]
                            
                            st.write(f"**Predicted Class:** {class_names[pred_class]}")
                            st.write(f"**Confidence:** {confidence:.3f}")
                            st.write(f"**Uncertainty:** {avg_uncertainty:.3f}")
                            
                            # Plot prediction distribution
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=class_names,
                                y=mean_pred[0].cpu().numpy(),
                                name="Mean Prediction"
                            ))
                            fig.add_trace(go.Bar(
                                x=class_names,
                                y=uncertainty[0].cpu().numpy(),
                                name="Uncertainty",
                                opacity=0.7
                            ))
                            
                            fig.update_layout(
                                title="Prediction Distribution",
                                xaxis_title="Class",
                                yaxis_title="Probability/Uncertainty"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            with torch.no_grad():
                                output = model(image_tensor)
                                probs = torch.softmax(output, dim=1)
                                pred_class = torch.argmax(probs, dim=1).item()
                                confidence = torch.max(probs, dim=1)[0].item()
                            
                            class_names = [str(i) for i in range(10)]
                            
                            st.write(f"**Predicted Class:** {class_names[pred_class]}")
                            st.write(f"**Confidence:** {confidence:.3f}")
                            
                            # Plot prediction distribution
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=class_names,
                                y=probs[0].cpu().numpy()
                            ))
                            
                            fig.update_layout(
                                title="Prediction Distribution",
                                xaxis_title="Class",
                                yaxis_title="Probability"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Uncertainty Metrics")
        
        st.markdown("""
        ### Understanding Uncertainty Metrics
        
        **Expected Calibration Error (ECE):** Measures how well-calibrated the model's confidence is.
        Lower values indicate better calibration.
        
        **Maximum Calibration Error (MCE):** The maximum difference between confidence and accuracy
        across all confidence bins.
        
        **Brier Score:** Measures the accuracy of probabilistic predictions. Lower is better.
        
        **Coverage:** The fraction of true labels that fall within the predicted confidence intervals.
        Should be close to the confidence level (e.g., 95% coverage for 95% confidence intervals).
        """)
        
        if st.button("Generate Sample Metrics"):
            # Generate sample metrics for demonstration
            sample_metrics = {
                "Expected Calibration Error": 0.045,
                "Maximum Calibration Error": 0.123,
                "Brier Score": 0.089,
                "Coverage (95%)": 0.942,
                "Coverage (90%)": 0.887,
                "Coverage (80%)": 0.798,
                "Average Uncertainty": 0.156
            }
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Calibration Metrics")
                for metric, value in list(sample_metrics.items())[:3]:
                    st.metric(metric, f"{value:.3f}")
            
            with col2:
                st.subheader("Coverage Metrics")
                for metric, value in list(sample_metrics.items())[3:6]:
                    st.metric(metric, f"{value:.3f}")
            
            st.metric("Average Uncertainty", f"{sample_metrics['Average Uncertainty']:.3f}")
            
            # Plot calibration curve
            fig = go.Figure()
            
            # Sample calibration curve data
            confidence_bins = np.linspace(0, 1, 11)
            accuracy_bins = confidence_bins + np.random.normal(0, 0.05, len(confidence_bins))
            accuracy_bins = np.clip(accuracy_bins, 0, 1)
            
            fig.add_trace(go.Scatter(
                x=confidence_bins,
                y=accuracy_bins,
                mode='markers+lines',
                name='Model',
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title="Calibration Curve",
                xaxis_title="Confidence",
                yaxis_title="Accuracy",
                width=600,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This Demo")
        
        st.markdown("""
        ### Bayesian Deep Learning
        
        This demo showcases the implementation of Bayesian Neural Networks using Pyro,
        a probabilistic programming library built on PyTorch. Bayesian Neural Networks
        treat model weights as random variables with distributions, enabling uncertainty
        quantification in predictions.
        
        ### Key Features
        
        - **Bayesian Neural Networks**: Full probabilistic treatment of weights
        - **Monte Carlo Dropout**: Efficient uncertainty estimation
        - **Uncertainty Metrics**: ECE, MCE, Brier Score, Coverage
        - **Interactive Visualization**: Real-time model comparison
        - **Multiple Datasets**: MNIST and CIFAR-10 support
        
        ### Models Implemented
        
        1. **Bayesian Neural Network**: Uses Pyro for variational inference
        2. **Monte Carlo Dropout**: Dropout-based uncertainty estimation
        3. **Simple CNN**: Baseline deterministic model
        
        ### Safety & Ethics
        
        ⚠️ **Important Disclaimers:**
        - This is a research/educational demonstration only
        - Not suitable for production decisions or control systems
        - Models may have biases and limitations
        - Always validate results with domain experts
        
        ### Author
        
        **kryptologyst** - [GitHub](https://github.com/kryptologyst)
        
        This project is part of the 1000 AI Projects series focusing on
        Meta and Hybrid AI implementations.
        """)


if __name__ == "__main__":
    main()
