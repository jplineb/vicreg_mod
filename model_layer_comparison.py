from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json

from custom_datasets import DATASETS
from torchvision.models import resnet50
from utils.log_config import configure_logging

import torch
import torch.nn as nn

logger = configure_logging()


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Compare feature maps between two models"
    )
    parser.add_argument(
        "--model1-path",
        type=Path,
        required=True,
        help="Path to the first model's weights",
    )
    parser.add_argument(
        "--model2-path",
        type=Path,
        required=True,
        help="Path to the second model's weights",
    )
    parser.add_argument(
        "--model1-name",
        type=str,
        default=None,
        help="Custom name for the first model (defaults to filename)",
    )
    parser.add_argument(
        "--model2-name",
        type=str,
        default=None,
        help="Custom name for the second model (defaults to filename)",
    )
    parser.add_argument(
        "--task-ds",
        type=str,
        required=True,
        help="Dataset to use for feature extraction",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to extract features from. "
             "Available indices: 0 (conv1), 1 (bn1), 2 (relu), 3 (maxpool), "
             "4 (layer1), 5 (layer2), 6 (layer3), 7 (layer4), 8 (avgpool). "
             "For inner blocks: 100-102 (layer1.0-2), 200-203 (layer2.0-3), "
             "300-305 (layer3.0-5), 400-402 (layer4.0-2). "
             "For components within blocks: e.g., 101 (layer1.0.conv1), "
             "102 (layer1.0.bn1), etc. See code for full mapping. "
             "If not specified, will use all main layers (0-8)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for dataset loading"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./layer_comparisons",
        help="Directory to save output visualizations",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of feature maps",
    )
    return parser.parse_args()


def load_model(model_path):
    """
    Load a model directly from path.

    Args:
        model_path (Path): Path to the model weights

    Returns:
        nn.Module: Loaded model
    """
    # Load the model state dict
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle different state dict formats
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Create ResNet50 model
    model = resnet50()

    # Replace the final FC layer with Identity for feature extraction
    model.fc = nn.Identity()  # type: ignore

    # Process state dict keys if needed (remove prefixes like 'backbone.')
    processed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module.backbone."):
            new_key = key.replace("module.backbone.", "")
            processed_state_dict[new_key] = value
        elif key.startswith("backbone."):
            new_key = key.replace("backbone.", "")
            processed_state_dict[new_key] = value
        else:
            processed_state_dict[key] = value

    # Load state dict
    model.load_state_dict(processed_state_dict, strict=False)

    return model.cuda()


def visualize_feature_maps(features1, features2, output_dir, sample_idx, layer_idx, original_image=None, layer_name=None):
    """
    Visualize and save comparisons of feature maps.

    Args:
        features1 (torch.Tensor): Features from model 1
        features2 (torch.Tensor): Features from model 2
        output_dir (Path): Directory to save visualizations
        sample_idx (int): Index of the sample
        layer_idx (int): Index of the layer
        original_image (torch.Tensor, optional): Original input image
        layer_name (str, optional): Name of the layer
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Take first image in batch if not already single image
    if len(features1.shape) > 3:  # If batch dimension exists
        features1 = features1[0]
        features2 = features2[0]
    
    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()

    # Calculate number of channels to visualize (up to 16 per model)
    num_channels = min(16, features1.shape[0])
    
    # Determine grid layout - we'll use 4 columns for a better layout
    num_cols = 4
    num_rows = (num_channels + num_cols - 1) // num_cols  # Ceiling division
    
    # Create a figure with appropriate layout
    if original_image is not None:
        # Create a figure with 2*num_rows + 1 rows (original image + features from both models)
        fig, axs = plt.subplots(2*num_rows + 1, num_cols, figsize=(20, 5 + 4*num_rows))
        
        # Display original image in the center of the first row
        for i in range(num_cols):
            if i == num_cols // 2 - 1 or i == num_cols // 2:
                # Convert image from tensor to numpy and adjust format
                img = original_image.cpu().numpy().transpose(1, 2, 0)
                # Normalize image for visualization
                img = (img - img.min()) / (img.max() - img.min())
                if i == num_cols // 2 - 1:  # Left-center position
                    axs[0, i].imshow(img)
                    axs[0, i].set_title("Original Image")
            if hasattr(axs[0, i], 'axis'):  # Check if the subplot exists
                axs[0, i].axis('off')
        
        # Display model 1 features
        for i in range(num_channels):
            row = 1 + (i // num_cols)
            col = i % num_cols
            if hasattr(axs[row, col], 'imshow'):  # Check if the subplot exists
                axs[row, col].imshow(features1[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 1")
                axs[row, col].axis('off')
        
        # Display model 2 features
        for i in range(num_channels):
            row = 1 + num_rows + (i // num_cols)
            col = i % num_cols
            if hasattr(axs[row, col], 'imshow'):  # Check if the subplot exists
                axs[row, col].imshow(features2[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 2")
                axs[row, col].axis('off')
    else:
        # Create a figure with 2*num_rows rows (features from both models)
        fig, axs = plt.subplots(2*num_rows, num_cols, figsize=(20, 4*num_rows))
        
        # Handle the case where num_rows is 1 (axs becomes 1D)
        if num_rows == 1:
            axs = axs.reshape(2, num_cols)
        
        # Display model 1 features
        for i in range(num_channels):
            row = i // num_cols
            col = i % num_cols
            if row < axs.shape[0] and col < axs.shape[1]:  # Check bounds
                axs[row, col].imshow(features1[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 1")
                axs[row, col].axis('off')
        
        # Display model 2 features
        for i in range(num_channels):
            row = num_rows + (i // num_cols)
            col = i % num_cols
            if row < axs.shape[0] and col < axs.shape[1]:  # Check bounds
                axs[row, col].imshow(features2[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 2")
                axs[row, col].axis('off')

    # Add a main title with layer information
    title = f"Feature Maps - Layer {layer_idx}"
    if layer_name:
        title += f" ({layer_name})"
    plt.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # type: ignore # Adjust for the suptitle
    plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}_layer_{layer_idx}.png"), dpi=150)
    plt.close()


def calculate_statistics(features1: torch.Tensor, features2: torch.Tensor) -> dict:
    """
    Calculate statistics to compare feature maps.

    Args:
        features1 (torch.Tensor): Features from model 1
        features2 (torch.Tensor): Features from model 2

    Returns:
        dict: Dictionary of statistics
    """
    # Calculate mean activations
    mean1 = features1.mean().item()
    mean2 = features2.mean().item()

    # Calculate standard deviation of activations
    std1 = features1.std().item()
    std2 = features2.std().item()

    # Calculate cosine similarity between flattened feature maps
    features1_flat = features1.view(features1.size(0), -1)
    features2_flat = features2.view(features2.size(0), -1)

    # Normalize features
    features1_norm = features1_flat / features1_flat.norm(dim=1, keepdim=True)
    features2_norm = features2_flat / features2_flat.norm(dim=1, keepdim=True)
    # Calculate coefficient of variation
    cov1 = std1 / mean1
    cov2 = std2 / mean1

    # Calculate l2 distance
    l2_dist = torch.norm(features1 - features2, dim=1).mean().item()

    # Calculate cosine similarity
    cosine_sim = torch.mean(torch.sum(features1_norm * features2_norm, dim=1)).item()

    return {
        "mean1": mean1,
        "mean2": mean2,
        "std1": std1,
        "std2": std2,
        "cosine_similarity": cosine_sim,
        "l2_distance": l2_dist,
        "cov1": cov1,
        "cov2": cov2,
    }


class FeatureExtractor:
    def __init__(self, model):
        """
        Initialize feature extractor for a given model.
        Automatically registers hooks for all layers.

        Args:
            model (nn.Module): The model to extract features from
        """
        self.model = model
        self.features = {}
        self.hooks = []
        self.layer_names = {}
        
        # Register hooks for all layers
        self._register_hooks(model)
        
    def _register_hooks(self, module, prefix=''):
        """
        Recursively register forward hooks on all layers.
        
        Args:
            module (nn.Module): Module to register hooks on
            prefix (str): Prefix for layer name
        """
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            
            # Register hook for this layer
            layer_id = len(self.layer_names)
            self.layer_names[layer_id] = layer_name
            
            self.hooks.append(
                layer.register_forward_hook(self._get_hook(layer_id))
            )
            logger.info(f"Registered hook for layer {layer_name} (index {layer_id})")
            
            # Recursively register hooks for children
            self._register_hooks(layer, layer_name)

    def _get_hook(self, layer_id):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            self.features[layer_id] = output.detach()
        return hook

    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Extract features for input x.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Dictionary of features for each layer
        """
        # Reset features
        self.features = {}
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            self.model(x)
        return self.features

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def visualize_feature_distributions(features1_all, features2_all, output_dir, layer_idx, layer_name, model1_name="Model 1", model2_name="Model 2"):
    """
    Visualize and save distributions of feature activations.

    Args:
        features1_all (list): List of feature tensors from model 1 for all samples
        features2_all (list): List of feature tensors from model 2 for all samples
        output_dir (Path): Directory to save visualizations
        layer_idx (int): Index of the layer
        layer_name (str): Name of the layer
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Concatenate all features
    features1_concat = torch.cat(features1_all, dim=0)
    features2_concat = torch.cat(features2_all, dim=0)
    
    # Convert to numpy for plotting
    features1_np = features1_concat.cpu().flatten().numpy()
    features2_np = features2_concat.cpu().flatten().numpy()
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot histograms of feature distributions
    bins = 100
    
    # Model 1 distribution
    ax1.hist(features1_np, bins=bins, alpha=0.7, color='blue', density=True)
    ax1.set_title(f"{model1_name} Feature Distribution")
    ax1.set_xlabel("Feature Value")
    ax1.set_ylabel("Density")
    
    # Model 2 distribution
    ax2.hist(features2_np, bins=bins, alpha=0.7, color='orange', density=True)
    ax2.set_title(f"{model2_name} Feature Distribution")
    ax2.set_xlabel("Feature Value")
    
    # Overlay both distributions
    ax3.hist(features1_np, bins=bins, alpha=0.5, color='blue', density=True, label=model1_name)
    ax3.hist(features2_np, bins=bins, alpha=0.5, color='orange', density=True, label=model2_name)
    ax3.set_title(f"Overlaid Feature Distributions")
    ax3.set_xlabel("Feature Value")
    ax3.legend()
    
    # Add statistics to the plot
    stats1 = f"{model1_name} - Mean: {features1_np.mean():.4f}, Std: {features1_np.std():.4f}"
    stats2 = f"{model2_name} - Mean: {features2_np.mean():.4f}, Std: {features2_np.std():.4f}"
    fig.text(0.5, 0.01, f"{stats1}\n{stats2}", ha='center', fontsize=12)
    
    # Add a main title
    plt.suptitle(f"Feature Distributions - Layer {layer_idx} ({layer_name})", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, f"feature_distribution_layer_{layer_idx}.png"), dpi=150)
    plt.close()
    
    # Also create a scatter plot of channel-wise statistics
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate channel-wise means and stds
    channel_means1 = []
    channel_stds1 = []
    channel_means2 = []
    channel_stds2 = []
    
    # Get number of channels (assuming first dimension is batch, second is channel)
    num_channels = features1_concat.shape[1]
    
    for c in range(num_channels):
        # Get all values for this channel across all samples
        channel_vals1 = features1_concat[:, c, ...].cpu().flatten().numpy()
        channel_vals2 = features2_concat[:, c, ...].cpu().flatten().numpy()
        
        channel_means1.append(channel_vals1.mean())
        channel_stds1.append(channel_vals1.std())
        channel_means2.append(channel_vals2.mean())
        channel_stds2.append(channel_vals2.std())
    
    # Plot channel-wise means vs stds
    ax.scatter(channel_means1, channel_stds1, alpha=0.7, label=model1_name, color='blue')
    ax.scatter(channel_means2, channel_stds2, alpha=0.7, label=model2_name, color='orange')
    
    # Add labels and title
    ax.set_xlabel("Channel Mean")
    ax.set_ylabel("Channel Standard Deviation")
    ax.set_title(f"Channel-wise Statistics - Layer {layer_idx} ({layer_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"channel_stats_layer_{layer_idx}.png"), dpi=150)
    plt.close()


def visualize_cosine_similarities(cosine_sims, layer_idx, layer_name, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Visualize cosine similarities between model features for each sample.
    
    Args:
        cosine_sims (list): List of cosine similarity values for each sample
        layer_idx (int): Index of the layer
        layer_name (str): Name of the layer
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cosine similarities
    sample_indices = list(range(len(cosine_sims)))
    ax.plot(sample_indices, cosine_sims, 'o-', linewidth=2, markersize=8)
    
    # Add horizontal line at mean
    mean_sim = np.mean(cosine_sims)
    ax.axhline(y=mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Per-Sample Cosine Similarity Between {model1_name} and {model2_name}\nLayer {layer_idx} ({layer_name})')
    
    # Set y-axis limits to better visualize the similarities
    y_min = max(min(cosine_sims) - 0.1, -1.0)
    y_max = min(max(cosine_sims) + 0.1, 1.0)
    ax.set_ylim(y_min, y_max)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cosine_similarity_per_sample_layer_{layer_idx}.png'), dpi=150)
    plt.close()


def calculate_per_sample_cosine_similarity(features1: torch.Tensor, features2: torch.Tensor) -> list:
    """
    Calculate cosine similarity for each sample between two feature tensors.
    
    Args:
        features1 (torch.Tensor): Features from model 1
        features2 (torch.Tensor): Features from model 2
        
    Returns:
        list: List of cosine similarity values for each sample
    """
    # Get batch size
    batch_size = features1.size(0)
    
    # Initialize list to store similarities
    similarities = []
    
    # Calculate similarity for each sample
    for i in range(batch_size):
        # Get features for this sample
        f1 = features1[i].view(-1)  # Flatten all dimensions except batch
        f2 = features2[i].view(-1)
        
        # Normalize features
        f1_norm = f1 / (f1.norm() + 1e-8)  # Add small epsilon to avoid division by zero
        f2_norm = f2 / (f2.norm() + 1e-8)
        
        # Calculate cosine similarity
        sim = torch.dot(f1_norm, f2_norm).item()
        similarities.append(sim)
    
    return similarities


def visualize_layer_comparison(layer_stats, output_dir, model1_name, model2_name):
    """
    Visualize comparison metrics across different layers.
    
    Args:
        layer_stats (dict): Dictionary with layer indices as keys and statistics as values
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model for display
        model2_name (str): Name of the second model for display
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract layer indices and names, and sort by layer index
    layers = sorted(layer_stats.keys())
    layer_names = [layer_stats[layer]['name'] for layer in layers]
    
    # Create shortened names for display (to avoid overcrowding)
    display_names = []
    for name in layer_names:
        if len(name) > 20:
            # Shorten long names
            parts = name.split('.')
            if len(parts) > 2:
                display_names.append(f"{parts[0]}...{parts[-1]}")
            else:
                display_names.append(name[:17] + "...")
        else:
            display_names.append(name)
    
    # Extract metrics
    l2_distances = [layer_stats[layer]['l2_distance'] for layer in layers]
    cosine_sims = [layer_stats[layer]['cosine_similarity'] for layer in layers]
    
    # Determine figure size based on number of layers
    fig_width = max(12, len(layers) * 0.5)  # Increase width for more layers
    
    # For line charts, identify parent layers to use as x-tick labels
    parent_layers = set()
    parent_layer_indices = []
    parent_layer_names = []
    
    for i, name in enumerate(layer_names):
        # Get the top-level parent (e.g., 'conv1', 'layer1', etc.)
        parent = name.split('.')[0]
        if parent not in parent_layers:
            parent_layers.add(parent)
            parent_layer_indices.append(layers[i])
            parent_layer_names.append(parent)
    
    # 1. L2 Distance Line Plot
    fig_l2_line, ax_l2_line = plt.subplots(figsize=(fig_width, 6))
    ax_l2_line.plot(layers, l2_distances, 'o-', linewidth=2, markersize=10, color='blue')
    ax_l2_line.set_xlabel('Layer')
    ax_l2_line.set_ylabel('L2 Distance')
    ax_l2_line.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
    
    # Set x-ticks at parent layer positions
    ax_l2_line.set_xticks(parent_layer_indices)
    ax_l2_line.set_xticklabels(parent_layer_names, rotation=45, ha='right')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax_l2_line.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax_l2_line.grid(True, alpha=0.3)
    
    # Add value labels (only for a reasonable number of points)
    if len(layers) <= 30:
        for i, l2 in enumerate(l2_distances):
            ax_l2_line.annotate(f'{l2:.2f}', 
                        (layers[i], l2),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l2_distance_line.png'), dpi=150)
    plt.close(fig_l2_line)
    
    # 2. Cosine Similarity Line Plot
    fig_cos_line, ax_cos_line = plt.subplots(figsize=(fig_width, 6))
    ax_cos_line.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=10, color='green')
    ax_cos_line.set_xlabel('Layer')
    ax_cos_line.set_ylabel('Cosine Similarity')
    ax_cos_line.set_title(f'Cosine Similarity Between {model1_name} and {model2_name} by Layer')
    
    # Set x-ticks at parent layer positions
    ax_cos_line.set_xticks(parent_layer_indices)
    ax_cos_line.set_xticklabels(parent_layer_names, rotation=45, ha='right')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax_cos_line.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax_cos_line.grid(True, alpha=0.3)
    
    # Add value labels (only for a reasonable number of points)
    if len(layers) <= 30:
        for i, sim in enumerate(cosine_sims):
            ax_cos_line.annotate(f'{sim:.2f}', 
                        (layers[i], sim),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    # Set y-axis limits for cosine similarity
    ax_cos_line.set_ylim(-1.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_line.png'), dpi=150)
    plt.close(fig_cos_line)
    
    # 3. L2 Distance Bar Chart - use horizontal bars for better label display
    fig_l2_bar, ax_l2_bar = plt.subplots(figsize=(10, max(8, len(layers) * 0.3)))
    
    # Reverse the order of layers, names, and values for the bar chart
    reversed_display_names = display_names[::-1]
    reversed_l2_distances = l2_distances[::-1]
    
    y_pos = np.arange(len(reversed_display_names))
    ax_l2_bar.barh(y_pos, reversed_l2_distances, color='skyblue')
    ax_l2_bar.set_xlabel('L2 Distance')
    ax_l2_bar.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
    ax_l2_bar.set_yticks(y_pos)
    ax_l2_bar.set_yticklabels(reversed_display_names)
    
    # Add value labels
    for i, l2 in enumerate(reversed_l2_distances):
        ax_l2_bar.text(l2 + max(reversed_l2_distances)*0.02, i, f'{l2:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l2_distance_bar.png'), dpi=150)
    plt.close(fig_l2_bar)
    
    # 4. Cosine Similarity Bar Chart - use horizontal bars for better label display
    fig_cos_bar, ax_cos_bar = plt.subplots(figsize=(10, max(8, len(layers) * 0.3)))
    
    # Reverse the order for cosine similarity as well
    reversed_cosine_sims = cosine_sims[::-1]
    
    ax_cos_bar.barh(y_pos, reversed_cosine_sims, color='lightgreen')
    ax_cos_bar.set_xlabel('Cosine Similarity')
    ax_cos_bar.set_title(f'Cosine Similarity Between {model1_name} and {model2_name} by Layer')
    ax_cos_bar.set_yticks(y_pos)
    ax_cos_bar.set_yticklabels(reversed_display_names)
    
    # Add value labels
    for i, sim in enumerate(reversed_cosine_sims):
        label_pos = sim + 0.05 if sim > 0 else sim - 0.1
        ax_cos_bar.text(label_pos, i, f'{sim:.2f}', va='center')
    
    # Set x-axis limits for cosine similarity
    ax_cos_bar.set_xlim(-1.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_bar.png'), dpi=150)
    plt.close(fig_cos_bar)
    
    # 5. Combined plot - also update this to use parent layer names
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot L2 distances
    ax1.plot(layers, l2_distances, 'o-', linewidth=2, markersize=10, color='blue')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name}')
    
    # Set x-ticks at parent layer positions
    ax1.set_xticks(parent_layer_indices)
    ax1.set_xticklabels(parent_layer_names, rotation=45, ha='right')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax1.grid(True, alpha=0.3)
    
    # Plot Cosine similarities
    ax2.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Cosine Similarity Between {model1_name} and {model2_name}')
    
    # Set x-ticks at parent layer positions
    ax2.set_xticks(parent_layer_indices)
    ax2.set_xticklabels(parent_layer_names, rotation=45, ha='right')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax2.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limits for cosine similarity
    ax2.set_ylim(-1.05, 1.05)
    
    plt.suptitle(f'Layer-wise Comparison Between {model1_name} and {model2_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, 'layer_comparison.png'), dpi=150)
    plt.close()


def compare_model_weights(model1, model2, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Compare weights between two models and generate visualizations.
    
    Args:
        model1 (nn.Module): First model
        model2 (nn.Module): Second model
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a directory for weight comparisons
    weights_dir = os.path.join(output_dir, "weight_comparisons")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # Get state dictionaries
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Find common parameters
    common_params = set(state_dict1.keys()).intersection(set(state_dict2.keys()))
    logger.info(f"Found {len(common_params)} common parameters between models")
    
    # Group parameters by layer
    layer_params = {}
    for param_name in common_params:
        # Extract layer name (everything before the last dot)
        if '.' in param_name:
            layer_name = '.'.join(param_name.split('.')[:-1])
        else:
            layer_name = param_name
        
        if layer_name not in layer_params:
            layer_params[layer_name] = []
        layer_params[layer_name].append(param_name)
    
    # Collect statistics for each layer
    layer_stats = {}
    all_l2_distances = []
    all_cosine_sims = []
    all_layer_names = []
    
    for layer_name, param_names in layer_params.items():
        # Skip layers with no parameters
        if not param_names:
            continue
        
        # Collect all parameters for this layer
        params1 = []
        params2 = []
        
        for param_name in param_names:
            p1 = state_dict1[param_name].cpu().float().flatten()
            p2 = state_dict2[param_name].cpu().float().flatten()
            params1.append(p1)
            params2.append(p2)
        
        # Concatenate all parameters for this layer
        all_params1 = torch.cat(params1)
        all_params2 = torch.cat(params2)
        
        # Calculate statistics
        l2_dist = torch.norm(all_params1 - all_params2).item()
        
        # Normalize for cosine similarity
        norm1 = torch.norm(all_params1)
        norm2 = torch.norm(all_params2)
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = torch.dot(all_params1, all_params2) / (norm1 * norm2)
            cosine_sim = cosine_sim.item()
        else:
            cosine_sim = 0.0
        
        # Store statistics
        layer_stats[layer_name] = {
            'l2_distance': l2_dist,
            'cosine_similarity': cosine_sim,
            'mean_diff': (all_params1.mean() - all_params2.mean()).item(),
            'std_diff': (all_params1.std() - all_params2.std()).item(),
            'params1': all_params1,
            'params2': all_params2
        }
        
        # Store for overall comparison
        all_l2_distances.append(l2_dist)
        all_cosine_sims.append(cosine_sim)
        all_layer_names.append(layer_name)
        
        # Create visualizations for this layer
        visualize_weight_distributions(
            all_params1.numpy(), 
            all_params2.numpy(), 
            layer_name, 
            os.path.join(weights_dir, f"{layer_name.replace('.', '_')}"),
            model1_name,
            model2_name
        )
    
    # Create overall comparison visualizations
    visualize_weight_comparison_across_layers(
        all_layer_names, 
        all_l2_distances,
        weights_dir,
        model1_name,
        model2_name
    )
    
    return layer_stats


def visualize_weight_distributions(weights1, weights2, layer_name, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Visualize weight distributions for a layer.
    
    Args:
        weights1 (numpy.ndarray): Weights from model 1
        weights2 (numpy.ndarray): Weights from model 2
        layer_name (str): Name of the layer
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot histograms of weight distributions
    bins = 100
    
    # Model 1 distribution
    ax1.hist(weights1, bins=bins, alpha=0.7, color='blue', density=True)
    ax1.set_title(f"{model1_name} Weight Distribution")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Density")
    
    # Model 2 distribution
    ax2.hist(weights2, bins=bins, alpha=0.7, color='orange', density=True)
    ax2.set_title(f"{model2_name} Weight Distribution")
    ax2.set_xlabel("Weight Value")
    
    # Overlay both distributions
    ax3.hist(weights1, bins=bins, alpha=0.5, color='blue', density=True, label=model1_name)
    ax3.hist(weights2, bins=bins, alpha=0.5, color='orange', density=True, label=model2_name)
    ax3.set_title(f"Overlaid Weight Distributions")
    ax3.set_xlabel("Weight Value")
    ax3.legend()
    
    # Add statistics to the plot
    stats1 = f"{model1_name} - Mean: {weights1.mean():.4f}, Std: {weights1.std():.4f}"
    stats2 = f"{model2_name} - Mean: {weights2.mean():.4f}, Std: {weights2.std():.4f}"
    fig.text(0.5, 0.01, f"{stats1}\n{stats2}", ha='center', fontsize=12)
    
    # Add a main title
    plt.suptitle(f"Weight Distributions - Layer: {layer_name}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, "weight_distributions.png"), dpi=150)
    plt.close()
    
    # Create a scatter plot of weights
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # If there are too many weights, sample a subset
    max_points = 10000
    if len(weights1) > max_points:
        indices = np.random.choice(len(weights1), max_points, replace=False)
        sample1 = weights1[indices]
        sample2 = weights2[indices]
    else:
        sample1 = weights1
        sample2 = weights2
    
    # Plot scatter of weights
    ax.scatter(sample1, sample2, alpha=0.5, s=5)
    
    # Add diagonal line (perfect correlation)
    min_val = min(sample1.min(), sample2.min())
    max_val = max(sample1.max(), sample2.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(f"{model1_name} Weights")
    ax.set_ylabel(f"{model2_name} Weights")
    ax.set_title(f"Weight Comparison - Layer: {layer_name}")
    
    # Add correlation coefficient
    corr = np.corrcoef(sample1, sample2)[0, 1]
    ax.text(0.05, 0.95, f"Correlation: {corr:.4f}", transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_scatter.png"), dpi=150)
    plt.close()
    
    # # Create a histogram of weight differences
    # fig, ax = plt.subplots(figsize=(10, 6))
    
    # # Calculate differences
    # weight_diffs = weights1 - weights2
    
    # # Plot histogram of differences
    # ax.hist(weight_diffs, bins=100, alpha=0.7, color='purple')
    # ax.set_xlabel("Weight Difference")
    # ax.set_ylabel("Count")
    # ax.set_title(f"Weight Differences ({model1_name} - {model2_name}) - Layer: {layer_name}")
    
    # Add statistics
    # mean_diff = weight_diffs.mean()
    # std_diff = weight_diffs.std()
    # ax.text(0.05, 0.95, f"Mean Diff: {mean_diff:.4f}\nStd Diff: {std_diff:.4f}", 
    #         transform=ax.transAxes, verticalalignment='top', 
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "weight_differences.png"), dpi=150)
    # plt.close()


def visualize_weight_comparison_across_layers(layer_names, l2_distances, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Visualize weight comparison metrics across different layers.
    
    Args:
        layer_names (list): Names of layers
        l2_distances (list): L2 distances for each layer
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sort layers by name for better visualization
    sorted_indices = np.argsort(layer_names)
    sorted_names = [layer_names[i] for i in sorted_indices]
    sorted_l2 = [l2_distances[i] for i in sorted_indices]
    
    # Create shortened names for display
    display_names = []
    for name in sorted_names:
        if len(name) > 20:
            parts = name.split('.')
            if len(parts) > 2:
                display_names.append(f"{parts[0]}...{parts[-1]}")
            else:
                display_names.append(name[:17] + "...")
        else:
            display_names.append(name)
    
    # Create horizontal bar chart for L2 Distance
    fig_l2, ax_l2 = plt.subplots(figsize=(10, max(8, len(sorted_names) * 0.3)))
    y_pos = np.arange(len(display_names))
    ax_l2.barh(y_pos, sorted_l2, color='skyblue')
    ax_l2.set_xlabel('L2 Distance')
    ax_l2.set_title(f'L2 Distance Between {model1_name} and {model2_name} Weights by Layer')
    ax_l2.set_yticks(y_pos)
    ax_l2.set_yticklabels(display_names)
    
    # Add value labels
    for i, l2 in enumerate(sorted_l2):
        ax_l2.text(l2 + max(sorted_l2)*0.02, i, f'{l2:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_l2_distance.png'), dpi=150)
    plt.close(fig_l2)


def calculate_cumulative_feature_changes(all_stats_by_layer, layer_names, model1_name, model2_name, output_dir):
    """
    Calculate and visualize cumulative changes in features across layers.
    
    Args:
        all_stats_by_layer (dict): Dictionary with layer indices as keys and statistics as values
        layer_names (dict): Dictionary mapping layer indices to layer names
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
        output_dir (Path): Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sort layers by their index to maintain proper order
    sorted_layers = sorted(all_stats_by_layer.keys())
    
    # Extract layer names and metrics
    layer_indices = []
    layer_names_list = []
    l2_distances = []
    cosine_sims = []
    
    # Filter for batch normalization layers
    bn_layer_indices = []
    bn_layer_names = []
    bn_l2_distances = []
    bn_cosine_sims = []
    
    for layer_idx in sorted_layers:
        if all_stats_by_layer[layer_idx]:  # Check if we have statistics for this layer
            layer_indices.append(layer_idx)
            layer_name = layer_names[layer_idx]
            layer_names_list.append(layer_name)
            
            # Calculate average statistics for this layer
            layer_stats = all_stats_by_layer[layer_idx]
            avg_stats = {
                key: np.mean([stat[key] for stat in layer_stats]) for key in layer_stats[0].keys()
            }
            
            l2_distances.append(avg_stats['l2_distance'])
            cosine_sims.append(avg_stats['cosine_similarity'])
            
            # Check if this is a batch normalization layer
            if 'bn' in layer_name or 'BatchNorm' in layer_name:
                bn_layer_indices.append(layer_idx)
                bn_layer_names.append(layer_name)
                bn_l2_distances.append(avg_stats['l2_distance'])
                bn_cosine_sims.append(avg_stats['cosine_similarity'])
    
    # Calculate cumulative changes for all layers
    cumulative_l2 = np.cumsum(l2_distances)
    
    # Calculate cumulative changes for batch normalization layers
    bn_cumulative_l2 = np.cumsum(bn_l2_distances) if bn_l2_distances else []
    
    # Create figure for cumulative L2 distance (all layers)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative L2 distance
    ax.plot(layer_indices, cumulative_l2, 'o-', linewidth=2, markersize=10, color='purple')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cumulative L2 Distance')
    ax.set_title(f'Cumulative Feature Differences Between {model1_name} and {model2_name}')
    
    # Identify parent layers for x-tick labels
    parent_layers = set()
    parent_layer_indices = []
    parent_layer_names = []
    
    for i, name in enumerate(layer_names_list):
        # Get the top-level parent (e.g., 'conv1', 'layer1', etc.)
        parent = name.split('.')[0]
        if parent not in parent_layers:
            parent_layers.add(parent)
            parent_layer_indices.append(layer_indices[i])
            parent_layer_names.append(parent)
    
    # Set x-ticks at parent layer positions
    ax.set_xticks(parent_layer_indices)
    ax.set_xticklabels(parent_layer_names, rotation=45, ha='right')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, (idx, l2) in enumerate(zip(layer_indices, cumulative_l2)):
        if i % max(1, len(layer_indices) // 10) == 0 or i == len(layer_indices) - 1:  # Annotate every 10th point and the last point
            ax.annotate(f'{l2:.2f}', 
                        (idx, l2),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_l2_distance.png'), dpi=150)
    plt.close()
    
    # Create a figure showing both L2 distance and cumulative L2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot L2 distance per layer
    ax1.plot(layer_indices, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name} Features by Layer')
    
    # Set x-ticks at parent layer positions
    ax1.set_xticks(parent_layer_indices)
    ax1.set_xticklabels(parent_layer_names, rotation=45, ha='right')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative L2 distance
    ax2.plot(layer_indices, cumulative_l2, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cumulative L2 Distance')
    ax2.set_title(f'Cumulative Feature Differences')
    
    # Add vertical lines to separate parent layers
    for idx in parent_layer_indices[1:]:  # Skip the first one
        ax2.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
    
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature Differences Between {model1_name} and {model2_name} Across Layers', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, 'l2_and_cumulative.png'), dpi=150)
    plt.close()
    
    # Calculate rate of change (derivative of L2 distance)
    if len(l2_distances) > 1:
        rate_of_change = np.diff(l2_distances)
        
        # Create figure for rate of change
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot rate of change
        ax.plot(layer_indices[1:], rate_of_change, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Rate of Change in L2 Distance')
        ax.set_title(f'Rate of Feature Divergence Between {model1_name} and {model2_name}')
        
        # Set x-ticks at parent layer positions that are in the range
        valid_parent_indices = [idx for idx in parent_layer_indices if idx in layer_indices[1:]]
        valid_parent_names = [parent_layer_names[parent_layer_indices.index(idx)] for idx in valid_parent_indices]
        
        ax.set_xticks(valid_parent_indices)
        ax.set_xticklabels(valid_parent_names, rotation=45, ha='right')
        
        # Add vertical lines to separate parent layers
        for idx in valid_parent_indices[1:]:  # Skip the first one
            ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.3)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rate_of_change.png'), dpi=150)
        plt.close()
    
    # Create BatchNorm-specific visualizations if we have batch norm layers
    if bn_layer_indices:
        # Create figure for cumulative L2 distance (BatchNorm layers only)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot cumulative L2 distance for BatchNorm layers
        ax.plot(bn_layer_indices, bn_cumulative_l2, 'o-', linewidth=2, markersize=10, color='green')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Cumulative L2 Distance')
        ax.set_title(f'Cumulative BatchNorm Feature Differences Between {model1_name} and {model2_name}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key points
        for i, (idx, l2) in enumerate(zip(bn_layer_indices, bn_cumulative_l2)):
            if i % max(1, len(bn_layer_indices) // 5) == 0 or i == len(bn_layer_indices) - 1:  # Annotate every 5th point and the last point
                ax.annotate(f'{l2:.2f}', 
                            (idx, l2),
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center',
                            fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bn_cumulative_l2_distance.png'), dpi=150)
        plt.close()
        
        # Create shortened names for display
        shortened_bn_names = []
        for name in bn_layer_names:
            if len(name) > 20:
                parts = name.split('.')
                if len(parts) > 2:
                    shortened_bn_names.append(f"{parts[0]}...{parts[-1]}")
                else:
                    shortened_bn_names.append(name[:17] + "...")
            else:
                shortened_bn_names.append(name)
        
        # Calculate rate of change for BatchNorm layers
        if len(bn_l2_distances) > 1:
            bn_rate_of_change = np.diff(bn_l2_distances)
            
            # Create figure for BatchNorm rate of change
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot rate of change for BatchNorm layers
            ax.plot(bn_layer_indices[1:], bn_rate_of_change, 'o-', linewidth=2, markersize=8, color='red')
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Rate of Change in L2 Distance')
            ax.set_title(f'Rate of BatchNorm Feature Divergence Between {model1_name} and {model2_name}')
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.grid(True, alpha=0.3)
            
            # Add layer names as x-tick labels (shortened for readability)
            if len(bn_layer_indices) > 1:
                shortened_bn_names_diff = [shortened_bn_names[i] for i in range(1, len(shortened_bn_names))]
                
                # Only show a subset of tick labels if there are many BatchNorm layers
                if len(bn_layer_indices[1:]) > 10:
                    step = len(bn_layer_indices[1:]) // 10
                    tick_indices = bn_layer_indices[1::step]
                    tick_labels = [shortened_bn_names_diff[i] for i in range(0, len(shortened_bn_names_diff), step)]
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                else:
                    ax.set_xticks(bn_layer_indices[1:])
                    ax.set_xticklabels(shortened_bn_names_diff, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bn_rate_of_change.png'), dpi=150)
            plt.close()
            
            # Create a combined figure with L2, cumulative L2, and rate of change for BatchNorm layers
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Plot L2 distance per BatchNorm layer
            ax1.plot(bn_layer_indices, bn_l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
            ax1.set_ylabel('L2 Distance')
            ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name} BatchNorm Features')
            ax1.grid(True, alpha=0.3)
            
            # Plot cumulative L2 distance for BatchNorm layers
            ax2.plot(bn_layer_indices, bn_cumulative_l2, 'o-', linewidth=2, markersize=8, color='green')
            ax2.set_ylabel('Cumulative L2 Distance')
            ax2.set_title(f'Cumulative BatchNorm Feature Differences')
            ax2.grid(True, alpha=0.3)
            
            # Plot rate of change for BatchNorm layers
            ax3.plot(bn_layer_indices[1:], bn_rate_of_change, 'o-', linewidth=2, markersize=8, color='red')
            ax3.set_xlabel('Layer Index')
            ax3.set_ylabel('Rate of Change')
            ax3.set_title(f'Rate of BatchNorm Feature Divergence')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            
            # Add layer names as x-tick labels (shortened for readability)
            if len(bn_layer_indices) > 10:
                step = len(bn_layer_indices) // 10
                tick_indices = bn_layer_indices[::step]
                tick_labels = [shortened_bn_names[i] for i in range(0, len(shortened_bn_names), step)]
                ax1.set_xticks(tick_indices)
                ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
            else:
                ax1.set_xticks(bn_layer_indices)
                ax1.set_xticklabels(shortened_bn_names, rotation=45, ha='right')
            
            plt.suptitle(f'BatchNorm Feature Analysis Between {model1_name} and {model2_name}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
            plt.savefig(os.path.join(output_dir, 'bn_comprehensive_analysis.png'), dpi=150)
            plt.close()
            
            # Include rate of change in the statistics file
            with open(os.path.join(output_dir, "bn_layer_stats.txt"), "a") as f:
                f.write("\nRate of Change Between BatchNorm Layers:\n")
                for i in range(len(bn_rate_of_change)):
                    f.write(f"Between layers {bn_layer_indices[i]} and {bn_layer_indices[i+1]}: {bn_rate_of_change[i]:.4f}\n")
        
        # Create a figure showing both L2 distance and cumulative L2 for BatchNorm layers
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot L2 distance per BatchNorm layer
        ax1.plot(bn_layer_indices, bn_l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_ylabel('L2 Distance')
        ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name} BatchNorm Features')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative L2 distance for BatchNorm layers
        ax2.plot(bn_layer_indices, bn_cumulative_l2, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Cumulative L2 Distance')
        ax2.set_title(f'Cumulative BatchNorm Feature Differences')
        ax2.grid(True, alpha=0.3)
        
        # Add layer names as x-tick labels
        if len(bn_layer_indices) > 10:
            step = len(bn_layer_indices) // 10
            tick_indices = bn_layer_indices[::step]
            tick_labels = [shortened_bn_names[i] for i in range(0, len(shortened_bn_names), step)]
            ax2.set_xticks(tick_indices)
            ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        else:
            ax2.set_xticks(bn_layer_indices)
            ax2.set_xticklabels(shortened_bn_names, rotation=45, ha='right')
        
        plt.suptitle(f'BatchNorm Feature Analysis Between {model1_name} and {model2_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
        plt.savefig(os.path.join(output_dir, 'bn_l2_and_cumulative.png'), dpi=150)
        plt.close()
        
        # Save BatchNorm-specific statistics to file
        with open(os.path.join(output_dir, "bn_layer_stats.txt"), "w") as f:
            f.write(f"BatchNorm Layer Statistics Between {model1_name} and {model2_name}\n\n")
            for i, idx in enumerate(bn_layer_indices):
                f.write(f"Layer {idx} ({bn_layer_names[i]}):\n")
                f.write(f"  L2 Distance: {bn_l2_distances[i]:.4f}\n")
                f.write(f"  Cosine Similarity: {bn_cosine_sims[i]:.4f}\n")
                f.write(f"  Cumulative L2: {bn_cumulative_l2[i]:.4f}\n\n")
    
    return {
        'layer_indices': layer_indices,
        'layer_names': layer_names_list,
        'l2_distances': l2_distances,
        'cumulative_l2': cumulative_l2.tolist(),
        'cosine_similarities': cosine_sims,
        'bn_layer_indices': bn_layer_indices,
        'bn_layer_names': bn_layer_names,
        'bn_l2_distances': bn_l2_distances,
        'bn_cumulative_l2': bn_cumulative_l2.tolist() if len(bn_cumulative_l2) > 0 else [],
        'bn_cosine_similarities': bn_cosine_sims
    }


def main():
    args = get_arguments()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create a directory for feature comparisons
    features_dir = os.path.join(args.output_dir, "feature_comparisons")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # Extract model names from checkpoint paths or use custom names if provided
    if args.model1_name:
        model1_name = args.model1_name
    else:
        model1_name = os.path.basename(args.model1_path).split('_')[0]
    
    if args.model2_name:
        model2_name = args.model2_name
    else:
        model2_name = os.path.basename(args.model2_path).split('_')[0]
    
    logger.info(f"Model 1 name: {model1_name}")
    logger.info(f"Model 2 name: {model2_name}")

    # Load models directly
    logger.info("Loading model 1...")
    model1 = load_model(args.model1_path)

    logger.info("Loading model 2...")
    model2 = load_model(args.model2_path)
    
    # Compare model weights
    logger.info("Comparing model weights...")
    weight_stats = compare_model_weights(model1, model2, args.output_dir, model1_name, model2_name)
    
    # Set up feature extractors
    logger.info("Setting up feature extractors...")
    extractor1 = FeatureExtractor(model1)
    extractor2 = FeatureExtractor(model2)
    
    # Get all layer IDs
    all_layer_ids = set(extractor1.layer_names.keys())
    
    # Filter layers if specified
    if args.layers:
        target_layers = [int(layer) for layer in args.layers.split(',')]
        # Check if specified layers exist
        for layer in target_layers:
            if layer not in all_layer_ids:
                logger.warning(f"Layer ID {layer} not found in model")
        # Filter to only include existing layers
        target_layers = [layer for layer in target_layers if layer in all_layer_ids]
    else:
        # Use all layers
        target_layers = sorted(list(all_layer_ids))

    logger.info(f"Extracting features from {len(target_layers)} layers")

    # Load dataset
    logger.info(f"Loading dataset: {args.task_ds}")
    dataset = DATASETS[args.task_ds](
        batch_size=args.batch_size,
        num_workers=4,
        gpu=torch.cuda.current_device(),
    )
    test_loader = dataset.get_dataloader(split="valid")

    # Extract features
    logger.info("Extracting features...")

    # Dictionary to store statistics for each layer
    all_stats_by_layer = {layer: [] for layer in target_layers}
    
    # Lists to store features for all samples for each layer
    all_features1_by_layer = {layer: [] for layer in target_layers}
    all_features2_by_layer = {layer: [] for layer in target_layers}
    
    # Lists to store per-sample cosine similarities for each layer
    per_sample_cosine_sims = {layer: [] for layer in target_layers}
    
    processed_samples = 0

    for batch_idx, data in enumerate(test_loader):
        if processed_samples >= args.num_samples:
            break

        images = data["img"].cuda()

        # Extract features from both models
        features1 = extractor1.extract_features(images)
        features2 = extractor2.extract_features(images)

        # Process each layer
        for layer_idx in target_layers:
            # Check if features were successfully extracted for this layer
            if layer_idx not in features1 or layer_idx not in features2:
                logger.warning(f"Features not extracted for layer {layer_idx}. Skipping layer.")
                continue

            # Get layer name for logging and directories
            layer_name = extractor1.layer_names[layer_idx]
            # Create a safe filename by replacing dots and other problematic characters
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            
            # Store features for distribution visualization
            all_features1_by_layer[layer_idx].append(features1[layer_idx].clone())
            all_features2_by_layer[layer_idx].append(features2[layer_idx].clone())
            
            # Calculate per-sample cosine similarities
            batch_sims = calculate_per_sample_cosine_similarity(features1[layer_idx], features2[layer_idx])
            per_sample_cosine_sims[layer_idx].extend(batch_sims)
            
            # Log statistics for this layer
            stats = calculate_statistics(features1[layer_idx], features2[layer_idx])
            all_stats_by_layer[layer_idx].append(stats)

            logger.info(f"Batch {batch_idx}, Layer {layer_idx} ({layer_name}), Stats: {stats}")

            # Visualize feature maps for this layer
            if args.visualize:
                # Create layer-specific output directory within features_dir
                layer_output_dir = os.path.join(features_dir, f"{safe_layer_name}")
                if not os.path.exists(layer_output_dir):
                    os.makedirs(layer_output_dir)
                
                for i in range(min(images.size(0), args.num_samples - processed_samples)):
                    sample_idx = batch_idx * args.batch_size + i
                    # Extract single image features for visualization
                    img_features1 = features1[layer_idx][i:i+1]
                    img_features2 = features2[layer_idx][i:i+1]
                    # Get the original image for this sample
                    original_img = images[i]
                    visualize_feature_maps(
                        img_features1,
                        img_features2,
                        layer_output_dir,
                        sample_idx,
                        layer_idx,
                        original_img,
                        extractor1.layer_names[layer_idx]
                    )

        processed_samples += images.size(0)

    # Clean up
    extractor1.remove_hooks()
    extractor2.remove_hooks()

    # Generate distribution visualizations for each layer
    for layer_idx in target_layers:
        if layer_idx in all_features1_by_layer and all_features1_by_layer[layer_idx]:
            # Get layer name for directories
            layer_name = extractor1.layer_names[layer_idx]
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            
            # Create layer-specific output directory within features_dir
            layer_output_dir = os.path.join(features_dir, f"{safe_layer_name}")
            if not os.path.exists(layer_output_dir):
                os.makedirs(layer_output_dir)
            
            # Visualize feature distributions
            logger.info(f"Generating feature distribution visualizations for layer {layer_idx} ({layer_name})...")
            visualize_feature_distributions(
                all_features1_by_layer[layer_idx],
                all_features2_by_layer[layer_idx],
                layer_output_dir,
                layer_idx,
                layer_name,
                model1_name,
                model2_name
            )
            
            # Visualize per-sample cosine similarities
            if layer_idx in per_sample_cosine_sims and per_sample_cosine_sims[layer_idx]:
                logger.info(f"Generating cosine similarity visualizations for layer {layer_idx} ({layer_name})...")
                visualize_cosine_similarities(
                    per_sample_cosine_sims[layer_idx],
                    layer_idx,
                    layer_name,
                    layer_output_dir,
                    model1_name,
                    model2_name
                )

    # Prepare data for layer comparison visualization
    layer_comparison_stats = {}
    
    for layer_idx in target_layers:
        if layer_idx in all_stats_by_layer and all_stats_by_layer[layer_idx]:
            # Calculate average statistics for this layer
            layer_stats = all_stats_by_layer[layer_idx]
            avg_stats = {
                key: np.mean([stat[key] for stat in layer_stats]) for key in layer_stats[0].keys()
            }
            
            # Store for layer comparison visualization
            layer_comparison_stats[layer_idx] = {
                'name': extractor1.layer_names[layer_idx],
                'l2_distance': avg_stats['l2_distance'],
                'cosine_similarity': avg_stats['cosine_similarity']
            }
            
            # Create layer-specific output directory within features_dir
            layer_output_dir = os.path.join(features_dir, f"{safe_layer_name}")
            if not os.path.exists(layer_output_dir):
                os.makedirs(layer_output_dir)

            # Save statistics to file
            with open(os.path.join(layer_output_dir, "statistics.txt"), "w") as f:
                f.write(f"Model 1: {args.model1_path}\n")
                f.write(f"Model 2: {args.model2_path}\n\n")
                f.write("Average statistics:\n")
                for key, value in avg_stats.items():
                    f.write(f"{key}: {value:.4f}\n")

            logger.info(f"Layer {layer_idx} ({extractor1.layer_names[layer_idx]}) - Average cosine similarity: {avg_stats['cosine_similarity']:.4f}")
        else:
            logger.warning(f"No statistics collected for layer {layer_idx}")

    # Generate layer comparison visualization if we have multiple layers
    if len(layer_comparison_stats) > 1:
        logger.info("Generating layer comparison visualization...")
        visualize_layer_comparison(layer_comparison_stats, features_dir, model1_name, model2_name)
        
        # Add cumulative feature change analysis
        logger.info("Calculating cumulative feature changes...")
        cumulative_stats = calculate_cumulative_feature_changes(
            all_stats_by_layer, 
            extractor1.layer_names, 
            model1_name, 
            model2_name, 
            features_dir
        )
        
        # Save cumulative statistics to file
        with open(os.path.join(features_dir, "cumulative_stats.json"), "w") as f:
            json.dump(cumulative_stats, f, indent=2)

    # Also save a summary file with statistics from all layers
    with open(os.path.join(features_dir, "all_layers_summary.txt"), "w") as f:
        f.write(f"Model 1: {args.model1_path}\n")
        f.write(f"Model 2: {args.model2_path}\n\n")
        f.write("Average statistics by layer:\n\n")
        
        for layer_idx in target_layers:
            if layer_idx in all_stats_by_layer and all_stats_by_layer[layer_idx]:
                layer_stats = all_stats_by_layer[layer_idx]
                avg_stats = {
                    key: np.mean([stat[key] for stat in layer_stats]) for key in layer_stats[0].keys()
                }
                
                f.write(f"Layer {layer_idx} ({extractor1.layer_names[layer_idx]}):\n")
                for key, value in avg_stats.items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            else:
                layer_name = extractor1.layer_names.get(layer_idx, f"unknown_layer_{layer_idx}")
                f.write(f"Layer {layer_idx} ({layer_name}): No statistics collected\n\n")

    logger.info("Feature extraction complete")


if __name__ == "__main__":
    main()
