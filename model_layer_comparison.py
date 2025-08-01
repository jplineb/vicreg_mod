from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import re

from custom_datasets import DATASETS
from torchvision.models import resnet50, ResNet50_Weights
from utils.log_config import configure_logging


import torch
import torch.nn as nn

logger = configure_logging()

torch.manual_seed(42)

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Compare feature maps between two models"
    )
    parser.add_argument(
        "--model1-path",
        type=Path,
        default=None,
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

    # Create ResNet50 model
    model = resnet50()
    if model_path == None:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return model.cuda()
    print(model_path)
    if model_path == Path("/project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar"):
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model = resnet50(weights=state_dict)
        return model.cuda()
    
    # Load the model state dict
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle different state dict formats
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Replace the final FC layer with Identity for feature extraction
    model.fc = nn.Identity()  # type: ignore

    # Process state dict keys if needed (remove prefixes like 'backbone.' or '0.')
    processed_state_dict = {}
    for key, value in state_dict.items():
        # Remove '0.' prefix if it exists
        if key.startswith("0."):
            new_key = key[2:]  # Remove the '0.' prefix
        elif key.startswith("module.backbone."):
            new_key = key.replace("module.backbone.", "")
        elif key.startswith("backbone."):
            new_key = key.replace("backbone.", "")
        else:
            new_key = key
        processed_state_dict[new_key] = value

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


def calculate_channel_cosine_similarity(feature_map1: torch.Tensor, 
                                      feature_map2: torch.Tensor) -> dict:
    """
    Calculate cosine similarity between two feature maps at each channel.
    
    Args:
        feature_map1: First feature map tensor (B, C, H, W)
        feature_map2: Second feature map tensor (B, C, H, W)
    
    Returns:
        Dictionary containing cosine similarity statistics per channel
    """
    # Ensure both feature maps have the same shape
    assert feature_map1.shape == feature_map2.shape, f"Feature maps must have same shape: {feature_map1.shape} vs {feature_map2.shape}"
    
    B, C, H, W = feature_map1.shape
    
    # Reshape to (B, C, H*W) for channel-wise processing
    fm1_reshaped = feature_map1.view(B, C, -1)  # (B, C, H*W)
    fm2_reshaped = feature_map2.view(B, C, -1)  # (B, C, H*W)
    
    # Channel-wise normalization (L2 norm)
    fm1_norm = torch.nn.functional.normalize(fm1_reshaped, p=2, dim=2)  # Normalize along spatial dimensions
    fm2_norm = torch.nn.functional.normalize(fm2_reshaped, p=2, dim=2)  # Normalize along spatial dimensions
    
    # Calculate cosine similarity for each channel
    # cosine_sim = sum(fm1_norm * fm2_norm) / (||fm1_norm|| * ||fm2_norm||)
    # Since we normalized, this simplifies to dot product
    cosine_similarities = torch.sum(fm1_norm * fm2_norm, dim=2)  # (B, C)
    
    # Calculate statistics across batch dimension
    mean_cosine_sim = torch.mean(cosine_similarities, dim=0)  # (C,)
    std_cosine_sim = torch.std(cosine_similarities, dim=0)    # (C,)
    min_cosine_sim = torch.min(cosine_similarities, dim=0)[0] # (C,)
    max_cosine_sim = torch.max(cosine_similarities, dim=0)[0] # (C,)
    
    # Convert to numpy for easier handling
    stats = {
        'channel_cosine_sim_mean': mean_cosine_sim.cpu().numpy(),
        'channel_cosine_sim_std': std_cosine_sim.cpu().numpy(),
        'channel_cosine_sim_min': min_cosine_sim.cpu().numpy(),
        'channel_cosine_sim_max': max_cosine_sim.cpu().numpy(),
        'channel_cosine_sim_all': cosine_similarities.cpu().numpy(),  # Full (B, C) array
    }
    
    # Add overall statistics
    stats['overall_cosine_sim_mean'] = float(torch.mean(cosine_similarities))
    stats['overall_cosine_sim_std'] = float(torch.std(cosine_similarities))
    stats['overall_cosine_sim_min'] = float(torch.min(cosine_similarities))
    stats['overall_cosine_sim_max'] = float(torch.max(cosine_similarities))
    
    return stats


def calculate_channel_cosine_similarity_batch(feature_maps1: list, 
                                            feature_maps2: list) -> dict:
    """
    Calculate cosine similarity statistics across multiple feature maps.
    
    Args:
        feature_maps1: List of feature map tensors from model 1
        feature_maps2: List of feature map tensors from model 2
    
    Returns:
        Dictionary containing aggregated cosine similarity statistics
    """
    all_cosine_sims = []
    
    for fm1, fm2 in zip(feature_maps1, feature_maps2):
        stats = calculate_channel_cosine_similarity(fm1, fm2)
        all_cosine_sims.append(stats['channel_cosine_sim_all'])
    
    # Concatenate all cosine similarities
    all_cosine_sims = np.concatenate(all_cosine_sims, axis=0)  # (Total_B, C)
    
    # Calculate aggregated statistics
    aggregated_stats = {
        'channel_cosine_sim_mean': np.mean(all_cosine_sims, axis=0),  # (C,)
        'channel_cosine_sim_std': np.std(all_cosine_sims, axis=0),    # (C,)
        'channel_cosine_sim_min': np.min(all_cosine_sims, axis=0),    # (C,)
        'channel_cosine_sim_max': np.max(all_cosine_sims, axis=0),    # (C,)
        'all_cosine_sims': all_cosine_sims,  # Save the full array
        'overall_cosine_sim_mean': float(np.mean(all_cosine_sims)),
        'overall_cosine_sim_std': float(np.std(all_cosine_sims)),
        'overall_cosine_sim_min': float(np.min(all_cosine_sims)),
        'overall_cosine_sim_max': float(np.max(all_cosine_sims)),
    }
    
    return aggregated_stats


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
    
    # Use full layer names for display and adjust figure size to accommodate longer names
    display_names = layer_names
    
    # Extract metrics
    l2_distances = [layer_stats[layer]['l2_distance'] for layer in layers]
    cosine_sims = [layer_stats[layer]['cosine_similarity'] for layer in layers]
    
    # Determine figure size based on number of layers - increase width significantly
    fig_width = max(20, len(layers) * 1.2)  # Much wider for better spacing
    
    # For line charts, create better x-tick spacing
    # Show every layer but with better spacing and rotation
    if len(layers) <= 15:
        # For fewer layers, show all layer names
        tick_indices = layers
        tick_labels = layer_names
    else:
        # For many layers, show every nth layer to avoid overcrowding
        step = max(1, len(layers) // 15)  # Show max 15 labels
        tick_indices = layers[::step]
        tick_labels = [layer_names[i] for i in range(0, len(layer_names), step)]
        # Always include the first and last layer
        if layers[0] not in tick_indices:
            tick_indices.insert(0, layers[0])
            tick_labels.insert(0, layer_names[0])
        if layers[-1] not in tick_indices:
            tick_indices.append(layers[-1])
            tick_labels.append(layer_names[-1])
    
    # 1. L2 Distance Line Plot
    fig_l2_line, ax_l2_line = plt.subplots(figsize=(fig_width, 6))
    ax_l2_line.plot(layers, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax_l2_line.set_xlabel('Layer')
    ax_l2_line.set_ylabel('L2 Distance')
    ax_l2_line.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
    
    # Set x-ticks with better spacing
    ax_l2_line.set_xticks(tick_indices)
    ax_l2_line.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    ax_l2_line.grid(True, alpha=0.3)
    
    # Add value labels (only for a reasonable number of points)
    if len(layers) <= 20:
        for i, l2 in enumerate(l2_distances):
            ax_l2_line.annotate(f'{l2:.2f}', 
                        (layers[i], l2),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l2_distance_line.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_l2_line)
    
    # 2. Cosine Similarity Line Plot
    fig_cos_line, ax_cos_line = plt.subplots(figsize=(fig_width, 6))
    ax_cos_line.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=8, color='green')
    ax_cos_line.set_xlabel('Layer')
    ax_cos_line.set_ylabel('Cosine Similarity')
    ax_cos_line.set_title(f'Cosine Similarity Between {model1_name} and {model2_name} by Layer')
    
    # Set x-ticks with better spacing
    ax_cos_line.set_xticks(tick_indices)
    ax_cos_line.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    ax_cos_line.grid(True, alpha=0.3)
    
    # Add value labels (only for a reasonable number of points)
    if len(layers) <= 20:
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
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_line.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_cos_line)
    
    # 3. L2 Distance Bar Chart - use horizontal bars for better label display
    fig_l2_bar, ax_l2_bar = plt.subplots(figsize=(12, max(10, len(layers) * 0.4)))
    
    # Reverse the order of layers, names, and values for the bar chart
    reversed_display_names = display_names[::-1]
    reversed_l2_distances = l2_distances[::-1]
    
    y_pos = np.arange(len(reversed_display_names))
    ax_l2_bar.barh(y_pos, reversed_l2_distances, color='skyblue')
    ax_l2_bar.set_xlabel('L2 Distance')
    ax_l2_bar.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
    ax_l2_bar.set_yticks(y_pos)
    ax_l2_bar.set_yticklabels(reversed_display_names, fontsize=10)
    
    # Add value labels
    for i, l2 in enumerate(reversed_l2_distances):
        ax_l2_bar.text(l2 + max(reversed_l2_distances)*0.02, i, f'{l2:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l2_distance_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_l2_bar)
    
    # 4. Cosine Similarity Bar Chart - use horizontal bars for better label display
    fig_cos_bar, ax_cos_bar = plt.subplots(figsize=(12, max(10, len(layers) * 0.4)))
    
    # Reverse the order for cosine similarity as well
    reversed_cosine_sims = cosine_sims[::-1]
    
    ax_cos_bar.barh(y_pos, reversed_cosine_sims, color='lightgreen')
    ax_cos_bar.set_xlabel('Cosine Similarity')
    ax_cos_bar.set_title(f'Cosine Similarity Between {model1_name} and {model2_name} by Layer')
    ax_cos_bar.set_yticks(y_pos)
    ax_cos_bar.set_yticklabels(reversed_display_names, fontsize=10)
    
    # Add value labels
    for i, sim in enumerate(reversed_cosine_sims):
        label_pos = sim + 0.05 if sim > 0 else sim - 0.1
        ax_cos_bar.text(label_pos, i, f'{sim:.2f}', va='center', fontsize=9)
    
    # Set x-axis limits for cosine similarity
    ax_cos_bar.set_xlim(-1.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_cos_bar)
    
    # 5. Combined plot with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(20, fig_width), 6))
    
    # Plot L2 distances
    ax1.plot(layers, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name}')
    
    # Set x-ticks with better spacing
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax1.grid(True, alpha=0.3)
    
    # Plot Cosine similarities
    ax2.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Cosine Similarity Between {model1_name} and {model2_name}')
    
    # Set x-ticks with better spacing
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limits for cosine similarity
    ax2.set_ylim(-1.05, 1.05)
    
    plt.suptitle(f'Layer-wise Comparison Between {model1_name} and {model2_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, 'layer_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compare_model_weights(model1, model2, output_dir, model1_name="Model 1", model2_name="Model 2", batch_size=1000):
    """
    Compare weights between two models and generate visualizations.
    
    Args:
        model1 (nn.Module): First model to compare
        model2 (nn.Module): Second model to compare
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model for display
        model2_name (str): Name of the second model for display
        batch_size (int): Number of parameters to process at once for memory efficiency
        
    Returns:
        dict: Dictionary containing statistics for each layer including:
            - l2_distance: L2 distance between weights
            - cosine_similarity: Cosine similarity between weights
            - mean_diff: Difference in mean values
            - std_diff: Difference in standard deviations
            - max_diff: Maximum absolute difference
            - min_diff: Minimum absolute difference
            - median_diff: Median absolute difference
            - param_count: Number of parameters in the layer
            
    Raises:
        ValueError: If inputs are not valid PyTorch models
        RuntimeError: If there are issues processing the models
    """
    try:
        # Input validation
        if not isinstance(model1, nn.Module) or not isinstance(model2, nn.Module):
            raise ValueError("Both inputs must be PyTorch models")
            
        # Create output directories
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
        
        total_layers = len(layer_params)
        for i, (layer_name, param_names) in enumerate(layer_params.items(), 1):
            try:
                logger.info(f"Processing layer {i}/{total_layers}: {layer_name}")
                
                # Skip layers with no parameters
                if not param_names:
                    continue
                
                # Process parameters in batches
                params1 = []
                params2 = []
                
                # For batch normalization layers, process weight and bias separately
                if 'bn' in layer_name.lower() or 'batchnorm' in layer_name.lower():
                    # Process weight and bias separately
                    weight_params = [p for p in param_names if p.endswith('.weight')]
                    bias_params = [p for p in param_names if p.endswith('.bias')]
                    
                    # Process weights
                    if weight_params:
                        weight1 = torch.cat([state_dict1[p].cpu().float().flatten() for p in weight_params])
                        weight2 = torch.cat([state_dict2[p].cpu().float().flatten() for p in weight_params])
                        
                        # Calculate statistics for weights
                        weight_stats = calculate_layer_statistics(weight1, weight2)
                        weight_stats['param_type'] = 'weight'
                        
                        # Create visualizations for weights
                        visualize_weight_distributions(
                            weight1.numpy(),
                            weight2.numpy(),
                            f"{layer_name}_weight",
                            os.path.join(weights_dir, f"{layer_name.replace('.', '_')}_weight"),
                            model1_name,
                            model2_name
                        )
                    
                    # Process biases
                    if bias_params:
                        bias1 = torch.cat([state_dict1[p].cpu().float().flatten() for p in bias_params])
                        bias2 = torch.cat([state_dict2[p].cpu().float().flatten() for p in bias_params])
                        
                        # Calculate statistics for biases
                        bias_stats = calculate_layer_statistics(bias1, bias2)
                        bias_stats['param_type'] = 'bias'
                        
                        # Create visualizations for biases
                        visualize_weight_distributions(
                            bias1.numpy(),
                            bias2.numpy(),
                            f"{layer_name}_bias",
                            os.path.join(weights_dir, f"{layer_name.replace('.', '_')}_bias"),
                            model1_name,
                            model2_name
                        )
                    
                    # Store combined statistics
                    layer_stats[layer_name] = {
                        'weight': weight_stats if weight_params else None,
                        'bias': bias_stats if bias_params else None
                    }
                    
                    # Store for overall comparison (use weight statistics)
                    if weight_params:
                        all_l2_distances.append(weight_stats['l2_distance'])
                        all_cosine_sims.append(weight_stats['cosine_similarity'])
                        all_layer_names.append(layer_name)
                    
                else:
                    # Process regular layers as before
                    for j in range(0, len(param_names), batch_size):
                        batch_params = param_names[j:j + batch_size]
                        
                        for param_name in batch_params:
                            p1 = state_dict1[param_name].cpu().float().flatten()
                            p2 = state_dict2[param_name].cpu().float().flatten()
                            params1.append(p1)
                            params2.append(p2)
                        
                        # Clean up batch memory
                        torch.cuda.empty_cache()
                    
                    # Concatenate all parameters for this layer
                    all_params1 = torch.cat(params1)
                    all_params2 = torch.cat(params2)
                    
                    # Calculate statistics
                    stats = calculate_layer_statistics(all_params1, all_params2)
                    layer_stats[layer_name] = stats
                    
                    # Store for overall comparison
                    all_l2_distances.append(stats['l2_distance'])
                    all_cosine_sims.append(stats['cosine_similarity'])
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
                
                # Clean up layer memory
                del params1, params2
                if 'all_params1' in locals():
                    del all_params1, all_params2
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing layer {layer_name}: {str(e)}")
                continue
        
        # Create overall comparison visualizations
        visualize_weight_comparison_across_layers(
            all_layer_names, 
            all_l2_distances,
            weights_dir,
            model1_name,
            model2_name
        )
        
        return layer_stats
        
    except Exception as e:
        logger.error(f"Error in compare_model_weights: {str(e)}")
        raise
    finally:
        # Ensure cleanup happens even if an error occurs
        torch.cuda.empty_cache()

def calculate_layer_statistics(params1, params2):
    """
    Calculate statistics between two parameter tensors.
    
    Args:
        params1 (torch.Tensor): First parameter tensor
        params2 (torch.Tensor): Second parameter tensor
        
    Returns:
        dict: Dictionary containing various statistics
    """
    # Calculate L2 distance
    l2_dist = torch.norm(params1 - params2).item()
    
    # Normalize for cosine similarity
    norm1 = torch.norm(params1)
    norm2 = torch.norm(params2)
    
    if norm1 > 0 and norm2 > 0:
        cosine_sim = torch.dot(params1, params2) / (norm1 * norm2)
        cosine_sim = cosine_sim.item()
    else:
        cosine_sim = 0.0
    
    # Calculate additional statistics
    abs_diff = torch.abs(params1 - params2)
    
    return {
        'l2_distance': l2_dist,
        'cosine_similarity': cosine_sim,
        'mean_diff': (params1.mean() - params2.mean()).item(),
        'std_diff': (params1.std() - params2.std()).item(),
        'max_diff': abs_diff.max().item(),
        'min_diff': abs_diff.min().item(),
        'median_diff': abs_diff.median().item(),
        'param_count': len(params1),
        'params1': params1,
        'params2': params2
    }


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
    
    # Use full layer names for display and adjust figure height
    display_names = sorted_names
    
    # Create horizontal bar chart for L2 Distance
    fig_l2, ax_l2 = plt.subplots(figsize=(12, max(8, len(sorted_names) * 0.5)))
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
    
    # Filter out parent layers
    filtered_layers = []
    for layer_idx in sorted_layers:
        layer_name = layer_names[layer_idx]
        # Skip if the layer name is just 'layerX'
        if not re.match(r'^layer[0-9]+$', layer_name):
            filtered_layers.append(layer_idx)
    
    # Extract layer names and metrics
    layer_indices = []
    layer_names_list = []
    l2_distances = []
    cosine_sims = []
    
    for layer_idx in filtered_layers:
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
    
    # Calculate cumulative changes for all layers
    cumulative_l2 = np.cumsum(l2_distances)
    
    # Create figure for cumulative L2 distance (all layers)
    fig, ax = plt.subplots(figsize=(max(20, len(layer_indices) * 1.2), 6))
    
    # Plot cumulative L2 distance
    ax.plot(layer_indices, cumulative_l2, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Cumulative L2 Distance')
    ax.set_title(f'Cumulative Feature Differences Between {model1_name} and {model2_name}')
    
    # Create better x-tick spacing
    if len(layer_indices) <= 15:
        # For fewer layers, show all layer names
        tick_indices = layer_indices
        tick_labels = layer_names_list
    else:
        # For many layers, show every nth layer to avoid overcrowding
        step = max(1, len(layer_indices) // 15)  # Show max 15 labels
        tick_indices = layer_indices[::step]
        tick_labels = [layer_names_list[i] for i in range(0, len(layer_names_list), step)]
        # Always include the first and last layer
        if layer_indices[0] not in tick_indices:
            tick_indices.insert(0, layer_indices[0])
            tick_labels.insert(0, layer_names_list[0])
        if layer_indices[-1] not in tick_indices:
            tick_indices.append(layer_indices[-1])
            tick_labels.append(layer_names_list[-1])
    
    # Set x-ticks with better spacing
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    
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
    plt.savefig(os.path.join(output_dir, 'cumulative_l2_distance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a figure showing both L2 distance and cumulative L2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(20, len(layer_indices) * 1.2), 10), sharex=True)
    
    # Plot L2 distance per layer
    ax1.plot(layer_indices, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name} Features by Layer')
    
    # Set x-ticks with better spacing
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative L2 distance
    ax2.plot(layer_indices, cumulative_l2, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cumulative L2 Distance')
    ax2.set_title(f'Cumulative Feature Differences')
    
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature Differences Between {model1_name} and {model2_name} Across Layers', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, 'l2_and_cumulative.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate rate of change (derivative of L2 distance)
    if len(l2_distances) > 1:
        rate_of_change = np.diff(l2_distances)
        
        # Create figure for rate of change
        fig, ax = plt.subplots(figsize=(max(20, len(layer_indices[1:]) * 1.2), 6))
        
        # Plot rate of change
        ax.plot(layer_indices[1:], rate_of_change, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Rate of Change in L2 Distance')
        ax.set_title(f'Rate of Feature Divergence Between {model1_name} and {model2_name}')
        
        # Create better x-tick spacing for rate of change
        if len(layer_indices[1:]) <= 15:
            tick_indices_roc = layer_indices[1:]
            tick_labels_roc = [layer_names_list[i] for i in range(1, len(layer_names_list))]
        else:
            step = max(1, len(layer_indices[1:]) // 15)
            tick_indices_roc = layer_indices[1::step]
            tick_labels_roc = [layer_names_list[i] for i in range(1, len(layer_names_list), step)]
            if layer_indices[1] not in tick_indices_roc:
                tick_indices_roc.insert(0, layer_indices[1])
                tick_labels_roc.insert(0, layer_names_list[1])
            if layer_indices[-1] not in tick_indices_roc:
                tick_indices_roc.append(layer_indices[-1])
                tick_labels_roc.append(layer_names_list[-1])
        
        ax.set_xticks(tick_indices_roc)
        ax.set_xticklabels(tick_labels_roc, rotation=45, ha='right', fontsize=10)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rate_of_change.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # Calculate successive layer ratios
    if len(l2_distances) > 1:
        # Calculate ratios between successive layers
        layer_ratios = np.array(l2_distances[1:]) / np.array(l2_distances[:-1])
        
        # Create figure for layer ratios
        fig, ax = plt.subplots(figsize=(max(20, len(layer_indices[1:]) * 1.2), 6))
        
        # Plot ratios
        ax.plot(layer_indices[1:], layer_ratios, 'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Ratio of L2 Distances (Layer_n / Layer_{n-1})')
        ax.set_title(f'Ratio Between Successive Layer Features\n{model1_name} vs {model2_name}')
        
        # Add horizontal line at y=1 (indicating no change)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        
        ax.grid(True, alpha=0.3)
        
        # Create better x-tick spacing for layer ratios
        if len(layer_indices[1:]) <= 15:
            tick_indices_ratio = layer_indices[1:]
            tick_labels_ratio = [layer_names_list[i] for i in range(1, len(layer_names_list))]
        else:
            step = max(1, len(layer_indices[1:]) // 15)
            tick_indices_ratio = layer_indices[1::step]
            tick_labels_ratio = [layer_names_list[i] for i in range(1, len(layer_names_list), step)]
            if layer_indices[1] not in tick_indices_ratio:
                tick_indices_ratio.insert(0, layer_indices[1])
                tick_labels_ratio.insert(0, layer_names_list[1])
            if layer_indices[-1] not in tick_indices_ratio:
                tick_indices_ratio.append(layer_indices[-1])
                tick_labels_ratio.append(layer_names_list[-1])
        
        ax.set_xticks(tick_indices_ratio)
        ax.set_xticklabels(tick_labels_ratio, rotation=45, ha='right', fontsize=10)
        
        # Add annotations for ratios
        for i, ratio in enumerate(layer_ratios):
            ax.annotate(f'{ratio:.2f}', 
                       (layer_indices[i+1], ratio),
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='center',
                       fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_ratios.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Include ratios in the statistics file
        with open(os.path.join(output_dir, "layer_stats.txt"), "a") as f:
            f.write("\nRatios Between Successive Layers:\n")
            for i in range(len(layer_ratios)):
                f.write(f"Ratio {layer_names_list[i]} to {layer_names_list[i+1]}: {layer_ratios[i]:.4f}\n")

    return {
        'layer_indices': layer_indices,
        'layer_names': layer_names_list,
        'l2_distances': l2_distances,
        'cumulative_l2': cumulative_l2.tolist(),
        'cosine_similarities': cosine_sims,
        'layer_ratios': layer_ratios.tolist() if len(l2_distances) > 1 else []
    }


def filter_target_layers(all_layer_ids, layer_names):
    """
    Filter layers to only include top-level layers and all bottleneck outputs in layer1, layer2, layer3.
    """
    target_layers = []
    for layer_id in sorted(all_layer_ids):
        layer_name = layer_names[layer_id]
        # Top-level layers
        if layer_name in ['conv1', 'bn1', 'relu', 'maxpool']:
            target_layers.append(layer_id)
            continue
        # All bottleneck outputs in layer1, layer2, layer3
        if re.match(r'^layer[123]\.\d+$', layer_name):
            target_layers.append(layer_id)
            continue
    return target_layers


def visualize_channel_cosine_similarity_distributions(channel_cosine_sims_by_layer, extractor_layer_names, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Visualize distributions of channel-wise cosine similarities across layers.
    
    Args:
        channel_cosine_sims_by_layer (dict): Dictionary with layer indices as keys and cosine similarity stats as values
        extractor_layer_names (dict): Dictionary mapping layer indices to layer names
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a 2x2 visualization (removing heatmap)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Collect data across all layers
    all_layer_names = []
    all_mean_sims = []
    all_std_sims = []
    all_min_sims = []
    all_max_sims = []
    
    for layer_idx in sorted(channel_cosine_sims_by_layer.keys()):
        if channel_cosine_sims_by_layer[layer_idx]:
            # Calculate average statistics across batches for this layer
            layer_stats = channel_cosine_sims_by_layer[layer_idx]
            avg_mean = np.mean([stat['channel_cosine_sim_mean'] for stat in layer_stats], axis=0)
            avg_std = np.mean([stat['channel_cosine_sim_std'] for stat in layer_stats], axis=0)
            avg_min = np.mean([stat['channel_cosine_sim_min'] for stat in layer_stats], axis=0)
            avg_max = np.mean([stat['channel_cosine_sim_max'] for stat in layer_stats], axis=0)
            
            # Use actual layer name instead of just index
            layer_name = extractor_layer_names.get(layer_idx, f"Layer_{layer_idx}")
            all_layer_names.append(layer_name)
            all_mean_sims.append(avg_mean)
            all_std_sims.append(avg_std)
            all_min_sims.append(avg_min)
            all_max_sims.append(avg_max)
    
    # Create better x-tick spacing for readability
    if len(all_layer_names) <= 10:
        tick_labels = all_layer_names
        tick_positions = range(len(all_layer_names))
        box_labels = all_layer_names
    else:
        # For many layers, show every nth layer to avoid overcrowding
        step = max(1, len(all_layer_names) // 10)
        tick_positions = list(range(0, len(all_layer_names), step))
        tick_labels = [all_layer_names[i] for i in tick_positions]
        # Always include the last layer
        if len(all_layer_names) - 1 not in tick_positions:
            tick_positions.append(len(all_layer_names) - 1)
            tick_labels.append(all_layer_names[-1])
        
        # For box plot, use shortened labels
        box_labels = [f"L{i}" for i in range(len(all_layer_names))]
    
    # 1. Box plot of channel cosine similarities across layers (use actual layer names)
    ax1.boxplot(all_mean_sims, labels=all_layer_names)
    ax1.set_title(f'Channel-wise Cosine Similarity Distribution Across Layers\n{model1_name} vs {model2_name}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cosine Similarity')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean cosine similarity per layer
    layer_means = [np.mean(means) for means in all_mean_sims]
    layer_stds = [np.std(means) for means in all_mean_sims]
    
    ax2.errorbar(range(len(all_layer_names)), layer_means, yerr=layer_stds, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax2.set_title(f'Average Channel Cosine Similarity per Layer\n{model1_name} vs {model2_name}')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average Cosine Similarity')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Min/Max range across layers (replacing heatmap)
    layer_mins = [np.min(mins) for mins in all_min_sims]
    layer_maxs = [np.max(maxs) for maxs in all_max_sims]
    
    ax3.fill_between(range(len(all_layer_names)), layer_mins, layer_maxs, alpha=0.5, color='green', label='Min-Max Range')
    ax3.plot(range(len(all_layer_names)), layer_means, 'o-', linewidth=2, markersize=6, color='red', label='Mean')
    ax3.set_title(f'Channel Cosine Similarity Range Across Layers\n{model1_name} vs {model2_name}')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of cosine similarities across all channels and layers
    all_similarities = []
    for means in all_mean_sims:
        all_similarities.extend(means)
    
    ax4.hist(all_similarities, bins=50, alpha=0.7, color='purple', density=True)
    ax4.set_title(f'Distribution of Channel Cosine Similarities\n{model1_name} vs {model2_name}')
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_sim = np.mean(all_similarities)
    std_sim = np.std(all_similarities)
    ax4.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_cosine_similarity_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()

def visualize_channel_cosine_similarity_batch_stats(channel_cosine_sims_by_layer, extractor_layer_names, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Visualize batch statistics for channel-wise cosine similarities.
    
    Args:
        channel_cosine_sims_by_layer (dict): Dictionary with layer indices as keys and cosine similarity stats as values
        extractor_layer_names (dict): Dictionary mapping layer indices to layer names
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create visualizations for each layer
    for layer_idx in sorted(channel_cosine_sims_by_layer.keys()):
        if not channel_cosine_sims_by_layer[layer_idx]:
            continue
            
        layer_stats = channel_cosine_sims_by_layer[layer_idx]
        
        # Get actual layer name
        layer_name = extractor_layer_names.get(layer_idx, f"Layer_{layer_idx}")
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        
        # Create figure with 2x2 subplots (removing heatmap)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract batch-wise statistics
        batch_means = [stat['channel_cosine_sim_mean'] for stat in layer_stats]
        batch_stds = [stat['channel_cosine_sim_std'] for stat in layer_stats]
        batch_mins = [stat['channel_cosine_sim_min'] for stat in layer_stats]
        batch_maxs = [stat['channel_cosine_sim_max'] for stat in layer_stats]
        
        # Convert to numpy arrays
        batch_means = np.array(batch_means)  # Shape: (num_batches, num_channels)
        batch_stds = np.array(batch_stds)
        batch_mins = np.array(batch_mins)
        batch_maxs = np.array(batch_maxs)
        
        # 1. Batch-wise mean cosine similarity per channel
        channel_means = np.mean(batch_means, axis=0)
        channel_stds = np.std(batch_means, axis=0)
        
        channels = range(len(channel_means))
        ax1.errorbar(channels, channel_means, yerr=channel_stds, 
                    marker='o', capsize=3, capthick=1, linewidth=1, markersize=4)
        ax1.set_title(f'Channel-wise Mean Cosine Similarity ({layer_name})\n{model1_name} vs {model2_name}')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Cosine Similarity')
        ax1.grid(True, alpha=0.3)
        
        # 2. Batch-wise standard deviation per channel
        channel_std_means = np.mean(batch_stds, axis=0)
        channel_std_stds = np.std(batch_stds, axis=0)
        
        ax2.errorbar(channels, channel_std_means, yerr=channel_std_stds,
                    marker='s', capsize=3, capthick=1, linewidth=1, markersize=4, color='orange')
        ax2.set_title(f'Channel-wise Standard Deviation ({layer_name})\n{model1_name} vs {model2_name}')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Min/Max range per channel (replacing heatmap)
        channel_mins = np.mean(batch_mins, axis=0)
        channel_maxs = np.mean(batch_maxs, axis=0)
        
        ax3.fill_between(channels, channel_mins, channel_maxs, alpha=0.5, color='lightblue', label='Min-Max Range')
        ax3.plot(channels, channel_means, 'o-', linewidth=1, markersize=4, color='blue', label='Mean')
        ax3.set_title(f'Channel-wise Range ({layer_name})\n{model1_name} vs {model2_name}')
        ax3.set_xlabel('Channel Index')
        ax3.set_ylabel('Cosine Similarity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution of cosine similarities across all batches and channels
        all_similarities = batch_means.flatten()
        
        ax4.hist(all_similarities, bins=30, alpha=0.7, color='green', density=True)
        ax4.set_title(f'Distribution of Cosine Similarities ({layer_name})\n{model1_name} vs {model2_name}')
        ax4.set_xlabel('Cosine Similarity')
        ax4.set_ylabel('Density')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_sim = np.mean(all_similarities)
        std_sim = np.std(all_similarities)
        ax4.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'channel_cosine_similarity_{safe_layer_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def visualize_channel_cosine_similarity_comparison(channel_cosine_sims_by_layer, extractor_layer_names, output_dir, model1_name="Model 1", model2_name="Model 2"):
    """
    Create comparison visualizations for channel-wise cosine similarities.
    
    Args:
        channel_cosine_sims_by_layer (dict): Dictionary with layer indices as keys and cosine similarity stats as values
        extractor_layer_names (dict): Dictionary mapping layer indices to layer names
        output_dir (Path): Directory to save visualizations
        model1_name (str): Name of the first model
        model2_name (str): Name of the second model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect summary statistics across all layers
    layer_summaries = []
    
    for layer_idx in sorted(channel_cosine_sims_by_layer.keys()):
        if channel_cosine_sims_by_layer[layer_idx]:
            layer_stats = channel_cosine_sims_by_layer[layer_idx]
            
            # Calculate overall statistics for this layer
            all_similarities = []
            for stat in layer_stats:
                all_similarities.extend(stat['channel_cosine_sim_mean'])
            
            # Get actual layer name
            layer_name = extractor_layer_names.get(layer_idx, f"Layer_{layer_idx}")
            
            layer_summaries.append({
                'layer_idx': layer_idx,
                'layer_name': layer_name,
                'mean': np.mean(all_similarities),
                'std': np.std(all_similarities),
                'min': np.min(all_similarities),
                'max': np.max(all_similarities),
                'num_channels': len(layer_stats[0]['channel_cosine_sim_mean'])
            })
    
    if not layer_summaries:
        return
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    layers = [summary['layer_idx'] for summary in layer_summaries]
    layer_names = [summary['layer_name'] for summary in layer_summaries]
    means = [summary['mean'] for summary in layer_summaries]
    stds = [summary['std'] for summary in layer_summaries]
    mins = [summary['min'] for summary in layer_summaries]
    maxs = [summary['max'] for summary in layer_summaries]
    num_channels = [summary['num_channels'] for summary in layer_summaries]
    
    # Create better x-tick spacing
    if len(layers) <= 10:
        tick_indices = layers
        tick_labels = layer_names
    else:
        step = max(1, len(layers) // 10)
        tick_indices = layers[::step]
        tick_labels = [layer_names[i] for i in range(0, len(layer_names), step)]
        if layers[-1] not in tick_indices:
            tick_indices.append(layers[-1])
            tick_labels.append(layer_names[-1])
    
    # 1. Mean cosine similarity across layers
    ax1.plot(layers, means, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.fill_between(layers, [m-s for m, s in zip(means, stds)], 
                     [m+s for m, s in zip(means, stds)], alpha=0.3, color='blue')
    ax1.set_title(f'Mean Channel Cosine Similarity Across Layers\n{model1_name} vs {model2_name}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Cosine Similarity')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Min/Max range across layers
    ax2.fill_between(layers, mins, maxs, alpha=0.5, color='green', label='Min-Max Range')
    ax2.plot(layers, means, 'o-', linewidth=2, markersize=6, color='red', label='Mean')
    ax2.set_title(f'Channel Cosine Similarity Range Across Layers\n{model1_name} vs {model2_name}')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Standard deviation across layers
    ax3.plot(layers, stds, 's-', linewidth=2, markersize=8, color='orange')
    ax3.set_title(f'Channel Cosine Similarity Standard Deviation Across Layers\n{model1_name} vs {model2_name}')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of channels vs mean similarity
    scatter = ax4.scatter(num_channels, means, s=100, alpha=0.7, c=layers, cmap='viridis')
    ax4.set_title(f'Number of Channels vs Mean Similarity\n{model1_name} vs {model2_name}')
    ax4.set_xlabel('Number of Channels')
    ax4.set_ylabel('Mean Cosine Similarity')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for layer indices
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Layer Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_cosine_similarity_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def convert_numpy_to_lists(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    
    Args:
        obj: Object that may contain numpy arrays
        
    Returns:
        Object with all numpy arrays converted to lists
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

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
        # Use the new filtering function to get only top-level layers and bottleneck outputs
        target_layers = filter_target_layers(all_layer_ids, extractor1.layer_names)

    logger.info(f"Extracting features from {len(target_layers)} layers")


    # Load dataset
    logger.info(f"Loading dataset: {args.task_ds}")
    dataset = DATASETS[args.task_ds](
        batch_size=args.batch_size,
        num_workers=8,
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
    
    # Lists to store channel-wise cosine similarities for each layer
    channel_cosine_sims_by_layer = {layer: [] for layer in target_layers}
    
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
            
            # Calculate channel-wise cosine similarities
            channel_cosine_stats = calculate_channel_cosine_similarity(features1[layer_idx], features2[layer_idx])
            channel_cosine_sims_by_layer[layer_idx].append(channel_cosine_stats)
            
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
        
        # Calculate aggregated channel-wise cosine similarity statistics
        logger.info("Calculating aggregated channel-wise cosine similarity statistics...")
        for layer_idx in target_layers:
            if layer_idx in channel_cosine_sims_by_layer and channel_cosine_sims_by_layer[layer_idx]:
                # Get all feature maps for this layer
                layer_features1 = [all_features1_by_layer[layer_idx][i] for i in range(len(all_features1_by_layer[layer_idx]))]
                layer_features2 = [all_features2_by_layer[layer_idx][i] for i in range(len(all_features2_by_layer[layer_idx]))]
                
                # Calculate aggregated statistics
                aggregated_channel_stats = calculate_channel_cosine_similarity_batch(layer_features1, layer_features2)
                
                # Add to cumulative stats
                layer_name = extractor1.layer_names[layer_idx]
                cumulative_stats[f'channel_cosine_sim_{layer_name}'] = aggregated_channel_stats
        
        # Save cumulative statistics to file
        cumulative_stats_serializable = convert_numpy_to_lists(cumulative_stats)
        with open(os.path.join(features_dir, "cumulative_stats.json"), "w") as f:
            json.dump(cumulative_stats_serializable, f, indent=2)

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
                
                # Add channel-wise cosine similarity statistics
                if layer_idx in channel_cosine_sims_by_layer and channel_cosine_sims_by_layer[layer_idx]:
                    f.write("  Channel-wise cosine similarity statistics:\n")
                    # Calculate average channel cosine similarity stats across batches
                    channel_stats = channel_cosine_sims_by_layer[layer_idx]
                    avg_channel_stats = {
                        'overall_cosine_sim_mean': np.mean([stat['overall_cosine_sim_mean'] for stat in channel_stats]),
                        'overall_cosine_sim_std': np.mean([stat['overall_cosine_sim_std'] for stat in channel_stats]),
                        'overall_cosine_sim_min': np.mean([stat['overall_cosine_sim_min'] for stat in channel_stats]),
                        'overall_cosine_sim_max': np.mean([stat['overall_cosine_sim_max'] for stat in channel_stats]),
                    }
                    for key, value in avg_channel_stats.items():
                        f.write(f"    {key}: {value:.4f}\n")
                
                f.write("\n")
            else:
                layer_name = extractor1.layer_names.get(layer_idx, f"unknown_layer_{layer_idx}")
                f.write(f"Layer {layer_idx} ({layer_name}): No statistics collected\n\n")

    logger.info("Feature extraction complete")

    # Generate channel-wise cosine similarity visualizations
    if channel_cosine_sims_by_layer:
        logger.info("Generating channel-wise cosine similarity visualizations...")
        
        # Create directory for channel cosine similarity visualizations
        channel_cosine_dir = os.path.join(features_dir, "channel_cosine_similarity")
        if not os.path.exists(channel_cosine_dir):
            os.makedirs(channel_cosine_dir)
        
        # Generate overview visualizations
        visualize_channel_cosine_similarity_distributions(
            channel_cosine_sims_by_layer,
            extractor1.layer_names,  # Pass the layer names
            channel_cosine_dir,
            model1_name,
            model2_name
        )
        
        # Generate batch statistics visualizations
        visualize_channel_cosine_similarity_batch_stats(
            channel_cosine_sims_by_layer,
            extractor1.layer_names,  # Pass the layer names
            channel_cosine_dir,
            model1_name,
            model2_name
        )
        
        # Generate comparison visualizations
        visualize_channel_cosine_similarity_comparison(
            channel_cosine_sims_by_layer,
            extractor1.layer_names,  # Pass the layer names
            channel_cosine_dir,
            model1_name,
            model2_name
        )


if __name__ == "__main__":
    main()
