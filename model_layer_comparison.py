from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

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
        "--task-ds",
        type=str,
        required=True,
        help="Dataset to use for feature extraction",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="7",
        help="Comma-separated list of layer indices to extract features from. "
             "Available indices: 0 (conv1), 1 (bn1), 2 (relu), 3 (maxpool), "
             "4 (layer1), 5 (layer2), 6 (layer3), 7 (layer4)",
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


def visualize_feature_maps(features1, features2, output_dir, sample_idx, layer_idx, original_image=None):
    """
    Visualize and save comparisons of feature maps.

    Args:
        features1 (torch.Tensor): Features from model 1
        features2 (torch.Tensor): Features from model 2
        output_dir (Path): Directory to save visualizations
        sample_idx (int): Index of the sample
        layer_idx (int): Index of the layer
        original_image (torch.Tensor, optional): Original input image
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
    plt.suptitle(f"Feature Maps - Layer {layer_idx} - Sample {sample_idx}", fontsize=16)
    
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
    def __init__(self, model, target_layers):
        """
        Initialize feature extractor for a given model.

        Args:
            model (nn.Module): The model to extract features from
            target_layers (list): List of layer indices to extract features from
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {layer: None for layer in target_layers}
        self.hooks = []
        
        # Define mapping from layer indices to actual layers (starting at 0)
        self.layer_mapping = {
            0: 'conv1',     # First layer
            1: 'bn1',       # BatchNorm after conv1
            2: 'relu',      # ReLU after bn1
            3: 'maxpool',   # MaxPool after relu
            4: 'layer1',    # First residual block
            5: 'layer2',    # Second residual block
            6: 'layer3',    # Third residual block
            7: 'layer4',    # Fourth residual block
        }
        
        # Create reverse mapping for printing layer names
        self.index_to_name = {idx: name for idx, name in self.layer_mapping.items()}
        
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on the target layers."""
        for layer_idx in self.target_layers:
            # Get the layer name from the mapping
            if layer_idx in self.layer_mapping:
                layer_name = self.layer_mapping[layer_idx]
                if hasattr(self.model, layer_name):
                    layer = getattr(self.model, layer_name)
                    self.hooks.append(
                        layer.register_forward_hook(self._get_hook(layer_idx))
                    )
                    logger.info(f"Registered hook for layer {layer_name} (index {layer_idx})")
                else:
                    logger.warning(f"Layer {layer_name} not found in model")
            else:
                logger.warning(f"Layer index {layer_idx} not in mapping, skipping")

    def _get_hook(self, layer_idx):
        """Create a hook function for a specific layer."""

        def hook(module, input, output):
            self.features[layer_idx] = output.detach()

        return hook

    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Extract features for input x.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Dictionary of features for each target layer
        """
        # Reset features
        self.features = {layer: None for layer in self.target_layers}
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


def visualize_feature_distributions(features1_all, features2_all, output_dir, layer_idx, layer_name):
    """
    Visualize and save distributions of feature activations.

    Args:
        features1_all (list): List of feature tensors from model 1 for all samples
        features2_all (list): List of feature tensors from model 2 for all samples
        output_dir (Path): Directory to save visualizations
        layer_idx (int): Index of the layer
        layer_name (str): Name of the layer
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Concatenate all features
    features1_concat = torch.cat(features1_all, dim=0)
    features2_concat = torch.cat(features2_all, dim=0)
    
    # Convert to numpy for plotting
    features1_np = features1_concat.cpu().flatten().numpy()
    features2_np = features2_concat.cpu().flatten().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot histograms of feature distributions
    bins = 100
    
    # Model 1 distribution
    ax1.hist(features1_np, bins=bins, alpha=0.7, color='blue', density=True)
    ax1.set_title(f"Model 1 Feature Distribution")
    ax1.set_xlabel("Feature Value")
    ax1.set_ylabel("Density")
    
    # Model 2 distribution
    ax2.hist(features2_np, bins=bins, alpha=0.7, color='orange', density=True)
    ax2.set_title(f"Model 2 Feature Distribution")
    ax2.set_xlabel("Feature Value")
    
    # Overlay both distributions
    ax3.hist(features1_np, bins=bins, alpha=0.5, color='blue', density=True, label='Model 1')
    ax3.hist(features2_np, bins=bins, alpha=0.5, color='orange', density=True, label='Model 2')
    ax3.set_title(f"Overlaid Feature Distributions")
    ax3.set_xlabel("Feature Value")
    ax3.legend()
    
    # Add statistics to the plot
    stats1 = f"Model 1 - Mean: {features1_np.mean():.4f}, Std: {features1_np.std():.4f}"
    stats2 = f"Model 2 - Mean: {features2_np.mean():.4f}, Std: {features2_np.std():.4f}"
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
    ax.scatter(channel_means1, channel_stds1, alpha=0.7, label='Model 1', color='blue')
    ax.scatter(channel_means2, channel_stds2, alpha=0.7, label='Model 2', color='orange')
    
    # Add labels and title
    ax.set_xlabel("Channel Mean")
    ax.set_ylabel("Channel Standard Deviation")
    ax.set_title(f"Channel-wise Statistics - Layer {layer_idx} ({layer_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"channel_stats_layer_{layer_idx}.png"), dpi=150)
    plt.close()


def visualize_cosine_similarities(cosine_sims, layer_idx, layer_name, output_dir):
    """
    Visualize cosine similarities between model features for each sample.
    
    Args:
        cosine_sims (list): List of cosine similarity values for each sample
        layer_idx (int): Index of the layer
        layer_name (str): Name of the layer
        output_dir (Path): Directory to save visualizations
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
    ax.set_title(f'Per-Sample Cosine Similarity - Layer {layer_idx} ({layer_name})')
    
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


def visualize_layer_comparison(layer_stats, output_dir):
    """
    Visualize comparison metrics across different layers.
    
    Args:
        layer_stats (dict): Dictionary with layer indices as keys and statistics as values
        output_dir (Path): Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract layer indices and names
    layers = sorted(layer_stats.keys())
    layer_names = [layer_stats[layer]['name'] for layer in layers]
    
    # Extract metrics
    l2_distances = [layer_stats[layer]['l2_distance'] for layer in layers]
    cosine_sims = [layer_stats[layer]['cosine_similarity'] for layer in layers]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot L2 distances
    ax1.plot(layers, l2_distances, 'o-', linewidth=2, markersize=10, color='blue')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title('L2 Distance Between Models by Layer')
    ax1.set_xticks(layers)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, l2 in enumerate(l2_distances):
        ax1.annotate(f'{l2:.2f}', 
                    (layers[i], l2),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Plot Cosine similarities
    ax2.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity Between Models by Layer')
    ax2.set_xticks(layers)
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, sim in enumerate(cosine_sims):
        ax2.annotate(f'{sim:.2f}', 
                    (layers[i], sim),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Set y-axis limits for cosine similarity
    ax2.set_ylim(-1.05, 1.05)
    
    plt.suptitle('Layer-wise Comparison Between Models', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, 'layer_comparison.png'), dpi=150)
    plt.close()
    
    # Create a bar chart version for better comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # L2 distance bar chart
    ax1.bar(layer_names, l2_distances, color='skyblue')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title('L2 Distance Between Models by Layer')
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    
    # Add value labels
    for i, l2 in enumerate(l2_distances):
        ax1.text(i, l2 + max(l2_distances)*0.02, f'{l2:.2f}', ha='center')
    
    # Cosine similarity bar chart
    ax2.bar(layer_names, cosine_sims, color='lightgreen')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity Between Models by Layer')
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    
    # Add value labels
    for i, sim in enumerate(cosine_sims):
        ax2.text(i, sim + 0.05 if sim > 0 else sim - 0.1, f'{sim:.2f}', ha='center')
    
    # Set y-axis limits for cosine similarity
    ax2.set_ylim(-1.05, 1.05)
    
    plt.suptitle('Layer-wise Comparison Between Models (Bar Chart)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    plt.savefig(os.path.join(output_dir, 'layer_comparison_bar.png'), dpi=150)
    plt.close()


def main():
    args = get_arguments()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load models directly
    logger.info("Loading model 1...")
    model1 = load_model(args.model1_path)

    logger.info("Loading model 2...")
    model2 = load_model(args.model2_path)

    # Determine which layers to extract features from
    if args.layers:
        target_layers = [int(layer) for layer in args.layers.split(',')]
    else:
        target_layers = [7]  # Default to layer4 if not specified

    logger.info(f"Extracting features from layers: {target_layers}")

    # Set up feature extractors
    extractor1 = FeatureExtractor(model1, target_layers)
    extractor2 = FeatureExtractor(model2, target_layers)
    
    # Log which layers we're extracting from
    for layer_idx in target_layers:
        if layer_idx in extractor1.index_to_name:
            logger.info(f"Layer {layer_idx}: {extractor1.index_to_name[layer_idx]}")

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

            # Get layer name for logging
            layer_name = extractor1.index_to_name.get(layer_idx, f"unknown_layer_{layer_idx}")
            
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
                layer_output_dir = os.path.join(args.output_dir, f"layer_{layer_idx}_{layer_name}")
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
                    )

        processed_samples += images.size(0)

    # Clean up
    extractor1.remove_hooks()
    extractor2.remove_hooks()

    # Generate distribution visualizations for each layer
    for layer_idx in target_layers:
        if layer_idx in all_features1_by_layer and all_features1_by_layer[layer_idx]:
            # Get layer name
            layer_name = extractor1.index_to_name.get(layer_idx, f"unknown_layer_{layer_idx}")
            
            # Create layer-specific output directory
            layer_output_dir = os.path.join(args.output_dir, f"layer_{layer_idx}_{layer_name}")
            if not os.path.exists(layer_output_dir):
                os.makedirs(layer_output_dir)
            
            # Visualize feature distributions
            logger.info(f"Generating feature distribution visualizations for layer {layer_idx} ({layer_name})...")
            visualize_feature_distributions(
                all_features1_by_layer[layer_idx],
                all_features2_by_layer[layer_idx],
                layer_output_dir,
                layer_idx,
                layer_name
            )
            
            # Visualize per-sample cosine similarities
            if layer_idx in per_sample_cosine_sims and per_sample_cosine_sims[layer_idx]:
                logger.info(f"Generating cosine similarity visualizations for layer {layer_idx} ({layer_name})...")
                visualize_cosine_similarities(
                    per_sample_cosine_sims[layer_idx],
                    layer_idx,
                    layer_name,
                    layer_output_dir
                )

    # Prepare data for layer comparison visualization
    layer_comparison_stats = {}
    
    for layer_idx in target_layers:
        if layer_idx in all_stats_by_layer and all_stats_by_layer[layer_idx]:
            # Get layer name
            layer_name = extractor1.index_to_name.get(layer_idx, f"unknown_layer_{layer_idx}")
            
            # Calculate average statistics for this layer
            layer_stats = all_stats_by_layer[layer_idx]
            avg_stats = {
                key: np.mean([stat[key] for stat in layer_stats]) for key in layer_stats[0].keys()
            }
            
            # Store for layer comparison visualization
            layer_comparison_stats[layer_idx] = {
                'name': layer_name,
                'l2_distance': avg_stats['l2_distance'],
                'cosine_similarity': avg_stats['cosine_similarity']
            }
            
            # Create layer-specific output directory
            layer_output_dir = os.path.join(args.output_dir, f"layer_{layer_idx}_{layer_name}")
            if not os.path.exists(layer_output_dir):
                os.makedirs(layer_output_dir)

            # Save statistics to file
            with open(os.path.join(layer_output_dir, "statistics.txt"), "w") as f:
                f.write(f"Model 1: {args.model1_path}\n")
                f.write(f"Model 2: {args.model2_path}\n")
                f.write(f"Layer {layer_idx}: {layer_name}\n\n")
                f.write("Average statistics:\n")
                for key, value in avg_stats.items():
                    f.write(f"{key}: {value:.4f}\n")

            logger.info(f"Layer {layer_idx} ({layer_name}) - Average cosine similarity: {avg_stats['cosine_similarity']:.4f}")
        else:
            logger.warning(f"No statistics collected for layer {layer_idx}")

    # Generate layer comparison visualization if we have multiple layers
    if len(layer_comparison_stats) > 1:
        logger.info("Generating layer comparison visualization...")
        visualize_layer_comparison(layer_comparison_stats, args.output_dir)

    # Also save a summary file with statistics from all layers
    with open(os.path.join(args.output_dir, "all_layers_summary.txt"), "w") as f:
        f.write(f"Model 1: {args.model1_path}\n")
        f.write(f"Model 2: {args.model2_path}\n\n")
        f.write("Average statistics by layer:\n\n")
        
        for layer_idx in target_layers:
            if layer_idx in all_stats_by_layer and all_stats_by_layer[layer_idx]:
                layer_name = extractor1.index_to_name.get(layer_idx, f"unknown_layer_{layer_idx}")
                layer_stats = all_stats_by_layer[layer_idx]
                avg_stats = {
                    key: np.mean([stat[key] for stat in layer_stats]) for key in layer_stats[0].keys()
                }
                
                f.write(f"Layer {layer_idx} ({layer_name}):\n")
                for key, value in avg_stats.items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
            else:
                layer_name = extractor1.index_to_name.get(layer_idx, f"unknown_layer_{layer_idx}")
                f.write(f"Layer {layer_idx} ({layer_name}): No statistics collected\n\n")

    logger.info("Feature extraction complete")


if __name__ == "__main__":
    main()
