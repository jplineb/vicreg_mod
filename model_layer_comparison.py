import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from custom_datasets import DATASETS
from utils.logging import configure_logging
from torchvision.models import resnet50
from utils.construct_model import FeatureExtractor
logger = configure_logging()

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Compare feature maps between two models"
    )
    parser.add_argument(
        "--model1-path", 
        type=Path, 
        required=True,
        help="Path to the first model's weights"
    )
    parser.add_argument(
        "--model2-path", 
        type=Path, 
        required=True,
        help="Path to the second model's weights"
    )
    parser.add_argument(
        "--task-ds",
        type=str,
        required=True,
        help="Dataset to use for feature extraction"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=3,
        help="Layer number to extract features from (0-based indexing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for dataset loading"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./layer_comparisons",
        help="Directory to save output visualizations"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of feature maps"
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
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Create ResNet50 model
    model = resnet50()
    
    # Replace the final FC layer with Identity for feature extraction
    model.fc = nn.Identity()
    
    # Process state dict keys if needed (remove prefixes like 'backbone.')
    processed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.backbone.'):
            new_key = key.replace('module.backbone.', '')
            processed_state_dict[new_key] = value
        elif key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            processed_state_dict[new_key] = value
        else:
            processed_state_dict[key] = value
    
    # Load state dict
    model.load_state_dict(processed_state_dict, strict=False)
    
    return model.cuda()

def visualize_feature_maps(features1, features2, output_dir, sample_idx, layer_idx):
    """
    Visualize and save comparisons of feature maps.
    
    Args:
        features1 (torch.Tensor): Features from model 1
        features2 (torch.Tensor): Features from model 2
        output_dir (Path): Directory to save visualizations
        sample_idx (int): Index of the sample
        layer_idx (int): Index of the layer
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Take first image in batch
    features1 = features1[0].cpu().numpy()
    features2 = features2[0].cpu().numpy()
    
    # Calculate number of channels to visualize (up to 16)
    num_channels = min(16, features1.shape[0])
    
    plt.figure(figsize=(16, 8))
    for i in range(num_channels):
        # Model 1 feature map
        plt.subplot(4, 8, i+1)
        plt.imshow(features1[i], cmap='viridis')
        plt.axis('off')
        if i == 0:
            plt.title('Model 1')
        
        # Model 2 feature map
        plt.subplot(4, 8, i+1+num_channels)
        plt.imshow(features2[i], cmap='viridis')
        plt.axis('off')
        if i == 0:
            plt.title('Model 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}_layer_{layer_idx}.png"))
    plt.close()

def calculate_statistics(features1:torch.Tensor, features2:torch.Tensor) -> dict:
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
        "cov2": cov2
    }

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
    
    # Set up feature extractors
    extractor1 = FeatureExtractor(model1, [args.layer])
    extractor2 = FeatureExtractor(model2, [args.layer])
    
    # Load dataset
    logger.info(f"Loading dataset: {args.task_ds}")
    dataset = DATASETS[args.task_ds](
        batch_size=args.batch_size,
        num_workers=4,
        gpu=torch.cuda.current_device(),
    )
    test_loader = dataset.get_dataloader(split="test")
    
    # Extract features
    logger.info("Extracting features...")
    
    all_stats = []
    processed_samples = 0
    
    for batch_idx, data in enumerate(test_loader):
        if processed_samples >= args.num_samples:
            break
        
        images = data["img"].cuda()

        # Store image ids and similarity metrics
        # Look at the class of images where adaptation is the larges
        # Find coefficient of variation of the features
        
        # Extract features from both models
        features1 = extractor1.extract_features(images)
        features2 = extractor2.extract_features(images)
        
        # Log statistics
        stats = calculate_statistics(features1[args.layer], features2[args.layer])
        all_stats.append(stats)
        
        logger.info(f"Batch {batch_idx}, All stats: {stats}")
        
        # Visualize feature maps
        if args.visualize:
            for i in range(min(images.size(0), args.num_samples - processed_samples)):
                sample_idx = batch_idx * args.batch_size + i
                visualize_feature_maps(
                    features1[args.layer][i:i+1],
                    features2[args.layer][i:i+1],
                    args.output_dir,
                    sample_idx,
                    args.layer
                )
        
        processed_samples += images.size(0)
    
    # Clean up
    extractor1.remove_hooks()
    extractor2.remove_hooks()
    
    # Calculate average statistics
    avg_stats = {
        key: np.mean([stat[key] for stat in all_stats]) 
        for key in all_stats[0].keys()
    }
    
    # Save statistics to file
    with open(os.path.join(args.output_dir, "statistics.txt"), "w") as f:
        f.write(f"Model 1: {args.model1_path}\n")
        f.write(f"Model 2: {args.model2_path}\n")
        f.write(f"Layer: {args.layer}\n\n")
        f.write("Average statistics:\n")
        for key, value in avg_stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    logger.info("Feature extraction complete")
    logger.info(f"Average cosine similarity: {avg_stats['cosine_similarity']:.4f}")

if __name__ == "__main__":
    main()
