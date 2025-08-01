#!/usr/bin/env python3
"""
Script to create comprehensive comparison plots from multiple model comparison outputs.
Combines figures from different pretrained datasets and training strategies.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import json
import re
import numpy as np

# Centralized color and style mapping for consistency across all plots
# Colors for pretraining datasets
IMAGENET_COLOR = '#1f77b4'  # Blue
RADIMAGENET_COLOR = '#ff7f0e'  # Orange

# Line styles for training strategies
SUPERVISED_LINESTYLE = '-'  # Solid line
VICREG_LINESTYLE = '-.'  # Dashed line

def get_model_style_mapping(comparison_names):
    """
    Create consistent color and line style mapping for all models.
    Uses consistent colors for pretraining datasets and line styles for training strategies.
    Returns a dictionary mapping model names to (color, linestyle) tuples.
    """
    style_mapping = {}
    
    for name in comparison_names:
        # Determine color based on pretraining dataset
        if "ImageNet" in name and "RadImageNet" not in name:
            color = IMAGENET_COLOR
        elif "RadImageNet" in name:
            color = RADIMAGENET_COLOR
        else:
            # Fallback color if dataset is unclear
            color = '#2ca02c'  # Green
        
        # Determine line style based on training strategy
        if "Supervised" in name and "VICREG" not in name:
            linestyle = SUPERVISED_LINESTYLE
        elif "VICREG" in name and "Supervised" not in name:
            linestyle = VICREG_LINESTYLE
        else:
            # Fallback line style if strategy is unclear
            linestyle = '-.'
        
        style_mapping[name] = (color, linestyle)
    
    return style_mapping

def get_model_color(comparison_name, style_mapping):
    """Get the color for a specific model."""
    return style_mapping.get(comparison_name, (IMAGENET_COLOR, SUPERVISED_LINESTYLE))[0]

def get_model_linestyle(comparison_name, style_mapping):
    """Get the line style for a specific model."""
    return style_mapping.get(comparison_name, (IMAGENET_COLOR, SUPERVISED_LINESTYLE))[1]



def load_cumulative_stats(comparison_dir):
    """Load cumulative statistics from a comparison directory."""
    stats_file = os.path.join(comparison_dir, "feature_comparisons", "cumulative_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None

def create_comprehensive_comparison_plot(comparison_dirs, output_dir="comprehensive_comparison"):
    """
    Create a comprehensive comparison plot from multiple model comparison directories.
    Displays model names on the left side in a grid style and shows them only once.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the number of comparisons
    num_comparisons = len(comparison_dirs)
    
    # Define the figure layout with space for model names on the left
    fig = plt.figure(figsize=(48, 32))
    # Create a 4-column grid: 1 for model names, 3 for plots
    gs = fig.add_gridspec(num_comparisons, 4, hspace=0.3, wspace=0.2, 
                         width_ratios=[0.2, 1, 1, 1])
    
    # Create consistent style mapping for all models
    style_mapping = get_model_style_mapping(comparison_dirs.keys())
    
    # Collect all L2 distance data for shared y-axis limits
    all_l2_data = []
    for comparison_name, comparison_path in comparison_dirs.items():
        l2_file = os.path.join(comparison_path, "feature_comparisons", "l2_distance_line.png")
        if os.path.exists(l2_file):
            # Load the L2 distance data from the JSON file instead of the image
            stats_file = os.path.join(comparison_path, "feature_comparisons", "cumulative_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    if stats and 'l2_distances' in stats:
                        all_l2_data.extend(stats['l2_distances'])
    
    # Calculate shared y-axis limits for L2 distance
    if all_l2_data:
        l2_min = min(all_l2_data)
        l2_max = max(all_l2_data)
        l2_margin = (l2_max - l2_min) * 0.05
        shared_l2_ylim = (l2_min - l2_margin, l2_max + l2_margin)
    else:
        shared_l2_ylim = (0, 1)  # Default fallback
    
    for idx, (comparison_name, comparison_path) in enumerate(comparison_dirs.items()):
        print(f"Processing {comparison_name}...")
        
        # Get consistent color for this model
        model_color = get_model_color(comparison_name, style_mapping)
        
        # Add model name on the left side (rotated 90 degrees)
        ax_name = fig.add_subplot(gs[idx, 0])
        ax_name.text(0.5, 0.5, comparison_name, fontsize=12, fontweight='bold', 
                    ha='center', va='center', rotation=90, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=model_color, alpha=0.3))
        ax_name.axis('off')
        
        # Create actual plots instead of displaying images for better control
        ax_l2 = fig.add_subplot(gs[idx, 1])
        ax_cosine = fig.add_subplot(gs[idx, 2])
        ax_ratios = fig.add_subplot(gs[idx, 3])
        
        # Load and plot L2 Distance data
        stats_file = os.path.join(comparison_path, "feature_comparisons", "cumulative_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if stats and 'l2_distances' in stats:
                    layer_names = stats['layer_names']
                    l2_distances = stats['l2_distances']
                    ax_l2.plot(range(len(layer_names)), l2_distances, 'o', linewidth=2, markersize=4, 
                              color=get_model_color(comparison_name, style_mapping), 
                              linestyle=get_model_linestyle(comparison_name, style_mapping))
                    ax_l2.set_xticks(range(len(layer_names)))
                    ax_l2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
                    ax_l2.set_ylim(shared_l2_ylim)
                    ax_l2.grid(True, alpha=0.3)
                    
                    # Plot cosine similarity
                    if 'cosine_similarities' in stats:
                        cosine_sims = stats['cosine_similarities']
                        ax_cosine.plot(range(len(layer_names)), cosine_sims, 'o', linewidth=2, markersize=4,
                                      color=get_model_color(comparison_name, style_mapping),
                                      linestyle=get_model_linestyle(comparison_name, style_mapping))
                        ax_cosine.set_xticks(range(len(layer_names)))
                        ax_cosine.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
                        ax_cosine.set_ylim(0, 1.05)
                        ax_cosine.grid(True, alpha=0.3)
                    
                    # Plot layer ratios
                    if 'layer_ratios' in stats and len(stats['layer_ratios']) > 0:
                        ratios = stats['layer_ratios']
                        ax_ratios.plot(range(len(ratios)), ratios, 'o', linewidth=2, markersize=4,
                                      color=get_model_color(comparison_name, style_mapping),
                                      linestyle=get_model_linestyle(comparison_name, style_mapping))
                        ax_ratios.set_xticks(range(len(ratios)))
                        ax_ratios.set_xticklabels(layer_names[1:], rotation=45, ha='right', fontsize=8)
                        ax_ratios.set_ylim(0, 4)
                        ax_ratios.grid(True, alpha=0.3)
                        ax_ratios.axhline(y=1, color='black', linestyle='--', alpha=0.3)
                else:
                    # Create empty subplots if no data
                    for ax, title in [(ax_l2, "L2 Distance"), (ax_cosine, "Cosine Similarity"), (ax_ratios, "Layer Ratios")]:
                        ax.text(0.5, 0.5, f"No {title}\nData Available", 
                               fontsize=10, ha='center', va='center', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
                        ax.axis('off')
        else:
            # Create empty subplots if file doesn't exist
            for ax, title in [(ax_l2, "L2 Distance"), (ax_cosine, "Cosine Similarity"), (ax_ratios, "Layer Ratios")]:
                ax.text(0.5, 0.5, f"No {title}\nData Available", 
                       fontsize=10, ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
                ax.axis('off')
    
    # Add main title
    fig.suptitle("Comprehensive Model Comparison: Pretrained Datasets vs Training Strategies", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Add column labels (skip Model column)
    column_labels = ["L2 Distance", "Cosine Similarity", "Layer Ratios"]
    for col_idx, label in enumerate(column_labels):
        # Plot column labels - adjusted positions for new layout
        x_pos = 0.3 + col_idx * 0.23
        fig.text(x_pos, 0.95, label, fontsize=14, fontweight='bold', ha='center')
    
    plt.savefig(os.path.join(output_dir, "comprehensive_model_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive comparison saved to {output_dir}/comprehensive_model_comparison.png")

def filter_stats_for_target_layers(stats):
    """
    Filter statistics to only include top-level layers and bottleneck outputs.
    Uses the same logic as filter_target_layers in model_layer_comparison.py.
    """
    if not stats or 'layer_names' not in stats:
        return None
    
    # Get the layer names and indices
    layer_names = stats['layer_names']
    layer_indices = stats['layer_indices']
    l2_distances = stats['l2_distances']
    cosine_sims = stats['cosine_similarities']
    
    # Filter to only include target layers
    filtered_indices = []
    filtered_names = []
    filtered_l2 = []
    filtered_cosine = []
    
    for i, layer_name in enumerate(layer_names):
        # Include top-level layers
        if layer_name in ['conv1', 'bn1', 'relu', 'maxpool']:
            filtered_indices.append(layer_indices[i])
            filtered_names.append(layer_name)
            filtered_l2.append(l2_distances[i])
            filtered_cosine.append(cosine_sims[i])
            continue
        
        # Include all bottleneck outputs in layer1, layer2, layer3
        if re.match(r'^layer[123]\.\d+$', layer_name):
            filtered_indices.append(layer_indices[i])
            filtered_names.append(layer_name)
            filtered_l2.append(l2_distances[i])
            filtered_cosine.append(cosine_sims[i])
            continue
    
    if not filtered_indices:
        return None
    
    # Recalculate cumulative L2
    cumulative_l2 = np.cumsum(filtered_l2)
    
    # Recalculate layer ratios if we have enough layers
    layer_ratios = []
    if len(filtered_l2) > 1:
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        layer_ratios = (np.array(filtered_l2[1:]) / (np.array(filtered_l2[:-1]) + epsilon)).tolist()
    
    return {
        'layer_indices': filtered_indices,
        'layer_names': filtered_names,
        'l2_distances': filtered_l2,
        'cumulative_l2': cumulative_l2.tolist(),
        'cosine_similarities': filtered_cosine,
        'layer_ratios': layer_ratios
    }

def create_cumulative_comparison_plot(comparison_dirs, output_dir="comprehensive_comparison"):
    """
    Create a comparison plot focusing on cumulative L2 distances across all comparisons.
    Uses the same layer filtering as model_layer_comparison.py (top-level + bottlenecks).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load cumulative statistics from all comparisons
    all_stats = {}
    for comparison_name, comparison_path in comparison_dirs.items():
        stats = load_cumulative_stats(comparison_path)
        if stats:
            # Filter to only include top-level layers and bottleneck outputs
            filtered_stats = filter_stats_for_target_layers(stats)
            if filtered_stats:
                all_stats[comparison_name] = filtered_stats
    
    if not all_stats:
        print("No cumulative statistics found!")
        return
    
    # Create consistent style mapping for all models
    style_mapping = get_model_style_mapping(all_stats.keys())
    
    # Calculate global min and max for consistent scaling
    all_cumulative_l2 = []
    for stats in all_stats.values():
        all_cumulative_l2.extend(stats['cumulative_l2'])
    
    global_min_l2 = min(all_cumulative_l2)
    global_max_l2 = max(all_cumulative_l2)
    l2_margin = (global_max_l2 - global_min_l2) * 0.05  # 5% margin
    
    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Plot 1: Cumulative L2 distances
    for comparison_name, stats in all_stats.items():
        layer_names = stats['layer_names']
        cumulative_l2 = stats['cumulative_l2']
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        ax1.plot(range(len(layer_names)), cumulative_l2, 'o', linewidth=2, markersize=6, 
                label=comparison_name, alpha=0.8, color=color, linestyle=linestyle)
    
    # Set x-axis labels to layer names
    ax1.set_xticks(range(len(all_stats[list(all_stats.keys())[0]]['layer_names'])))
    ax1.set_xticklabels(all_stats[list(all_stats.keys())[0]]['layer_names'], rotation=45, ha='right', fontsize=10)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Cumulative L2 Distance', fontsize=12)
    ax1.set_title('Cumulative Feature Differences Across All Comparisons\n(Top-level + Bottleneck Layers)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(global_min_l2 - l2_margin, global_max_l2 + l2_margin)
    
    # Plot 2: Cosine similarities
    for comparison_name, stats in all_stats.items():
        layer_names = stats['layer_names']
        cosine_sims = stats['cosine_similarities']
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        ax2.plot(range(len(layer_names)), cosine_sims, 'o', linewidth=2, markersize=6, 
                label=comparison_name, alpha=0.8, color=color, linestyle=linestyle)
    
    # Set x-axis labels to layer names
    ax2.set_xticks(range(len(all_stats[list(all_stats.keys())[0]]['layer_names'])))
    ax2.set_xticklabels(all_stats[list(all_stats.keys())[0]]['layer_names'], rotation=45, ha='right', fontsize=10)
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Cosine Similarities Across All Comparisons\n(Top-level + Bottleneck Layers)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cumulative comparison saved to {output_dir}/cumulative_comparison.png")

def create_layer_ratio_comparison_plot(comparison_dirs, output_dir="comprehensive_comparison"):
    """
    Create a comparison plot focusing on layer ratios across all comparisons.
    Uses the same layer filtering as model_layer_comparison.py (top-level + bottlenecks).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_stats = {}
    for comparison_name, comparison_path in comparison_dirs.items():
        stats = load_cumulative_stats(comparison_path)
        if stats and 'layer_ratios' in stats:
            # Filter to only include top-level layers and bottleneck outputs
            filtered_stats = filter_stats_for_target_layers(stats)
            if filtered_stats and 'layer_ratios' in filtered_stats:
                all_stats[comparison_name] = filtered_stats
    
    if not all_stats:
        print("No layer ratio statistics found!")
        return
    
    # Check if any stats have layer ratios
    has_ratios = any('layer_ratios' in stats and len(stats['layer_ratios']) > 0 for stats in all_stats.values())
    if not has_ratios:
        print("No valid layer ratios found!")
        return
    
    # Create consistent style mapping for all models
    style_mapping = get_model_style_mapping(all_stats.keys())
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    for comparison_name, stats in all_stats.items():
        layer_names = stats['layer_names']
        ratios = stats['layer_ratios']
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        # Plot all ratios (they already skip the first layer in calculation)
        ax.plot(range(len(ratios)), ratios, 'o', linewidth=2, markersize=6, label=comparison_name, alpha=0.8, color=color, linestyle=linestyle)
    
    # Set x-axis labels to layer names (skip the first layer for ratios)
    if all_stats:
        first_stats = list(all_stats.values())[0]
        layer_names = first_stats['layer_names'][1:]  # Skip first layer for ratios
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Layer Ratio (L2_n / L2_{n-1})', fontsize=12)
    ax.set_title('Layer Ratios Across All Comparisons\n(Top-level + Bottleneck Layers)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4)
    
    # Add horizontal line at y=1 (indicating no change)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_ratio_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Layer ratio comparison saved to {output_dir}/layer_ratio_comparison.png")

def create_strategy_comparison_plot(comparison_dirs, output_dir="comprehensive_comparison"):
    """
    Create a plot comparing different training strategies (Supervised vs VICREG) for the same pretrained dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group comparisons by pretrained dataset (ImageNet vs RadImageNet)
    imagenet_comparisons = []
    radimagenet_comparisons = []
    
    for comparison_name, comparison_path in comparison_dirs.items():
        stats = load_cumulative_stats(comparison_path)
        if stats:
            filtered_stats = filter_stats_for_target_layers(stats)
            if filtered_stats:
                # Check if it's an ImageNet comparison (contains ImageNet but not RadImageNet)
                if "ImageNet" in comparison_name and "RadImageNet" not in comparison_name:
                    imagenet_comparisons.append((comparison_name, filtered_stats))
                # Check if it's a RadImageNet comparison (contains RadImageNet)
                elif "RadImageNet" in comparison_name:
                    radimagenet_comparisons.append((comparison_name, filtered_stats))
    
    # Create consistent style mapping for all models
    all_comparison_names = [name for name, _ in imagenet_comparisons + radimagenet_comparisons]
    style_mapping = get_model_style_mapping(all_comparison_names)
    
    # Calculate global min and max for consistent scaling across all L2 plots
    all_cumulative_l2 = []
    for _, stats in imagenet_comparisons + radimagenet_comparisons:
        all_cumulative_l2.extend(stats['cumulative_l2'])
    
    global_min_l2 = min(all_cumulative_l2)
    global_max_l2 = max(all_cumulative_l2)
    l2_margin = (global_max_l2 - global_min_l2) * 0.05  # 5% margin
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot ImageNet comparisons (Supervised vs VICREG)
    for comparison_name, stats in imagenet_comparisons:
        layer_names = stats['layer_names']
        cumulative_l2 = stats['cumulative_l2']
        cosine_sims = stats['cosine_similarities']
        
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        ax1.plot(range(len(layer_names)), cumulative_l2, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
        ax2.plot(range(len(layer_names)), cosine_sims, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
    
    if imagenet_comparisons:
        # Set x-axis labels to layer names
        layer_names = imagenet_comparisons[0][1]['layer_names']
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        
        ax1.set_title('ImageNet: Supervised vs VICREG\nCumulative L2 (Top-level + Bottleneck Layers)')
        ax2.set_title('ImageNet: Supervised vs VICREG\nCosine Similarity (Top-level + Bottleneck Layers)')
        ax1.legend()
        ax2.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.set_ylim(global_min_l2 - l2_margin, global_max_l2 + l2_margin)
        ax2.set_ylim(-1.05, 1.05)
    
    # Plot RadImageNet comparisons (Supervised vs VICREG)
    for comparison_name, stats in radimagenet_comparisons:
        layer_names = stats['layer_names']
        cumulative_l2 = stats['cumulative_l2']
        cosine_sims = stats['cosine_similarities']
        
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        ax3.plot(range(len(layer_names)), cumulative_l2, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
        ax4.plot(range(len(layer_names)), cosine_sims, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
    
    if radimagenet_comparisons:
        # Set x-axis labels to layer names
        layer_names = radimagenet_comparisons[0][1]['layer_names']
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        ax4.set_xticks(range(len(layer_names)))
        ax4.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        
        ax3.set_title('RadImageNet: Supervised vs VICREG\nCumulative L2 (Top-level + Bottleneck Layers)')
        ax4.set_title('RadImageNet: Supervised vs VICREG\nCosine Similarity (Top-level + Bottleneck Layers)')
        ax3.legend()
        ax4.legend()
        ax3.grid(True, alpha=0.3)
        ax4.grid(True, alpha=0.3)
        ax3.set_ylim(global_min_l2 - l2_margin, global_max_l2 + l2_margin)
        ax4.set_ylim(-1.05, 1.05)
    
    plt.suptitle('Training Strategy Comparison: Supervised vs VICREG', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Strategy comparison saved to {output_dir}/strategy_comparison.png")

def create_pretrained_dataset_comparison_plot(comparison_dirs, output_dir="comprehensive_comparison"):
    """
    Create a plot comparing different pretrained datasets (ImageNet vs RadImageNet) for the same training strategy.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group comparisons by training strategy (Supervised vs VICREG)
    supervised_comparisons = []
    vicreg_comparisons = []
    
    for comparison_name, comparison_path in comparison_dirs.items():
        stats = load_cumulative_stats(comparison_path)
        if stats:
            filtered_stats = filter_stats_for_target_layers(stats)
            if filtered_stats:
                # Check if it's a Supervised comparison (contains Supervised but not VICREG)
                if "Supervised" in comparison_name and "VICREG" not in comparison_name:
                    supervised_comparisons.append((comparison_name, filtered_stats))
                # Check if it's a VICREG comparison (contains VICREG but not Supervised)
                elif "VICREG" in comparison_name and "Supervised" not in comparison_name:
                    vicreg_comparisons.append((comparison_name, filtered_stats))
    
    # Create consistent style mapping for all models
    all_comparison_names = [name for name, _ in supervised_comparisons + vicreg_comparisons]
    style_mapping = get_model_style_mapping(all_comparison_names)
    
    # Calculate global min and max for consistent scaling across all L2 plots
    all_cumulative_l2 = []
    for _, stats in supervised_comparisons + vicreg_comparisons:
        all_cumulative_l2.extend(stats['cumulative_l2'])
    
    global_min_l2 = min(all_cumulative_l2)
    global_max_l2 = max(all_cumulative_l2)
    l2_margin = (global_max_l2 - global_min_l2) * 0.05  # 5% margin
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot Supervised comparisons (ImageNet vs RadImageNet)
    for comparison_name, stats in supervised_comparisons:
        layer_names = stats['layer_names']
        cumulative_l2 = stats['cumulative_l2']
        cosine_sims = stats['cosine_similarities']
        
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        ax1.plot(range(len(layer_names)), cumulative_l2, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
        ax2.plot(range(len(layer_names)), cosine_sims, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
    
    if supervised_comparisons:
        # Set x-axis labels to layer names
        layer_names = supervised_comparisons[0][1]['layer_names']
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        
        ax1.set_title('Supervised: ImageNet vs RadImageNet\nCumulative L2 (Top-level + Bottleneck Layers)')
        ax2.set_title('Supervised: ImageNet vs RadImageNet\nCosine Similarity (Top-level + Bottleneck Layers)')
        ax1.legend()
        ax2.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.set_ylim(global_min_l2 - l2_margin, global_max_l2 + l2_margin)
        ax2.set_ylim(-1.05, 1.05)
    
    # Plot VICREG comparisons (ImageNet vs RadImageNet)
    for comparison_name, stats in vicreg_comparisons:
        layer_names = stats['layer_names']
        cumulative_l2 = stats['cumulative_l2']
        cosine_sims = stats['cosine_similarities']
        
        color = get_model_color(comparison_name, style_mapping)
        linestyle = get_model_linestyle(comparison_name, style_mapping)
        ax3.plot(range(len(layer_names)), cumulative_l2, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
        ax4.plot(range(len(layer_names)), cosine_sims, 'o', linewidth=2, markersize=6, 
                label=comparison_name, color=color, linestyle=linestyle)
    
    if vicreg_comparisons:
        # Set x-axis labels to layer names
        layer_names = vicreg_comparisons[0][1]['layer_names']
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        ax4.set_xticks(range(len(layer_names)))
        ax4.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
        
        ax3.set_title('VICREG: ImageNet vs RadImageNet\nCumulative L2 (Top-level + Bottleneck Layers)')
        ax4.set_title('VICREG: ImageNet vs RadImageNet\nCosine Similarity (Top-level + Bottleneck Layers)')
        ax3.legend()
        ax4.legend()
        ax3.grid(True, alpha=0.3)
        ax4.grid(True, alpha=0.3)
        ax3.set_ylim(global_min_l2 - l2_margin, global_max_l2 + l2_margin)
        ax4.set_ylim(-1.05, 1.05)
    
    plt.suptitle('Pretrained Dataset Comparison: ImageNet vs RadImageNet', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pretrained_dataset_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pretrained dataset comparison saved to {output_dir}/pretrained_dataset_comparison.png")

def main():
    """Main function to create all comprehensive comparison plots."""
    
    # Define the comparison directories for ChexPert
    chexpert_comparison_dirs = {
        "Chexpert Supervised ImageNet": "./layer_comparisons/chexpert_Base_Supervised_ImageNet_VS_Supervised_Imagenet",
        "Chexpert VICREG ImageNet": "./layer_comparisons/chexpert_Base_VICREG_ImageNet_VS_VICREG_ImageNet",
        "Chexpert Supervised RadImageNet": "./layer_comparisons/chexpert_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet",
        "Chexpert VICREG RadImageNet": "./layer_comparisons/chexpert_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet"
    }
    
    # Define the comparison directories for VINDR-CXR
    vindrcxr_comparison_dirs = {
        "VINDR-CXR Supervised ImageNet": "./layer_comparisons/vindrcxr_Base_Supervised_ImageNet_VS_Supervised_Imagenet",
        "VINDR-CXR VICREG ImageNet": "./layer_comparisons/vindrcxr_Base_VICREG_ImageNet_VS_VICREG_ImageNet",
        "VINDR-CXR Supervised RadImageNet": "./layer_comparisons/vindrcxr_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet",
        "VINDR-CXR VICREG RadImageNet": "./layer_comparisons/vindrcxr_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet"
    }
    
    # Define the comparison directories for BCN2K
    bcn2k_comparison_dirs = {
        "BCN2K Supervised ImageNet": "./layer_comparisons/bcn2k_Base_Supervised_ImageNet_VS_Supervised_Imagenet",
        "BCN2K VICREG ImageNet": "./layer_comparisons/bcn2k_Base_VICREG_ImageNet_VS_VICREG_ImageNet",
        "BCN2K Supervised RadImageNet": "./layer_comparisons/bcn2k_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet",
        "BCN2K VICREG RadImageNet": "./layer_comparisons/bcn2k_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet"
    }
    
    # Check which ChexPert directories exist
    existing_chexpert_comparisons = {}
    for name, path in chexpert_comparison_dirs.items():
        if os.path.exists(path):
            existing_chexpert_comparisons[name] = path
            print(f"Found ChexPert comparison: {name}")
        else:
            print(f"Warning: ChexPert comparison directory not found: {path}")
    
    # Check which VINDR-CXR directories exist
    existing_vindrcxr_comparisons = {}
    for name, path in vindrcxr_comparison_dirs.items():
        if os.path.exists(path):
            existing_vindrcxr_comparisons[name] = path
            print(f"Found VINDR-CXR comparison: {name}")
        else:
            print(f"Warning: VINDR-CXR comparison directory not found: {path}")
    
    # Check which BCN2K directories exist
    existing_bcn2k_comparisons = {}
    for name, path in bcn2k_comparison_dirs.items():
        if os.path.exists(path):
            existing_bcn2k_comparisons[name] = path
            print(f"Found BCN2K comparison: {name}")
        else:
            print(f"Warning: BCN2K comparison directory not found: {path}")
    
    # Create output directories
    chexpert_output_dir = "comprehensive_comparison_chexpert"
    vindrcxr_output_dir = "comprehensive_comparison_vindrcxr"
    bcn2k_output_dir = "comprehensive_comparison_bcn2k"
    
    # Process ChexPert comparisons if any exist
    if existing_chexpert_comparisons:
        print(f"\nCreating ChexPert comprehensive comparison plots...")
        
        # 1. Main comprehensive comparison
        create_comprehensive_comparison_plot(existing_chexpert_comparisons, chexpert_output_dir)
        
        # 2. Cumulative comparison
        create_cumulative_comparison_plot(existing_chexpert_comparisons, chexpert_output_dir)
        
        # 2b. Layer ratio comparison
        create_layer_ratio_comparison_plot(existing_chexpert_comparisons, chexpert_output_dir)
        
        # 3. Strategy comparison (Supervised vs VICREG)
        create_strategy_comparison_plot(existing_chexpert_comparisons, chexpert_output_dir)
        
        # 4. Pretrained dataset comparison (ImageNet vs RadImageNet)
        create_pretrained_dataset_comparison_plot(existing_chexpert_comparisons, chexpert_output_dir)
        
        print(f"\nCheXpert comparison plots saved to {chexpert_output_dir}/")
        print("Files created:")
        print("- comprehensive_model_comparison.png")
        print("- cumulative_comparison.png") 
        print("- layer_ratio_comparison.png")
        print("- strategy_comparison.png")
        print("- pretrained_dataset_comparison.png")
    else:
        print("No ChexPert comparison directories found!")
    
    # Process VINDR-CXR comparisons if any exist
    if existing_vindrcxr_comparisons:
        print(f"\nCreating VINDR-CXR comprehensive comparison plots...")
        
        # 1. Main comprehensive comparison
        create_comprehensive_comparison_plot(existing_vindrcxr_comparisons, vindrcxr_output_dir)
        
        # 2. Cumulative comparison
        create_cumulative_comparison_plot(existing_vindrcxr_comparisons, vindrcxr_output_dir)
        
        # 2b. Layer ratio comparison
        create_layer_ratio_comparison_plot(existing_vindrcxr_comparisons, vindrcxr_output_dir)
        
        # 3. Strategy comparison (Supervised vs VICREG)
        create_strategy_comparison_plot(existing_vindrcxr_comparisons, vindrcxr_output_dir)
        
        # 4. Pretrained dataset comparison (ImageNet vs RadImageNet)
        create_pretrained_dataset_comparison_plot(existing_vindrcxr_comparisons, vindrcxr_output_dir)
        
        print(f"\nVINDR-CXR comparison plots saved to {vindrcxr_output_dir}/")
        print("Files created:")
        print("- comprehensive_model_comparison.png")
        print("- cumulative_comparison.png") 
        print("- layer_ratio_comparison.png")
        print("- strategy_comparison.png")
        print("- pretrained_dataset_comparison.png")
    else:
        print("No VINDR-CXR comparison directories found!")
    
    # Process BCN2K comparisons if any exist
    if existing_bcn2k_comparisons:
        print(f"\nCreating BCN2K comprehensive comparison plots...")
        
        # 1. Main comprehensive comparison
        create_comprehensive_comparison_plot(existing_bcn2k_comparisons, bcn2k_output_dir)
        
        # 2. Cumulative comparison
        create_cumulative_comparison_plot(existing_bcn2k_comparisons, bcn2k_output_dir)
        
        # 2b. Layer ratio comparison
        create_layer_ratio_comparison_plot(existing_bcn2k_comparisons, bcn2k_output_dir)
        
        # 3. Strategy comparison (Supervised vs VICREG)
        create_strategy_comparison_plot(existing_bcn2k_comparisons, bcn2k_output_dir)
        
        # 4. Pretrained dataset comparison (ImageNet vs RadImageNet)
        create_pretrained_dataset_comparison_plot(existing_bcn2k_comparisons, bcn2k_output_dir)
        
        print(f"\nBCN2K comparison plots saved to {bcn2k_output_dir}/")
        print("Files created:")
        print("- comprehensive_model_comparison.png")
        print("- cumulative_comparison.png") 
        print("- layer_ratio_comparison.png")
        print("- strategy_comparison.png")
        print("- pretrained_dataset_comparison.png")
    else:
        print("No BCN2K comparison directories found!")
    

if __name__ == "__main__":
    main() 