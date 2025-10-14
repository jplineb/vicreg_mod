#!/usr/bin/env python3
"""
Script to create comprehensive comparison plots from multiple model comparison outputs.
Combines figures from different pretrained datasets and training strategies.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np
import os
import json
import re

# Centralized color and style mapping for consistency across all plots
# Colors for pretraining datasets
IMAGENET_COLOR = '#1f77b4'  # Blue
RADIMAGENET_COLOR = '#ff7f0e'  # Orange

# Line styles for training strategies
SUPERVISED_LINESTYLE = '-'  # Solid line
VICREG_LINESTYLE = '-.'  # Dashed line

AUROC_SCORES_BY_TASK = {
        "Chexpert": {
            "Chexpert Supervised ImageNet": 0.8383,
            "Chexpert VICREG ImageNet": 0.8012,
            "Chexpert Supervised RadImageNet": 0.8374,
            "Chexpert VICREG RadImageNet": 0.8402,
        },
        "MIMIC-CXR": {
            "MIMIC-CXR Supervised ImageNet": 0.7620,
            "MIMIC-CXR VICREG ImageNet": 0.7768,
            "MIMIC-CXR Supervised RadImageNet": 0.7951,
            "MIMIC-CXR VICREG RadImageNet": 0.7953,
        },
        "VINDR-CXR": {
            "VINDR-CXR Supervised ImageNet": 0.6349,
            "VINDR-CXR VICREG ImageNet": 0.6557,
            "VINDR-CXR Supervised RadImageNet": 0.6405,
            "VINDR-CXR VICREG RadImageNet": 0.6288,
        },
        "Messidor": {
            "Messidor Supervised ImageNet": 0.8444,
            "Messidor VICREG ImageNet": 0.8382,
            "Messidor Supervised RadImageNet": 0.8004,
            "Messidor VICREG RadImageNet": 0.7991,
        },
        "BCN2K": {
            "BCN2K Supervised ImageNet": 0.9310,
            "BCN2K VICREG ImageNet": 0.9519,
            "BCN2K Supervised RadImageNet": 0.9009,
            "BCN2K VICREG RadImageNet": 0.9233
        }
    }

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

# New: Load aggregated channel cosine stats

def load_aggregated_channel_stats(comparison_dir):
    """Load aggregated channel cosine statistics and layer names from a comparison directory.
    Returns None if missing.
    """
    agg_file = os.path.join(comparison_dir, "feature_comparisons", "aggregated_channel_stats_by_layer.json")
    names_file = os.path.join(comparison_dir, "feature_comparisons", "layer_names.json")
    if not (os.path.exists(agg_file) and os.path.exists(names_file)):
        return None
    with open(agg_file, 'r') as f:
        agg = json.load(f)
    with open(names_file, 'r') as f:
        layer_names = {int(k): v for k, v in json.load(f).items()}
    return {"aggregated": {int(k): v for k, v in agg.items()}, "layer_names": layer_names}


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

# Helper: filter target layers (top-level + bottleneck outputs layer1-3)

def filter_layer_indices_for_channel_cos(layer_names_map, layer_indices_sorted):
    filtered_pairs = []
    for idx in layer_indices_sorted:
        name = layer_names_map.get(idx, "")
        if name in ['conv1', 'bn1', 'relu', 'maxpool']:
            filtered_pairs.append((idx, name))
            continue
        if re.match(r'^layer[123]\.\d+$', name):
            filtered_pairs.append((idx, name))
            continue
    return filtered_pairs

def create_all_tasks_channel_cosine_page(
    dataset_to_dirs: dict,
    output_dir: str = "comprehensive_comparison",
    filename: str = "all_tasks_channel_cosine.png",
):
    """Create a 2x2 page: one subplot per task dataset, overlaying channel-wise cosine similarity per comparison.
    dataset_to_dirs maps dataset name (Chexpert, VINDR-CXR, BCN2K, Messidor) -> {comparison_name: path}.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Rank models within each task
    ranked_models_by_task = {}
    for task, scores in AUROC_SCORES_BY_TASK.items():
        sorted_models = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_models_by_task[task] = {name: rank + 1 for rank, (name, _) in enumerate(sorted_models)}

    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    axes = axes.flatten()

    # Maintain consistent style mapping across all comparisons
    all_comp_names = []
    for dirs in dataset_to_dirs.values():
        all_comp_names.extend(list(dirs.keys()))
    style_mapping = get_model_style_mapping(all_comp_names)

    dataset_names = list(dataset_to_dirs.keys())
    for ax_idx, dataset_name in enumerate(dataset_names):
        ax = axes[ax_idx]
        comparisons = dataset_to_dirs.get(dataset_name, {})
        if not comparisons:
            ax.text(0.5, 0.5, f"No data for {dataset_name}", ha='center', va='center')
            ax.axis('off')
            continue

        plotted_any = False
        x_ticks = None
        x_labels = None

        for comp_name, comp_path in comparisons.items():
            loaded = load_aggregated_channel_stats(comp_path)
            if not loaded:
                continue
            agg = loaded["aggregated"]
            layer_names_map = loaded["layer_names"]
            # Sort layer indices, then filter to target layers
            sorted_layer_indices = sorted(agg.keys())
            filtered = filter_layer_indices_for_channel_cos(layer_names_map, sorted_layer_indices)
            if not filtered:
                continue

            # Prepare X (layer names) and Y (mean over channels per layer)
            layer_indices = [idx for idx, _ in filtered]
            layer_labels = [layer_names_map[idx] for idx in layer_indices]

            y_vals = []
            for idx in layer_indices:
                stats = agg.get(idx)
                if not stats:
                    y_vals.append(np.nan)
                    continue
                # channel_cosine_sim_mean is an array per channel; average across channels
                chan_means = stats.get('channel_cosine_sim_mean')
                if isinstance(chan_means, list) and len(chan_means) > 0:
                    y_vals.append(float(np.mean(chan_means)))
                else:
                    # fallback to overall mean if available
                    overall = stats.get('overall_cosine_sim_mean')
                    y_vals.append(float(overall) if overall is not None else np.nan)

            color = get_model_color(comp_name, style_mapping)
            linestyle = get_model_linestyle(comp_name, style_mapping)
            # Update legend with rank
            rank = ranked_models_by_task[dataset_name].get(comp_name, "N/A")
            ax.plot(range(len(layer_indices)), y_vals, 'o-', linewidth=2, markersize=4,
                    label=f"{comp_name} ({rank})", color=color, linestyle=linestyle)
            plotted_any = True

            if x_ticks is None:
                x_ticks = list(range(len(layer_indices)))
                x_labels = layer_labels

        if plotted_any and x_ticks is not None and x_labels is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(-0.2, 1.0)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{dataset_name}: Channel-wise Cosine Similarity (mean across channels)")
            ax.legend(loc='best', fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No channel cosine stats for {dataset_name}", ha='center', va='center')
            ax.axis('off')

    plt.suptitle("Channel-wise Cosine Similarity Across Tasks", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Channel cosine comparison saved to {out_path}")


def create_pretraining_strategy_performance_plot(
    dataset_to_dirs: dict,
    output_dir: str = "comprehensive_comparison",
    filename: str = "pretraining_strategy_performance.png",
):
    """Create a plot showing how each pretraining strategy + dataset combination performs across all downstream tasks.
    Each subplot shows one pretraining combination (e.g., "Supervised ImageNet") across all tasks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract unique pretraining combinations
    pretraining_combinations = set()
    for task_scores in AUROC_SCORES_BY_TASK.values():
        for model_name in task_scores.keys():
            # Extract the pretraining strategy + dataset part
            if "Supervised ImageNet" in model_name:
                pretraining_combinations.add("Supervised ImageNet")
            elif "VICREG ImageNet" in model_name:
                pretraining_combinations.add("VICREG ImageNet")
            elif "Supervised RadImageNet" in model_name:
                pretraining_combinations.add("Supervised RadImageNet")
            elif "VICREG RadImageNet" in model_name:
                pretraining_combinations.add("VICREG RadImageNet")

    pretraining_combinations = sorted(list(pretraining_combinations))
    
    # Create style mapping for pretraining combinations
    style_mapping = get_model_style_mapping(pretraining_combinations)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Get all available tasks
    available_tasks = list(dataset_to_dirs.keys())
    
    for ax_idx, pretraining_combo in enumerate(pretraining_combinations):
        ax = axes[ax_idx]
        
        # Collect AUROC scores for this pretraining combination across all tasks
        task_names = []
        auroc_scores = []
        
        for task_name in available_tasks:
            task_scores = AUROC_SCORES_BY_TASK.get(task_name, {})
            # Find the model name that matches this pretraining combination
            matching_model = None
            for model_name in task_scores.keys():
                if pretraining_combo in model_name:
                    matching_model = model_name
                    break
            
            if matching_model and matching_model in task_scores:
                task_names.append(task_name)
                auroc_scores.append(task_scores[matching_model])
        
        if not task_names:
            ax.text(0.5, 0.5, f"No data for {pretraining_combo}", ha='center', va='center')
            ax.axis('off')
            continue

        # Create bar plot
        bars = ax.bar(range(len(task_names)), auroc_scores, 
                     color=get_model_color(pretraining_combo, style_mapping),
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for i, (bar, score) in enumerate(zip(bars, auroc_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize the plot
        ax.set_xticks(range(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('AUROC Score', fontsize=12)
        ax.set_title(f'{pretraining_combo}\nPerformance Across Tasks', fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.0)  # AUROC range
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.legend(fontsize=8)

    # Remove empty subplots if we have fewer than 4 combinations
    for ax_idx in range(len(pretraining_combinations), 4):
        axes[ax_idx].axis('off')

    plt.suptitle('Pretraining Strategy Performance Across Downstream Tasks', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pretraining strategy performance plot saved to {out_path}")


def create_pretraining_strategy_cosine_similarity_plot(
    dataset_to_dirs: dict,
    output_dir: str = "comprehensive_comparison",
    filename: str = "pretraining_strategy_cosine_similarity.png",
):
    """Create a plot showing cosine similarity at each layer for each pretraining strategy + dataset combination.
    Each subplot shows one pretraining combination with cosine similarity across layers for all available tasks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract unique pretraining combinations
    pretraining_combinations = set()
    for task_scores in AUROC_SCORES_BY_TASK.values():
        for model_name in task_scores.keys():
            # Extract the pretraining strategy + dataset part
            if "Supervised ImageNet" in model_name:
                pretraining_combinations.add("Supervised ImageNet")
            elif "VICREG ImageNet" in model_name:
                pretraining_combinations.add("VICREG ImageNet")
            elif "Supervised RadImageNet" in model_name:
                pretraining_combinations.add("Supervised RadImageNet")
            elif "VICREG RadImageNet" in model_name:
                pretraining_combinations.add("VICREG RadImageNet")

    pretraining_combinations = sorted(list(pretraining_combinations))
    
    # Create style mapping for pretraining combinations
    style_mapping = get_model_style_mapping(pretraining_combinations)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # Get all available tasks
    available_tasks = list(dataset_to_dirs.keys())
    
    for ax_idx, pretraining_combo in enumerate(pretraining_combinations):
        ax = axes[ax_idx]
        
        plotted_any = False
        x_ticks = None
        x_labels = None
        
        # Plot cosine similarity for each task that uses this pretraining combination
        for task_name in available_tasks:
            task_comparisons = dataset_to_dirs.get(task_name, {})
            if not task_comparisons:
                continue
                
            # Find the comparison that matches this pretraining combination
            matching_comparison = None
            for comp_name in task_comparisons.keys():
                if pretraining_combo in comp_name:
                    matching_comparison = comp_name
                    break
            
            if not matching_comparison:
                continue
                
            comp_path = task_comparisons[matching_comparison]
            
            # Load cumulative stats for cosine similarity
            stats = load_cumulative_stats(comp_path)
            if not stats or 'cosine_similarities' not in stats:
                continue
                
            # Filter to target layers
            filtered_stats = filter_stats_for_target_layers(stats)
            if not filtered_stats:
                continue
                
            layer_names = filtered_stats['layer_names']
            cosine_sims = filtered_stats['cosine_similarities']
            
            # Plot the cosine similarity line
            ax.plot(range(len(layer_names)), cosine_sims, 'o-', linewidth=2, markersize=4,
                   label=task_name, alpha=0.8)
            plotted_any = True
            
            if x_ticks is None:
                x_ticks = list(range(len(layer_names)))
                x_labels = layer_names
        
        if plotted_any and x_ticks is not None and x_labels is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(-0.2, 1.0)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title(f'{pretraining_combo}\nCosine Similarity Across Layers', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
        else:
            ax.text(0.5, 0.5, f"No cosine similarity data for {pretraining_combo}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')

    # Remove empty subplots if we have fewer than 4 combinations
    for ax_idx in range(len(pretraining_combinations), 4):
        axes[ax_idx].axis('off')

    plt.suptitle('Cosine Similarity Across Layers by Pretraining Strategy', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pretraining strategy cosine similarity plot saved to {out_path}")


def create_pretraining_strategy_channel_cosine_boxplot(
    dataset_to_dirs: dict,
    output_dir: str = "comprehensive_comparison",
    filename: str = "pretraining_strategy_channel_cosine_boxplot.png",
):
    """Create a plot showing box plots of channel cosine similarity at each layer for each pretraining strategy + dataset combination.
    Each subplot shows one pretraining combination with separate box plots for each downstream task.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract unique pretraining combinations
    pretraining_combinations = set()
    for task_scores in AUROC_SCORES_BY_TASK.values():
        for model_name in task_scores.keys():
            # Extract the pretraining strategy + dataset part
            if "Supervised ImageNet" in model_name:
                pretraining_combinations.add("Supervised ImageNet")
            elif "VICREG ImageNet" in model_name:
                pretraining_combinations.add("VICREG ImageNet")
            elif "Supervised RadImageNet" in model_name:
                pretraining_combinations.add("Supervised RadImageNet")
            elif "VICREG RadImageNet" in model_name:
                pretraining_combinations.add("VICREG RadImageNet")

    pretraining_combinations = sorted(list(pretraining_combinations))
    
    # Create style mapping for pretraining combinations
    style_mapping = get_model_style_mapping(pretraining_combinations)

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes = axes.flatten()

    # Get all available tasks
    available_tasks = list(dataset_to_dirs.keys())
    
    for ax_idx, pretraining_combo in enumerate(pretraining_combinations):
        ax = axes[ax_idx]
        
        plotted_any = False
        all_task_data = {}  # task_name -> {layer_name -> list of channel cosine similarities}
        layer_names = None
        
        # Collect channel cosine similarity data for each task that uses this pretraining combination
        for task_name in available_tasks:
            task_comparisons = dataset_to_dirs.get(task_name, {})
            if not task_comparisons:
                continue
                
            # Find the comparison that matches this pretraining combination
            matching_comparison = None
            for comp_name in task_comparisons.keys():
                if pretraining_combo in comp_name:
                    matching_comparison = comp_name
                    break
            
            if not matching_comparison:
                continue
                
            comp_path = task_comparisons[matching_comparison]
            
            # Load aggregated channel stats
            loaded = load_aggregated_channel_stats(comp_path)
            if not loaded:
                continue
                
            agg = loaded["aggregated"]
            layer_names_map = loaded["layer_names"]
            
            # Sort layer indices, then filter to target layers
            sorted_layer_indices = sorted(agg.keys())
            filtered = filter_layer_indices_for_channel_cos(layer_names_map, sorted_layer_indices)
            if not filtered:
                continue
                
            # Store layer names (should be the same for all tasks)
            if layer_names is None:
                layer_names = [layer_name for _, layer_name in filtered]
            
            # Collect channel cosine similarities for each layer for this task
            task_layer_data = {}
            for layer_idx, layer_name in filtered:
                stats = agg.get(layer_idx)
                if not stats:
                    continue
                    
                # Get channel cosine similarities (array per channel)
                chan_means = stats.get('channel_cosine_sim_mean')
                if isinstance(chan_means, list) and len(chan_means) > 0:
                    task_layer_data[layer_name] = chan_means
                    plotted_any = True
            
            if task_layer_data:
                all_task_data[task_name] = task_layer_data
        
        if plotted_any and all_task_data and layer_names:
            # Create grouped box plots
            # Each group will be a layer, and within each group we'll have boxes for each task
            n_layers = len(layer_names)
            n_tasks = len(all_task_data)
            task_names = list(all_task_data.keys())
            
            # Calculate positions for grouped box plots
            box_width = 0.8 / n_tasks  # Width of each box
            layer_positions = np.arange(n_layers)
            
            # Define colors for different tasks
            task_colors = cm.get_cmap('Set3')(np.linspace(0, 1, n_tasks))
            
            # Plot box plots for each task
            for task_idx, task_name in enumerate(task_names):
                task_data = all_task_data[task_name]
                task_positions = layer_positions + (task_idx - n_tasks/2 + 0.5) * box_width
                
                # Prepare data for this task
                layer_data = []
                positions = []
                for layer_name in layer_names:
                    if layer_name in task_data:
                        layer_data.append(task_data[layer_name])
                        positions.append(task_positions[layer_names.index(layer_name)])
                
                if layer_data:
                    # Create box plot for this task
                    box_plot = ax.boxplot(layer_data, positions=positions, widths=box_width*0.8, 
                                        patch_artist=True, labels=[''] * len(layer_data))
                    
                    # Color the boxes for this task
                    for patch in box_plot['boxes']:
                        patch.set_facecolor(task_colors[task_idx])
                        patch.set_alpha(0.7)
            
            # Customize the plot
            ax.set_xticks(layer_positions)
            ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(-0.2, 1.0)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Channel Cosine Similarity', fontsize=12)
            ax.set_title(f'{pretraining_combo}\nChannel Cosine Similarity Distribution Across Layers', 
                        fontsize=12, fontweight='bold')
            
            # Create custom legend for tasks
            legend_elements = [patches.Rectangle((0,0),1,1, facecolor=task_colors[i], alpha=0.7, label=task_name) 
                             for i, task_name in enumerate(task_names)]
            ax.legend(handles=legend_elements, fontsize=8, loc='best')
        else:
            ax.text(0.5, 0.5, f"No channel cosine similarity data for {pretraining_combo}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')

    # Remove empty subplots if we have fewer than 4 combinations
    for ax_idx in range(len(pretraining_combinations), 4):
        axes[ax_idx].axis('off')

    plt.suptitle('Channel Cosine Similarity Distribution Across Layers by Pretraining Strategy', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pretraining strategy channel cosine boxplot saved to {out_path}")


def create_comprehensive_channel_cosine_heatmap(
    dataset_to_dirs: dict,
    output_dir: str = "comprehensive_comparison",
    filename: str = "comprehensive_channel_cosine_heatmap.png",
):
    """Create a comprehensive heatmap showing channel cosine similarity for each combination of 
    pretraining strategy + pretraining dataset + downstream task.
    Y-axis: Layers, X-axis: Channel Cosine Similarity, Each subplot: One combination
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all combinations
    all_combinations = []
    for task_name, task_comparisons in dataset_to_dirs.items():
        for comp_name, comp_path in task_comparisons.items():
            # Extract pretraining strategy and dataset from comparison name
            if "Supervised ImageNet" in comp_name and "RadImageNet" not in comp_name:
                pretraining_strategy = "Supervised"
                pretraining_dataset = "ImageNet"
            elif "VICREG ImageNet" in comp_name and "RadImageNet" not in comp_name:
                pretraining_strategy = "VICREG"
                pretraining_dataset = "ImageNet"
            elif "Supervised RadImageNet" in comp_name:
                pretraining_strategy = "Supervised"
                pretraining_dataset = "RadImageNet"
            elif "VICREG RadImageNet" in comp_name:
                pretraining_strategy = "VICREG"
                pretraining_dataset = "RadImageNet"
            else:
                continue
            
            all_combinations.append({
                'task_name': task_name,
                'pretraining_strategy': pretraining_strategy,
                'pretraining_dataset': pretraining_dataset,
                'comp_name': comp_name,
                'comp_path': comp_path
            })

    if not all_combinations:
        print("No valid combinations found!")
        return

    # Sort combinations for consistent ordering
    all_combinations.sort(key=lambda x: (x['pretraining_strategy'], x['pretraining_dataset'], x['task_name']))

    # Calculate grid dimensions
    n_combinations = len(all_combinations)
    n_cols = 4  # Fixed number of columns
    n_rows = (n_combinations + n_cols - 1) // n_cols  # Ceiling division

    # Create the large figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Get layer names from first available combination
    layer_names = None
    for combo in all_combinations:
        loaded = load_aggregated_channel_stats(combo['comp_path'])
        if loaded:
            layer_names_map = loaded["layer_names"]
            sorted_layer_indices = sorted(loaded["aggregated"].keys())
            filtered = filter_layer_indices_for_channel_cos(layer_names_map, sorted_layer_indices)
            if filtered:
                layer_names = [layer_name for _, layer_name in filtered]
                break

    if not layer_names:
        print("No layer data found!")
        return

    # Create style mapping for pretraining strategies and datasets
    strategy_colors = {'Supervised': '#1f77b4', 'VICREG': '#ff7f0e'}
    dataset_colors = {'ImageNet': '#2ca02c', 'RadImageNet': '#d62728'}

    for ax_idx, combo in enumerate(all_combinations):
        ax = axes[ax_idx]
        
        # Load data for this combination
        loaded = load_aggregated_channel_stats(combo['comp_path'])
        if not loaded:
            ax.text(0.5, 0.5, f"No data for\n{combo['comp_name']}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue

        agg = loaded["aggregated"]
        layer_names_map = loaded["layer_names"]
        
        # Sort layer indices, then filter to target layers
        sorted_layer_indices = sorted(agg.keys())
        filtered = filter_layer_indices_for_channel_cos(layer_names_map, sorted_layer_indices)
        if not filtered:
            ax.text(0.5, 0.5, f"No filtered data for\n{combo['comp_name']}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue

        # Collect data for box plots
        layer_data = []
        layer_labels = []
        
        for layer_idx, layer_name in filtered:
            stats = agg.get(layer_idx)
            if not stats:
                continue
                
            # Get channel cosine similarities (array per channel)
            chan_means = stats.get('channel_cosine_sim_mean')
            if isinstance(chan_means, list) and len(chan_means) > 0:
                layer_data.append(chan_means)
                layer_labels.append(layer_name)

        if not layer_data:
            ax.text(0.5, 0.5, f"No channel data for\n{combo['comp_name']}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue

        # Create horizontal box plots (layers on y-axis, cosine similarity on x-axis)
        positions = np.arange(len(layer_labels))
        box_plot = ax.boxplot(layer_data, positions=positions, vert=False, patch_artist=True)
        
        # Color the boxes based on pretraining strategy and dataset
        strategy_color = strategy_colors[combo['pretraining_strategy']]
        dataset_color = dataset_colors[combo['pretraining_dataset']]
        
        # Create a mixed color for the boxes
        for patch in box_plot['boxes']:
            patch.set_facecolor(strategy_color)
            patch.set_alpha(0.7)
            # Add a border in the dataset color
            patch.set_edgecolor(dataset_color)
            patch.set_linewidth(2)

        # Customize the plot
        ax.set_yticks(positions)
        ax.set_yticklabels(layer_labels, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Channel Cosine Similarity', fontsize=10)
        ax.set_ylabel('Layer', fontsize=10)
        
        # Create title with strategy and dataset info
        title = f"{combo['task_name']}\n{combo['pretraining_strategy']} + {combo['pretraining_dataset']}"
        ax.set_title(title, fontsize=10, fontweight='bold')

    # Hide unused subplots
    for ax_idx in range(n_combinations, len(axes)):
        axes[ax_idx].axis('off')

    # Add overall title and legend
    plt.suptitle('Channel Cosine Similarity Distribution by Layer for Each Pretraining Strategy + Dataset + Task Combination', 
                 fontsize=16, fontweight='bold')
    
    # Create custom legend
    legend_elements = [
        patches.Rectangle((0,0),1,1, facecolor=strategy_colors['Supervised'], alpha=0.7, label='Supervised'),
        patches.Rectangle((0,0),1,1, facecolor=strategy_colors['VICREG'], alpha=0.7, label='VICREG'),
        patches.Rectangle((0,0),1,1, facecolor='white', edgecolor=dataset_colors['ImageNet'], linewidth=2, label='ImageNet'),
        patches.Rectangle((0,0),1,1, facecolor='white', edgecolor=dataset_colors['RadImageNet'], linewidth=2, label='RadImageNet')
    ]
    
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

    plt.tight_layout(rect=(0, 0, 0.95, 0.95))
    
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive channel cosine heatmap saved to {out_path}")


def create_task_specific_channel_cosine_plots(
    dataset_to_dirs: dict,
    output_dir: str = "comprehensive_comparison",
):
    """Create separate figures for each downstream task, showing all pretraining strategy + dataset combinations for that task.
    Each figure shows horizontal box plots with layers on y-axis and channel cosine similarity on x-axis.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Process each task separately
    for task_name, task_comparisons in dataset_to_dirs.items():
        if not task_comparisons:
            continue
            
        print(f"Creating channel cosine plot for {task_name}...")
        
        # Collect combinations for this task
        task_combinations = []
        for comp_name, comp_path in task_comparisons.items():
            # Extract pretraining strategy and dataset from comparison name
            if "Supervised ImageNet" in comp_name and "RadImageNet" not in comp_name:
                pretraining_strategy = "Supervised"
                pretraining_dataset = "ImageNet"
            elif "VICREG ImageNet" in comp_name and "RadImageNet" not in comp_name:
                pretraining_strategy = "VICREG"
                pretraining_dataset = "ImageNet"
            elif "Supervised RadImageNet" in comp_name:
                pretraining_strategy = "Supervised"
                pretraining_dataset = "RadImageNet"
            elif "VICREG RadImageNet" in comp_name:
                pretraining_strategy = "VICREG"
                pretraining_dataset = "RadImageNet"
            else:
                continue
            
            task_combinations.append({
                'pretraining_strategy': pretraining_strategy,
                'pretraining_dataset': pretraining_dataset,
                'comp_name': comp_name,
                'comp_path': comp_path
            })

        if not task_combinations:
            print(f"No valid combinations found for {task_name}")
            continue

        # Sort combinations for consistent ordering
        task_combinations.sort(key=lambda x: (x['pretraining_strategy'], x['pretraining_dataset']))

        # Calculate grid dimensions (2x2 for 4 combinations)
        n_combinations = len(task_combinations)
        n_cols = 2
        n_rows = 2

        # Create figure for this task
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
        axes = axes.flatten()

        # Get layer names from first available combination
        layer_names = None
        for combo in task_combinations:
            loaded = load_aggregated_channel_stats(combo['comp_path'])
            if loaded:
                layer_names_map = loaded["layer_names"]
                sorted_layer_indices = sorted(loaded["aggregated"].keys())
                filtered = filter_layer_indices_for_channel_cos(layer_names_map, sorted_layer_indices)
                if filtered:
                    layer_names = [layer_name for _, layer_name in filtered]
                    break

        if not layer_names:
            print(f"No layer data found for {task_name}")
            continue

        for ax_idx, combo in enumerate(task_combinations):
            ax = axes[ax_idx]
            
            # Load data for this combination
            loaded = load_aggregated_channel_stats(combo['comp_path'])
            if not loaded:
                ax.text(0.5, 0.5, f"No data for\n{combo['comp_name']}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue

            agg = loaded["aggregated"]
            layer_names_map = loaded["layer_names"]
            
            # Sort layer indices, then filter to target layers
            sorted_layer_indices = sorted(agg.keys())
            filtered = filter_layer_indices_for_channel_cos(layer_names_map, sorted_layer_indices)
            if not filtered:
                ax.text(0.5, 0.5, f"No filtered data for\n{combo['comp_name']}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue

            # Collect data for box plots
            layer_data = []
            layer_labels = []
            
            for layer_idx, layer_name in filtered:
                stats = agg.get(layer_idx)
                if not stats:
                    continue
                    
                # Get channel cosine similarities (array per channel)
                chan_means = stats.get('channel_cosine_sim_mean')
                if isinstance(chan_means, list) and len(chan_means) > 0:
                    layer_data.append(chan_means)
                    layer_labels.append(layer_name)

            if not layer_data:
                ax.text(0.5, 0.5, f"No channel data for\n{combo['comp_name']}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue

            # Create horizontal box plots (layers on y-axis, cosine similarity on x-axis)
            positions = np.arange(len(layer_labels))
            box_plot = ax.boxplot(layer_data, positions=positions, vert=False, patch_artist=True)

            # Customize the plot
            ax.set_yticks(positions)
            ax.set_yticklabels(layer_labels, fontsize=10)
            ax.set_xlim(0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Channel Cosine Similarity', fontsize=12)
            ax.set_ylabel('Layer', fontsize=12)
            
            # Create title with strategy and dataset info
            title = f"{combo['pretraining_strategy']} + {combo['pretraining_dataset']}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            

        # Hide unused subplots
        for ax_idx in range(n_combinations, len(axes)):
            axes[ax_idx].axis('off')

        # Add overall title and legend
        plt.suptitle(f'{task_name}: Channel Cosine Similarity Distribution by Layer', 
                     fontsize=16, fontweight='bold')
        

        plt.tight_layout(rect=(0, 0, 0.95, 0.95))
        
        # Save figure for this task
        filename = f"{task_name.lower()}_channel_cosine_boxplots.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Channel cosine boxplots for {task_name} saved to {out_path}")


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

def get_task_comparison_dirs(base_dir):
    """Dynamically parse task comparison directories from the base directory."""
    task_dirs = {}
    for task_name in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_name)
        if os.path.isdir(task_path):
            comparisons = {}
            for comp_name in os.listdir(task_path):
                comp_path = os.path.join(task_path, comp_name)
                if os.path.isdir(comp_path):
                    comparisons[comp_name] = comp_path
            if comparisons:
                task_dirs[task_name] = comparisons
    return task_dirs


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
    
    # Define the comparison directories for Messidor
    messidor_comparison_dirs = {
        "Messidor Supervised ImageNet": "./layer_comparisons/messidor_Base_Supervised_ImageNet_VS_Supervised_Imagenet",
        "Messidor VICREG ImageNet": "./layer_comparisons/messidor_Base_VICREG_ImageNet_VS_VICREG_ImageNet",
        "Messidor Supervised RadImageNet": "./layer_comparisons/messidor_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet",
        "Messidor VICREG RadImageNet": "./layer_comparisons/messidor_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet"
    }

    # Define the comparison directories for MIMIC-CXR
    mimiccxr_comparison_dirs = {
        "MIMIC-CXR Supervised ImageNet": "./layer_comparisons/mimiccxr_Base_Supervised_ImageNet_VS_Supervised_Imagenet",
        "MIMIC-CXR VICREG ImageNet": "./layer_comparisons/mimiccxr_Base_VICREG_ImageNet_VS_VICREG_ImageNet",
        "MIMIC-CXR Supervised RadImageNet": "./layer_comparisons/mimiccxr_Base_Supervised_RadImageNet_VS_Supervised_RadImageNet",
        "MIMIC-CXR VICREG RadImageNet": "./layer_comparisons/mimiccxr_Base_VICREG_RadImageNet_VS_VICREG_RadImageNet"
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
    
    # Check which Messidor directories exist
    existing_messidor_comparisons = {}
    for name, path in messidor_comparison_dirs.items():
        if os.path.exists(path):
            existing_messidor_comparisons[name] = path
            print(f"Found Messidor comparison: {name}")
        else:
            print(f"Warning: Messidor comparison directory not found: {path}")
    
    # Check which MIMIC-CXR directories exist
    existing_mimiccxr_comparisons = {}
    for name, path in mimiccxr_comparison_dirs.items():
        if os.path.exists(path):
            existing_mimiccxr_comparisons[name] = path
            print(f"Found MIMIC-CXR comparison: {name}")
        else:
            print(f"Warning: MIMIC-CXR comparison directory not found: {path}")
    
    # Create output directories
    chexpert_output_dir = "comprehensive_comparison_chexpert"
    vindrcxr_output_dir = "comprehensive_comparison_vindrcxr"
    bcn2k_output_dir = "comprehensive_comparison_bcn2k"
    messidor_output_dir = "comprehensive_comparison_messidor"
    mimiccxr_output_dir = "comprehensive_comparison_mimiccxr"
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
    
    # Process Messidor comparisons if any exist
    if existing_messidor_comparisons:
        print(f"\nCreating Messidor comprehensive comparison plots...")
        
        # 1. Main comprehensive comparison
        create_comprehensive_comparison_plot(existing_messidor_comparisons, messidor_output_dir)
        
        # 2. Cumulative comparison
        create_cumulative_comparison_plot(existing_messidor_comparisons, messidor_output_dir)
        
        # 2b. Layer ratio comparison
        create_layer_ratio_comparison_plot(existing_messidor_comparisons, messidor_output_dir)
        
        # 3. Strategy comparison (Supervised vs VICREG)
        create_strategy_comparison_plot(existing_messidor_comparisons, messidor_output_dir)
        
        # 4. Pretrained dataset comparison (ImageNet vs RadImageNet)
        create_pretrained_dataset_comparison_plot(existing_messidor_comparisons, messidor_output_dir)
        
        print(f"\nMessidor comparison plots saved to {messidor_output_dir}/")
        print("Files created:")
        print("- comprehensive_model_comparison.png")
        print("- cumulative_comparison.png") 
        print("- layer_ratio_comparison.png")
        print("- strategy_comparison.png")
        print("- pretrained_dataset_comparison.png")
    else:
        print("No Messidor comparison directories found!")
    
    if existing_mimiccxr_comparisons:
        print(f"\nCreating MIMIC-CXR comprehensive comparison plots...")
        
        # 1. Main comprehensive comparison
        create_comprehensive_comparison_plot(existing_mimiccxr_comparisons, mimiccxr_output_dir)
        
        # 2. Cumulative comparison
        create_cumulative_comparison_plot(existing_mimiccxr_comparisons, mimiccxr_output_dir)
        
        # 2b. Layer ratio comparison
        create_layer_ratio_comparison_plot(existing_mimiccxr_comparisons, mimiccxr_output_dir)
        
        # 3. Strategy comparison (Supervised vs VICREG)
        create_strategy_comparison_plot(existing_mimiccxr_comparisons, mimiccxr_output_dir)
        
        # 4. Pretrained dataset comparison (ImageNet vs RadImageNet)
        create_pretrained_dataset_comparison_plot(existing_mimiccxr_comparisons, mimiccxr_output_dir)
        
        print(f"\nMIMIC-CXR comparison plots saved to {mimiccxr_output_dir}/")
        print("Files created:")
        print("- comprehensive_model_comparison.png")
        print("- cumulative_comparison.png") 
        print("- layer_ratio_comparison.png")
        print("- strategy_comparison.png")
        print("- pretrained_dataset_comparison.png")
    else:
        print("No MIMIC-CXR comparison directories found!")

    # New: Combined page across tasks (Chexpert, VINDR-CXR, BCN2K, Messidor)
    dataset_to_dirs = {}
    if existing_chexpert_comparisons:
        dataset_to_dirs["Chexpert"] = existing_chexpert_comparisons
    if existing_vindrcxr_comparisons:
        dataset_to_dirs["VINDR-CXR"] = existing_vindrcxr_comparisons
    if existing_mimiccxr_comparisons:
        dataset_to_dirs["MIMIC-CXR"] = existing_mimiccxr_comparisons
    if existing_bcn2k_comparisons:
        dataset_to_dirs["BCN2K"] = existing_bcn2k_comparisons
    if existing_messidor_comparisons:
        dataset_to_dirs["Messidor"] = existing_messidor_comparisons

    if dataset_to_dirs:
        print("\nCreating all-tasks channel cosine similarity page...")
       
        create_all_tasks_channel_cosine_page(dataset_to_dirs, output_dir="comprehensive_comparison", filename="all_tasks_channel_cosine.png")
        
        print("\nCreating pretraining strategy performance plot...")
        
        create_pretraining_strategy_performance_plot(dataset_to_dirs, output_dir="comprehensive_comparison", filename="pretraining_strategy_performance.png")
        
        print("\nCreating pretraining strategy cosine similarity plot...")
        
        create_pretraining_strategy_cosine_similarity_plot(dataset_to_dirs, output_dir="comprehensive_comparison", filename="pretraining_strategy_cosine_similarity.png")
        
        print("\nCreating pretraining strategy channel cosine boxplot...")
        
        create_pretraining_strategy_channel_cosine_boxplot(dataset_to_dirs, output_dir="comprehensive_comparison", filename="pretraining_strategy_channel_cosine_boxplot.png")
        
        print("\nCreating comprehensive channel cosine heatmap...")
        
        create_comprehensive_channel_cosine_heatmap(dataset_to_dirs, output_dir="comprehensive_comparison", filename="comprehensive_channel_cosine_heatmap.png")
        
        print("\nCreating task-specific channel cosine boxplots...")
        
        create_task_specific_channel_cosine_plots(dataset_to_dirs, output_dir="comprehensive_comparison")
    else:
        print("No datasets found to create all-tasks channel cosine page.")


if __name__ == "__main__":
    main() 