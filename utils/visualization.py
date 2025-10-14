from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any, Optional


def visualize_feature_maps(features1, features2, output_dir, sample_idx, layer_idx, original_image=None, layer_name=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(features1.shape) > 3:
        features1 = features1[0]
        features2 = features2[0]

    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()

    num_channels = min(16, features1.shape[0])
    num_cols = 4
    num_rows = (num_channels + num_cols - 1) // num_cols

    if original_image is not None:
        fig, axs = plt.subplots(2 * num_rows + 1, num_cols, figsize=(20, 5 + 4 * num_rows))
        for i in range(num_cols):
            if i == num_cols // 2 - 1 or i == num_cols // 2:
                img = original_image.cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                if i == num_cols // 2 - 1:
                    axs[0, i].imshow(img)
                    axs[0, i].set_title("Original Image")
            if hasattr(axs[0, i], "axis"):
                axs[0, i].axis("off")
        for i in range(num_channels):
            row = 1 + (i // num_cols)
            col = i % num_cols
            if hasattr(axs[row, col], "imshow"):
                axs[row, col].imshow(features1[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 1")
                axs[row, col].axis("off")
        for i in range(num_channels):
            row = 1 + num_rows + (i // num_cols)
            col = i % num_cols
            if hasattr(axs[row, col], "imshow"):
                axs[row, col].imshow(features2[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 2")
                axs[row, col].axis("off")
    else:
        fig, axs = plt.subplots(2 * num_rows, num_cols, figsize=(20, 4 * num_rows))
        if num_rows == 1:
            axs = axs.reshape(2, num_cols)
        for i in range(num_channels):
            row = i // num_cols
            col = i % num_cols
            if row < axs.shape[0] and col < axs.shape[1]:
                axs[row, col].imshow(features1[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 1")
                axs[row, col].axis("off")
        for i in range(num_channels):
            row = num_rows + (i // num_cols)
            col = i % num_cols
            if row < axs.shape[0] and col < axs.shape[1]:
                axs[row, col].imshow(features2[i], cmap="viridis")
                if i == 0:
                    axs[row, col].set_title("Model 2")
                axs[row, col].axis("off")

    title = f"Feature Maps - Layer {layer_idx}"
    if layer_name:
        title += f" ({layer_name})"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}_layer_{layer_idx}.png"), dpi=150)
    plt.close()


def visualize_feature_distributions(features1_all, features2_all, output_dir, layer_idx, layer_name, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    features1_concat = torch.cat(features1_all, dim=0)
    features2_concat = torch.cat(features2_all, dim=0)
    features1_np = features1_concat.cpu().flatten().numpy()
    features2_np = features2_concat.cpu().flatten().numpy()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    bins = 100
    ax1.hist(features1_np, bins=bins, alpha=0.7, color='blue', density=True)
    ax1.set_title(f"{model1_name} Feature Distribution")
    ax1.set_xlabel("Feature Value")
    ax1.set_ylabel("Density")
    ax2.hist(features2_np, bins=bins, alpha=0.7, color='orange', density=True)
    ax2.set_title(f"{model2_name} Feature Distribution")
    ax2.set_xlabel("Feature Value")
    ax3.hist(features1_np, bins=bins, alpha=0.5, color='blue', density=True, label=model1_name)
    ax3.hist(features2_np, bins=bins, alpha=0.5, color='orange', density=True, label=model2_name)
    ax3.set_title(f"Overlaid Feature Distributions")
    ax3.set_xlabel("Feature Value")
    ax3.legend()
    stats1 = f"{model1_name} - Mean: {features1_np.mean():.4f}, Std: {features1_np.std():.4f}"
    stats2 = f"{model2_name} - Mean: {features2_np.mean():.4f}, Std: {features2_np.std():.4f}"
    fig.text(0.5, 0.01, f"{stats1}\n{stats2}", ha='center', fontsize=12)
    plt.suptitle(f"Feature Distributions - Layer {layer_idx} ({layer_name})", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(os.path.join(output_dir, f"feature_distribution_layer_{layer_idx}.png"), dpi=150)
    plt.close()


def visualize_cosine_similarities(cosine_sims, layer_idx, layer_name, output_dir, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_indices = list(range(len(cosine_sims)))
    ax.plot(sample_indices, cosine_sims, 'o-', linewidth=2, markersize=8)
    mean_sim = np.mean(cosine_sims) if len(cosine_sims) else 0.0
    ax.axhline(y=mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.4f}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Per-Sample Cosine Similarity Between {model1_name} and {model2_name}\nLayer {layer_idx} ({layer_name})')
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cosine_similarity_per_sample_layer_{layer_idx}.png'), dpi=150)
    plt.close()


def visualize_layer_comparison(layer_stats, output_dir, model1_name, model2_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    layers = sorted(layer_stats.keys())
    layer_names = [layer_stats[layer]['name'] for layer in layers]
    l2_distances = [layer_stats[layer]['l2_distance'] for layer in layers]
    cosine_sims = [layer_stats[layer]['cosine_similarity'] for layer in layers]
    fig_width = max(20, len(layers) * 1.2)
    if len(layers) <= 15:
        tick_indices = layers
        tick_labels = layer_names
    else:
        step = max(1, len(layers) // 15)
        tick_indices = layers[::step]
        tick_labels = [layer_names[i] for i in range(0, len(layer_names), step)]
        if layers[0] not in tick_indices:
            tick_indices.insert(0, layers[0])
            tick_labels.insert(0, layer_names[0])
        if layers[-1] not in tick_indices:
            tick_indices.append(layers[-1])
            tick_labels.append(layer_names[-1])
    fig_l2_line, ax_l2_line = plt.subplots(figsize=(fig_width, 6))
    ax_l2_line.plot(layers, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax_l2_line.set_xlabel('Layer')
    ax_l2_line.set_ylabel('L2 Distance')
    ax_l2_line.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
    ax_l2_line.set_xticks(tick_indices)
    ax_l2_line.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax_l2_line.grid(True, alpha=0.3)
    if len(layers) <= 20:
        for i, l2 in enumerate(l2_distances):
            ax_l2_line.annotate(f'{l2:.2f}', (layers[i], l2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l2_distance_line.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_l2_line)

    fig_cos_line, ax_cos_line = plt.subplots(figsize=(fig_width, 6))
    ax_cos_line.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=8, color='green')
    ax_cos_line.set_xlabel('Layer')
    ax_cos_line.set_ylabel('Cosine Similarity')
    ax_cos_line.set_title(f'Cosine Similarity Between {model1_name} and {model2_name} by Layer')
    ax_cos_line.set_xticks(tick_indices)
    ax_cos_line.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)
    ax_cos_line.grid(True, alpha=0.3)
    if len(layers) <= 20:
        for i, sim in enumerate(cosine_sims):
            ax_cos_line.annotate(f'{sim:.2f}', (layers[i], sim), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax_cos_line.set_ylim(-0.2, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_line.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_cos_line)

    fig_l2_bar, ax_l2_bar = plt.subplots(figsize=(12, max(10, len(layers) * 0.4)))
    reversed_display_names = layer_names[::-1]
    reversed_l2_distances = l2_distances[::-1]
    y_pos = np.arange(len(reversed_display_names))
    ax_l2_bar.barh(y_pos, reversed_l2_distances, color='skyblue')
    ax_l2_bar.set_xlabel('L2 Distance')
    ax_l2_bar.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
    ax_l2_bar.set_yticks(y_pos)
    ax_l2_bar.set_yticklabels(reversed_display_names, fontsize=10)
    for i, l2 in enumerate(reversed_l2_distances):
        ax_l2_bar.text(l2 + max(reversed_l2_distances) * 0.02, i, f'{l2:.2f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'l2_distance_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_l2_bar)

    fig_cos_bar, ax_cos_bar = plt.subplots(figsize=(12, max(10, len(layers) * 0.4)))
    reversed_cosine_sims = cosine_sims[::-1]
    ax_cos_bar.barh(y_pos, reversed_cosine_sims, color='lightgreen')
    ax_cos_bar.set_xlabel('Cosine Similarity')
    ax_cos_bar.set_title(f'Cosine Similarity Between {model1_name} and {model2_name} by Layer')
    ax_cos_bar.set_yticks(y_pos)
    ax_cos_bar.set_yticklabels(reversed_display_names, fontsize=10)
    for i, sim in enumerate(reversed_cosine_sims):
        label_pos = sim + 0.05 if sim > 0 else sim - 0.1
        ax_cos_bar.text(label_pos, i, f'{sim:.2f}', va='center', fontsize=9)
    ax_cos_bar.set_xlim(-0.2, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_cos_bar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(20, fig_width), 6))
    ax1.plot(layers, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('L2 Distance')
    ax1.set_title(f'L2 Distance Between {model1_name} and {model2_name}')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax2.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Cosine Similarity Between {model1_name} and {model2_name}')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.0)
    plt.suptitle(f'Layer-wise Comparison Between {model1_name} and {model2_name}', fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(output_dir, 'layer_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_weight_distributions(weights1, weights2, layer_name, output_dir, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    bins = 100
    ax1.hist(weights1, bins=bins, alpha=0.7, color='blue', density=True)
    ax1.set_title(f"{model1_name} Weight Distribution")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Density")
    ax2.hist(weights2, bins=bins, alpha=0.7, color='orange', density=True)
    ax2.set_title(f"{model2_name} Weight Distribution")
    ax2.set_xlabel("Weight Value")
    ax3.hist(weights1, bins=bins, alpha=0.5, color='blue', density=True, label=model1_name)
    ax3.hist(weights2, bins=bins, alpha=0.5, color='orange', density=True, label=model2_name)
    ax3.set_title(f"Overlaid Weight Distributions")
    ax3.set_xlabel("Weight Value")
    ax3.legend()
    stats1 = f"{model1_name} - Mean: {np.mean(weights1):.4f}, Std: {np.std(weights1):.4f}"
    stats2 = f"{model2_name} - Mean: {np.mean(weights2):.4f}, Std: {np.std(weights2):.4f}"
    fig.text(0.5, 0.01, f"{stats1}\n{stats2}", ha='center', fontsize=12)
    plt.suptitle(f"Weight Distributions - Layer: {layer_name}", fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(os.path.join(output_dir, "weight_distributions.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    max_points = 10000
    if len(weights1) > max_points:
        indices = np.random.choice(len(weights1), max_points, replace=False)
        sample1 = np.asarray(weights1)[indices]
        sample2 = np.asarray(weights2)[indices]
    else:
        sample1 = np.asarray(weights1)
        sample2 = np.asarray(weights2)
    ax.scatter(sample1, sample2, alpha=0.5, s=5)
    min_val = min(sample1.min(), sample2.min())
    max_val = max(sample1.max(), sample2.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    ax.set_xlabel(f"{model1_name} Weights")
    ax.set_ylabel(f"{model2_name} Weights")
    ax.set_title(f"Weight Comparison - Layer: {layer_name}")
    corr = np.corrcoef(sample1, sample2)[0, 1]
    ax.text(0.05, 0.95, f"Correlation: {corr:.4f}", transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_scatter.png"), dpi=150)
    plt.close()


def visualize_weight_comparison_across_layers(layer_names, l2_distances, output_dir, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sorted_indices = np.argsort(layer_names)
    sorted_names = [layer_names[i] for i in sorted_indices]
    sorted_l2 = [l2_distances[i] for i in sorted_indices]
    fig_l2, ax_l2 = plt.subplots(figsize=(12, max(8, len(sorted_names) * 0.5)))
    y_pos = np.arange(len(sorted_names))
    ax_l2.barh(y_pos, sorted_l2, color='skyblue')
    ax_l2.set_xlabel('L2 Distance')
    ax_l2.set_title(f'L2 Distance Between {model1_name} and {model2_name} Weights by Layer')
    ax_l2.set_yticks(y_pos)
    ax_l2.set_yticklabels(sorted_names)
    for i, l2 in enumerate(sorted_l2):
        ax_l2.text(l2 + max(sorted_l2) * 0.02, i, f'{l2:.2f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_l2_distance.png'), dpi=150)
    plt.close(fig_l2)


def visualize_channel_cosine_similarity_distributions(channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]], extractor_layer_names: Dict[int, str], output_dir: str, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    all_layer_names: List[str] = []
    all_mean_sims: List[np.ndarray] = []
    all_std_sims: List[np.ndarray] = []
    all_min_sims: List[np.ndarray] = []
    all_max_sims: List[np.ndarray] = []
    for layer_idx in sorted(channel_cosine_sims_by_layer.keys()):
        if channel_cosine_sims_by_layer[layer_idx]:
            layer_stats = channel_cosine_sims_by_layer[layer_idx]
            avg_mean = np.mean([stat['channel_cosine_sim_mean'] for stat in layer_stats], axis=0)
            avg_std = np.mean([stat['channel_cosine_sim_std'] for stat in layer_stats], axis=0)
            avg_min = np.mean([stat['channel_cosine_sim_min'] for stat in layer_stats], axis=0)
            avg_max = np.mean([stat['channel_cosine_sim_max'] for stat in layer_stats], axis=0)
            layer_name = extractor_layer_names.get(layer_idx, f"Layer_{layer_idx}")
            all_layer_names.append(layer_name)
            all_mean_sims.append(avg_mean)
            all_std_sims.append(avg_std)
            all_min_sims.append(avg_min)
            all_max_sims.append(avg_max)
    if len(all_layer_names) <= 10:
        tick_labels = all_layer_names
        tick_positions = range(len(all_layer_names))
    else:
        step = max(1, len(all_layer_names) // 10)
        tick_positions = list(range(0, len(all_layer_names), step))
        tick_labels = [all_layer_names[i] for i in tick_positions]
        if len(all_layer_names) - 1 not in tick_positions:
            tick_positions.append(len(all_layer_names) - 1)
            tick_labels.append(all_layer_names[-1])
    ax1.boxplot(all_mean_sims, labels=all_layer_names)
    ax1.set_ylim(-0.2, 1.0)
    ax1.set_title(f'Channel-wise Cosine Similarity Distribution Across Layers\n{model1_name} vs {model2_name}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cosine Similarity')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    layer_means = [np.mean(means) for means in all_mean_sims]
    layer_stds = [np.std(means) for means in all_mean_sims]
    ax2.errorbar(range(len(all_layer_names)), layer_means, yerr=layer_stds, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax2.set_ylim(-0.2, 1.0)
    ax2.set_title(f'Average Channel Cosine Similarity per Layer\n{model1_name} vs {model2_name}')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average Cosine Similarity')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    layer_mins = [np.min(mins) for mins in all_min_sims]
    layer_maxs = [np.max(maxs) for maxs in all_max_sims]
    ax3.fill_between(range(len(all_layer_names)), layer_mins, layer_maxs, alpha=0.5, color='green', label='Min-Max Range')
    ax3.set_ylim(-0.2, 1.0)
    ax3.plot(range(len(all_layer_names)), layer_means, 'o-', linewidth=2, markersize=6, color='red', label='Mean')
    ax3.set_title(f'Channel Cosine Similarity Range Across Layers\n{model1_name} vs {model2_name}')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    all_similarities = []
    for means in all_mean_sims:
        all_similarities.extend(means)
    ax4.hist(all_similarities, bins=50, alpha=0.7, color='purple', density=True)
    ax4.set_xlim(-0.2, 1.0)
    ax4.set_title(f'Distribution of Channel Cosine Similarities\n{model1_name} vs {model2_name}')
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    mean_sim = np.mean(all_similarities) if len(all_similarities) else 0.0
    ax4.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    ax4.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_cosine_similarity_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_channel_cosine_similarity_batch_stats(channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]], extractor_layer_names: Dict[int, str], output_dir: str, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for layer_idx in sorted(channel_cosine_sims_by_layer.keys()):
        if not channel_cosine_sims_by_layer[layer_idx]:
            continue
        layer_stats = channel_cosine_sims_by_layer[layer_idx]
        layer_name = extractor_layer_names.get(layer_idx, f"Layer_{layer_idx}")
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        batch_means = [stat['channel_cosine_sim_mean'] for stat in layer_stats]
        batch_stds = [stat['channel_cosine_sim_std'] for stat in layer_stats]
        batch_mins = [stat['channel_cosine_sim_min'] for stat in layer_stats]
        batch_maxs = [stat['channel_cosine_sim_max'] for stat in layer_stats]
        batch_means = np.array(batch_means)
        batch_stds = np.array(batch_stds)
        batch_mins = np.array(batch_mins)
        batch_maxs = np.array(batch_maxs)
        channel_means = np.mean(batch_means, axis=0)
        channel_stds = np.std(batch_means, axis=0)
        channels = range(len(channel_means))
        ax1.errorbar(channels, channel_means, yerr=channel_stds, marker='o', capsize=3, capthick=1, linewidth=1, markersize=4)
        ax1.set_ylim(-0.2, 1.0)
        ax1.set_title(f'Channel-wise Mean Cosine Similarity ({layer_name})\n{model1_name} vs {model2_name}')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Cosine Similarity')
        ax1.grid(True, alpha=0.3)
        channel_std_means = np.mean(batch_stds, axis=0)
        channel_std_stds = np.std(batch_stds, axis=0)
        ax2.errorbar(channels, channel_std_means, yerr=channel_std_stds, marker='s', capsize=3, capthick=1, linewidth=1, markersize=4, color='orange')
        ax2.set_title(f'Channel-wise Standard Deviation ({layer_name})\n{model1_name} vs {model2_name}')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        channel_mins = np.mean(batch_mins, axis=0)
        channel_maxs = np.mean(batch_maxs, axis=0)
        ax3.fill_between(channels, channel_mins, channel_maxs, alpha=0.5, color='lightblue', label='Min-Max Range')
        ax3.set_ylim(-0.2, 1.0)
        ax3.plot(channels, channel_means, 'o-', linewidth=1, markersize=4, color='blue', label='Mean')
        ax3.set_title(f'Channel-wise Range ({layer_name})\n{model1_name} vs {model2_name}')
        ax3.set_xlabel('Channel Index')
        ax3.set_ylabel('Cosine Similarity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        all_similarities = batch_means.flatten()
        ax4.hist(all_similarities, bins=30, alpha=0.7, color='green', density=True)
        ax4.set_xlim(-0.2, 1.0)
        ax4.set_title(f'Distribution of Cosine Similarities ({layer_name})\n{model1_name} vs {model2_name}')
        ax4.set_xlabel('Cosine Similarity')
        ax4.set_ylabel('Density')
        ax4.grid(True, alpha=0.3)
        mean_sim = np.mean(all_similarities) if all_similarities.size > 0 else 0.0
        ax4.axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        ax4.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'channel_cosine_similarity_{safe_layer_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def visualize_channel_cosine_similarity_comparison(channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]], extractor_layer_names: Dict[int, str], output_dir: str, model1_name="Model 1", model2_name="Model 2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    layer_summaries: List[Dict[str, Any]] = []
    for layer_idx in sorted(channel_cosine_sims_by_layer.keys()):
        if channel_cosine_sims_by_layer[layer_idx]:
            layer_stats = channel_cosine_sims_by_layer[layer_idx]
            all_similarities: List[float] = []
            for stat in layer_stats:
                all_similarities.extend(stat['channel_cosine_sim_mean'])
            layer_name = extractor_layer_names.get(layer_idx, f"Layer_{layer_idx}")
            layer_summaries.append({
                'layer_idx': layer_idx,
                'layer_name': layer_name,
                'mean': float(np.mean(all_similarities)) if len(all_similarities) else 0.0,
                'std': float(np.std(all_similarities)) if len(all_similarities) else 0.0,
                'min': float(np.min(all_similarities)) if len(all_similarities) else 0.0,
                'max': float(np.max(all_similarities)) if len(all_similarities) else 0.0,
                'num_channels': int(len(layer_stats[0]['channel_cosine_sim_mean'])) if len(layer_stats) else 0,
            })
    if not layer_summaries:
        return
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    layers = [summary['layer_idx'] for summary in layer_summaries]
    layer_names = [summary['layer_name'] for summary in layer_summaries]
    means = [summary['mean'] for summary in layer_summaries]
    stds = [summary['std'] for summary in layer_summaries]
    mins = [summary['min'] for summary in layer_summaries]
    maxs = [summary['max'] for summary in layer_summaries]
    num_channels = [summary['num_channels'] for summary in layer_summaries]
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
    ax1.plot(layers, means, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_ylim(-0.2, 1.0)
    ax1.fill_between(layers, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.3, color='blue')
    ax1.set_title(f'Mean Channel Cosine Similarity Across Layers\n{model1_name} vs {model2_name}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Cosine Similarity')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(layers, mins, maxs, alpha=0.5, color='green', label='Min-Max Range')
    ax2.set_ylim(-0.2, 1.0)
    ax2.plot(layers, means, 'o-', linewidth=2, markersize=6, color='red', label='Mean')
    ax2.set_title(f'Channel Cosine Similarity Range Across Layers\n{model1_name} vs {model2_name}')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3.plot(layers, stds, 's-', linewidth=2, markersize=8, color='orange')
    ax3.set_title(f'Channel Cosine Similarity Standard Deviation Across Layers\n{model1_name} vs {model2_name}')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    scatter = ax4.scatter(num_channels, means, s=100, alpha=0.7, c=layers, cmap='viridis')
    ax4.set_ylim(-0.2, 1.0)
    ax4.set_title(f'Number of Channels vs Mean Similarity\n{model1_name} vs {model2_name}')
    ax4.set_xlabel('Number of Channels')
    ax4.set_ylabel('Mean Cosine Similarity')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Layer Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_cosine_similarity_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# New: Orchestrator for STATISTICS-only visualizations

def render_statistics_visualizations(
    output_dir: str,
    model1_name: str,
    model2_name: str,
    layer_names: Dict[int, str],
    average_stats_by_layer: Dict[int, Dict[str, float]],
    per_sample_cosine_sims: Dict[int, List[float]],
    channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]],
    cumulative_stats: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Layer-wise comparison from averages
    layer_comparison_stats = {}
    for layer_idx, avg_stats in average_stats_by_layer.items():
        if avg_stats:
            layer_comparison_stats[layer_idx] = {
                'name': layer_names.get(layer_idx, f'Layer_{layer_idx}'),
                'l2_distance': float(avg_stats.get('l2_distance', 0.0)),
                'cosine_similarity': float(avg_stats.get('cosine_similarity', 0.0)),
            }
    if len(layer_comparison_stats) > 1:
        visualize_layer_comparison(layer_comparison_stats, output_dir, model1_name, model2_name)

    # Per-layer per-sample cosine plots
    for layer_idx, sims in per_sample_cosine_sims.items():
        if not sims:
            continue
        layer_name = layer_names.get(layer_idx, f'Layer_{layer_idx}')
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        layer_out = os.path.join(output_dir, safe_layer_name)
        os.makedirs(layer_out, exist_ok=True)
        visualize_cosine_similarities(sims, layer_idx, layer_name, layer_out, model1_name, model2_name)

    # Channel-wise distributions, batch stats, comparisons
    if channel_cosine_sims_by_layer:
        channel_dir = os.path.join(output_dir, 'channel_cosine_similarity')
        os.makedirs(channel_dir, exist_ok=True)
        visualize_channel_cosine_similarity_distributions(channel_cosine_sims_by_layer, layer_names, channel_dir, model1_name, model2_name)
        visualize_channel_cosine_similarity_batch_stats(channel_cosine_sims_by_layer, layer_names, channel_dir, model1_name, model2_name)
        visualize_channel_cosine_similarity_comparison(channel_cosine_sims_by_layer, layer_names, channel_dir, model1_name, model2_name)

    # Cumulative changes visualization (if provided)
    if cumulative_stats:
        visualize_cumulative_changes(cumulative_stats, output_dir, model1_name, model2_name)


# New: Orchestrator for MODEL OUTPUT visualizations (feature maps)

def render_model_output_visualizations(
    output_dir: str,
    model1_name: str,
    model2_name: str,
    layer_names: Dict[int, str],
    features1_by_layer: Dict[int, List[torch.Tensor]] | None,
    features2_by_layer: Dict[int, List[torch.Tensor]] | None,
    max_samples_per_layer: int = 1,
) -> None:
    if not features1_by_layer or not features2_by_layer:
        return
    os.makedirs(output_dir, exist_ok=True)

    for layer_idx in sorted(set(features1_by_layer.keys()).intersection(set(features2_by_layer.keys()))):
        layer_name = layer_names.get(layer_idx, f'Layer_{layer_idx}')
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        layer_out = os.path.join(output_dir, safe_layer_name)
        os.makedirs(layer_out, exist_ok=True)

        # Use the first batch(es) and first sample to visualize representative feature maps
        f1_batches = features1_by_layer[layer_idx]
        f2_batches = features2_by_layer[layer_idx]
        num_batches = min(len(f1_batches), len(f2_batches))
        samples_done = 0
        for b in range(num_batches):
            if samples_done >= max_samples_per_layer:
                break
            f1 = f1_batches[b]
            f2 = f2_batches[b]
            # visualize_feature_maps expects tensors with batch dimension; it will take [0]
            visualize_feature_maps(
                f1, f2, layer_out, sample_idx=samples_done, layer_idx=layer_idx, original_image=None, layer_name=layer_name
            )
            samples_done += 1


def visualize_cumulative_changes(cumulative_stats: Dict[str, Any], output_dir: str, model1_name: str, model2_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    layer_indices = cumulative_stats.get('layer_indices', [])
    layer_names = cumulative_stats.get('layer_names', [])
    cumulative_l2 = cumulative_stats.get('cumulative_l2', [])
    l2_distances = cumulative_stats.get('l2_distances', [])
    layer_ratios = cumulative_stats.get('layer_ratios', [])

    # Cumulative L2 plot
    fig, ax = plt.subplots(figsize=(max(20, len(layer_indices) * 1.2), 6))
    ax.plot(layer_indices, cumulative_l2, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cumulative L2 Distance')
    ax.set_title(f'Cumulative Feature Differences Between {model1_name} and {model2_name}')

    # X-ticks for readability
    if len(layer_indices) > 0:
        if len(layer_indices) <= 15:
            tick_indices = layer_indices
            tick_labels = layer_names
        else:
            step = max(1, len(layer_indices) // 15)
            tick_indices = layer_indices[::step]
            tick_labels = [layer_names[i] for i in range(0, len(layer_names), step)]
            if layer_indices[0] not in tick_indices:
                tick_indices.insert(0, layer_indices[0])
                tick_labels.insert(0, layer_names[0])
            if layer_indices[-1] not in tick_indices:
                tick_indices.append(layer_indices[-1])
                tick_labels.append(layer_names[-1])
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_l2_distance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Optional: L2 per layer for context
    if len(l2_distances) > 0:
        fig2, ax2 = plt.subplots(figsize=(max(20, len(layer_indices) * 1.2), 6))
        ax2.plot(layer_indices, l2_distances, 'o-', linewidth=2, markersize=8, color='blue')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('L2 Distance')
        ax2.set_title(f'L2 Distance Between {model1_name} and {model2_name} by Layer')
        if len(layer_indices) > 0:
            ax2.set_xticks(tick_indices)
            ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'l2_distance_by_layer_from_cumulative.png'), dpi=150, bbox_inches='tight')
        plt.close(fig2)

    # Optional: Ratios between successive layers
    if len(layer_ratios) > 0:
        fig3, ax3 = plt.subplots(figsize=(max(20, len(layer_ratios) * 1.2), 6))
        ax3.plot(layer_indices[1:], layer_ratios, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Ratio of L2 Distances (Layer_n / Layer_{n-1})')
        ax3.set_title(f'Ratio Between Successive Layer Features\n{model1_name} vs {model2_name}')
        if len(layer_indices) > 0:
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_ratios.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)


def render_all_visualizations(
    output_dir: str,
    model1_name: str,
    model2_name: str,
    layer_names: Dict[int, str],
    average_stats_by_layer: Dict[int, Dict[str, float]],
    per_sample_cosine_sims: Dict[int, List[float]],
    channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]],
    features1_by_layer: Optional[Dict[int, List[torch.Tensor]]] = None,
    features2_by_layer: Optional[Dict[int, List[torch.Tensor]]] = None,
    cumulative_stats: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # First: statistics visualizations
    render_statistics_visualizations(
        output_dir=output_dir,
        model1_name=model1_name,
        model2_name=model2_name,
        layer_names=layer_names,
        average_stats_by_layer=average_stats_by_layer,
        per_sample_cosine_sims=per_sample_cosine_sims,
        channel_cosine_sims_by_layer=channel_cosine_sims_by_layer,
        cumulative_stats=cumulative_stats,
    )

    # Then: model output (feature map) visualizations if features provided
    render_model_output_visualizations(
        output_dir=output_dir,
        model1_name=model1_name,
        model2_name=model2_name,
        layer_names=layer_names,
        features1_by_layer=features1_by_layer,
        features2_by_layer=features2_by_layer,
        max_samples_per_layer=1,
    ) 