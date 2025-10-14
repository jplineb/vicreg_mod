from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import os
import json

from custom_datasets import DATASETS
from torchvision.models import resnet50, ResNet50_Weights
from utils.log_config import configure_logging

import torch
import torch.nn as nn

# Analysis utilities (computation only)
from utils.feature_analysis import (
    FeatureExtractor,
    filter_target_layers,
    compute_feature_and_similarity_stats,
    compute_cumulative_feature_changes,
    compute_weight_stats,
    convert_numpy_to_lists,
)

# Visualization utilities (plotting only)
from utils.visualization import (
    visualize_feature_maps,
    visualize_feature_distributions,
    visualize_cosine_similarities,
    visualize_layer_comparison,
    visualize_weight_distributions,
    visualize_weight_comparison_across_layers,
    visualize_channel_cosine_similarity_distributions,
    visualize_channel_cosine_similarity_batch_stats,
    visualize_channel_cosine_similarity_comparison,
)

logger = configure_logging()

torch.manual_seed(42)


def get_arguments():
    parser = argparse.ArgumentParser(description="Compare feature maps and weights between two models")
    parser.add_argument("--model1-path", type=Path, default=None, help="Path to the first model's weights; if omitted uses ImageNet pretrained")
    parser.add_argument("--model2-path", type=Path, required=True, help="Path to the second model's weights")
    parser.add_argument("--model1-name", type=str, default=None, help="Custom name for the first model (defaults to filename or preset)")
    parser.add_argument("--model2-name", type=str, default=None, help="Custom name for the second model (defaults to filename)")
    parser.add_argument("--task-ds", type=str, required=True, help="Dataset to use for feature extraction")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated list of layer indices to extract (default: filtered main layers)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for dataset loading")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to process")
    parser.add_argument("--output-dir", type=Path, default="./layer_comparisons", help="Directory to save outputs")

    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--skip-plots", action="store_true", help="Skip all plotting regardless of --visualize")
    parser.add_argument("--save-data", action="store_true", help="Save computed statistics to JSON files")
    parser.add_argument("--save-features", action="store_true", help="Also save raw feature tensors (large)")
    parser.add_argument("--weights-only", action="store_true", help="Only compute and save weight comparisons, skip features")

    return parser.parse_args()


def load_model(model_path: Path | None) -> nn.Module:
    """Load a ResNet50 model and its weights.
    If model_path is None, return ImageNet pretrained model.
    Special-case for known RadImageNet checkpoint file.
    """
    model = resnet50()
    if model_path is None:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return model.cuda()

    if model_path == Path("/project/dane2/wficai/BenchMD/models/pretrained/supervised/radimagenet/checkpoint-159.pth.tar"):
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model = resnet50(weights=state_dict)
        return model.cuda()

    state_dict = torch.load(model_path, map_location="cpu")

    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.fc = nn.Identity()  # type: ignore

    processed_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("0."):
            new_key = key[2:]
        elif key.startswith("module.backbone."):
            new_key = key.replace("module.backbone.", "")
        elif key.startswith("backbone."):
            new_key = key.replace("backbone.", "")
        else:
            new_key = key
        processed_state_dict[new_key] = value

    model.load_state_dict(processed_state_dict, strict=False)
    return model.cuda()


def main():
    args = get_arguments()

    # Prepare output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    features_dir = os.path.join(args.output_dir, "feature_comparisons")
    weights_dir = os.path.join(args.output_dir, "weight_comparisons")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Resolve names
    if args.model1_name:
        model1_name = args.model1_name
    else:
        model1_name = "ImageNet Pretrained" if args.model1_path is None else os.path.basename(str(args.model1_path))
    if args.model2_name:
        model2_name = args.model2_name
    else:
        model2_name = os.path.basename(str(args.model2_path))

    logger.info(f"Model 1 name: {model1_name}")
    logger.info(f"Model 2 name: {model2_name}")

    # Load models
    logger.info("Loading model 1...")
    model1 = load_model(args.model1_path)
    logger.info("Loading model 2...")
    model2 = load_model(args.model2_path)

    # Compare weights (compute only)
    logger.info("Comparing model weights (compute only)...")
    weight_stats = compute_weight_stats(model1, model2)

    # Save weight stats data
    if args.save_data:
        weight_stats_path = os.path.join(weights_dir, "weights_stats.json")
        with open(weight_stats_path, "w") as f:
            json.dump(convert_numpy_to_lists(weight_stats), f, indent=2)
    
    # Optional weight visualizations
    if args.visualize and not args.skip_plots:
        visualize_weight_comparison_across_layers(
            weight_stats["summary_layer_names"],
            weight_stats["summary_l2"],
            weights_dir,
            model1_name,
            model2_name,
        )
        # Per-layer distributions
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        for layer_name, stats in weight_stats["by_layer"].items():
            out_dir = os.path.join(weights_dir, layer_name.replace(".", "_"))
            os.makedirs(out_dir, exist_ok=True)
            # Try to reconstruct arrays for plotting when possible
            try:
                if isinstance(stats, dict) and "l2_distance" in stats:
                    # Recreate concatenated tensors for generic layers
                    params1 = []
                    params2 = []
                    for k in state_dict1.keys():
                        if k.startswith(layer_name):
                            params1.append(state_dict1[k].cpu().float().flatten())
                    for k in state_dict2.keys():
                        if k.startswith(layer_name):
                            params2.append(state_dict2[k].cpu().float().flatten())
                    if params1 and params2:
                        w1 = torch.cat(params1).numpy()
                        w2 = torch.cat(params2).numpy()
                        visualize_weight_distributions(w1, w2, layer_name, out_dir, model1_name, model2_name)
            except Exception as e:
                logger.warning(f"Skipping weight distribution plot for {layer_name}: {e}")

    if args.weights_only:
        logger.info("Weights-only flag set. Skipping feature extraction.")
        return

    # Feature extractor setup
    logger.info("Setting up feature extractors...")
    extractor1 = FeatureExtractor(model1)
    extractor2 = FeatureExtractor(model2)

    all_layer_ids = set(extractor1.layer_names.keys())

    if args.layers:
        target_layers = [int(layer) for layer in args.layers.split(',')]
        target_layers = [layer for layer in target_layers if layer in all_layer_ids]
    else:
        target_layers = filter_target_layers(all_layer_ids, extractor1.layer_names)

    logger.info(f"Extracting features from {len(target_layers)} layers")

    # Dataset
    logger.info(f"Loading dataset: {args.task_ds}")
    dataset = DATASETS[args.task_ds](batch_size=args.batch_size, num_workers=8, gpu=torch.cuda.current_device())
    test_loader = dataset.get_dataloader(split="valid")

    # Compute features and stats (no plotting)
    logger.info("Extracting features and computing statistics...")
    feat_result = compute_feature_and_similarity_stats(
        extractor1,
        extractor2,
        test_loader,
        target_layers,
        args.num_samples,
        include_features=args.save_features,
    )

    # Remove hooks
    extractor1.remove_hooks()
    extractor2.remove_hooks()

    # Save data results
    if args.save_data:
        # Save layer names mapping
        layer_names_path = os.path.join(features_dir, "layer_names.json")
        with open(layer_names_path, "w") as f:
            json.dump({int(k): v for k, v in extractor1.layer_names.items()}, f, indent=2)

        # Average stats per layer
        with open(os.path.join(features_dir, "average_stats_by_layer.json"), "w") as f:
            json.dump(convert_numpy_to_lists(feat_result.average_stats_by_layer), f, indent=2)
        # Per-sample cosine
        with open(os.path.join(features_dir, "per_sample_cosine_sims.json"), "w") as f:
            json.dump(convert_numpy_to_lists(feat_result.per_sample_cosine_sims), f, indent=2)
        # Channel-wise per-batch stats
        with open(os.path.join(features_dir, "channel_cosine_sims_by_layer.json"), "w") as f:
            json.dump(convert_numpy_to_lists(feat_result.channel_cosine_sims_by_layer), f, indent=2)
        # Aggregated channel stats
        with open(os.path.join(features_dir, "aggregated_channel_stats_by_layer.json"), "w") as f:
            json.dump(convert_numpy_to_lists(feat_result.aggregated_channel_stats_by_layer), f, indent=2)

        # Cumulative metrics
        cumulative_stats = compute_cumulative_feature_changes(
            feat_result.all_stats_by_layer, extractor1.layer_names
        )
        with open(os.path.join(features_dir, "cumulative_stats.json"), "w") as f:
            json.dump(convert_numpy_to_lists(cumulative_stats), f, indent=2)

    # Save raw features if requested
    if args.save_features:
        raw_dir = os.path.join(features_dir, "raw_features")
        os.makedirs(raw_dir, exist_ok=True)
        for layer_idx in feat_result.target_layers:
            layer_name = extractor1.layer_names.get(layer_idx, f"layer_{layer_idx}")
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            f1 = feat_result.features1_by_layer.get(layer_idx, [])
            f2 = feat_result.features2_by_layer.get(layer_idx, [])
            if f1:
                torch.save(f1, os.path.join(raw_dir, f"{safe_layer_name}_model1.pt"))
            if f2:
                torch.save(f2, os.path.join(raw_dir, f"{safe_layer_name}_model2.pt"))

    # Optional plots
    if args.visualize and not args.skip_plots:
        # For each layer, create distributions and per-sample plots
        for layer_idx in feat_result.target_layers:
            if layer_idx not in feat_result.all_stats_by_layer or not feat_result.all_stats_by_layer[layer_idx]:
                continue
            layer_name = extractor1.layer_names[layer_idx]
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            layer_output_dir = os.path.join(features_dir, f"{safe_layer_name}")
            os.makedirs(layer_output_dir, exist_ok=True)

            # Distributions if features stored
            if args.save_features:
                f1_list = feat_result.features1_by_layer.get(layer_idx, [])
                f2_list = feat_result.features2_by_layer.get(layer_idx, [])
                if f1_list and f2_list:
                    visualize_feature_distributions(
                        f1_list, f2_list, layer_output_dir, layer_idx, layer_name, model1_name, model2_name
                    )

            # Per-sample cosine similarities
            sims = feat_result.per_sample_cosine_sims.get(layer_idx, [])
            if sims:
                visualize_cosine_similarities(sims, layer_idx, layer_name, layer_output_dir, model1_name, model2_name)

        # Layer-wise comparison from averages
        layer_comparison_stats = {}
        for layer_idx in feat_result.target_layers:
            if feat_result.average_stats_by_layer.get(layer_idx):
                avg_stats = feat_result.average_stats_by_layer[layer_idx]
                layer_comparison_stats[layer_idx] = {
                    "name": extractor1.layer_names[layer_idx],
                    "l2_distance": float(avg_stats.get("l2_distance", 0.0)),
                    "cosine_similarity": float(avg_stats.get("cosine_similarity", 0.0)),
                }
        if len(layer_comparison_stats) > 1:
            visualize_layer_comparison(layer_comparison_stats, features_dir, model1_name, model2_name)

        # Channel-wise cosine similarity visualizations
        if feat_result.channel_cosine_sims_by_layer:
            channel_dir = os.path.join(features_dir, "channel_cosine_similarity")
            os.makedirs(channel_dir, exist_ok=True)
            visualize_channel_cosine_similarity_distributions(
                feat_result.channel_cosine_sims_by_layer, extractor1.layer_names, channel_dir, model1_name, model2_name
            )
            visualize_channel_cosine_similarity_batch_stats(
                feat_result.channel_cosine_sims_by_layer, extractor1.layer_names, channel_dir, model1_name, model2_name
            )
            visualize_channel_cosine_similarity_comparison(
                feat_result.channel_cosine_sims_by_layer, extractor1.layer_names, channel_dir, model1_name, model2_name
            )

    logger.info("Processing complete")


if __name__ == "__main__":
    main()
