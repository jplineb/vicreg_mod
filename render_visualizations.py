from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

from utils.visualization import (
    render_all_visualizations,
    render_statistics_visualizations,
    render_model_output_visualizations,
)


def get_args():
    p = argparse.ArgumentParser(description="Render visualizations from saved feature/weight data")
    p.add_argument("--input-dir", type=Path, required=True, help="Path to experiment output dir containing feature_comparisons/")
    p.add_argument("--model1-name", type=str, default="Model 1")
    p.add_argument("--model2-name", type=str, default="Model 2")
    # Selection flags
    p.add_argument("--statistics", action="store_true", help="Render only statistics visualizations")
    p.add_argument("--model-outputs", action="store_true", help="Render only model output (feature map) visualizations")
    # Raw features
    p.add_argument("--use-features", action="store_true", help="Load raw feature tensors if available (needed for model outputs)")
    p.add_argument("--max-samples-per-layer", type=int, default=1, help="Max sample batches per layer to visualize for model outputs")
    p.add_argument(
        "--output-subdir",
        type=str,
        default="feature_comparisons",
        help="Subdirectory with saved JSONs (default: feature_comparisons)",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def main():
    args = get_args()
    base_dir = Path(args.input_dir)
    features_dir = base_dir / args.output_subdir
    if not features_dir.exists():
        raise FileNotFoundError(f"Could not find {features_dir}")

    # Load JSON data
    avg_stats_path = features_dir / "average_stats_by_layer.json"
    per_sample_path = features_dir / "per_sample_cosine_sims.json"
    channel_sims_path = features_dir / "channel_cosine_sims_by_layer.json"
    layer_names_path = features_dir / "layer_names.json"
    cumulative_stats_path = features_dir / "cumulative_stats.json"

    if not (avg_stats_path.exists() and per_sample_path.exists() and channel_sims_path.exists() and layer_names_path.exists()):
        missing = [str(p) for p in [avg_stats_path, per_sample_path, channel_sims_path, layer_names_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required files: {missing}")

    average_stats_by_layer: Dict[int, Dict[str, float]] = {int(k): v for k, v in load_json(avg_stats_path).items()}
    per_sample_cosine_sims: Dict[int, List[float]] = {int(k): v for k, v in load_json(per_sample_path).items()}
    channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]] = {int(k): v for k, v in load_json(channel_sims_path).items()}
    layer_names: Dict[int, str] = {int(k): v for k, v in load_json(layer_names_path).items()}
    cumulative_stats: Optional[Dict[str, Any]] = load_json(cumulative_stats_path) if cumulative_stats_path.exists() else None

    # Determine which visualizations to render
    render_stats_only = args.statistics and not args.model_outputs
    render_outputs_only = args.model_outputs and not args.statistics
    render_both = not args.statistics and not args.model_outputs

    # Optionally load raw features (only if needed)
    features1_by_layer: Optional[Dict[int, List[torch.Tensor]]] = None
    features2_by_layer: Optional[Dict[int, List[torch.Tensor]]] = None

    needs_features = render_outputs_only or render_both
    if needs_features and args.use_features:
        raw_dir = features_dir / "raw_features"
        if raw_dir.exists():
            features1_by_layer = {}
            features2_by_layer = {}
            # Load all .pt files if present
            for pt_path in raw_dir.glob("*_model1.pt"):
                key_name = pt_path.name.replace("_model1.pt", "")
                layer_idx = next((idx for idx, name in layer_names.items() if name.replace('.', '_').replace('/', '_') == key_name), None)
                if layer_idx is not None:
                    features1_by_layer[layer_idx] = torch.load(pt_path)
            for pt_path in raw_dir.glob("*_model2.pt"):
                key_name = pt_path.name.replace("_model2.pt", "")
                layer_idx = next((idx for idx, name in layer_names.items() if name.replace('.', '_').replace('/', '_') == key_name), None)
                if layer_idx is not None:
                    features2_by_layer[layer_idx] = torch.load(pt_path)
        else:
            print(f"Raw features dir not found: {raw_dir}")

    # Route based on selection
    if render_stats_only:
        render_statistics_visualizations(
            output_dir=str(features_dir),
            model1_name=args.model1_name,
            model2_name=args.model2_name,
            layer_names=layer_names,
            average_stats_by_layer=average_stats_by_layer,
            per_sample_cosine_sims=per_sample_cosine_sims,
            channel_cosine_sims_by_layer=channel_cosine_sims_by_layer,
            cumulative_stats=cumulative_stats,
        )
        return

    if render_outputs_only:
        render_model_output_visualizations(
            output_dir=str(features_dir),
            model1_name=args.model1_name,
            model2_name=args.model2_name,
            layer_names=layer_names,
            features1_by_layer=features1_by_layer,
            features2_by_layer=features2_by_layer,
            max_samples_per_layer=args.max_samples_per_layer,
        )
        return

    # Default: render both
    render_all_visualizations(
        output_dir=str(features_dir),
        model1_name=args.model1_name,
        model2_name=args.model2_name,
        layer_names=layer_names,
        average_stats_by_layer=average_stats_by_layer,
        per_sample_cosine_sims=per_sample_cosine_sims,
        channel_cosine_sims_by_layer=channel_cosine_sims_by_layer,
        features1_by_layer=features1_by_layer,
        features2_by_layer=features2_by_layer,
        cumulative_stats=cumulative_stats,
    )


if __name__ == "__main__":
    main() 