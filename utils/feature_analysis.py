from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import re
import numpy as np
import torch
import torch.nn as nn


@dataclass
class FeatureExtractionResult:
    all_stats_by_layer: Dict[int, List[Dict[str, float]]]
    per_sample_cosine_sims: Dict[int, List[float]]
    channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]]
    aggregated_channel_stats_by_layer: Dict[int, Dict[str, Any]]
    average_stats_by_layer: Dict[int, Dict[str, float]]
    target_layers: List[int]
    # Optional raw features for downstream use (can be large)
    features1_by_layer: Dict[int, List[torch.Tensor]]
    features2_by_layer: Dict[int, List[torch.Tensor]]


class FeatureExtractor:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.features: Dict[int, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_names: Dict[int, str] = {}
        self._register_hooks(model)

    def _register_hooks(self, module: nn.Module, prefix: str = "") -> None:
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            layer_id = len(self.layer_names)
            self.layer_names[layer_id] = layer_name
            self.hooks.append(layer.register_forward_hook(self._get_hook(layer_id)))
            self._register_hooks(layer, layer_name)

    def _get_hook(self, layer_id: int):
        def hook(module, input, output):  # type: ignore[no-redef]
            self.features[layer_id] = output.detach()
        return hook

    def extract_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        self.features = {}
        self.model.eval()
        with torch.no_grad():
            self.model(x)
        return self.features

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def filter_target_layers(all_layer_ids: List[int] | set[int], layer_names: Dict[int, str]) -> List[int]:
    target_layers: List[int] = []
    for layer_id in sorted(all_layer_ids):
        layer_name = layer_names[layer_id]
        if layer_name in ["conv1", "bn1", "relu", "maxpool"]:
            target_layers.append(layer_id)
            continue
        if re.match(r"^layer[123]\.\d+$", layer_name):
            target_layers.append(layer_id)
            continue
    return target_layers


def calculate_statistics(features1: torch.Tensor, features2: torch.Tensor) -> Dict[str, float]:
    mean1 = features1.mean().item()
    mean2 = features2.mean().item()
    std1 = features1.std().item()
    std2 = features2.std().item()

    features1_flat = features1.view(features1.size(0), -1)
    features2_flat = features2.view(features2.size(0), -1)

    features1_norm = features1_flat / (features1_flat.norm(dim=1, keepdim=True) + 1e-8)
    features2_norm = features2_flat / (features2_flat.norm(dim=1, keepdim=True) + 1e-8)

    # Note: use abs to avoid div-by-zero
    cov1 = std1 / (abs(mean1) + 1e-8)
    cov2 = std2 / (abs(mean2) + 1e-8)

    l2_dist = torch.norm(features1 - features2, dim=1).mean().item()
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


def calculate_per_sample_cosine_similarity(features1: torch.Tensor, features2: torch.Tensor) -> List[float]:
    batch_size = features1.size(0)
    similarities: List[float] = []
    for i in range(batch_size):
        f1 = features1[i].view(-1)
        f2 = features2[i].view(-1)
        f1_norm = f1 / (f1.norm() + 1e-8)
        f2_norm = f2 / (f2.norm() + 1e-8)
        sim = torch.dot(f1_norm, f2_norm).item()
        similarities.append(sim)
    return similarities


def calculate_channel_cosine_similarity(feature_map1: torch.Tensor, feature_map2: torch.Tensor) -> Dict[str, Any]:
    assert feature_map1.shape == feature_map2.shape, (
        f"Feature maps must have same shape: {feature_map1.shape} vs {feature_map2.shape}"
    )
    B, C, H, W = feature_map1.shape
    fm1_reshaped = feature_map1.view(B, C, -1)
    fm2_reshaped = feature_map2.view(B, C, -1)

    fm1_norm = torch.nn.functional.normalize(fm1_reshaped, p=2, dim=2)
    fm2_norm = torch.nn.functional.normalize(fm2_reshaped, p=2, dim=2)

    cosine_similarities = torch.sum(fm1_norm * fm2_norm, dim=2)  # (B, C)

    mean_cosine_sim = torch.mean(cosine_similarities, dim=0)
    std_cosine_sim = torch.std(cosine_similarities, dim=0)
    min_cosine_sim = torch.min(cosine_similarities, dim=0)[0]
    max_cosine_sim = torch.max(cosine_similarities, dim=0)[0]

    stats: Dict[str, Any] = {
        "channel_cosine_sim_mean": mean_cosine_sim.cpu().numpy(),
        "channel_cosine_sim_std": std_cosine_sim.cpu().numpy(),
        "channel_cosine_sim_min": min_cosine_sim.cpu().numpy(),
        "channel_cosine_sim_max": max_cosine_sim.cpu().numpy(),
        "channel_cosine_sim_all": cosine_similarities.cpu().numpy(),
    }

    stats["overall_cosine_sim_mean"] = float(torch.mean(cosine_similarities))
    stats["overall_cosine_sim_std"] = float(torch.std(cosine_similarities))
    stats["overall_cosine_sim_min"] = float(torch.min(cosine_similarities))
    stats["overall_cosine_sim_max"] = float(torch.max(cosine_similarities))

    return stats


def calculate_channel_cosine_similarity_batch(feature_maps1: List[torch.Tensor], feature_maps2: List[torch.Tensor]) -> Dict[str, Any]:
    all_cosine_sims: List[np.ndarray] = []
    for fm1, fm2 in zip(feature_maps1, feature_maps2):
        stats = calculate_channel_cosine_similarity(fm1, fm2)
        all_cosine_sims.append(stats["channel_cosine_sim_all"])  # (B, C)

    if len(all_cosine_sims) == 0:
        return {
            "channel_cosine_sim_mean": [],
            "channel_cosine_sim_std": [],
            "channel_cosine_sim_min": [],
            "channel_cosine_sim_max": [],
            "all_cosine_sims": [],
            "overall_cosine_sim_mean": 0.0,
            "overall_cosine_sim_std": 0.0,
            "overall_cosine_sim_min": 0.0,
            "overall_cosine_sim_max": 0.0,
        }

    all_cosine_sims_np = np.concatenate(all_cosine_sims, axis=0)  # (Total_B, C)

    aggregated_stats: Dict[str, Any] = {
        "channel_cosine_sim_mean": np.mean(all_cosine_sims_np, axis=0),
        "channel_cosine_sim_std": np.std(all_cosine_sims_np, axis=0),
        "channel_cosine_sim_min": np.min(all_cosine_sims_np, axis=0),
        "channel_cosine_sim_max": np.max(all_cosine_sims_np, axis=0),
        "all_cosine_sims": all_cosine_sims_np,
        "overall_cosine_sim_mean": float(np.mean(all_cosine_sims_np)),
        "overall_cosine_sim_std": float(np.std(all_cosine_sims_np)),
        "overall_cosine_sim_min": float(np.min(all_cosine_sims_np)),
        "overall_cosine_sim_max": float(np.max(all_cosine_sims_np)),
    }

    return aggregated_stats


def calculate_layer_statistics(params1: torch.Tensor, params2: torch.Tensor) -> Dict[str, Any]:
    l2_dist = torch.norm(params1 - params2).item()
    norm1 = torch.norm(params1)
    norm2 = torch.norm(params2)
    if norm1 > 0 and norm2 > 0:
        cosine_sim = torch.dot(params1, params2) / (norm1 * norm2)
        cosine_sim_val = float(cosine_sim)
    else:
        cosine_sim_val = 0.0

    abs_diff = torch.abs(params1 - params2)

    return {
        "l2_distance": l2_dist,
        "cosine_similarity": cosine_sim_val,
        "mean_diff": (params1.mean() - params2.mean()).item(),
        "std_diff": (params1.std() - params2.std()).item(),
        "max_diff": abs_diff.max().item(),
        "min_diff": abs_diff.min().item(),
        "median_diff": abs_diff.median().item(),
        "param_count": len(params1),
    }


def compute_weight_stats(model1: nn.Module, model2: nn.Module, batch_size: int = 1000) -> Dict[str, Any]:
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    common_params = set(state_dict1.keys()).intersection(set(state_dict2.keys()))

    layer_params: Dict[str, List[str]] = {}
    for param_name in common_params:
        layer_name = ".".join(param_name.split(".")[:-1]) if "." in param_name else param_name
        layer_params.setdefault(layer_name, []).append(param_name)

    layer_stats: Dict[str, Any] = {}
    summary_names: List[str] = []
    summary_l2: List[float] = []
    summary_cos: List[float] = []

    for layer_name, param_names in layer_params.items():
        if not param_names:
            continue

        if "bn" in layer_name.lower() or "batchnorm" in layer_name.lower():
            weight_params = [p for p in param_names if p.endswith(".weight")]
            bias_params = [p for p in param_names if p.endswith(".bias")]

            weight_stats = None
            bias_stats = None

            if weight_params:
                weight1 = torch.cat([state_dict1[p].cpu().float().flatten() for p in weight_params])
                weight2 = torch.cat([state_dict2[p].cpu().float().flatten() for p in weight_params])
                weight_stats = calculate_layer_statistics(weight1, weight2)
                summary_names.append(layer_name)
                summary_l2.append(weight_stats["l2_distance"])  # type: ignore[index]
                summary_cos.append(weight_stats["cosine_similarity"])  # type: ignore[index]

            if bias_params:
                bias1 = torch.cat([state_dict1[p].cpu().float().flatten() for p in bias_params])
                bias2 = torch.cat([state_dict2[p].cpu().float().flatten() for p in bias_params])
                bias_stats = calculate_layer_statistics(bias1, bias2)

            layer_stats[layer_name] = {"weight": weight_stats, "bias": bias_stats}
        else:
            params1: List[torch.Tensor] = []
            params2: List[torch.Tensor] = []
            for j in range(0, len(param_names), batch_size):
                batch_params = param_names[j : j + batch_size]
                for param_name in batch_params:
                    p1 = state_dict1[param_name].cpu().float().flatten()
                    p2 = state_dict2[param_name].cpu().float().flatten()
                    params1.append(p1)
                    params2.append(p2)
                torch.cuda.empty_cache()

            all_params1 = torch.cat(params1) if params1 else torch.tensor([])
            all_params2 = torch.cat(params2) if params2 else torch.tensor([])
            stats = calculate_layer_statistics(all_params1, all_params2)
            layer_stats[layer_name] = stats
            summary_names.append(layer_name)
            summary_l2.append(stats["l2_distance"])  # type: ignore[index]
            summary_cos.append(stats["cosine_similarity"])  # type: ignore[index]

            del params1, params2
            if "all_params1" in locals():
                del all_params1, all_params2
            torch.cuda.empty_cache()

    return {
        "by_layer": layer_stats,
        "summary_layer_names": summary_names,
        "summary_l2": summary_l2,
        "summary_cosine": summary_cos,
    }


def compute_feature_and_similarity_stats(
    extractor1: FeatureExtractor,
    extractor2: FeatureExtractor,
    dataloader,
    target_layers: List[int],
    num_samples: int,
    include_features: bool = False,
) -> FeatureExtractionResult:
    all_stats_by_layer: Dict[int, List[Dict[str, float]]] = {layer: [] for layer in target_layers}
    per_sample_cosine_sims: Dict[int, List[float]] = {layer: [] for layer in target_layers}
    channel_cosine_sims_by_layer: Dict[int, List[Dict[str, Any]]] = {layer: [] for layer in target_layers}

    all_features1_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}
    all_features2_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}

    processed_samples = 0

    for batch_idx, data in enumerate(dataloader):
        if processed_samples >= num_samples:
            break

        images = data["img"].cuda()

        features1 = extractor1.extract_features(images)
        features2 = extractor2.extract_features(images)

        for layer_idx in target_layers:
            if layer_idx not in features1 or layer_idx not in features2:
                continue

            if include_features:
                all_features1_by_layer[layer_idx].append(features1[layer_idx].clone())
                all_features2_by_layer[layer_idx].append(features2[layer_idx].clone())

            batch_sims = calculate_per_sample_cosine_similarity(features1[layer_idx], features2[layer_idx])
            per_sample_cosine_sims[layer_idx].extend(batch_sims)

            channel_cosine_stats = calculate_channel_cosine_similarity(features1[layer_idx], features2[layer_idx])
            channel_cosine_sims_by_layer[layer_idx].append(channel_cosine_stats)

            stats = calculate_statistics(features1[layer_idx], features2[layer_idx])
            all_stats_by_layer[layer_idx].append(stats)

        processed_samples += images.size(0)

    # Aggregate stats per layer
    average_stats_by_layer: Dict[int, Dict[str, float]] = {}
    aggregated_channel_stats_by_layer: Dict[int, Dict[str, Any]] = {}

    for layer_idx in target_layers:
        if all_stats_by_layer[layer_idx]:
            keys = list(all_stats_by_layer[layer_idx][0].keys())
            avg_stats: Dict[str, float] = {key: float(np.mean([stat[key] for stat in all_stats_by_layer[layer_idx]])) for key in keys}
            average_stats_by_layer[layer_idx] = avg_stats
        else:
            average_stats_by_layer[layer_idx] = {}

        # Prefer aggregating from per-batch channel cosine stats to avoid storing full feature maps
        if channel_cosine_sims_by_layer[layer_idx]:
            try:
                all_cosine_sims_list = [np.asarray(stat['channel_cosine_sim_all']) for stat in channel_cosine_sims_by_layer[layer_idx] if 'channel_cosine_sim_all' in stat]
                if len(all_cosine_sims_list) > 0:
                    all_cosine_sims = np.concatenate(all_cosine_sims_list, axis=0)  # (Total_B, C)
                    aggregated_channel_stats_by_layer[layer_idx] = {
                        'channel_cosine_sim_mean': np.mean(all_cosine_sims, axis=0),
                        'channel_cosine_sim_std': np.std(all_cosine_sims, axis=0),
                        'channel_cosine_sim_min': np.min(all_cosine_sims, axis=0),
                        'channel_cosine_sim_max': np.max(all_cosine_sims, axis=0),
                        'overall_cosine_sim_mean': float(np.mean(all_cosine_sims)),
                        'overall_cosine_sim_std': float(np.std(all_cosine_sims)),
                        'overall_cosine_sim_min': float(np.min(all_cosine_sims)),
                        'overall_cosine_sim_max': float(np.max(all_cosine_sims)),
                    }
                    continue
            except Exception:
                # Fallback to feature-based aggregation if available
                pass

        # Fallback: if features were requested and present, aggregate using features
        if include_features and all_features1_by_layer[layer_idx]:
            layer_features1 = [all_features1_by_layer[layer_idx][i] for i in range(len(all_features1_by_layer[layer_idx]))]
            layer_features2 = [all_features2_by_layer[layer_idx][i] for i in range(len(all_features2_by_layer[layer_idx]))]
            aggregated_channel_stats = calculate_channel_cosine_similarity_batch(layer_features1, layer_features2)
            aggregated_channel_stats_by_layer[layer_idx] = aggregated_channel_stats
        else:
            aggregated_channel_stats_by_layer[layer_idx] = {}

    return FeatureExtractionResult(
        all_stats_by_layer=all_stats_by_layer,
        per_sample_cosine_sims=per_sample_cosine_sims,
        channel_cosine_sims_by_layer=channel_cosine_sims_by_layer,
        aggregated_channel_stats_by_layer=aggregated_channel_stats_by_layer,
        average_stats_by_layer=average_stats_by_layer,
        target_layers=target_layers,
        features1_by_layer=all_features1_by_layer if include_features else {},
        features2_by_layer=all_features2_by_layer if include_features else {},
    )


def compute_cumulative_feature_changes(
    all_stats_by_layer: Dict[int, List[Dict[str, float]]],
    layer_names: Dict[int, str],
) -> Dict[str, Any]:
    sorted_layers = sorted(all_stats_by_layer.keys())

    filtered_layers: List[int] = []
    for layer_idx in sorted_layers:
        layer_name = layer_names[layer_idx]
        if not re.match(r"^layer[0-9]+$", layer_name):
            filtered_layers.append(layer_idx)

    layer_indices: List[int] = []
    layer_names_list: List[str] = []
    l2_distances: List[float] = []
    cosine_sims: List[float] = []

    for layer_idx in filtered_layers:
        if all_stats_by_layer[layer_idx]:
            layer_indices.append(layer_idx)
            layer_name = layer_names[layer_idx]
            layer_names_list.append(layer_name)
            layer_stats = all_stats_by_layer[layer_idx]
            avg_stats = {key: np.mean([stat[key] for stat in layer_stats]) for key in layer_stats[0].keys()}
            l2_distances.append(float(avg_stats["l2_distance"]))
            cosine_sims.append(float(avg_stats["cosine_similarity"]))

    cumulative_l2 = np.cumsum(l2_distances).tolist()

    layer_ratios: List[float] = []
    if len(l2_distances) > 1:
        layer_ratios = [float(l2_distances[i] / (l2_distances[i - 1] + 1e-12)) for i in range(1, len(l2_distances))]

    return {
        "layer_indices": layer_indices,
        "layer_names": layer_names_list,
        "l2_distances": l2_distances,
        "cumulative_l2": cumulative_l2,
        "cosine_similarities": cosine_sims,
        "layer_ratios": layer_ratios,
    }


def convert_numpy_to_lists(obj: Any) -> Any:
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