from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from util.losses_pheno import PHENOTYPE_COLS, phenotype_regression_metrics, safe_pearson


PHENOTYPE_GROUPS = OrderedDict([
    ("structure", ("LVEDV", "LVESV", "LVM", "RVEDV", "RVESV")),
    ("function", ("LVSV", "LVEF", "LVCO", "RVSV", "RVEF")),
    ("wall_thickness", tuple(["WT_AHA_%d" % i for i in range(1, 17)] + ["WT_Global"])),
])


def _parse_group_names(groups: str) -> List[str]:
    if not isinstance(groups, str) or not groups.strip():
        raise ValueError("phenotype groups must be a non-empty comma-separated string")
    requested = [item.strip() for item in groups.split(",")]
    if any(not item for item in requested):
        raise ValueError("phenotype groups cannot contain empty names")
    if "all" in requested:
        if requested != ["all"]:
            raise ValueError("'all' cannot be combined with named phenotype groups")
        return list(PHENOTYPE_GROUPS)
    unknown = sorted(set(requested) - set(PHENOTYPE_GROUPS))
    if unknown:
        raise ValueError("unknown phenotype group(s): %s" % ", ".join(unknown))
    return [name for name in PHENOTYPE_GROUPS if name in requested]


def resolve_phenotype_indices(groups: str) -> List[int]:
    """Resolve named phenotype groups to indices in canonical PHENOTYPE_COLS order."""
    selected_groups = _parse_group_names(groups)
    selected_names = {
        phenotype
        for group_name in selected_groups
        for phenotype in PHENOTYPE_GROUPS[group_name]
    }
    return [index for index, name in enumerate(PHENOTYPE_COLS) if name in selected_names]


def phenotype_selection_metadata(groups: str) -> Dict[str, object]:
    selected_groups = _parse_group_names(groups)
    indices = resolve_phenotype_indices(groups)
    all_selected = len(indices) == len(PHENOTYPE_COLS)
    return {
        "phenotype_groups": list(PHENOTYPE_GROUPS) if all_selected else selected_groups,
        "phenotype_group_spec": "all" if all_selected else ",".join(selected_groups),
        "phenotype_group_sizes": {
            name: len(PHENOTYPE_GROUPS[name]) for name in selected_groups
        },
        "phenotype_indices": indices,
        "phenotype_names": [PHENOTYPE_COLS[index] for index in indices],
        "phenotype_total_count": len(PHENOTYPE_COLS),
        "phenotype_selected_count": len(indices),
    }


def configure_phenotype_selection(args):
    metadata = phenotype_selection_metadata(args.phenotype_groups)
    args.phenotype_groups = metadata["phenotype_group_spec"]
    args.phenotype_indices = metadata["phenotype_indices"]
    args.phenotype_names = metadata["phenotype_names"]
    args.phenotype_group_metadata = metadata
    return metadata


def write_run_config(output_dir: str, args, filename: str = "run_config.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload = dict(vars(args)) if hasattr(args, "__dict__") else dict(args)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError("%r is not JSON serializable" % (value,))


def selected_dimensions(value: torch.Tensor, indices: Optional[Sequence[int]]) -> torch.Tensor:
    if indices is None:
        return value
    return value[..., list(indices)]


def selected_phenotype_metrics(
    pred_norm: np.ndarray,
    target_raw: np.ndarray,
    pheno_mean: torch.Tensor,
    pheno_std: torch.Tensor,
    indices: Optional[Sequence[int]],
) -> Dict[str, float]:
    selected = list(range(len(PHENOTYPE_COLS))) if indices is None else list(indices)
    if selected == list(range(len(PHENOTYPE_COLS))):
        legacy, _ = phenotype_regression_metrics(
            pred_norm, target_raw, pheno_mean, pheno_std
        )
        return {
            "pheno_selected_mae_raw_mean": legacy["pheno_mae_raw_mean"],
            "pheno_selected_mae_norm_mean": legacy["pheno_mae_norm_mean"],
            "pheno_selected_pearson_mean": legacy["pheno_pearson_mean"],
            "pheno_selected_count": float(len(selected)),
        }
    pred_norm = np.nan_to_num(np.asarray(pred_norm), nan=0.0, posinf=0.0, neginf=0.0)
    target_raw = np.asarray(target_raw)
    mean_np = pheno_mean.detach().cpu().numpy()
    std_np = pheno_std.detach().cpu().numpy()
    target_raw = np.where(np.isfinite(target_raw), target_raw, mean_np)
    pred_raw = pred_norm * std_np + mean_np
    target_norm = (target_raw - mean_np) / np.maximum(std_np, 1e-6)
    mae_raw = np.mean(np.abs(pred_raw[:, selected] - target_raw[:, selected]), axis=0)
    mae_norm = np.mean(np.abs(pred_norm[:, selected] - target_norm[:, selected]), axis=0)
    pearson = safe_pearson(pred_raw[:, selected], target_raw[:, selected])
    return {
        "pheno_selected_mae_raw_mean": float(np.mean(mae_raw)),
        "pheno_selected_mae_norm_mean": float(np.mean(mae_norm)),
        "pheno_selected_pearson_mean": float(np.mean(pearson)),
        "pheno_selected_count": float(len(selected)),
    }


def _validate_weight(config: Mapping[str, object], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("%s must be finite and nonnegative, got %r" % (key, value))
    return value


def validate_stage1_weights(config: Mapping[str, object]) -> Dict[str, float]:
    weights = {
        "sem_loss_weight": _validate_weight(config, "sem_loss_weight", 1.0),
        "pheno_loss_weight": _validate_weight(config, "pheno_loss_weight", 1.0),
        "rel_ecg_weight": _validate_weight(config, "rel_ecg_weight", 0.5),
    }
    if not any(value > 0.0 for value in weights.values()):
        raise ValueError("stage1 requires at least one active loss weight")
    return weights


def validate_stage2_weights(config: Mapping[str, object]) -> Dict[str, float]:
    sem_weight = _validate_weight(config, "sem_loss_weight", 1.0)
    if sem_weight <= 0.0:
        raise ValueError("stage2 sem_loss_weight must be positive")
    return {"sem_loss_weight": sem_weight}


def validate_teacher_target_match(
    checkpoint_or_path: Union[str, os.PathLike, Mapping[str, object]],
    groups: str,
    fold: int,
) -> None:
    if isinstance(checkpoint_or_path, Mapping):
        checkpoint = checkpoint_or_path
        source = "teacher checkpoint"
    else:
        source = os.fspath(checkpoint_or_path)
        checkpoint = torch.load(source, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("%s has no configuration metadata for fold validation" % source)
    teacher_fold = config.get("fold")
    if teacher_fold is None or int(teacher_fold) != int(fold):
        raise ValueError(
            "%s fold mismatch: teacher=%r, requested=%r" % (source, teacher_fold, fold)
        )

    requested = resolve_phenotype_indices(groups)
    teacher_indices = config.get("phenotype_indices")
    if teacher_indices is None:
        metadata = config.get("phenotype_group_metadata")
        if isinstance(metadata, Mapping):
            teacher_indices = metadata.get("phenotype_indices")
    if teacher_indices is None and "phenotype_groups" in config:
        teacher_indices = resolve_phenotype_indices(str(config["phenotype_groups"]))
    if teacher_indices is None:
        if requested != list(range(len(PHENOTYPE_COLS))):
            raise ValueError(
                "%s predates phenotype metadata and is only compatible with phenotype_groups=all" % source
            )
        return
    teacher_indices = [int(index) for index in teacher_indices]
    if teacher_indices != requested:
        raise ValueError(
            "%s phenotype target mismatch: teacher=%s, requested=%s"
            % (source, teacher_indices, requested)
        )


def should_save_checkpoint(checkpoint_selection: str, is_best: bool, is_final: bool) -> bool:
    if checkpoint_selection == "legacy":
        return bool(is_best)
    if checkpoint_selection == "last":
        return bool(is_final)
    raise ValueError("unknown checkpoint selection: %s" % checkpoint_selection)


def should_stop_early(patient: int, best_patience: int) -> bool:
    return best_patience >= 0 and patient > best_patience


def teacher_checkpoint_mae(stats: Mapping[str, float], indices: Sequence[int]) -> float:
    if list(indices) == list(range(len(PHENOTYPE_COLS))):
        return float(stats["pheno_mae_raw_mean"])
    return float(stats["pheno_selected_mae_raw_mean"])
