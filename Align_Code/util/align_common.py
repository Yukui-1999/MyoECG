from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml

from models import encoder
from models.lora import apply_lora_from_config
import util.misc as misc
from util.losses_pheno import PHENOTYPE_COLS


def set_seed(seed: int) -> int:
    seed = seed + misc.get_rank()
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    return seed


def build_ecg_model_from_config(ecg_config_path: str, num_classes=None, lora_config_override: Optional[dict] = None) -> torch.nn.Module:
    with open(os.path.realpath(ecg_config_path), "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["model"]["num_classes"] = num_classes
    if lora_config_override is not None:
        config["lora"] = lora_config_override
    lora_config = config.get("lora")
    lora_enabled = bool(lora_config and lora_config.get("enabled", False))
    model_name = config["model_name"]
    if model_name not in encoder.__dict__:
        raise ValueError(f"Unsupported model name: {model_name}")

    ecg_model = encoder.__dict__[model_name](**config["model"])
    if config["mode"] == "scratch":
        print("Training ECG model from scratch")
        if lora_enabled:
            apply_lora_from_config(ecg_model, lora_config)
        return ecg_model

    checkpoint_path = config.get("afterAlign_path") if config["mode"] == "align" else config.get("encoder_path")
    if checkpoint_path is None:
        raise ValueError(f"Missing checkpoint path in {ecg_config_path}")
    if lora_enabled and config["mode"] == "align":
        apply_lora_from_config(ecg_model, lora_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Load pre-trained ECG checkpoint from: {checkpoint_path}")
    checkpoint_model = checkpoint["model"]
    state_dict = ecg_model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Remove key {key} from pre-trained checkpoint")
            del checkpoint_model[key]
    msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
    print(f"Load pre-trained ECG model: {msg}")
    if lora_enabled and config["mode"] != "align":
        apply_lora_from_config(ecg_model, lora_config)
    return ecg_model


def build_pheno_head(embed_dim: int = 768, hidden_dim: int = 256, num_phenotypes: int = 27) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_phenotypes),
    )


def build_semantic_head(embed_dim: int = 768, hidden_dim: int = 768) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, embed_dim),
        nn.LayerNorm(embed_dim),
    )


def compute_phenotype_norm(train_set):
    phenotypes = np.asarray(train_set.phenotypes, dtype=np.float32)
    mean = np.nanmean(phenotypes, axis=0)
    std = np.nanstd(phenotypes, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def save_phenotype_norm(path: str, mean, std, source: str = "UKB train split") -> None:
    payload = {
        "phenotype_names": PHENOTYPE_COLS,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "source": source,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_phenotype_norm(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("phenotype_names") != PHENOTYPE_COLS:
        raise ValueError(f"Phenotype columns in {path} do not match PHENOTYPE_COLS")
    return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)


def save_ecg_checkpoint(
    config: dict,
    checkpoint_path: str,
    epoch: int,
    ecg_model: torch.nn.Module,
    optimizer=None,
    loss_scaler=None,
    metrics: Optional[dict] = None,
    extra_state: Optional[dict] = None,
) -> None:
    to_save = {
        "epoch": epoch,
        "model": ecg_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "config": config,
    }
    if metrics is not None:
        to_save["metrics"] = metrics
    if extra_state is not None:
        to_save.update(extra_state)
    misc.save_on_master(to_save, checkpoint_path)


def save_multi_module_checkpoint(
    config: dict,
    checkpoint_path: str,
    epoch: int,
    modules: Dict[str, torch.nn.Module],
    optimizer=None,
    loss_scaler=None,
    metrics: Optional[dict] = None,
) -> None:
    to_save = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "config": config,
    }
    for name, module in modules.items():
        to_save[name] = module.state_dict()
    if metrics is not None:
        to_save["metrics"] = metrics
    misc.save_on_master(to_save, checkpoint_path)


def load_module_state(module: torch.nn.Module, checkpoint: dict, key: str, strict: bool = True) -> None:
    if key not in checkpoint:
        print(f"Checkpoint has no key `{key}`, skip loading it")
        return
    msg = module.load_state_dict(checkpoint[key], strict=strict)
    print(f"Load `{key}`: {msg}")


def write_downstream_config(
    output_dir: str,
    config_name: str,
    ckpt_name: str,
    encoder_path: str = "/home/liziyu/CMRGEN/ECGCMR/st_mem_vit_base_encoder.pth",
    lora_config: Optional[dict] = None,
) -> str:
    config = {
        "encoder_path": encoder_path,
        "afterAlign_path": os.path.join(output_dir, ckpt_name),
        "model_name": "st_mem_vit_base",
        "mode": "align",
        "model": {
            "seq_len": 2250,
            "patch_size": 75,
            "num_leads": 12,
            "num_classes": 1,
            "pred_metric": False,
        },
    }
    if lora_config is not None:
        config["lora"] = lora_config
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Saved downstream ECG config to {config_path}")
    return config_path


def append_phenotype_metrics_csv(csv_path: str, split: str, epoch: int, checkpoint_name: str, details: List[dict]) -> None:
    fieldnames = ["split", "epoch", "checkpoint", "phenotype", "mae_raw", "mae_norm", "pearson"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for item in details:
            row = {"split": split, "epoch": epoch, "checkpoint": checkpoint_name}
            row.update(item)
            writer.writerow(row)


def append_metrics_csv(csv_path: str, split: str, epoch: int, checkpoint_name: str, stats: Dict[str, float]) -> None:
    base_fields = ["split", "epoch", "checkpoint"]
    fields = base_fields + sorted(k for k in stats.keys())
    file_exists = os.path.exists(csv_path)
    existing_fields = None
    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            existing_fields = header.split(",") if header else None
    fieldnames = existing_fields or fields
    row = {"split": split, "epoch": epoch, "checkpoint": checkpoint_name}
    row.update(stats)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
