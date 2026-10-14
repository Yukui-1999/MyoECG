from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PHENOTYPE_COLS = [
    "LVEDV", "LVESV", "LVSV", "LVEF", "LVCO", "LVM",
    "RVEDV", "RVESV", "RVSV", "RVEF",
    "WT_AHA_1", "WT_AHA_2", "WT_AHA_3", "WT_AHA_4",
    "WT_AHA_5", "WT_AHA_6", "WT_AHA_7", "WT_AHA_8",
    "WT_AHA_9", "WT_AHA_10", "WT_AHA_11", "WT_AHA_12",
    "WT_AHA_13", "WT_AHA_14", "WT_AHA_15", "WT_AHA_16",
    "WT_Global",
]

KEY_PHENOTYPE_COLS = [
    "LVEDV", "LVESV", "LVSV", "LVEF", "LVM",
    "RVEDV", "RVESV", "RVSV", "RVEF", "WT_Global",
]

RANK_PHENOTYPE_COLS = ["LVM", "LVEF", "LVEDV", "RVEDV", "RVEF", "WT_Global"]
RANK_PHENOTYPE_INDICES = [PHENOTYPE_COLS.index(name) for name in RANK_PHENOTYPE_COLS]


def normalize_phenotypes(
    phenotypes: torch.Tensor,
    pheno_mean: torch.Tensor,
    pheno_std: torch.Tensor,
) -> torch.Tensor:
    phenotypes = phenotypes.float()
    normalized = (phenotypes - pheno_mean) / pheno_std.clamp_min(1e-6)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def fill_raw_phenotypes(phenotypes: torch.Tensor, pheno_mean: torch.Tensor) -> torch.Tensor:
    phenotypes = phenotypes.float()
    return torch.where(torch.isfinite(phenotypes), phenotypes, pheno_mean)


def pairwise_l2(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    return torch.sqrt(dist2.clamp_min(0.0) + 1e-12)


def phenotype_soft_targets(
    pheno_norm: torch.Tensor,
    tau_phi: float = 1.0,
    mask_self: bool = False,
) -> torch.Tensor:
    batch_size = pheno_norm.size(0)
    if mask_self and batch_size <= 1:
        return torch.zeros((batch_size, batch_size), device=pheno_norm.device, dtype=pheno_norm.dtype)

    logits = -pairwise_l2(pheno_norm) / tau_phi
    if mask_self:
        eye = torch.eye(batch_size, device=pheno_norm.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, -1e4)
    return F.softmax(logits, dim=-1)


def cosine_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float,
    mask_self: bool = False,
) -> torch.Tensor:
    query = F.normalize(query.float(), dim=-1)
    key = F.normalize(key.float(), dim=-1)
    logits = query @ key.t()
    logits = logits / temperature
    if mask_self:
        batch_size = logits.size(0)
        if batch_size <= 1:
            return logits
        eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, -1e4)
    return logits


def relational_kl_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    target_prob: torch.Tensor,
    temperature: float,
    mask_self: bool = False,
) -> torch.Tensor:
    if mask_self and query.size(0) <= 1:
        return query.new_zeros(())
    logits = cosine_logits(query, key, temperature=temperature, mask_self=mask_self)
    log_prob = F.log_softmax(logits, dim=-1)
    target_prob = target_prob.detach().float()
    return F.kl_div(log_prob, target_prob, reduction="batchmean")


def phenotype_relational_losses(
    ecg_embed: torch.Tensor,
    cmr_embed: torch.Tensor,
    pheno_norm: torch.Tensor,
    tau_phi: float = 1.0,
    tau_cross: float = 0.07,
    tau_ecg: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    target_cross = phenotype_soft_targets(pheno_norm, tau_phi=tau_phi, mask_self=False)
    target_ecg = phenotype_soft_targets(pheno_norm, tau_phi=tau_phi, mask_self=True)

    rel_cross = relational_kl_loss(
        ecg_embed,
        cmr_embed,
        target_cross,
        temperature=tau_cross,
        mask_self=False,
    )
    rel_ecg = relational_kl_loss(
        ecg_embed,
        ecg_embed,
        target_ecg,
        temperature=tau_ecg,
        mask_self=True,
    )
    metrics = {
        "rel_cross_loss": float(rel_cross.detach().item()),
        "rel_ecg_loss": float(rel_ecg.detach().item()),
    }
    return rel_cross, rel_ecg, metrics


def safe_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.nan_to_num(pred.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pred = pred - pred.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    numerator = (pred * target).sum(axis=0)
    denominator = np.sqrt((pred ** 2).sum(axis=0) * (target ** 2).sum(axis=0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def phenotype_regression_metrics(
    pred_norm: np.ndarray,
    target_raw: np.ndarray,
    pheno_mean: torch.Tensor,
    pheno_std: torch.Tensor,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    mean_np = pheno_mean.detach().cpu().numpy()
    std_np = pheno_std.detach().cpu().numpy()
    pred_norm = np.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)
    target_raw = np.where(np.isfinite(target_raw), target_raw, mean_np)
    pred_raw = pred_norm * std_np + mean_np
    target_norm = (target_raw - mean_np) / np.maximum(std_np, 1e-6)

    mae_raw = np.mean(np.abs(pred_raw - target_raw), axis=0)
    mae_norm = np.mean(np.abs(pred_norm - target_norm), axis=0)
    pearson = safe_pearson(pred_raw, target_raw)

    key_indices = [PHENOTYPE_COLS.index(name) for name in KEY_PHENOTYPE_COLS]
    summary = {
        "pheno_mae_raw_mean": float(np.mean(mae_raw)),
        "pheno_mae_norm_mean": float(np.mean(mae_norm)),
        "pheno_pearson_mean": float(np.mean(pearson)),
        "pheno_key_pearson": float(np.mean(pearson[key_indices])),
    }

    details = []
    for idx, name in enumerate(PHENOTYPE_COLS):
        details.append({
            "phenotype": name,
            "mae_raw": float(mae_raw[idx]),
            "mae_norm": float(mae_norm[idx]),
            "pearson": float(pearson[idx]),
        })
    return summary, details


def alignment_retrieval_metrics(ecg_outputs: torch.Tensor, cmr_outputs: torch.Tensor) -> Dict[str, float]:
    ecg_embed = F.normalize(ecg_outputs.float(), dim=-1)
    cmr_embed = F.normalize(cmr_outputs.float(), dim=-1)
    logits = ecg_embed @ cmr_embed.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    l2 = torch.linalg.norm(ecg_embed - cmr_embed, dim=1).mean().item()
    cos = (ecg_embed * cmr_embed).sum(dim=1).mean().item()
    return {"align_acc": acc, "align_l2": l2, "align_cos": cos}


def phenotype_embedding_retrieval_metrics(
    ecg_outputs: torch.Tensor,
    pheno_norm: torch.Tensor,
    topk: int = 5,
) -> Dict[str, float]:
    batch_size = ecg_outputs.size(0)
    if batch_size <= 1:
        return {
            "pheno_retrieval_top1_l2": 0.0,
            "pheno_retrieval_top5_l2": 0.0,
        }

    topk = min(topk, batch_size - 1)
    ecg_embed = F.normalize(ecg_outputs.float(), dim=-1)
    sim = ecg_embed @ ecg_embed.t()
    eye = torch.eye(batch_size, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    neighbor_idx = sim.topk(k=topk, dim=-1).indices

    pheno_dist = pairwise_l2(pheno_norm.float())
    gathered = pheno_dist.gather(dim=1, index=neighbor_idx)
    return {
        "pheno_retrieval_top1_l2": float(gathered[:, 0].mean().item()),
        "pheno_retrieval_top5_l2": float(gathered.mean().item()),
    }


def phenotype_hard_negative_rank_loss(
    ecg_embed: torch.Tensor,
    cmr_embed: torch.Tensor,
    pheno_norm: torch.Tensor,
    age: torch.Tensor,
    sex: torch.Tensor,
    margin: float = 0.2,
    top_fraction: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch_size = ecg_embed.size(0)
    if batch_size <= 1:
        zero = ecg_embed.new_zeros(())
        return zero, {"rank_loss": 0.0, "rank_valid_frac": 0.0}

    ecg_norm = F.normalize(ecg_embed.float(), dim=-1)
    cmr_norm = F.normalize(cmr_embed.float(), dim=-1)
    sim = ecg_norm @ cmr_norm.t()
    pos = sim.diag()

    age = age.float().view(-1)
    sex = sex.view(-1)
    same_sex = sex[:, None] == sex[None, :]
    close_age = torch.abs(age[:, None] - age[None, :]) <= 5.0
    not_self = ~torch.eye(batch_size, device=ecg_embed.device, dtype=torch.bool)
    valid = same_sex & close_age & not_self

    key_phi = pheno_norm[:, RANK_PHENOTYPE_INDICES]
    pheno_dist = pairwise_l2(key_phi)
    dist_masked = pheno_dist.masked_fill(~valid, -1.0)

    valid_count = valid.sum(dim=1)
    max_k = max(1, int(round((batch_size - 1) * top_fraction)))
    neg_indices = []
    anchor_indices = []
    for anchor_idx in range(batch_size):
        if valid_count[anchor_idx].item() == 0:
            continue
        k = min(max_k, int(valid_count[anchor_idx].item()))
        top_idx = dist_masked[anchor_idx].topk(k=k, largest=True).indices
        # Among phenotype-different candidates, use the hardest CMR similarity.
        hardest_local = sim[anchor_idx, top_idx].argmax()
        neg_indices.append(top_idx[hardest_local])
        anchor_indices.append(anchor_idx)

    if not anchor_indices:
        zero = ecg_embed.new_zeros(())
        return zero, {"rank_loss": 0.0, "rank_valid_frac": 0.0}

    anchors = torch.tensor(anchor_indices, device=ecg_embed.device, dtype=torch.long)
    negatives = torch.stack(neg_indices).long()
    neg = sim[anchors, negatives]
    loss = F.relu(neg - pos[anchors] + margin).mean()
    return loss, {
        "rank_loss": float(loss.detach().item()),
        "rank_valid_frac": float(len(anchor_indices) / batch_size),
    }
