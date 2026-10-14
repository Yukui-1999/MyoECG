"""Align raw ECG features to the frozen short-axis semantic teacher."""
import torch
import torch.nn.functional as F

from util.losses_pheno import (
    fill_raw_phenotypes, normalize_phenotypes, phenotype_soft_targets, relational_kl_loss,
)
from util.sem_ablation import selected_dimensions, validate_stage1_weights

from .engine_common import (
    add_regression_summary, amp_context, evaluate_batches, full_precision,
    phenotype_tensors, regression_batch_metrics, train_batches,
)


def _forward_ecg(ecg_model, ecg):
    if ecg.ndim == 4:
        return torch.stack([ecg_model(ecg[:, view]) for view in range(ecg.shape[1])], dim=1).mean(dim=1)
    return ecg_model(ecg)


def compute_alignment_losses(ecg_features, teacher_semantic, phenotype_prediction,
                             phenotype_target, config):
    """The long-axis stage-1 objective, without unused cross-modal relational KL."""
    weights = validate_stage1_weights(config)
    selected = config.get("phenotype_indices")
    with full_precision(ecg_features.device):
        ecg = ecg_features.float()
        losses = {}
        # Retain a gradient path for a relational-only singleton batch, whose KL is zero.
        total = ecg.sum() * 0.0
        if weights["sem_loss_weight"] > 0:
            losses["sem_loss"] = 1 - F.cosine_similarity(ecg, teacher_semantic.float(), dim=-1).mean()
            losses["weighted_sem_loss"] = weights["sem_loss_weight"] * losses["sem_loss"]
            total = total + losses["weighted_sem_loss"]
        target = selected_dimensions(phenotype_target.float(), selected)
        if weights["pheno_loss_weight"] > 0:
            if phenotype_prediction is None:
                raise ValueError("pheno_head is required when pheno_loss_weight is positive")
            losses["reg_loss"] = F.smooth_l1_loss(selected_dimensions(phenotype_prediction.float(), selected), target)
            losses["weighted_reg_loss"] = weights["pheno_loss_weight"] * losses["reg_loss"]
            total = total + losses["weighted_reg_loss"]
        if weights["rel_ecg_weight"] > 0:
            tau_phi, tau_ecg = float(config.get("tau_phi", 1.0)), float(config.get("tau_ecg", 0.07))
            if tau_phi <= 0 or tau_ecg <= 0:
                raise ValueError("tau_phi and tau_ecg must be positive")
            target_prob = phenotype_soft_targets(target, tau_phi=tau_phi, mask_self=True)
            losses["rel_ecg_loss"] = relational_kl_loss(ecg, ecg, target_prob, temperature=tau_ecg, mask_self=True)
            losses["weighted_rel_ecg_loss"] = weights["rel_ecg_weight"] * losses["rel_ecg_loss"]
            total = total + losses["weighted_rel_ecg_loss"]
        losses["loss"] = total
    return losses


def _compute_batch(ecg_model, pheno_head, teacher, batch, device, mean, std, config, cache):
    ecg = batch["ecg"].to(device=device, dtype=torch.float32, non_blocking=True)
    cmr = batch["sa_cmr"].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = batch["slice_valid_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    raw = batch["phenotype"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = normalize_phenotypes(raw, mean, std)
    with amp_context(device, config):
        features = _forward_ecg(ecg_model, ecg)
        with torch.no_grad():
            semantic, _ = teacher(cmr, mask)
        prediction = pheno_head(features) if pheno_head is not None else None
    losses = compute_alignment_losses(features, semantic, prediction, target, config)
    filled = fill_raw_phenotypes(raw, mean)
    if prediction is not None:
        losses.update(regression_batch_metrics(prediction, target, filled, mean, std,
                                               config.get("phenotype_indices")))
    batch_cache = None
    if cache:
        batch_cache = {"ecg_out": features, "cmr_sem": semantic, "pheno_norm": target}
        if prediction is not None:
            batch_cache.update(pheno_pred=prediction, pheno_raw=filled)
    return losses, batch_cache


@torch.no_grad()
def chunked_retrieval_metrics(ecg_outputs, cmr_outputs, pheno_norm=None,
                               phenotype_indices=None, chunk_size=256, topk=5):
    """Exact row-blocked retrieval on CPU, with no dense full-cohort GPU matrix.

    Phenotype distances are evaluated only for retrieved neighbours, avoiding a
    second quadratic matrix. All cached embeddings remain on CPU.
    """
    if chunk_size < 1:
        raise ValueError("retrieval chunk_size must be positive")
    ecg = F.normalize(ecg_outputs.detach().float().cpu(), dim=-1)
    cmr = F.normalize(cmr_outputs.detach().float().cpu(), dim=-1)
    count = ecg.shape[0]
    if count == 0 or cmr.shape[0] != count:
        raise ValueError("retrieval requires equal nonempty ECG and CMR cohorts")
    phenotype = None if pheno_norm is None else selected_dimensions(pheno_norm.detach().float().cpu(), phenotype_indices)
    if phenotype is not None and phenotype.shape[0] != count:
        raise ValueError("retrieval phenotype count must match the embedding count")
    correct, top1_sum, topk_sum = 0, 0.0, 0.0
    neighbors = min(topk, count - 1)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        query = ecg[start:end]
        labels = torch.arange(start, end)
        scores = query @ cmr.t()
        correct += int((scores.argmax(dim=1) == labels).sum().item())
        if phenotype is not None and neighbors > 0:
            scores = query @ ecg.t()
            scores[torch.arange(end - start), labels] = -1e4
            nearest = scores.topk(k=neighbors, dim=1).indices
            # Same squared-distance identity and epsilon as util.losses_pheno.pairwise_l2.
            anchors = phenotype[start:end, None, :]
            retrieved = phenotype[nearest]
            distances_squared = anchors.square().sum(-1) + retrieved.square().sum(-1) - 2 * (anchors * retrieved).sum(-1)
            distances = (distances_squared.clamp_min(0) + 1e-12).sqrt()
            top1_sum += distances[:, 0].sum().item()
            topk_sum += distances.sum().item()
    metrics = {"align_acc": correct / count,
               "align_l2": torch.linalg.norm(ecg - cmr, dim=1).mean().item(),
               "align_cos": (ecg * cmr).sum(dim=1).mean().item()}
    if phenotype is not None:
        metrics.update(pheno_retrieval_top1_l2=top1_sum / count,
                       pheno_retrieval_top5_l2=topk_sum / (count * max(neighbors, 1)))
    return metrics


def train_one_epoch(ecg_model, pheno_head, teacher, data_loader, optimizer, device,
                    epoch, scaler, pheno_mean, pheno_std, config):
    config = config or {}
    ecg_model.train()
    if pheno_head is not None:
        pheno_head.train()
    teacher.eval()
    teacher.requires_grad_(False)
    mean, std = phenotype_tensors(pheno_mean, pheno_std, device)
    compute = lambda batch, cache: _compute_batch(ecg_model, pheno_head, teacher, batch, device, mean, std, config, cache)
    return train_batches(compute, data_loader, optimizer, device, epoch, scaler, config)


@torch.no_grad()
def evaluate(ecg_model, pheno_head, teacher, data_loader, device, epoch,
             pheno_mean, pheno_std, config, split="val"):
    config = config or {}
    ecg_model.eval()
    if pheno_head is not None:
        pheno_head.eval()
    teacher.eval()
    mean, std = phenotype_tensors(pheno_mean, pheno_std, device)
    compute = lambda batch, cache: _compute_batch(ecg_model, pheno_head, teacher, batch, device, mean, std, config, cache)
    stats, cache = evaluate_batches(compute, data_loader, device, epoch, config, split)
    stats.update(chunked_retrieval_metrics(torch.cat(cache["ecg_out"]), torch.cat(cache["cmr_sem"]),
                                           torch.cat(cache["pheno_norm"]), config.get("phenotype_indices"),
                                           chunk_size=int(config.get("retrieval_chunk_size", 256))))
    details = None
    if "pheno_pred" in cache:
        details = add_regression_summary(stats, cache, mean, std, config.get("phenotype_indices"))
    return stats, details


def test_evaluate(*args, **kwargs):
    kwargs["split"] = "test"
    return evaluate(*args, **kwargs)
