"""Phenotype-supervised short-axis CMR semantic teacher engine."""
import torch
import torch.nn.functional as F

from util.losses_pheno import fill_raw_phenotypes, normalize_phenotypes
from util.sem_ablation import selected_dimensions

from .engine_common import (
    add_regression_summary, amp_context, evaluate_batches, full_precision,
    phenotype_tensors, regression_batch_metrics, train_batches,
)


def teacher_regression_loss(pred, target, phenotype_indices=None):
    with full_precision(pred.device):
        return F.smooth_l1_loss(selected_dimensions(pred.float(), phenotype_indices),
                                selected_dimensions(target.float(), phenotype_indices))


def _compute_batch(teacher, batch, device, mean, std, config, cache):
    cmr = batch["sa_cmr"].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = batch["slice_valid_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    raw = batch["phenotype"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = normalize_phenotypes(raw, mean, std)
    filled = fill_raw_phenotypes(raw, mean)
    with amp_context(device, config):
        _, prediction = teacher(cmr, mask)
    loss = teacher_regression_loss(prediction, target, config.get("phenotype_indices"))
    losses = {"loss": loss, "reg_loss": loss}
    losses.update(regression_batch_metrics(prediction, target, filled, mean, std,
                                           config.get("phenotype_indices")))
    batch_cache = {"pheno_pred": prediction, "pheno_raw": filled} if cache else None
    return losses, batch_cache


def train_one_epoch(teacher, data_loader, optimizer, device, epoch, scaler,
                    pheno_mean, pheno_std, config):
    config = config or {}
    teacher.train()
    teacher.cmr_encoder.train(any(parameter.requires_grad for parameter in teacher.cmr_encoder.parameters()))
    teacher.sem_head.train()
    teacher.pheno_head.train()
    mean, std = phenotype_tensors(pheno_mean, pheno_std, device)
    compute = lambda batch, cache: _compute_batch(teacher, batch, device, mean, std, config, cache)
    return train_batches(compute, data_loader, optimizer, device, epoch, scaler, config)


@torch.no_grad()
def evaluate(teacher, data_loader, device, epoch, pheno_mean, pheno_std, config, split="val"):
    config = config or {}
    teacher.eval()
    mean, std = phenotype_tensors(pheno_mean, pheno_std, device)
    compute = lambda batch, cache: _compute_batch(teacher, batch, device, mean, std, config, cache)
    stats, cache = evaluate_batches(compute, data_loader, device, epoch, config, split)
    details = add_regression_summary(stats, cache, mean, std, config.get("phenotype_indices"))
    return stats, details


def test_evaluate(*args, **kwargs):
    kwargs["split"] = "test"
    return evaluate(*args, **kwargs)
