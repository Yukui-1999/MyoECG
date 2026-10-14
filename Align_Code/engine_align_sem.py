import math
import sys
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

import util.lr_sched as lr_sched
import util.misc as misc
from engine_cmr_sem_teacher import forward_teacher
from util.losses_pheno import (
    alignment_retrieval_metrics,
    fill_raw_phenotypes,
    normalize_phenotypes,
    phenotype_embedding_retrieval_metrics,
    phenotype_regression_metrics,
    phenotype_relational_losses,
)


def _forward_ecg(ecg_model, ecg):
    if ecg.ndim == 4:
        outputs = [ecg_model(ecg[:, i]) for i in range(ecg.size(1))]
        return torch.stack(outputs, dim=1).mean(dim=1)
    return ecg_model(ecg)


def _batch_to_ecg_cmr(batch, device):
    if isinstance(batch, dict):
        return batch["ecg"].to(device, non_blocking=True), batch["la_cmr"].to(device, non_blocking=True)
    return batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)


def _compute_loss(ECG_model, pheno_head, cmr_encoder, sem_head, teacher_pheno_head, batch, device, pheno_mean, pheno_std, config, use_amp):
    sem_weight = (config or {}).get("sem_loss_weight", 1.0)
    reg_weight = (config or {}).get("pheno_loss_weight", 1.0)
    rel_weight = (config or {}).get("rel_ecg_weight", 0.5)
    tau_phi = (config or {}).get("tau_phi", 1.0)
    tau_ecg = (config or {}).get("tau_ecg", 0.07)
    ecg, cmr = _batch_to_ecg_cmr(batch, device)
    has_pheno = isinstance(batch, dict) and "phenotypes" in batch

    with torch.cuda.amp.autocast(enabled=use_amp):
        ecg_out = _forward_ecg(ECG_model, ecg)
        with torch.no_grad():
            _, cmr_sem, _ = forward_teacher(cmr_encoder, sem_head, teacher_pheno_head, cmr)
        sem_loss = 1.0 - F.cosine_similarity(ecg_out.float(), cmr_sem.float(), dim=-1).mean()
        loss = sem_weight * sem_loss
        loss_dict = {"loss": loss, "sem_loss": sem_loss}
        cache = {"ecg_out": ecg_out.detach().float().cpu(), "cmr_sem": cmr_sem.detach().float().cpu()}
        if has_pheno and pheno_head is not None:
            pheno_raw = batch["phenotypes"].to(device, non_blocking=True).float()
            pheno_target = normalize_phenotypes(pheno_raw, pheno_mean, pheno_std)
            pheno_pred = pheno_head(ecg_out)
            reg_loss = F.smooth_l1_loss(pheno_pred, pheno_target)
            _, rel_ecg, _ = phenotype_relational_losses(ecg_out, cmr_sem, pheno_target, tau_phi=tau_phi, tau_cross=tau_ecg, tau_ecg=tau_ecg)
            loss = loss + reg_weight * reg_loss + rel_weight * rel_ecg
            loss_dict.update({"loss": loss, "reg_loss": reg_loss, "rel_ecg_loss": rel_ecg})
            raw_filled = fill_raw_phenotypes(pheno_raw, pheno_mean)
            pred_raw = pheno_pred.detach().float() * pheno_std + pheno_mean
            loss_dict.update({
                "pheno_mae_raw": torch.mean(torch.abs(pred_raw - raw_filled)),
                "pheno_mae_norm": torch.mean(torch.abs(pheno_pred.detach().float() - pheno_target)),
            })
            cache.update({
                "pheno_pred": pheno_pred.detach().float().cpu(),
                "pheno_raw": raw_filled.detach().float().cpu(),
                "pheno_norm": pheno_target.detach().float().cpu(),
            })
    return loss, loss_dict, cache


def train_one_epoch(ECG_model, pheno_head, cmr_encoder, sem_head, teacher_pheno_head, data_loader: Iterable, optimizer, device, epoch, loss_scaler, pheno_mean=None, pheno_std=None, log_writer=None, config=None, use_amp=True) -> Dict[str, float]:
    ECG_model.train()
    if pheno_head is not None:
        pheno_head.train()
    cmr_encoder.eval()
    sem_head.eval()
    teacher_pheno_head.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    accum_iter = (config or {}).get("accum_iter", 1)
    max_norm = (config or {}).get("max_norm", None)
    params = list(ECG_model.parameters()) + ([] if pheno_head is None else list(pheno_head.parameters()))
    optimizer.zero_grad()
    for step, batch in enumerate(metric_logger.log_every(data_loader, 20, f"Epoch: [{epoch}]")):
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, config)
        loss, loss_dict, _ = _compute_loss(ECG_model, pheno_head, cmr_encoder, sem_head, teacher_pheno_head, batch, device, pheno_mean, pheno_std, config, use_amp)
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping semantic ECG training")
            sys.exit(1)
        loss_scaler(loss / accum_iter, optimizer, clip_grad=max_norm, parameters=params, update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + step / len(data_loader)) * 1000)
            for key, value in loss_dict.items():
                log_writer.add_scalar(f"train/{key}", value.item(), epoch_1000x)
            log_writer.add_scalar("train/lr", lr, epoch_1000x)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(ECG_model, pheno_head, cmr_encoder, sem_head, teacher_pheno_head, data_loader, device, epoch, pheno_mean=None, pheno_std=None, log_writer=None, config=None, use_amp=True, split="Valid") -> Tuple[Dict[str, float], Optional[list]]:
    ECG_model.eval()
    if pheno_head is not None:
        pheno_head.eval()
    cmr_encoder.eval()
    sem_head.eval()
    teacher_pheno_head.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    all_ecg, all_sem, all_pred, all_raw, all_norm = [], [], [], [], []
    for batch in metric_logger.log_every(data_loader, 10, f"{split}:"):
        loss, loss_dict, cache = _compute_loss(ECG_model, pheno_head, cmr_encoder, sem_head, teacher_pheno_head, batch, device, pheno_mean, pheno_std, config, use_amp)
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping semantic ECG eval")
            sys.exit(1)
        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        all_ecg.append(cache["ecg_out"])
        all_sem.append(cache["cmr_sem"])
        if "pheno_pred" in cache:
            all_pred.append(cache["pheno_pred"])
            all_raw.append(cache["pheno_raw"])
            all_norm.append(cache["pheno_norm"])
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update(alignment_retrieval_metrics(torch.cat(all_ecg).to(device), torch.cat(all_sem).to(device)))
    details = None
    if all_pred:
        summary, details = phenotype_regression_metrics(torch.cat(all_pred).numpy(), torch.cat(all_raw).numpy(), pheno_mean, pheno_std)
        stats.update(summary)
        stats.update(phenotype_embedding_retrieval_metrics(torch.cat(all_ecg).to(device), torch.cat(all_norm).to(device), topk=5))
    print(f"{split} stats:", metric_logger)
    if log_writer is not None:
        for key, value in stats.items():
            log_writer.add_scalar(f"{split.lower()}/{key}", value, epoch)
    return stats, details


def test_evaluate(*args, **kwargs):
    kwargs["split"] = "Test"
    return evaluate(*args, **kwargs)
