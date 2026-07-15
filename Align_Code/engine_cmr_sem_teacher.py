import math
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

import util.lr_sched as lr_sched
import util.misc as misc
from util.losses_pheno import fill_raw_phenotypes, normalize_phenotypes, phenotype_regression_metrics


def forward_teacher(cmr_encoder, sem_head, pheno_head, cmr):
    raw = cmr_encoder(cmr)
    sem = sem_head(raw)
    pred = pheno_head(sem)
    return raw, sem, pred


def train_one_epoch(cmr_encoder, sem_head, pheno_head, data_loader: Iterable, optimizer, device, epoch, loss_scaler, pheno_mean, pheno_std, log_writer=None, config=None, use_amp=True) -> Dict[str, float]:
    cmr_encoder.train(any(p.requires_grad for p in cmr_encoder.parameters()))
    sem_head.train()
    pheno_head.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    accum_iter = (config or {}).get("accum_iter", 1)
    max_norm = (config or {}).get("max_norm", None)
    params = [p for group in [cmr_encoder.parameters(), sem_head.parameters(), pheno_head.parameters()] for p in group]
    optimizer.zero_grad()
    for step, data_dict in enumerate(metric_logger.log_every(data_loader, 20, f"Epoch: [{epoch}]")):
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, config)
        cmr = data_dict["la_cmr"].to(device, non_blocking=True)
        pheno_raw = data_dict["phenotypes"].to(device, non_blocking=True).float()
        target = normalize_phenotypes(pheno_raw, pheno_mean, pheno_std)
        raw_filled = fill_raw_phenotypes(pheno_raw, pheno_mean)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, _, pred = forward_teacher(cmr_encoder, sem_head, pheno_head, cmr)
            loss = F.smooth_l1_loss(pred, target)
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping CMR teacher training")
            sys.exit(1)
        loss_scaler(loss / accum_iter, optimizer, clip_grad=max_norm, parameters=params, update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pred_raw = pred.detach().float() * pheno_std + pheno_mean
        metric_logger.update(loss=loss.item(), pheno_mae_raw=torch.mean(torch.abs(pred_raw - raw_filled)).item(), pheno_mae_norm=torch.mean(torch.abs(pred.detach().float() - target)).item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + step / len(data_loader)) * 1000)
            log_writer.add_scalar("train/loss", loss.item(), epoch_1000x)
            log_writer.add_scalar("train/lr", lr, epoch_1000x)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(cmr_encoder, sem_head, pheno_head, data_loader, device, epoch, pheno_mean, pheno_std, log_writer=None, config=None, use_amp=True, split="Valid") -> Tuple[Dict[str, float], List[dict]]:
    cmr_encoder.eval()
    sem_head.eval()
    pheno_head.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    all_pred, all_target = [], []
    for data_dict in metric_logger.log_every(data_loader, 10, f"{split}:"):
        cmr = data_dict["la_cmr"].to(device, non_blocking=True)
        pheno_raw = data_dict["phenotypes"].to(device, non_blocking=True).float()
        target = normalize_phenotypes(pheno_raw, pheno_mean, pheno_std)
        raw_filled = fill_raw_phenotypes(pheno_raw, pheno_mean)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, _, pred = forward_teacher(cmr_encoder, sem_head, pheno_head, cmr)
            loss = F.smooth_l1_loss(pred, target)
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping CMR teacher eval")
            sys.exit(1)
        pred_raw = pred.detach().float() * pheno_std + pheno_mean
        metric_logger.update(loss=loss.item(), pheno_mae_raw=torch.mean(torch.abs(pred_raw - raw_filled)).item(), pheno_mae_norm=torch.mean(torch.abs(pred.detach().float() - target)).item())
        all_pred.append(pred.detach().float().cpu())
        all_target.append(raw_filled.detach().float().cpu())
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    summary, details = phenotype_regression_metrics(torch.cat(all_pred).numpy(), torch.cat(all_target).numpy(), pheno_mean, pheno_std)
    stats.update(summary)
    print(f"{split} stats:", metric_logger)
    print(f"{split}: loss={stats['loss']:.4f}, key_r={stats['pheno_key_pearson']:.4f}, mean_r={stats['pheno_pearson_mean']:.4f}")
    if log_writer is not None:
        for key, value in stats.items():
            log_writer.add_scalar(f"{split.lower()}/{key}", value, epoch)
    return stats, details


def test_evaluate(*args, **kwargs):
    kwargs["split"] = "Test"
    return evaluate(*args, **kwargs)
