"""Small shared helpers for single-device short-axis training and evaluation."""
import json
import math
import time
from contextlib import nullcontext

import torch

from util.losses_pheno import phenotype_regression_metrics
from util.sem_ablation import selected_dimensions, selected_phenotype_metrics


def amp_context(device, config):
    if torch.device(device).type == "cuda" and config.get("use_amp", True):
        return torch.cuda.amp.autocast()
    return nullcontext()


def full_precision(device):
    """Disable an enclosing autocast context, including CPU autocast in tests."""
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


def phenotype_tensors(mean, std, device):
    return (torch.as_tensor(mean, dtype=torch.float32, device=device),
            torch.as_tensor(std, dtype=torch.float32, device=device))


def regression_batch_metrics(pred, target, raw, mean, std, indices):
    pred = pred.detach().float()
    pred_raw = pred * std + mean
    return {
        "pheno_mae_raw": (pred_raw - raw).abs().mean(),
        "pheno_mae_norm": (pred - target).abs().mean(),
        "pheno_selected_mae_raw": (selected_dimensions(pred_raw, indices)
                                   - selected_dimensions(raw, indices)).abs().mean(),
        "pheno_selected_mae_norm": (selected_dimensions(pred, indices)
                                    - selected_dimensions(target, indices)).abs().mean(),
    }


def add_regression_summary(stats, cache, mean, std, indices):
    predictions = torch.cat(cache["pheno_pred"]).numpy()
    targets = torch.cat(cache["pheno_raw"]).numpy()
    summary, details = phenotype_regression_metrics(predictions, targets, mean, std)
    stats.update(summary)
    stats.update(selected_phenotype_metrics(predictions, targets, mean, std, indices))
    return details


def adjust_learning_rate(optimizer, progress, config):
    base_lr = float(config.get("lr", optimizer.param_groups[0]["lr"]))
    minimum = float(config.get("min_lr", 0.0))
    warmup = float(config.get("warmup_epochs", 0))
    epochs = float(config.get("epochs", 1))
    if warmup > 0 and progress < warmup:
        lr = base_lr * progress / warmup
    else:
        fraction = min(1.0, max(0.0, (progress - warmup) / max(epochs - warmup, 1e-12)))
        lr = minimum + (base_lr - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


class EpochStats:
    def __init__(self, device):
        self.device = torch.device(device)
        self.sums = {}
        self.samples = 0
        self.started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def update(self, losses, count):
        for name, value in losses.items():
            scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            self.sums[name] = self.sums.get(name, 0.0) + count * scalar
        self.samples += count

    def means(self):
        return {name: value / self.samples for name, value in self.sums.items()}

    def finish(self, optimizer_steps=0):
        if not self.samples:
            raise ValueError("the data loader produced no samples")
        stats = self.means()
        stats.update(samples=self.samples, optimizer_steps=optimizer_steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            stats["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
            stats["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / 2**20
        else:
            stats.update(gpu_peak_allocated_mb=0.0, gpu_peak_reserved_mb=0.0)
        stats["seconds"] = time.perf_counter() - self.started
        return stats


def _log_progress(metrics, epoch, step, split, config, optimizer_steps=0):
    interval = int(config.get("log_interval", 20))
    if step == 0 or (interval > 0 and (step + 1) % interval == 0):
        payload = dict(event="step", split=split, epoch=epoch, step=step + 1,
                       samples=metrics.samples, optimizer_steps=optimizer_steps)
        payload.update(metrics.means())
        print(json.dumps(payload, sort_keys=True), flush=True)


def _optimizer_step(optimizer, scaler, sample_count, reference_count, max_norm):
    """Correct the nominal group denominator before clipping, including tails."""
    scaler.unscale_(optimizer)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                  if parameter.grad is not None]
    for parameter in parameters:
        parameter.grad.mul_(reference_count / sample_count)
    if max_norm is not None and float(max_norm) > 0:
        torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    did_step = scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    return int(did_step)


def train_batches(compute_batch, loader, optimizer, device, epoch, scaler, config):
    accumulation = int(config.get("accum_iter", 1))
    if accumulation < 1:
        raise ValueError("accum_iter must be positive")
    nominal_batch_size = config.get("batch_size")
    if nominal_batch_size is None:
        nominal_batch_size = getattr(loader, "batch_size", None)
    if nominal_batch_size is None:
        nominal_batch_size = getattr(getattr(loader, "batch_sampler", None), "batch_size", 1)
    if nominal_batch_size < 1:
        raise ValueError("batch_size must be positive")
    reference_count = nominal_batch_size * accumulation
    metrics = EpochStats(device)
    optimizer.zero_grad(set_to_none=True)
    group_samples = 0
    optimizer_steps = 0
    total_batches = len(loader)
    for step, batch in enumerate(loader):
        if step % accumulation == 0:
            adjust_learning_rate(optimizer, epoch + step / max(total_batches, 1), config)
        losses, _ = compute_batch(batch, False)
        loss = losses["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite training loss at epoch %s, step %s" % (epoch, step))
        count = batch["sa_cmr"].shape[0]
        # Keep the backward loss at the ordinary mean-loss scale for AMP. The
        # actual group size can differ because of partial batches or a tail.
        scaler.scale(loss * (count / reference_count)).backward()
        group_samples += count
        metrics.update(losses, count)
        metrics.sums["lr"] = metrics.sums.get("lr", 0.0) + optimizer.param_groups[0]["lr"] * count
        if (step + 1) % accumulation == 0 or step + 1 == total_batches:
            optimizer_steps += _optimizer_step(optimizer, scaler, group_samples, reference_count,
                                               config.get("max_norm"))
            group_samples = 0
        _log_progress(metrics, epoch, step, "train", config, optimizer_steps)
    # Also support loaders whose advertised length exceeds the yielded batch count.
    if group_samples:
        optimizer_steps += _optimizer_step(optimizer, scaler, group_samples, reference_count,
                                           config.get("max_norm"))
    return metrics.finish(optimizer_steps)


@torch.no_grad()
def evaluate_batches(compute_batch, loader, device, epoch, config, split):
    metrics = EpochStats(device)
    cache = {}
    for step, batch in enumerate(loader):
        losses, batch_cache = compute_batch(batch, True)
        if not bool(torch.isfinite(losses["loss"]).item()):
            raise FloatingPointError("non-finite %s loss at step %s" % (split, step))
        metrics.update(losses, batch["sa_cmr"].shape[0])
        for name, value in batch_cache.items():
            cache.setdefault(name, []).append(value.detach().float().cpu())
        _log_progress(metrics, epoch, step, split, config)
    return metrics.finish(), cache
