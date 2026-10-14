"""Single-process training utilities for short-axis MAE pretraining."""

import hashlib
import json
import math
import os
import random
import tempfile
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


RESUME_CONFIG_KEYS = (
    "model", "mask_ratio", "norm_pix_loss", "size", "cohort_fingerprint",
    "seed", "batch_size", "accum_iter", "lr", "blr", "effective_lr",
    "warmup_epochs", "min_lr", "weight_decay", "no_amp",
)


def cosine_learning_rate(progress, epochs, warmup_epochs, lr, min_lr):
    if warmup_epochs > 0 and progress < warmup_epochs:
        return lr * progress / warmup_epochs
    denominator = max(1.0, epochs - warmup_epochs)
    phase = min(1.0, max(0.0, (progress - warmup_epochs) / denominator))
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * phase))


def set_learning_rate(optimizer, value):
    for group in optimizer.param_groups:
        group["lr"] = value * group.get("lr_scale", 1.0)


def _optimizer_step(model, optimizer, scaler, sample_count):
    if scaler is not None:
        scaler.unscale_(optimizer)
    inverse_samples = 1.0 / sample_count
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(inverse_samples)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_one_epoch(model, data_loader, optimizer, device, epoch, accum_iter=1,
                    mask_ratio=0.75, amp=True, scaler=None, print_freq=20,
                    max_steps=None, lr_schedule=None):
    """Train one epoch, weighting every optimizer window and metric by samples."""
    if accum_iter <= 0:
        raise ValueError("accum_iter must be positive")
    model.train(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    samples = 0
    window_samples = 0
    optimizer_steps = 0
    started = time.time()
    use_amp = bool(amp and device.type == "cuda")
    total_batches = len(data_loader)
    if max_steps is not None:
        total_batches = min(total_batches, max_steps)

    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break
        if lr_schedule is not None and window_samples == 0:
            set_learning_rate(optimizer, lr_schedule(step, total_batches))
        image_dtype = torch.float16 if use_amp else None
        images = batch["image"].to(device, dtype=image_dtype, non_blocking=True)
        valid_mask = batch["slice_valid_mask"].to(device, non_blocking=True)
        batch_size = images.shape[0]
        amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with amp_context:
            loss, _, _ = model(images, valid_mask, mask_ratio=mask_ratio,
                               return_predictions=False)
        if not torch.isfinite(loss).item():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite loss at epoch {} step {}: {}".format(
                epoch, step, loss.detach().item()))
        # Model loss is a patient mean. Backpropagate its sample sum, then
        # normalize gradients by the actual window size immediately before step.
        weighted_loss = loss * batch_size
        if scaler is None:
            weighted_loss.backward()
        else:
            scaler.scale(weighted_loss).backward()
        window_samples += batch_size
        samples += batch_size
        loss_sum += loss.detach().item() * batch_size
        end_of_window = (step + 1) % accum_iter == 0
        end_of_epoch = step + 1 == total_batches
        if end_of_window or end_of_epoch:
            _optimizer_step(model, optimizer, scaler, window_samples)
            window_samples = 0
            optimizer_steps += 1
        if print_freq and (step == 0 or (step + 1) % print_freq == 0 or end_of_epoch):
            print(json.dumps({
                "event": "train_step", "epoch": epoch, "step": step + 1,
                "steps": total_batches, "loss": loss_sum / samples,
                "lr": optimizer.param_groups[0]["lr"],
            }), flush=True)

    if samples == 0:
        raise ValueError("data loader produced no samples")
    stats = {
        "loss": loss_sum / samples, "samples": samples,
        "optimizer_steps": optimizer_steps, "seconds": time.time() - started,
        "lr": optimizer.param_groups[0]["lr"],
    }
    if device.type == "cuda":
        stats["peak_memory_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return stats


def _rng_state():
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


def _atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp",
                                         dir=str(destination.parent))
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, str(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(output_dir, model, optimizer, scaler, epoch, config,
                    periodic=False):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "config": dict(config), "rng_state": _rng_state(),
    }
    output_dir = Path(output_dir)
    _atomic_torch_save(payload, output_dir / "checkpoint-latest.pth")
    if periodic:
        _atomic_torch_save(payload, output_dir / "checkpoint-{:04d}.pth".format(epoch))


def _validate_resume_config(saved, current):
    mismatches = []
    for key in RESUME_CONFIG_KEYS:
        if key in saved and key in current and saved[key] != current[key]:
            mismatches.append("{}={!r} (checkpoint {!r})".format(
                key, current[key], saved[key]))
    if mismatches:
        raise ValueError("Checkpoint configuration mismatch: " + ", ".join(mismatches))


def _cohort_relocation(saved, current, canonical_files):
    """Prove an ordered cohort was rerooted without accessing its old storage."""
    saved_hash = saved.get("cohort_fingerprint")
    current_hash = current.get("cohort_fingerprint")
    if canonical_files is None or not saved_hash or not current_hash or saved_hash == current_hash:
        return None
    try:
        # The old root is checkpoint metadata: do not resolve it, since its
        # mount may no longer exist or may now point somewhere else.
        old_root_text = os.fspath(saved["data_root"])
        if not isinstance(old_root_text, str) or old_root_text != os.path.normpath(old_root_text):
            return None
        old_root = Path(old_root_text)
        if not old_root.is_absolute():
            return None
        new_root = Path(current["data_root"]).resolve()
        if old_root == new_root:
            return None
        files = [os.fspath(path) for path in canonical_files]
        if hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest() != current_hash:
            return None
        old_files = []
        for filename in files:
            path = Path(filename)
            # The dataset supplies already resolved paths. Require canonical
            # absolute spelling before using lexical containment/relative_to.
            if not path.is_absolute() or filename != os.path.normpath(filename):
                return None
            relative = path.relative_to(new_root)
            if relative == Path("."):
                return None
            old_files.append(str(old_root / relative))
        if hashlib.sha256("\n".join(old_files).encode("utf-8")).hexdigest() != saved_hash:
            return None
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        return None
    return str(old_root), str(new_root)


def load_checkpoint(path, model, optimizer, scaler, config, device, canonical_files=None):
    # Always deserialize on CPU. This keeps CPU RNG ByteTensors valid and
    # load_state_dict moves optimizer tensors to their parameter devices.
    checkpoint = torch.load(path, map_location="cpu")
    saved_config = checkpoint.get("config", {})
    relocation = _cohort_relocation(saved_config, config, canonical_files)
    resume_config = config
    if relocation is not None:
        resume_config = dict(config, cohort_fingerprint=saved_config["cohort_fingerprint"])
    _validate_resume_config(saved_config, resume_config)
    if relocation is not None:
        print(json.dumps({
            "event": "resume_data_root_relocation", "old_data_root": relocation[0],
            "new_data_root": relocation[1],
        }), flush=True)
    saved_epochs = saved_config.get("epochs")
    current_epochs = config.get("epochs")
    if saved_epochs is not None and current_epochs != saved_epochs:
        warnings.warn(
            "epochs changed from {} to {}; the cosine schedule horizon changes".format(
                saved_epochs, current_epochs), RuntimeWarning)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("rng_state") is not None:
        _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1
