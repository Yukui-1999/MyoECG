"""Phenotype-free Henan ECG adaptation to the frozen short-axis teacher."""
import torch
import torch.nn.functional as F

from util.sem_ablation import validate_stage2_weights

from .engine_align_sem import _forward_ecg, chunked_retrieval_metrics
from .engine_common import amp_context, evaluate_batches, full_precision, train_batches


class ValidBatches:
    """Count unreadable samples and omit batches with no surviving pairs.

    The raw length remains an upper bound for learning-rate scheduling. Shared
    training code flushes pending gradients after iteration, including when
    skipping the final batches leaves an incomplete accumulation group.
    """

    def __init__(self, loader):
        self.loader = loader
        self.batch_size = getattr(loader, "batch_size", None)
        self.skipped_samples = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.skipped_samples = 0
        yielded = False
        for batch in self.loader:
            self.skipped_samples += int(batch.get("_skipped_samples", 0))
            if "sa_cmr" in batch:
                yielded = True
                yield batch
        if not yielded:
            raise ValueError("Stage2 data loader produced no readable samples; skipped %s samples"
                             % self.skipped_samples)


def compute_semantic_losses(ecg_features, teacher_semantic, config):
    """Match the long-axis stage2 cosine objective in float32."""
    weight = validate_stage2_weights(config or {})["sem_loss_weight"]
    with full_precision(ecg_features.device):
        semantic_loss = 1 - F.cosine_similarity(ecg_features.float(),
                                               teacher_semantic.float(), dim=-1).mean()
        weighted = weight * semantic_loss
    return {"sem_loss": semantic_loss, "weighted_sem_loss": weighted, "loss": weighted}


def _compute_batch(ecg_model, teacher, batch, device, config, cache):
    ecg = batch["ecg"].to(device=device, dtype=torch.float32, non_blocking=True)
    cmr = batch["sa_cmr"].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = batch["slice_valid_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
    with amp_context(device, config):
        features = _forward_ecg(ecg_model, ecg)
        with torch.no_grad():
            semantic, _ = teacher(cmr, mask)
    losses = compute_semantic_losses(features, semantic, config)
    batch_cache = {"ecg_out": features, "cmr_sem": semantic} if cache else None
    return losses, batch_cache


def _sample_counts(stats, batches):
    stats.update(processed_samples=stats["samples"], skipped_samples=batches.skipped_samples)
    return stats


def train_one_epoch(ecg_model, teacher, loader, optimizer, device, epoch, scaler, config):
    config = config or {}
    validate_stage2_weights(config)
    ecg_model.train()
    teacher.eval()
    teacher.requires_grad_(False)
    batches = ValidBatches(loader)
    compute = lambda batch, cache: _compute_batch(ecg_model, teacher, batch, device, config, cache)
    stats = train_batches(compute, batches, optimizer, device, epoch, scaler, config)
    return _sample_counts(stats, batches)


@torch.no_grad()
def evaluate(ecg_model, teacher, loader, device, epoch, config, split="val"):
    config = config or {}
    validate_stage2_weights(config)
    ecg_model.eval()
    teacher.eval()
    teacher.requires_grad_(False)
    batches = ValidBatches(loader)
    compute = lambda batch, cache: _compute_batch(ecg_model, teacher, batch, device, config, cache)
    stats, cache = evaluate_batches(compute, batches, device, epoch, config, split)
    stats.update(chunked_retrieval_metrics(torch.cat(cache["ecg_out"]),
                                           torch.cat(cache["cmr_sem"]),
                                           chunk_size=int(config.get("retrieval_chunk_size", 256))))
    return _sample_counts(stats, batches), None
