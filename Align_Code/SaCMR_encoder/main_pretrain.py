"""Command-line entry point for short-axis early-fusion MAE pretraining."""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler


def get_args_parser():
    parser = argparse.ArgumentParser("Short-axis MAE pretraining", add_help=False)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_manifest")
    parser.add_argument("--auto_train_manifest",
                        help="When --train_manifest is omitted, reuse this manifest or prepare it using --auto_manifest_validation")
    parser.add_argument("--auto_manifest_validation", choices=("nonempty", "full"), default="nonempty",
                        help="Validation for automatic manifests: file size only, or full decoding of the required arrays")
    parser.add_argument("--manifest_is_canonical", action="store_true",
                        help="Reuse a generated absolute-path manifest without per-file NFS metadata checks; files are validated when read")
    parser.add_argument("--output_dir", default="./output_sa_mae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="mae_vit_base_patch16",
                        choices=("mae_vit_base_patch16", "mae_vit_tiny_patch16"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_iter", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1,
                        help="Batches prepared per worker; each float32 batch of 256 is about 4.39 GiB in RAM")
    parser.add_argument("--blr", type=float, default=0.001)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--resume")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--decoder_slice_chunk_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--size", type=int, default=96)
    return parser


def validate_args(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise ValueError("This trainer is single-process only; WORLD_SIZE must be 1")
    positive = ("batch_size", "accum_iter", "epochs", "num_workers", "prefetch_factor",
                "save_freq", "decoder_slice_chunk_size", "size")
    for name in positive:
        value = getattr(args, name)
        if name == "num_workers":
            if value < 0:
                raise ValueError("num_workers must be non-negative")
        elif value <= 0:
            raise ValueError("{} must be positive".format(name))
    for name in ("max_steps", "max_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError("{} must be positive".format(name))
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if args.print_freq < 0:
        raise ValueError("print_freq must be non-negative")
    if args.size != 96:
        raise ValueError("size must be 96 for the fixed 6x6-token MAE")
    float_values = ("mask_ratio", "blr", "min_lr", "weight_decay")
    if args.lr is not None:
        float_values += ("lr",)
    for name in float_values:
        if not math.isfinite(getattr(args, name)):
            raise ValueError("{} must be finite".format(name))
    if int(36 * (1.0 - args.mask_ratio)) < 1:
        raise ValueError("mask_ratio must leave at least one visible token")
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if args.lr is not None and args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.blr <= 0 or args.min_lr < 0 or args.weight_decay < 0:
        raise ValueError("learning rates must be positive and weight decay non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")
    if args.manifest_is_canonical and not (args.train_manifest or args.auto_train_manifest):
        raise ValueError("--manifest_is_canonical requires --train_manifest or --auto_train_manifest")
    return args


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config(args):
    config = vars(args).copy()
    config["use_checkpoint"] = not args.no_checkpoint
    return config


def cohort_fingerprint(files, canonical=False):
    # Discovery excludes descendant symlinks and canonicalizes the root once.
    # Avoid walking every NFS path again for this pure hash.
    resolved = "\n".join(str(path) if canonical else str(Path(path).resolve())
                         for path in files)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def parameter_groups(model, weight_decay):
    """Apply decay to matrix weights, excluding biases/norms/special tokens."""
    decay, no_decay = [], []
    special_suffixes = ("cls_token", "mask_token", "pos_embed")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or name.endswith(special_suffixes):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def data_loader_configuration(args, device):
    """Describe image payload capacity, excluding collation and pinned copies."""
    batch_bytes = args.batch_size * 10 * 50 * args.size * args.size * 4
    prefetch_batches = args.num_workers * args.prefetch_factor
    return {
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers else None,
        "max_prefetch_batches": prefetch_batches,
        "pin_memory": device.type == "cuda",
        "image_dtype": "float32",
        "batch_image_gib": batch_bytes / (1024.0 ** 3),
        "prefetch_image_gib": batch_bytes * prefetch_batches / (1024.0 ** 3),
    }


def build_data_loader(dataset, args, device, epoch):
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    sampler = RandomSampler(dataset, generator=generator)
    options = {}
    if args.num_workers:
        # PyTorch 1.12 rejects a non-default prefetch_factor with zero workers.
        options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False, **options,
    )


def guard_output_directory(output_dir, resume):
    """Reject a fresh launch that would overwrite artifacts from an earlier run."""
    output_dir = Path(output_dir)
    if resume:
        return
    artifacts = []
    for name in ("config.json", "log.jsonl", "train_manifest.txt"):
        path = output_dir / name
        if path.exists():
            artifacts.append(path.name)
    artifacts.extend(path.name for path in output_dir.glob("checkpoint-*.pth"))
    if artifacts:
        raise FileExistsError(
            "Output contains existing run artifact(s): {}. Use --resume or a new "
            "output_dir".format(", ".join(sorted(set(artifacts)))))


def _write_manifest_snapshot(dataset, output_dir):
    destination = Path(output_dir) / "train_manifest.txt"
    temporary = destination.with_suffix(".txt.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        canonical_files = getattr(dataset, 'canonical_files', None)
        if canonical_files is None:
            canonical_files = [Path(path).resolve() for path in dataset.files]
        for path in canonical_files:
            handle.write(str(path) + "\n")
    os.replace(str(temporary), str(destination))


def main(args):
    if not __package__:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from SaCMR_encoder.startup import StartupProgress

    with StartupProgress() as progress:
        progress.update('imports')
        return _main_with_progress(args, progress)


def _main_with_progress(args, progress):
    if __package__:
        from .data.dataset import ShortAxisNPZDataset
        from .data.manifest import ensure_nonempty_manifest, ensure_validated_manifest
        from .engine_pretrain import (cosine_learning_rate, load_checkpoint,
                                      save_checkpoint, train_one_epoch)
        from . import models_mae
    else:  # Support: python SaCMR_encoder/main_pretrain.py
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from SaCMR_encoder.data.dataset import ShortAxisNPZDataset
        from SaCMR_encoder.data.manifest import ensure_nonempty_manifest, ensure_validated_manifest
        from SaCMR_encoder.engine_pretrain import (cosine_learning_rate,
                                                   load_checkpoint,
                                                   save_checkpoint,
                                                   train_one_epoch)
        from SaCMR_encoder import models_mae
    progress.update('environment', device=args.device)
    validate_args(args)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_output_directory(output_dir, args.resume)

    if args.auto_train_manifest and not args.train_manifest:
        progress.update('prepare_manifest', data_root=args.data_root,
                        manifest=args.auto_train_manifest)
        prepare_manifest = (ensure_validated_manifest if args.auto_manifest_validation == 'full'
                            else ensure_nonempty_manifest)
        args.train_manifest = str(prepare_manifest(
            args.data_root, args.auto_train_manifest, progress=progress.update))

    progress.update('dataset_index', data_root=args.data_root, manifest=args.train_manifest)
    dataset = ShortAxisNPZDataset(args.data_root, manifest=args.train_manifest,
                                  size=args.size, max_samples=args.max_samples,
                                  progress=progress.update,
                                  manifest_is_canonical=args.manifest_is_canonical)
    effective_batch = args.batch_size * args.accum_iter
    learning_rate = args.lr if args.lr is not None else args.blr * effective_batch / 256.0
    progress.update('model_init', model=args.model, samples=len(dataset), device=str(device))
    model = getattr(models_mae, args.model)(
        norm_pix_loss=args.norm_pix_loss,
        use_checkpoint=not args.no_checkpoint,
        decoder_slice_chunk_size=args.decoder_slice_chunk_size,
    ).to(device)
    if args.init_checkpoint:
        progress.update('load_initial_weights', checkpoint=args.init_checkpoint)
        report = model.load_long_axis_checkpoint(args.init_checkpoint)
        print(json.dumps({"event": "init_checkpoint", "report": report}, default=str))
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay),
                                  lr=learning_rate, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    config = _config(args)
    config["effective_lr"] = learning_rate
    progress.update('cohort_fingerprint', samples=len(dataset))
    config["cohort_fingerprint"] = cohort_fingerprint(dataset.canonical_files, canonical=True)
    start_epoch = 0
    if args.resume:
        progress.update('resume_checkpoint', checkpoint=args.resume)
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler,
                                      config, device, canonical_files=dataset.canonical_files)
    progress.update('write_manifest', samples=len(dataset), output_dir=str(output_dir))
    _write_manifest_snapshot(dataset, output_dir)

    print(json.dumps({
        "event": "configuration", "samples": len(dataset),
        "effective_batch_size": effective_batch, "learning_rate": learning_rate,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model": args.model, "device": str(device),
        "data_loader": data_loader_configuration(args, device),
    }), flush=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    # Stop before DataLoader workers are forked. Training has its own logs.
    progress.update('ready', samples=len(dataset), start_epoch=start_epoch)
    progress.stop()
    for epoch in range(start_epoch, args.epochs):
        print(json.dumps({"event": "epoch_start", "epoch": epoch,
                          "samples": len(dataset), "num_workers": args.num_workers,
                          "prefetch_factor": args.prefetch_factor if args.num_workers else None}), flush=True)
        loader = build_data_loader(dataset, args, device, epoch)
        def schedule(step, steps):
            progress = epoch + step / max(1, steps)
            return cosine_learning_rate(progress, args.epochs, args.warmup_epochs,
                                        learning_rate, args.min_lr)
        stats = train_one_epoch(
            model, loader, optimizer, device, epoch, args.accum_iter,
            args.mask_ratio, not args.no_amp, scaler, args.print_freq,
            args.max_steps, schedule,
        )
        record = {"event": "epoch_end", "epoch": epoch, **stats}
        print(json.dumps(record), flush=True)
        with (output_dir / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        periodic = (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs
        save_checkpoint(output_dir, model, optimizer, scaler, epoch, config, periodic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Short-axis MAE pretraining",
                                     parents=[get_args_parser()])
    main(parser.parse_args())
