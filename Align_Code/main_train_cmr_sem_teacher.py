import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import yaml

from CMR_encoder import models_vit
from data.dataset import UKB_SingleSaxCMR_Lax4chCMR_ECG_test, UKB_SingleSaxCMR_Lax4chCMR_ECG_train, UKB_SingleSaxCMR_Lax4chCMR_ECG_valid
from engine_cmr_sem_teacher import evaluate, test_evaluate, train_one_epoch
from util.align_common import append_metrics_csv, append_phenotype_metrics_csv, build_pheno_head, build_semantic_head, compute_phenotype_norm, save_multi_module_checkpoint, save_phenotype_norm, set_seed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config


def get_args_parser():
    parser = argparse.ArgumentParser("Train LAX CMR semantic bottleneck teacher", add_help=False)
    parser.add_argument("--cmr_model", default="vit_base_patch16", type=str)
    parser.add_argument("--Lax_cmr_pretrained_weights", default="/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth", type=str)
    parser.add_argument("--drop_path", default=0.1, type=float)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--sem_hidden_dim", default=768, type=int)
    parser.add_argument("--pheno_hidden_dim", default=256, type=int)
    parser.add_argument("--output_dir", default="/mnt/sda1/dingzhengyao/Work/Align/ECG_CMR", type=str)
    parser.add_argument("--exp_name", default="cmr_sem_teacher", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--drop_last", default=False, type=bool)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--blr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=1e-7, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--warmup_epochs", default=20, type=int)
    parser.add_argument("--freeze_encoder_epochs", default=50, type=int)
    parser.add_argument("--unfreeze_last_blocks", default=2, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_amp", default=True, type=bool)
    parser.add_argument("--best_patience", default=10, type=int)
    return parser


def set_cmr_trainable(cmr_encoder, train_last_blocks: bool, last_blocks: int):
    for p in cmr_encoder.parameters():
        p.requires_grad = False
    if train_last_blocks and hasattr(cmr_encoder, "blocks"):
        for block in cmr_encoder.blocks[-last_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        if hasattr(cmr_encoder, "norm"):
            for p in cmr_encoder.norm.parameters():
                p.requires_grad = True


def main(args):
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(yaml.dump(vars(args), default_flow_style=False, sort_keys=False))
    set_seed(args.seed)
    train_set, valid_set, test_set = UKB_SingleSaxCMR_Lax4chCMR_ECG_train(), UKB_SingleSaxCMR_Lax4chCMR_ECG_valid(), UKB_SingleSaxCMR_Lax4chCMR_ECG_test()
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    loader_valid = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    pheno_mean_np, pheno_std_np = compute_phenotype_norm(train_set)
    pheno_mean = torch.tensor(pheno_mean_np, dtype=torch.float32, device=args.device)
    pheno_std = torch.tensor(pheno_std_np, dtype=torch.float32, device=args.device)
    args.exp_name = args.exp_name + f"_bs{args.batch_size}_lr{args.blr}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    save_phenotype_norm(os.path.join(output_dir, "phenotype_norm.json"), pheno_mean_np, pheno_std_np)
    log_writer = SummaryWriter(log_dir=output_dir)

    cmr_encoder = models_vit.__dict__[args.cmr_model](drop_path_rate=args.drop_path)
    checkpoint = torch.load(args.Lax_cmr_pretrained_weights, map_location="cpu")
    print(f"Load pre-trained CMR model: {cmr_encoder.load_state_dict(checkpoint['model'], strict=False)}")
    sem_head = build_semantic_head(args.embed_dim, args.sem_hidden_dim)
    pheno_head = build_pheno_head(args.embed_dim, args.pheno_hidden_dim)
    cmr_encoder.to(args.device)
    sem_head.to(args.device)
    pheno_head.to(args.device)

    set_cmr_trainable(cmr_encoder, train_last_blocks=False, last_blocks=args.unfreeze_last_blocks)
    trainable = nn.ModuleDict({"cmr_encoder": cmr_encoder, "sem_head": sem_head, "pheno_head": pheno_head})
    args.lr = args.blr * args.batch_size * args.accum_iter / 256.0
    optimizer = get_optimizer_from_config(args, trainable)
    loss_scaler = NativeScaler()
    best_mae, patient = float("inf"), 0
    phenotype_csv = os.path.join(output_dir, "cmr_teacher_phenotype_metrics.csv")
    metrics_csv = os.path.join(output_dir, "cmr_teacher_metrics.csv")
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        set_cmr_trainable(cmr_encoder, train_last_blocks=epoch >= args.freeze_encoder_epochs, last_blocks=args.unfreeze_last_blocks)
        train_stats = train_one_epoch(cmr_encoder, sem_head, pheno_head, loader_train, optimizer, args.device, epoch, loss_scaler, pheno_mean, pheno_std, log_writer, vars(args), args.use_amp)
        valid_stats, valid_details = evaluate(cmr_encoder, sem_head, pheno_head, loader_valid, args.device, epoch, pheno_mean, pheno_std, log_writer, vars(args), args.use_amp)
        append_phenotype_metrics_csv(phenotype_csv, "valid", epoch, "current", valid_details)
        append_metrics_csv(metrics_csv, "valid", epoch, "current", valid_stats)
        patient += 1
        if valid_stats["pheno_mae_raw_mean"] < best_mae:
            best_mae = valid_stats["pheno_mae_raw_mean"]
            patient = 0
            save_multi_module_checkpoint(vars(args), os.path.join(output_dir, "best-cmr-pheno.pth"), epoch, {"cmr_encoder": cmr_encoder, "sem_head": sem_head, "pheno_head": pheno_head}, optimizer, loss_scaler, {"pheno_mae_raw_mean": best_mae, **valid_stats})
        with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps({**{f"train_{k}": v for k, v in train_stats.items()}, **{f"valid_{k}": v for k, v in valid_stats.items()}, "epoch": epoch}) + "\n")
        if patient > args.best_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Total training time: {str(datetime.timedelta(seconds=int(time.time() - start)))}, best MAE: {best_mae}")
    ckpt = torch.load(os.path.join(output_dir, "best-cmr-pheno.pth"), map_location="cpu")
    cmr_encoder.load_state_dict(ckpt["cmr_encoder"], strict=True)
    sem_head.load_state_dict(ckpt["sem_head"], strict=True)
    pheno_head.load_state_dict(ckpt["pheno_head"], strict=True)
    test_stats, test_details = test_evaluate(cmr_encoder, sem_head, pheno_head, loader_test, args.device, ckpt["epoch"], pheno_mean, pheno_std, log_writer, vars(args), args.use_amp)
    append_phenotype_metrics_csv(phenotype_csv, "test", ckpt["epoch"], "best-cmr-pheno.pth", test_details)
    append_metrics_csv(metrics_csv, "test", ckpt["epoch"], "best-cmr-pheno.pth", test_stats)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
