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

from data.dataset import HeNan_ECGCMRTest, HeNan_ECGCMRTrain, HeNan_ECGCMRValid
from engine_align_sem import evaluate, test_evaluate, train_one_epoch
from main_align_sem_stage1UKB import load_teacher
from util.align_common import build_ecg_model_from_config, save_multi_module_checkpoint, set_seed, write_downstream_config
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config


def get_args_parser():
    parser = argparse.ArgumentParser("ECG semantic alignment stage2 FCCCH", add_help=False)
    parser.add_argument("--cmr_model", default="vit_base_patch16", type=str)
    parser.add_argument("--cmr_teacher_path", required=True, type=str)
    parser.add_argument("--ecg_config_path", required=True, type=str)
    parser.add_argument("--ecg_model", default="stmem", type=str)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--sem_hidden_dim", default=768, type=int)
    parser.add_argument("--pheno_hidden_dim", default=256, type=int)
    parser.add_argument("--sem_loss_weight", default=1.0, type=float)
    parser.add_argument("--output_dir", default="/mnt/sda1/dingzhengyao/Work/Align/ECG_CMR", type=str)
    parser.add_argument("--exp_name", default="ecg_sem_stage2FCCCH", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)
    parser.add_argument("--drop_last", default=False, type=bool)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--blr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-7, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_amp", default=True, type=bool)
    parser.add_argument("--best_patience", default=5, type=int)
    return parser


def main(args):
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(yaml.dump(vars(args), default_flow_style=False, sort_keys=False))
    set_seed(args.seed)
    train_set, valid_set, test_set = HeNan_ECGCMRTrain(), HeNan_ECGCMRValid(), HeNan_ECGCMRTest()
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    loader_valid = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=args.drop_last)
    args.stage1_source_dir = os.path.basename(os.path.dirname(args.ecg_config_path))
    args.exp_name = args.exp_name + f"_bs{args.batch_size}_lr{args.blr:g}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
    write_downstream_config(output_dir, "st_mem_aligned_ecg_sem_stage2.yaml", "best-loss.pth")

    ECG_model = build_ecg_model_from_config(args.ecg_config_path, num_classes=None).to(args.device)
    cmr_encoder, sem_head, teacher_pheno_head = load_teacher(args)
    trainable = nn.ModuleDict({"ecg": ECG_model})
    args.lr = args.blr * args.batch_size * args.accum_iter / 256.0
    optimizer = get_optimizer_from_config(args, trainable)
    loss_scaler = NativeScaler()
    best_loss, patient = float("inf"), 0
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(ECG_model, None, cmr_encoder, sem_head, teacher_pheno_head, loader_train, optimizer, args.device, epoch, loss_scaler, None, None, log_writer, vars(args), args.use_amp)
        valid_stats, _ = evaluate(ECG_model, None, cmr_encoder, sem_head, teacher_pheno_head, loader_valid, args.device, epoch, None, None, log_writer, vars(args), args.use_amp)
        patient += 1
        if valid_stats["loss"] < best_loss:
            best_loss = valid_stats["loss"]
            patient = 0
            save_multi_module_checkpoint(vars(args), os.path.join(output_dir, "best-loss.pth"), epoch, {"model": ECG_model}, optimizer, loss_scaler, {"loss": best_loss, **valid_stats})
        with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps({**{f"train_{k}": v for k, v in train_stats.items()}, **{f"valid_{k}": v for k, v in valid_stats.items()}, "epoch": epoch}) + "\n")
        if patient > args.best_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Total training time: {str(datetime.timedelta(seconds=int(time.time() - start)))}, best loss: {best_loss}")
    ckpt = torch.load(os.path.join(output_dir, "best-loss.pth"), map_location="cpu")
    ECG_model.load_state_dict(ckpt["model"], strict=True)
    test_stats, _ = test_evaluate(ECG_model, None, cmr_encoder, sem_head, teacher_pheno_head, loader_test, args.device, ckpt["epoch"], None, None, log_writer, vars(args), args.use_amp)
    with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps({**{f"test_{k}": v for k, v in test_stats.items()}, "epoch": ckpt["epoch"]}) + "\n")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
