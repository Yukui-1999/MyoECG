import argparse
import datetime
import json
import os
import pickle
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_ecg_cmr")

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir

        def add_scalar(self, *args, **kwargs):
            pass

        def flush(self):
            pass

from data.dataset import (
    ECGBaseMIMIC_CMthree,
    ECGzheyi_three_Base,
    Harvard_ECGBaseDis,
    HeNan_ECGBaseDis,
    ShaoyifuCardiomyopathyDataset,
)
from engine_saliency_three import test_evaluate_saliency
from models import encoder
import util.misc as misc


def str_true_false(x):
    if x.lower() == "true":
        return True
    if x.lower() == "false":
        return False
    raise ValueError(f'Invalid value for bool flag {x}, should be "true" or "false"')


def get_args_parser():
    parser = argparse.ArgumentParser("Three-class saliency for downstream task")

    # model
    parser.add_argument("--cmr_model", default="vit_base_patch16", type=str, help="model name")
    parser.add_argument(
        "--cmr_pretrained_weights",
        default="/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth",
        type=str,
        help="pretrained weights path",
    )
    parser.add_argument("--drop_path", default=0, type=float, help="drop path rate")
    parser.add_argument("--input_modality", default="ECG", type=str, help="ECG or CMR")
    parser.add_argument(
        "--ecg_config_path",
        default="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/configs/Cla/st_mem_align_CLIP_Bin.yaml",
        type=str,
        help="ecg config path",
    )
    parser.add_argument("--num_classes", default=3, type=int, help="number of classes")
    parser.add_argument("--ecg_model", default="stmem", type=str, help="model name")

    # log / saliency
    parser.add_argument("--output_dir", default="/mnt/sda1/dingzhengyao/Work/ECG_CMR_Rework_v1/", type=str)
    parser.add_argument("--test_dir_name", default="test", type=str, help="test dir name")
    parser.add_argument(
        "--saliency_dir",
        default="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/Explain/results/Three_saliency_maps",
        type=str,
    )
    parser.add_argument("--saliency_mode", default="integrated_gradients", type=str, help="smoothgrad, integrated_gradients, or scratch")
    parser.add_argument("--saliency_target", default="label", type=str, help="label, pred, RCM, DCM, HCM, 0, 1, or 2")
    parser.add_argument("--smoothgrad_samples", default=16, type=int, help="number of SmoothGrad noisy samples")
    parser.add_argument("--smoothgrad_noise_std", default=0.05, type=float, help="SmoothGrad noise std")
    parser.add_argument("--ig_steps", default=32, type=int, help="Integrated Gradients steps")

    # data
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers")
    parser.add_argument("--pin_memory", default=True, type=bool, help="pin memory")
    parser.add_argument("--drop_last", default=False, type=bool, help="drop last batch")
    parser.add_argument("--dis", default="cm_three", type=str, help="disease label type")
    parser.add_argument("--health_magnification", default=1, type=int, help="health magnification")
    parser.add_argument("--ECG_length", default=10000, type=int, help="original ECG length after merging crops")
    parser.add_argument("--cm_three_dataset", default="Henan", type=str, help="Henan, Zheyi, MIMIC, Shaoyifu, or Harvard")
    parser.add_argument("--group_analysis_csv", default=None, type=str, help="optional subgroup CSV for supported datasets")

    # training/checkpoint
    parser.add_argument("--optimizer", default="adamw", type=str, help="optimizer name")
    parser.add_argument("--blr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="minimum learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--accum_iter", default=1, type=int, help="accumulation iterations")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
    parser.add_argument("--resume", default=None, type=str, help="resume from checkpoint")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing testing")
    parser.add_argument("--use_amp", default=True, type=bool, help="use amp for forward metrics")
    parser.add_argument("--best_patience", default=10, type=int, help="best patience")
    parser.add_argument("--only_test", action="store_true", help="kept for compatibility; saliency always uses test mode")
    parser.add_argument("--use_pretrained_CMR", default="False", type=str_true_false, help="use pre-trained CMR model")
    parser.add_argument("--record_eid", action="store_true", help="record eid during testing")
    parser.add_argument("--renji", action="store_true", help="use reader-study three-class CSV")
    parser.add_argument("--cal_popular_index", action="store_true", help="calculate popular index")
    return parser


def build_ecg_model(ecg_config_path):
    with open(os.path.realpath(ecg_config_path), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["model"]["num_classes"] = 3
    model_name = config["model_name"]

    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config["model"])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if config["mode"] == "pretrain":
        checkpoint = torch.load(config["encoder_path"], map_location="cpu")
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
    elif config["mode"] == "align":
        checkpoint = torch.load(config["afterAlign_path"], map_location="cpu")
        print(f"Load pre-trained checkpoint from: {config['afterAlign_path']}")
    elif config["mode"] == "scratch":
        print("Training from scratch")
        return ecg_model
    else:
        raise ValueError(f'Unsupported mode: {config["mode"]}')

    checkpoint_model = checkpoint["model"]
    state_dict = ecg_model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Remove key {k} from pre-trained checkpoint")
            del checkpoint_model[k]
    msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
    print(f"Load pre-trained ECG model: {msg}")
    return ecg_model


def build_test_dataset(args):
    if args.renji:
        data_excel = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407/test.xlsx"
        print(f"Reader-study data: {data_excel}")
        return HeNan_ECGBaseDis(data_excel=data_excel, isTrain=False, args=args)

    dataset_name = args.cm_three_dataset.lower()

    if dataset_name == "henan":
        henan_data_root = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407"
        print(f"Henan_data_root: {henan_data_root}")
        return HeNan_ECGBaseDis(data_excel=os.path.join(henan_data_root, "test.xlsx"), isTrain=False, args=args)

    if dataset_name == "zheyi":
        final_eid = pickle.load(open("/mnt/data2/ECG_CMR/zheyi_data/Final_data/rework/all_data_final_addPath_valid.pkl", "rb"))
        random.shuffle(final_eid)
        return ECGzheyi_three_Base(data=final_eid, isTrain=False, args=args)

    if dataset_name == "mimic":
        mimic_json_dir = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/MIMIC/Three"
        if not os.path.exists(os.path.join(mimic_json_dir, "mimic_data_path_RCM.json")):
            mimic_json_dir = "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json"
        RCM_eid = json.load(open(os.path.join(mimic_json_dir, "mimic_data_path_RCM.json")))
        DCM_eid = json.load(open(os.path.join(mimic_json_dir, "mimic_data_path_DCM.json")))
        HCM_eid = json.load(open(os.path.join(mimic_json_dir, "mimic_data_path_HCM.json")))
        final_eid = RCM_eid + DCM_eid + HCM_eid
        label = [0] * len(RCM_eid) + [1] * len(DCM_eid) + [2] * len(HCM_eid)
        final_eid = list(zip(final_eid, label))
        random.shuffle(final_eid)
        return ECGBaseMIMIC_CMthree(data=final_eid, isTrain=False, args=args)

    if dataset_name in ["shaoyifu", "srrsh"]:
        return ShaoyifuCardiomyopathyDataset(
            table_path="/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/ShaoyifuData/FirstWholeSummary0330/extval_three_class_patient_unique_with_quality.csv",
            task="three_class",
            isTrain=False,
            args=args,
        )

    if dataset_name == "harvard":
        df_all = pd.read_csv(
            "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HafoData_adjust2/norm_process/HarvardECG_step6.csv"
        )
        return Harvard_ECGBaseDis(data_df=df_all, isTrain=False, args=args)

    raise ValueError(f"Unsupported cm_three_dataset: {args.cm_three_dataset}")


def build_model(args):
    if args.input_modality == "ECG":
        if args.ecg_model == "stmem":
            model = build_ecg_model(args.ecg_config_path)
            model.to(args.device)
            return model

        if args.ecg_model == "ecg_found":
            from ECG_baselines.ECGFounder.finetune_model import ft_12lead_ECGFounder

            return ft_12lead_ECGFounder(
                args.device,
                "/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/ECG_baselines/ECGFounder/checkpoint/12_lead_ECGFounder.pth",
                n_classes=3,
            )

        if args.ecg_model == "ecgfm_ked":
            from ECG_baselines.ECGFM_KED.models.xresnet1d_101 import xresnet1d101

            checkpoint = torch.load(
                "/mnt/sda1/dingzhengyao/Work/ECG_CMR_CMAI/ECG_baselines/ECGFM-KED/best_valid_all_increase_with_augment_epoch_3.pt",
                map_location="cpu",
            )
            model = xresnet1d101(
                num_classes=3,
                input_channels=12,
                kernel_size=5,
                ps_head=0.5,
                lin_ftrs_head=[768],
                use_ecgNet_Diagnosis="ecgNet",
            )
            msg = model.load_state_dict(checkpoint["ecg_model"], strict=False)
            print(msg)
            model.to(args.device)
            return model

        if args.ecg_model == "fg_clep":
            from ECG_baselines.FG_CLEP.clep.modeling_clep import ECGModel

            model = ECGModel(
                output_class_num=3,
                encoder="resnet50",
                clep_checkpoint="/mnt/sda1/dingzhengyao/Work/ECG_CMR_CMAI/ECG_baselines/FGCLEP",
            )
            model.to(args.device)
            return model

        if args.ecg_model == "merl":
            from ECG_baselines.MERL_ICML2024.finetune.models.vit1d import vit_tiny

            model = vit_tiny(num_leads=12, num_classes=3, seq_len=5000, patch_size=50)
            msg = model.load_state_dict(
                torch.load("/mnt/sda1/dingzhengyao/Work/ECG_CMR_CMAI/ECG_baselines/MERL-ICML2024/vit_tiny_best_encoder.pth"),
                strict=False,
            )
            print(msg)
            model.to(args.device)
            return model

        raise ValueError(f"Unsupported ECG model: {args.ecg_model}")

    if args.input_modality == "CMR":
        from CMR_encoder import models_vit

        model = models_vit.__dict__[args.cmr_model](
            drop_path_rate=args.drop_path,
            num_classes=3,
        )
        if args.use_pretrained_CMR:
            checkpoint = torch.load(args.cmr_pretrained_weights, map_location="cpu")
            msg = model.load_state_dict(checkpoint["model"], strict=False)
            print(f"Load pre-trained CMR model: {msg}")
        else:
            print("No pre-trained CMR model, training from scratch")
        model.to(args.device)
        return model

    raise ValueError(f"Unsupported input modality: {args.input_modality}")


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def main(args):
    args.num_classes = 3
    if args.dis != "cm_three":
        print(f"Override dis from {args.dis} to cm_three for three-class saliency")
        args.dis = "cm_three"

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(yaml.dump(args, default_flow_style=False, sort_keys=False))

    seed = args.seed + misc.get_rank()
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

    test_set = build_test_dataset(args)
    data_loader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    print(f"Test dataset size: {len(test_set)}")

    model = build_model(args)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
    criterion = torch.nn.CrossEntropyLoss()

    if not args.only_test:
        print("This entry is for saliency extraction; running checkpoint test mode.")

    print("Start three-class saliency testing")
    ckpt_path = os.path.join(output_dir, "best-auc.pth")
    print(f'load {ckpt_path} for saliency testing')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"])
    print(f"load best auc model: {msg}")

    start_time = time.time()
    test_stats, metrics_df, opt_list, tgt_list, saliency_maps, sample_ids = test_evaluate_saliency(
        model,
        criterion,
        data_loader_test,
        args.device,
        log_writer,
        use_amp=args.use_amp,
        args=args,
        num_classes=args.num_classes,
    )

    test_dir = os.path.join(output_dir, args.test_dir_name)
    os.makedirs(test_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(test_dir, "saliency_metrics.csv"), index=False)

    if args.record_eid:
        df = pd.DataFrame(
            {
                "eid": sample_ids,
                "prob_RCM": opt_list[:, 0].squeeze(),
                "prob_DCM": opt_list[:, 1].squeeze(),
                "prob_HCM": opt_list[:, 2].squeeze(),
                "pred_label": np.array(opt_list).argmax(axis=1).squeeze(),
                "label": tgt_list.squeeze(),
            }
        )
        df.to_excel(os.path.join(test_dir, "Eid_prob_saliency_three.xlsx"), index=False)

    test_log_stats = {f"test_{k}": json_safe(v) for k, v in test_stats.items()}
    if output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(test_log_stats) + "\n\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total saliency time: {total_time_str}")
    print(f"Saved merged saliency shape: {saliency_maps.shape}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
