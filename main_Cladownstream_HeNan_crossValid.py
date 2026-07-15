import os
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np
import random
import time
import datetime
import json
import pandas as pd
import tempfile
from engine_Caldownstream import train_one_epoch, evaluate, test_evaluate
from models import encoder
from CMR_encoder import models_vit
from typing import Tuple, Optional, List, Dict
import util.misc as misc
from util.val_result import process_val_result
from util.Val_class import find_optimal_threshold
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import HeNan_ECGBaseDis
from tqdm import tqdm


def str_true_false(x):
    if x.lower() == 'true':
        return True
    elif x.lower() == 'false':
        return False
    else:
        raise ValueError(f'Invalid value for bool flag {x}, should be "true" or "false"')


def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)
    
    # model
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default=None, type=str, help='pretrained weights path')
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--input_modality', default='ECG', type=str, help='ECG or CMR')
    parser.add_argument('--ecg_config_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/configs/Cla/st_mem_align_CLIP_woHenan.yaml', type=str, help='ecg config path')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--ecg_model', default='stmem', type=str, help='model name')
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_Rework_v1/', type=str, help='number of classes')
    parser.add_argument('--test_dir_name', default='test_debug', type=str, help='test dir name')
    
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--dis', default='cad', type=str, help='dis')
    
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--blr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--accum_iter', default=1, type=int, help='accumulation iterations')
    
    # training
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--device', default='cuda:0', type=str,)
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    parser.add_argument('--best_patience', default=10, type=int, help='best patience')
    parser.add_argument('--fold', default=0, type=int, help='fold 0,1,2,3,4')
    parser.add_argument('--only_test', default=False, type=bool, help='only test')
    parser.add_argument('--use_pretrained_CMR', default='False', type=str_true_false, help='use pre-trained CMR model')
    parser.add_argument('--threshold_method', default='youden', type=str, help='threshold method')
    return parser


def build_ecg_model(ecg_config_path):
    
    with open(os.path.realpath(ecg_config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = 1
    model_name = config['model_name']
    if model_name in encoder.__dict__:
        ecg_model = encoder.__dict__[model_name](**config['model'])
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] == "pretrain":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
    elif config['mode'] == "align":
        checkpoint = torch.load(config['afterAlign_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['afterAlign_path']}")
    elif config['mode'] == "scratch":
        print('Training from scratch')
        return ecg_model
    else:
        raise ValueError(f'Unsupported mode: {config["mode"]}')
    
    # load pre-trained weights
    checkpoint_model = checkpoint['model']
    state_dict = ecg_model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Remove key {k} from pre-trained checkpoint")
            del checkpoint_model[k]
    msg = ecg_model.load_state_dict(checkpoint_model, strict=False)
    print(f'Load pre-trained ECG model: {msg}')


    return ecg_model

def _assign_groups_to_folds_greedy(
    group_sizes: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[int, int]:
    """
    Greedy bin packing by group size to make folds as balanced as possible by total rows.
    Returns: dict {group_id -> fold_id}
    """
    rng = np.random.default_rng(seed)

    # group_sizes: index=group_id, value=size (#rows)
    # Sort groups by size desc; shuffle within same size for reproducibility
    df_g = group_sizes.reset_index()
    df_g.columns = ["group", "size"]

    # Shuffle first, then stable sort by size desc (so ties are randomized but deterministic)
    df_g = df_g.sample(frac=1.0, random_state=seed).sort_values("size", ascending=False, kind="mergesort")

    fold_loads = np.zeros(n_splits, dtype=np.int64)
    assignment: Dict[int, int] = {}

    for _, row in df_g.iterrows():
        g = row["group"]
        sz = int(row["size"])
        # assign to fold with minimum current load
        f = int(np.argmin(fold_loads))
        assignment[g] = f
        fold_loads[f] += sz

    return assignment


def split_excel_5fold_to_val_test(
    excel_path: str,
    fold: int,
    seed: int = 42,
    sheet_name: Optional[str] = 0,  # 默认第一个sheet；也可传具体名字
    keep_original_order_in_output: bool = False,  # True则不打乱输出（但分折仍按打乱后的索引）
) -> Tuple[str, str]:
    """
    将一个Excel按行随机划分为尽可能均等的5折；指定fold作为测试集，其余为验证集。
    返回：(val_excel_path, test_excel_path)

    参数
    - excel_path: 总Excel路径
    - fold: 0~4，指定哪一折作为测试集
    - seed: 随机种子，保证可复现
    - sheet_name: pandas.read_excel 的 sheet_name；默认0表示第一个sheet
    - keep_original_order_in_output: 输出是否保持原始行顺序（默认False，输出也按打乱后的顺序）

    说明
    - 默认只处理一个sheet（sheet_name指定的那个）
    - 表头保留；不写入DataFrame索引
    """
    if fold not in [0, 1, 2, 3, 4]:
        raise ValueError(f"fold must be in [0,1,2,3,4], got {fold}")

    excel_path = str(excel_path)
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # 读Excel（单sheet）
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.dropna(subset=['selected_data_HW_crop', 'ECG_data_path'], inplace=True)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Reading Excel returned multiple sheets (dict). "
            "Please pass a single sheet_name (e.g., 0 or 'Sheet1')."
        )

    n = len(df)
    if n == 0:
        raise ValueError("Excel sheet is empty (0 rows).")

    # 随机打乱索引
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=False)  # 保留原始行号用于可选恢复
    # 计算5折大小：前 remainder 折多1行
    k = 5
    base = n // k
    remainder = n % k
    fold_sizes = [base + (1 if i < remainder else 0) for i in range(k)]

    # 计算每折的起止
    boundaries = [0]
    for sz in fold_sizes:
        boundaries.append(boundaries[-1] + sz)

    start, end = boundaries[fold], boundaries[fold + 1]
    test_part = shuffled.iloc[start:end]
    val_part = pd.concat([shuffled.iloc[:start], shuffled.iloc[end:]], axis=0)

    # 输出顺序控制
    if keep_original_order_in_output:
        # 按原始行号排序，恢复到原Excel的行顺序
        test_part = test_part.sort_values("index")
        val_part = val_part.sort_values("index")

    # 去掉辅助列
    test_out = test_part.drop(columns=["index"]).reset_index(drop=True)
    val_out = val_part.drop(columns=["index"]).reset_index(drop=True)

    # 写入临时目录
    tmp_dir = Path(tempfile.mkdtemp(prefix="cv5fold_"))
    stem = Path(excel_path).stem
    test_path = tmp_dir / f"{stem}_fold{fold}_test.xlsx"
    val_path = tmp_dir / f"{stem}_fold{fold}_val.xlsx"

    test_out.to_excel(test_path, index=False, engine="openpyxl")
    val_out.to_excel(val_path, index=False, engine="openpyxl")

    return str(val_path), str(test_path)

def main(args):
    
    
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(args, default_flow_style=False, sort_keys=False))
    
    # reproducibility
    seed = args.seed + misc.get_rank()
    print(f'seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    
    # dataset
    train_path, valid_path = split_excel_5fold_to_val_test(
        excel_path='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/mnt_sda1_HENAN_CMR_all_output_with_Cine_LAX_4ch_process_dicom_parallel_selected_HW_crop_Withdiag_withECG.xlsx',
        fold=args.fold,
        # patient_col="PatientID",
        seed=args.seed,
        sheet_name=0,
        # shuffle_within_fold=True,
    )
    train_set = HeNan_ECGBaseDis(data_excel=train_path,isTrain=True,args=args)
    valid_set = HeNan_ECGBaseDis(data_excel=valid_path,isTrain=False,args=args)
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}")

    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}")
    
    # ECG model input shape (batchsize, 12, 2250)
    if args.input_modality == 'ECG':
        model = build_ecg_model(args.ecg_config_path)
        model.to(args.device)
    elif args.input_modality == 'CMR':
        model = models_vit.__dict__[args.cmr_model](
            drop_path_rate=args.drop_path,
            num_classes=1,
        )
        if args.use_pretrained_CMR:
            checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')
            checkpoint_model = checkpoint['model']
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(f'Load pre-trained CMR model: {msg}')
        else:
            print(f'No pre-trained CMR model, training from scratch')
        model.to(args.device)
    else:
        raise ValueError(f'Unsupported input modality: {args.input_modality}')
    
    # log
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)
        
    
    
    # optimizer

    args.lr = args.blr 
    print(f'learning rate: {args.lr}')
    optimizer = get_optimizer_from_config(args, model)
    use_amp = args.use_amp
    
    
    # loss
    ClaLoss = torch.nn.BCEWithLogitsLoss()
    loss_scaler = NativeScaler()
    best_auc = float('-inf')
    BEST_PATIENCE = args.best_patience
    patient = 0

    
    if not args.only_test:
        # Start training
        misc.load_model(vars(args), model, optimizer, loss_scaler)
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(args.start_epoch, args.epochs):
            
            train_stats = train_one_epoch(model,
                                        ClaLoss,
                                        data_loader_train,
                                        optimizer,
                                        args.device,
                                        epoch,
                                        loss_scaler,
                                        log_writer,
                                        vars(args),
                                        use_amp=use_amp,
                                        args=args,
                                        )

            valid_stats, log_dict, opt_list, tgt_list =  evaluate(model,
                                    ClaLoss,
                                    data_loader_valid,
                                    args.device,
                                    log_writer,
                                    epoch,
                                    use_amp=use_amp,
                                    args=args,
                                    )
            

            test_AUC = valid_stats['AUC'].astype(float)
            curr_AUC = test_AUC
            patient += 1               
            if output_dir and curr_AUC > best_auc:
                
                optimal_threshold = find_optimal_threshold(tgt_list, opt_list, method='youden')
                print(f"Optimal threshold found: {optimal_threshold}")
                best_auc = curr_AUC
                patient = 0
                misc.save_model(vars(args),
                                os.path.join(output_dir, 'best-auc.pth'),
                                epoch,
                                model,
                                optimizer,
                                loss_scaler,
                                metrics={'AUC': curr_AUC,'optimal_threshold': optimal_threshold},)
                
                
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'valid_{k}': v for k, v in valid_stats.items()},
                        'epoch': epoch}

            if output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + '\n\n')

            if patient > BEST_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        
        print(f'best auc: {best_auc}')
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Total training time: {total_time_str}")
        
        print(f'begin testing')
        ckpt = torch.load(os.path.join(output_dir, 'best-auc.pth'))
        msg = model.load_state_dict(ckpt['model'])
        optimal_threshold = ckpt['metrics']['optimal_threshold']
        print(f'Optimal threshold: {optimal_threshold}')
        print(f'load best auc model: {msg}')
        model.eval()
        test_stats, log_dict, opt_list, tgt_list = test_evaluate(model,
                                    ClaLoss,
                                    data_loader_valid,
                                    args.device,
                                    log_writer,
                                    use_amp=args.use_amp,
                                    args=args,
                                    )
        
        
        test_dir = os.path.join(output_dir, args.test_dir_name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        args.metric_save_path = test_dir
        if args.num_classes == 1:
            args.downtask_type = 'BCE'
        else:
            args.downtask_type = 'CE'
            
        process_val_result(tgt_list, opt_list, optimal_threshold, args)
        test_log_stats = {f'test_{k}': v for k, v in test_stats.items()}
        
        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                f.write(json.dumps(test_log_stats) + '\n\n')
        
        
    elif args.only_test:
        print(f"Start testing")
        ckpt = torch.load(os.path.join(output_dir, 'best-auc.pth'))
        msg = model.load_state_dict(ckpt['model'])
        optimal_threshold = ckpt['metrics']['optimal_threshold']
        print(f'Optimal threshold: {optimal_threshold}')
        print(f'load best auc model: {msg}')
        valid_stats, log_dict, opt_list, tgt_list = test_evaluate(model,
                                    ClaLoss,
                                    data_loader_valid,
                                    args.device,
                                    log_writer,
                                    epoch=100,
                                    use_amp=use_amp,
                                    args=args,
                                    )
        
        test_dir = os.path.join(output_dir, args.test_dir_name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        args.metric_save_path = test_dir
        if args.num_classes == 1:
            args.downtask_type = 'BCE'
        else:
            args.downtask_type = 'CE'
            
        process_val_result(tgt_list, opt_list, optimal_threshold, args)
        test_log_stats = {f'test_{k}': v for k, v in test_stats.items()}
        
        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                f.write(json.dumps(test_log_stats) + '\n\n')
        
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
