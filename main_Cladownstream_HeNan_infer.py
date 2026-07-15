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
import pickle
from engine_Caldownstream import train_one_epoch, evaluate, test_evaluate
from models import encoder
from CMR_encoder import models_vit
from util.val_result import process_val_result
from util.Val_class import find_optimal_threshold
import util.misc as misc
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
    parser.add_argument('--cmr_pretrained_weights', default='/mnt/sda1/liziyu/CMRMAR/output/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth', type=str, help='pretrained weights path')
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--input_modality', default='ECG', type=str, help='ECG or CMR')
    parser.add_argument('--ecg_config_path', default='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/configs/Cla/st_mem_align_CLIP_Bin.yaml', type=str, help='ecg config path')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--ecg_model', default='stmem', type=str, help='model name')
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_Rework_v1/', type=str, help='number of classes')
    parser.add_argument('--test_dir_name', default='test', type=str, help='test dir name')
    
    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--dis', default='cm', type=str, help='dis')
    parser.add_argument('--health_magnification', default=1, type=int, help='health magnification')
    
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
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--use_pretrained_CMR', default='False', type=str_true_false, help='use pre-trained CMR model')
    parser.add_argument('--threshold_method', default='youden', type=str, help='threshold method')
    parser.add_argument('--record_eid', action='store_true', help='record eid during testing')
    parser.add_argument('--renji', action='store_true', help='record eid during testing')
    parser.add_argument('--cal_popular_index', action='store_true', help='calculate popular index for UKB data')
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
    
    if args.renji:
        all_set = HeNan_ECGBaseDis(data_excel='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407/test.xlsx',isTrain=False,args=args)
    else:
        all_set = HeNan_ECGBaseDis(data_excel='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/all_set.xlsx',isTrain=False,args=args)
        
    
   
    # dataloader
    
    data_loader_all = torch.utils.data.DataLoader(
        all_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    print(f"All dataset size: {len(all_set)}")
    
    # ECG model input shape (batchsize, 12, 2250)
    if args.input_modality == 'ECG':
        if args.ecg_model == 'stmem':
            model = build_ecg_model(args.ecg_config_path)
            model.to(args.device)
        elif args.ecg_model == 'ecg_found':
            from ECG_baselines.ECGFounder.finetune_model import ft_12lead_ECGFounder
            model = ft_12lead_ECGFounder(args.device, 
            '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/ECG_baselines/ECGFounder/checkpoint/12_lead_ECGFounder.pth',
            n_classes=1)
        elif args.ecg_model == 'ecgfm_ked':
            from ECG_baselines.ECGFM_KED.models.xresnet1d_101 import xresnet1d101
            checkpoint = torch.load('/mnt/sda1/dingzhengyao/Work/ECG_CMR_CMAI/ECG_baselines/ECGFM-KED/best_valid_all_increase_with_augment_epoch_3.pt',
                                    map_location='cpu')
            ecg_model_state_dict = checkpoint['ecg_model']
            model = xresnet1d101(num_classes=1, input_channels=12, kernel_size=5,
                                ps_head=0.5, lin_ftrs_head=[768],
                                use_ecgNet_Diagnosis='ecgNet'
                                )
            msg = model.load_state_dict(ecg_model_state_dict, strict=False)
            print(msg)
            model.to(args.device)
        elif args.ecg_model == 'fg_clep':
            from ECG_baselines.FG_CLEP.clep.modeling_clep import ECGModel
            model = ECGModel(output_class_num=1, encoder='resnet50',
                     clep_checkpoint='/mnt/sda1/dingzhengyao/Work/ECG_CMR_CMAI/ECG_baselines/FGCLEP')
            model.to(args.device)
        elif args.ecg_model == 'merl':
            from ECG_baselines.MERL_ICML2024.finetune.models.vit1d import vit_tiny
            model = vit_tiny(num_leads=12, num_classes=1, seq_len=5000, patch_size=50)
            msg = model.load_state_dict(torch.load('/mnt/sda1/dingzhengyao/Work/ECG_CMR_CMAI/ECG_baselines/MERL-ICML2024/vit_tiny_best_encoder.pth'), strict=False)
            print(msg)
            model.to(args.device)
        else:
            raise ValueError(f'Unsupported ECG model: {args.ecg_model}')
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
        pass
        
        
    elif args.only_test:
        print(f"Start testing")
        ckpt = torch.load(os.path.join(output_dir, 'best-auc.pth'))
        msg = model.load_state_dict(ckpt['model'])
        optimal_threshold = ckpt['metrics']['optimal_threshold']
        print(f'Optimal threshold: {optimal_threshold}')
        print(f'load best auc model: {msg}')
        if args.test_dir_name != 'test':
            data_loader_test = data_loader_all
        test_stats, log_dict, opt_list, tgt_list = test_evaluate(model,
                                    ClaLoss,
                                    data_loader_test,
                                    args.device,
                                    log_writer,
                                    use_amp=use_amp,
                                    args=args,
                                    throshold=optimal_threshold
                                    )
        
        test_dir = os.path.join(output_dir, args.test_dir_name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if args.record_eid:
            # print(f'log_dict:{log_dict}')
            # print(f'opt_list:{opt_list}')
            # print(f'tgt_list:{tgt_list}')
            df = pd.DataFrame({
                'eid': log_dict,
                'prob': opt_list.squeeze(),
                'label': tgt_list.squeeze()
            })
            df.to_excel(os.path.join(test_dir, f'Eid_prob.xlsx'), index=False)
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
