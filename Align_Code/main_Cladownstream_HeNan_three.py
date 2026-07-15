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
from engine_Cal_three_downstream import train_one_epoch, evaluate, test_evaluate
from models import encoder
from CMR_encoder import models_vit
from util.val_result import process_val_result
from util.Val_class import find_optimal_threshold
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import HeNan_ECGBaseDis,ECGzheyi_three_Base,ECGBaseMIMIC_CMthree,ShaoyifuCardiomyopathyDataset
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
    parser.add_argument('--cm_three_dataset', default='Henan', type=str, help='dataset for cm three classification: Henan, Zheyi, MIMIC')
    
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
    parser.add_argument('--cal_popular_index', action='store_true', help='calculate popular index for UKB data')
    parser.add_argument('--record_eid', action='store_true', help='record eid during testing')
    return parser


def build_ecg_model(ecg_config_path):
    
    with open(os.path.realpath(ecg_config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model']['num_classes'] = 3
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
    if args.cm_three_dataset == 'Henan':
        Henan_data_root = '/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/HenanData/seed_3407'
        print(f"Henan_data_root: {Henan_data_root}")
        # dataset
        train_set = HeNan_ECGBaseDis(data_excel=os.path.join(Henan_data_root,'train.xlsx'),isTrain=True,args=args)
        valid_set = HeNan_ECGBaseDis(data_excel=os.path.join(Henan_data_root,'valid.xlsx'),isTrain=False,args=args)
        test_set = HeNan_ECGBaseDis(data_excel=os.path.join(Henan_data_root,'test.xlsx'),isTrain=False,args=args)
        
        print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}, test dataset size: {len(test_set)}")
    elif args.cm_three_dataset == 'Zheyi':
        final_eid = pickle.load(open("/mnt/data2/ECG_CMR/zheyi_data/Final_data/rework/all_data_final_addPath_valid.pkl", 'rb'))
        random.shuffle(final_eid)
        test_set = ECGzheyi_three_Base(data=final_eid, isTrain=False, args=args)
        print(f"test dataset size: {len(test_set)}")
    elif args.cm_three_dataset == 'MIMIC':
        RCM_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_RCM.json')) 
        DCM_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_DCM.json')) 
        HCM_eid = json.load(open('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/MIMIC_process/mimic_json/mimic_data_path_HCM.json'))
        final_eid = RCM_eid + DCM_eid + HCM_eid
        label = [0] * len(RCM_eid) + [1] * len(DCM_eid) + [2] * len(HCM_eid)
        final_eid = list(zip(final_eid, label))
        random.shuffle(final_eid)
        test_set = ECGBaseMIMIC_CMthree(data=final_eid,isTrain=False,args=args)
        print(f"test dataset size: {len(test_set)}")
    elif args.cm_three_dataset == 'Shaoyifu':
        test_set = ShaoyifuCardiomyopathyDataset(table_path='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/data/ShaoyifuData/FirstWholeSummary0330/extval_three_class_patient_unique_with_quality.csv',
                                            task='three_class', isTrain=False, args=args)
        print(f"test dataset size: {len(test_set)}")

    if args.cm_three_dataset == 'Henan':
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
        data_loader_test = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
        )
        print(f"Train dataset size: {len(train_set)}, valid dataset size: {len(valid_set)}, test dataset size: {len(test_set)}")
    else:
        data_loader_test = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
        )
        print(f"test dataset size: {len(test_set)}")
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
    ClaLoss = torch.nn.CrossEntropyLoss()
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
                
                
                best_auc = curr_AUC
                patient = 0
                misc.save_model(vars(args),
                                os.path.join(output_dir, 'best-auc.pth'),
                                epoch,
                                model,
                                optimizer,
                                loss_scaler,
                                metrics={'AUC': curr_AUC},)
                
                
        
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
        
        print(f'load best auc model: {msg}')
        model.eval()
        test_stats, log_dict, opt_list, tgt_list = test_evaluate(model,
                                    ClaLoss,
                                    data_loader_test,
                                    args.device,
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
            
        process_val_result(tgt_list, opt_list, None, args)
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
        
        print(f'load best auc model: {msg}')
        test_stats, log_dict, opt_list, tgt_list = test_evaluate(model,
                                    ClaLoss,
                                    data_loader_test,
                                    args.device,
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
            
        process_val_result(tgt_list, opt_list, None, args)
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
