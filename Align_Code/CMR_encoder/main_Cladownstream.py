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

from engine_Caldownstream import train_one_epoch, evaluate

import models_vit

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from data.dataset import SaxCMRBaseDis, Lax4chCMRBaseDis, Lax4chCMRBaseDis_ZiyuData




def get_args_parser():
    parser = argparse.ArgumentParser('Classification for downstramtask', add_help=False)

    # model
    parser.add_argument('--cmr_model', default='vit_base_patch16', type=str, help='model name')
    parser.add_argument('--cmr_pretrained_weights', default=None, type=str, help='pretrained weights path')
    parser.add_argument('--drop_path', default=0, type=float, help='drop path rate')
    parser.add_argument('--input_modality', default='ECG', type=str, help='ECG or CMR')

    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    # log
    parser.add_argument('--output_dir', default='/mnt/sda1/dingzhengyao/Work/ECG_CMR_Rework_v1/', type=str, help='number of classes')
    parser.add_argument('--test_dir_name', default='test', type=str, help='test dir name')

    # data
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--drop_last', default=False, type=bool, help='drop last batch')
    parser.add_argument('--dis', default='cad', type=str, help='dis')
    parser.add_argument('--health_magnification', default=1, type=int, help='health magnification')
    parser.add_argument('--useint8', action='store_true', )
    parser.add_argument('--cmr_type', default='SingleSaxCMR', type=str, help='cmr type' )

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--blr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--accum_iter', default=1, type=int, help='accumulation iterations')
    parser.add_argument('--max_norm', default=None, type=int, help='max_norm')
    # training
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--device', default='cuda:0', type=str, )
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
    parser.add_argument('--use_amp', default=True, type=bool, help='use amp for training')
    parser.add_argument('--best_patience', default=10, type=int, help='best patience')
    parser.add_argument('--fold', default=0, type=int, help='fold 0,1,2,3,4')
    parser.add_argument('--only_test', default=False, type=bool, help='only test')
    return parser



def cross_validation_split(eid, k, fold):
    n = len(eid)
    k = 5
    split_size = n // k
    remainder = n % k
    split_eid = []
    start = 0
    for i in range(k):
        end = start + split_size + (1 if i < remainder else 0)
        split_eid.append(eid[start:end])
        start = end
    # 检查结果
    for i, part in enumerate(split_eid):
        print(f"Part {i + 1}: {len(part)} samples, eid[0]:{part[0][0]}, eid[-1]:{part[-1][0]}")

    train = []
    valid = []
    for i in range(k):
        if i == fold:
            valid = split_eid[i]
        else:
            train += split_eid[i]
    print(f"Train size: {len(train)}, valid size: {len(valid)}")
    return train, valid


def set_true_false_eid(eid, dis, health_magnification):
    health_eid = json.load(open('/home/liziyu/CMR/data/cmr_files_wo_I.json', "r"))
    print(f'health_eid: {len(health_eid)}')
    print(f'health_eid: {health_eid[:5]}')
    eid = [str(i) for i in eid]

    if dis == 'cad':
        print(f'dis: {dis}')
        dis_eid = json.load(open('/home/liziyu/CMR/data/cad_v2.json', "r"))
        dis_eid = [i.split('_')[0] for i in dis_eid]
    elif dis == 'cm':
        print(f'dis: {dis}')
        dis_eid = json.load(open('/home/liziyu/CMR/data/xjb_v2.json', "r"))
        dis_eid = [i.split('_')[0] for i in dis_eid]
    elif dis == 'hf':
        print(f'dis: {dis}')
        dis_eid = json.load(open('/home/liziyu/CMR/data/xs_v2.json', "r"))
        dis_eid = [i.split('_')[0] for i in dis_eid]
    print(f'dis_eid: {dis_eid[:5]}')
    dis_eid = sorted(set(dis_eid) & set(eid))  # sorted is important 因为不受种子控制
    health_eid = sorted(set(health_eid) & set(eid))

    assert len(set(dis_eid) & set(health_eid)) == 0, f"Dis EID and Health EID overlap: {len(set(dis_eid) & set(health_eid))}"
    if health_magnification > 0:
        max_size = min(len(dis_eid) * health_magnification, len(health_eid))
        if max_size == len(health_eid):
            print(f'----------------------------------------------')
            print(f"Health EID size * health_magnification is greater than the original size")
            print(f'----------------------------------------------')
        health_eid = np.random.choice(health_eid, size=max_size, replace=False).tolist()
    print(f'----------------------------------------------')
    print(f"Dis {dis}: {len(dis_eid)}, Health EID: {len(health_eid)}")
    print(f'----------------------------------------------')

    # extact_info(dis_eid, pd.read_csv('/mnt/data/ukb_heartmri/all_select_v5.csv'), pd.read_csv('/mnt/data2/ECG_CMR/UKB_edu_work.csv'), os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/people_popular_index/ukb', f'{dis}_positive.csv'))
    # extact_info(health_eid, pd.read_csv('/mnt/data/ukb_heartmri/all_select_v5.csv'), pd.read_csv('/mnt/data2/ECG_CMR/UKB_edu_work.csv'), os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/people_popular_index/ukb', f'{dis}_nagetive.csv'))

    final_eid = dis_eid + health_eid
    # extact_info(final_eid, pd.read_csv('/mnt/data/ukb_heartmri/all_select_v5.csv'), pd.read_csv('/mnt/data2/ECG_CMR/UKB_edu_work.csv'), os.path.join('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CardioNets_v1/data/people_popular_index/ukb', f'{dis}_positive_nagetive.csv'))
    label = [1] * len(dis_eid) + [0] * len(health_eid)
    final_eid = list(zip(final_eid, label))
    random.shuffle(final_eid)

    return final_eid


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


    eid = pd.read_csv('/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_align_rework/Align_Code/data/UKB/cmr_data_addphen.csv')['Eid'].tolist()
    k = 5
    mix_eid = set_true_false_eid(eid, args.dis, args.health_magnification)

    train, valid = cross_validation_split(mix_eid, k, args.fold)
    if args.cmr_type == 'SingleSaxCMR':
        train_set = SaxCMRBaseDis(data=train,args=args)
        valid_set = SaxCMRBaseDis(data=valid,args=args)
    elif args.cmr_type == 'Lax4chCMR':
        train_set = Lax4chCMRBaseDis(data=train,args=args)
        valid_set = Lax4chCMRBaseDis(data=valid,args=args)
    elif args.cmr_type == 'Lax4chCMR_ziyu':
        train_set = Lax4chCMRBaseDis_ZiyuData(data=train,args=args)
        valid_set = Lax4chCMRBaseDis_ZiyuData(data=valid,args=args)

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

    model = models_vit.__dict__[args.cmr_model](
        drop_path_rate=args.drop_path,
        num_classes=1,
    )
    if args.cmr_pretrained_weights:
        checkpoint = torch.load(args.cmr_pretrained_weights, map_location='cpu')

        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f'Load pre-trained CMR model: {msg}')
    else:
        print(f'No pre-trained CMR model, training from scratch')
    model.to(args.device)

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
                                          use_amp=use_amp,
                                          args=args,
                                          )

            valid_stats, log_dict, opt_list, tgt_list = evaluate(model,
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
                                metrics={'AUC': curr_AUC})

                log_dict_csv = pd.DataFrame(log_dict, index=[0])
                test_dir = os.path.join(output_dir, args.test_dir_name)
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
                log_dict_csv.to_csv(os.path.join(test_dir, f'log_dict.csv'), index=False)
                np.save(os.path.join(test_dir, f'opt_list.npy'), opt_list)
                np.save(os.path.join(test_dir, f'tgt_list.npy'), tgt_list)

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


    elif args.only_test:
        print(f"Start testing")
        msg = model.load_state_dict(torch.load(os.path.join(output_dir, 'best-auc.pth'))['model'])
        print(f'Load pre-trained ECG model: {msg}')
        valid_stats, log_dict, opt_list, tgt_list = evaluate(model,
                                                             ClaLoss,
                                                             data_loader_valid,
                                                             args.device,
                                                             log_writer,
                                                             epoch=100,
                                                             use_amp=use_amp,
                                                             args=args,
                                                             )

        log_dict_csv = pd.DataFrame(log_dict, index=[0])
        test_dir = os.path.join(output_dir, args.test_dir_name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        log_dict_csv.to_csv(os.path.join(test_dir, f'log_dict.csv'), index=False)
        np.save(os.path.join(test_dir, f'opt_list.npy'), opt_list)
        np.save(os.path.join(test_dir, f'tgt_list.npy'), tgt_list)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
