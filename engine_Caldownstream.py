# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Dict, Iterable, Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix
import util.misc as misc
import util.lr_sched as lr_sched

def safe_divide(a, b, default=np.nan):
    return a / b if b != 0 else default

def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config: Optional[dict] = None,
                    use_amp: bool = True,
                    args=None,
                    ) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config.get('accum_iter', 1)
    max_norm = config.get('max_norm', None)

    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (ecg, cmr, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        if args.input_modality == 'ECG':
            samples = ecg
        elif args.input_modality == 'CMR':
            samples = cmr
        else:
            raise ValueError(f"Unsupported input modality: {args.input_modality}")
        samples = samples.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float().unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=use_amp):
            
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import numpy as np
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             log_writer=None,
             epoch: int = 0,
             use_amp: bool = True,
             args=None,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'valid:'
    
    tgt_list = np.array([])
    opt_list = np.array([])
    
    for ecg, cmr, target in metric_logger.log_every(data_loader, 10, header):
        if args.input_modality == 'ECG':
            sample = ecg
        elif args.input_modality == 'CMR':
            sample = cmr
        else:
            raise ValueError(f"Unsupported input modality: {args.input_modality}")
        sample = sample.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().unsqueeze(1)
        # print(sample.shape)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if sample.ndim == 4 and not 'CMRmode' in args.output_dir:  # batch_size, n_drops, n_channels, n_frames
                    logits_list = []
                    for i in range(sample.size(1)):
                        logits = model(sample[:, i])
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
            else:
                output = model(sample)
            loss = criterion(output, target)
        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output).cpu().numpy()
            tgt_list = target.cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output).cpu().numpy()])
            tgt_list = np.concatenate([tgt_list, target.cpu().numpy()])
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    
    pred_labels = (opt_list.squeeze() > 0.5).astype(int)  # 确保是1D
    true_labels = tgt_list.squeeze().astype(int)  
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sens = tp / (tp + fn)
    Spec = tn / (tn + fp)
    PPV = safe_divide(tp, tp + fp, default=0.0)   # 未预测正类时返回0
    NPV = safe_divide(tn, tn + fn, default=0.0)   # 未预测负类时返回0
    F1 = 2 * tp / (2 * tp + fp + fn)
    AUC = roc_auc_score(tgt_list, opt_list)
    metric_logger.synchronize_between_processes()
    print('* loss@all {losses.global_avg:.3f}, Acc {Acc:.3f}, Sens {Sens:.3f}, Spec {Spec:.3f}, PPV {PPV:.3f}, NPV {NPV:.3f}, F1 {F1:.3f}, AUC {AUC:.3f}'.format(
        losses=metric_logger.loss,
        Acc=Acc,
        Sens=Sens,
        Spec=Spec,
        PPV=PPV,
        NPV=NPV,
        F1=F1,
        AUC=AUC
    ))
    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['Acc'] = Acc
    test_state['Sens'] = Sens
    test_state['Spec'] = Spec
    test_state['PPV'] = PPV
    test_state['NPV'] = NPV
    test_state['F1'] = F1
    test_state['AUC'] = AUC
    
    if log_writer is not None:
        epoch_1000x = int((epoch) * 1000)
        log_writer.add_scalar('valid_loss', metric_logger.loss.global_avg, epoch_1000x)
        log_writer.add_scalar('Acc', Acc, epoch_1000x)
        log_writer.add_scalar('Sens', Sens, epoch_1000x)
        log_writer.add_scalar('Spec', Spec, epoch_1000x)
        log_writer.add_scalar('PPV', PPV, epoch_1000x)
        log_writer.add_scalar('NPV', NPV, epoch_1000x)
        log_writer.add_scalar('F1', F1, epoch_1000x)
        log_writer.add_scalar('AUC', AUC, epoch_1000x)

    log_dict = {}
    log_dict['Acc'] = Acc
    log_dict['Sens'] = Sens
    log_dict['Spec'] = Spec
    log_dict['PPV'] = PPV
    log_dict['NPV'] = NPV
    log_dict['F1'] = F1
    log_dict['AUC'] = AUC
            
            
            
            
    return test_state, log_dict, opt_list, tgt_list






import numpy as np
@torch.no_grad()
def test_evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             log_writer=None,
             use_amp: bool = True,
             args=None,
             throshold=0.5
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    tgt_list = np.array([])
    opt_list = np.array([])
    eid_list = []

    for ecg, cmr, target in metric_logger.log_every(data_loader, 10, header):
        if args.input_modality == 'ECG':
            sample = ecg
        elif args.input_modality == 'CMR':
            sample = cmr
        else:
            raise ValueError(f"Unsupported input modality: {args.input_modality}")
        if args.record_eid:
            eid = cmr #此时cmr存放eid
            eid_list.extend(eid.numpy().tolist())

        sample = sample.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().unsqueeze(1)
        # print(f'sample shape: {sample.shape}, target shape: {target.shape}')
        with torch.cuda.amp.autocast(enabled=use_amp):
            # print(f'sample shape: {sample.shape}, target shape: {target.shape}')
            if sample.ndim == 4 and not 'CMRmode' in args.output_dir:  # batch_size, n_drops, n_channels, n_frames
                    logits_list = []
                    for i in range(sample.size(1)):
                        logits = model(sample[:, i])
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
            else:
                output = model(sample)
            loss = criterion(output, target)
        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output).cpu().numpy()
            tgt_list = target.cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output).cpu().numpy()])
            tgt_list = np.concatenate([tgt_list, target.cpu().numpy()])
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    
    pred_labels = (opt_list.squeeze() > throshold).astype(int)  # 确保是1D
    true_labels = tgt_list.squeeze().astype(int)  
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sens = tp / (tp + fn)
    Spec = tn / (tn + fp)
    PPV = safe_divide(tp, tp + fp, default=0.0)   # 未预测正类时返回0
    NPV = safe_divide(tn, tn + fn, default=0.0)   # 未预测负类时返回0
    F1 = 2 * tp / (2 * tp + fp + fn)
    AUC = roc_auc_score(tgt_list, opt_list)
    metric_logger.synchronize_between_processes()
    print('* loss@all {losses.global_avg:.3f}, Acc {Acc:.3f}, Sens {Sens:.3f}, Spec {Spec:.3f}, PPV {PPV:.3f}, NPV {NPV:.3f}, F1 {F1:.3f}, AUC {AUC:.3f}'.format(
        losses=metric_logger.loss,
        Acc=Acc,
        Sens=Sens,
        Spec=Spec,
        PPV=PPV,
        NPV=NPV,
        F1=F1,
        AUC=AUC
    ))
    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['Acc'] = Acc
    test_state['Sens'] = Sens
    test_state['Spec'] = Spec
    test_state['PPV'] = PPV
    test_state['NPV'] = NPV
    test_state['F1'] = F1
    test_state['AUC'] = AUC
    
    if log_writer is not None:
        epoch_1000x = 0
        log_writer.add_scalar('test_loss', metric_logger.loss.global_avg, epoch_1000x)
        log_writer.add_scalar('test_Acc', Acc, epoch_1000x)
        log_writer.add_scalar('test_Sens', Sens, epoch_1000x)
        log_writer.add_scalar('test_Spec', Spec, epoch_1000x)
        log_writer.add_scalar('test_PPV', PPV, epoch_1000x)
        log_writer.add_scalar('test_NPV', NPV, epoch_1000x)
        log_writer.add_scalar('test_F1', F1, epoch_1000x)
        log_writer.add_scalar('test_AUC', AUC, epoch_1000x)

    
    if args.record_eid:
        log_dict = eid_list
    else:
        log_dict = {}
        log_dict['Acc'] = Acc
        log_dict['Sens'] = Sens
        log_dict['Spec'] = Spec
        log_dict['PPV'] = PPV
        log_dict['NPV'] = NPV
        log_dict['F1'] = F1
        log_dict['AUC'] = AUC
    
            
            
            
            
    return test_state, log_dict, opt_list, tgt_list


def get_positive_score(model, sample):
    """
    sample:
        [B, 3, 12, 2250]  多crop
        或 [B, 12, 2250] 单输入
    return:
        score_per_sample: [B]
    """
    if sample.ndim == 4:  # [B, 3, 12, 2250]
        logits_list = []
        for i in range(sample.size(1)):
            out = model(sample[:, i])   # [B,1] or [B,2] or [B]
            logits_list.append(out)
        out = torch.stack(logits_list, dim=1).mean(dim=1)
    else:
        out = model(sample)

    # BCE单logit
    if out.ndim == 1:
        score = out
    elif out.shape[-1] == 1:
        score = out.view(-1)
    # CE双logit，默认第1类是阳性
    elif out.shape[-1] == 2:
        score = out[:, 1]
    else:
        raise ValueError(f"Unexpected model output shape: {out.shape}")

    return score

def smoothgrad_saliency(model, sample, n_samples=16, noise_std=0.05):
    """
    sample: [B, 3, 12, 2250] or [B, 12, 2250]
    return: same shape as sample
    """
    model.eval()
    sample = sample.float()

    grad_accum = torch.zeros_like(sample)

    for _ in range(n_samples):
        noise = torch.randn_like(sample) * noise_std
        noisy_sample = (sample + noise).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        score = get_positive_score(model, noisy_sample).sum()

        grads = torch.autograd.grad(
            outputs=score,
            inputs=noisy_sample,
            retain_graph=False,
            create_graph=False
        )[0]

        grad_accum += grads.abs()

    saliency = grad_accum / n_samples
    return saliency

def integrated_gradients_saliency(model, sample, baseline=None, steps=32):
    """
    sample: [B, 3, 12, 2250] or [B, 12, 2250]
    return: same shape as sample
    """
    model.eval()
    sample = sample.float()

    if baseline is None:
        baseline = torch.zeros_like(sample)

    alphas = torch.linspace(0.0, 1.0, steps, device=sample.device)
    grad_sum = torch.zeros_like(sample)

    for alpha in alphas:
        x = (baseline + alpha * (sample - baseline)).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        score = get_positive_score(model, x).sum()

        grads = torch.autograd.grad(
            outputs=score,
            inputs=x,
            retain_graph=False,
            create_graph=False
        )[0]

        grad_sum += grads

    avg_grad = grad_sum / steps
    ig = (sample - baseline) * avg_grad
    saliency = ig.abs()
    return saliency

def test_evaluate_saliency(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             log_writer=None,
             use_amp: bool = True,
             args=None,
             throshold=0.5
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    tgt_list = np.array([])
    opt_list = np.array([])
    eid_list = []
    sample_list = []
    saliency_maps = []
    for ecg, cmr, target in metric_logger.log_every(data_loader, 10, header):
        if args.input_modality == 'ECG':
            sample = ecg
        elif args.input_modality == 'CMR':
            sample = cmr
        else:
            raise ValueError(f"Unsupported input modality: {args.input_modality}")
        if args.record_eid:
            eid = cmr #此时cmr存放eid
            eid_list.extend(eid.numpy().tolist())
        sample_list.append(sample)
        sample = sample.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float().unsqueeze(1)
        # print(f'sample shape: {sample.shape}, target shape: {target.shape}')
        with torch.cuda.amp.autocast(enabled=False):
            sample.requires_grad_()
            saliency = []
            if sample.ndim == 4 and not 'CMRmode' in args.output_dir:  # batch_size, n_drops, n_channels, n_frames
                    logits_list = []
                    for i in range(sample.size(1)):
                        logits = model(sample[:, i])
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
            else:
                output = model(sample)
            loss = criterion(output, target)
            model.zero_grad()
            if sample.grad is not None:
                sample.grad.zero_()
            loss.backward()
            if args.saliency_mode == 'smoothgrad':
                saliency = smoothgrad_saliency(model, sample, n_samples=16, noise_std=0.05)
            elif args.saliency_mode == 'integrated_gradients':
                saliency = integrated_gradients_saliency(model, sample, baseline=None, steps=32)
            elif args.saliency_mode == 'scratch':
                saliency = sample.grad.data.abs()
            else:
                raise ValueError()
            
            saliency_maps.append(saliency.detach().cpu().numpy())
        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output).detach().cpu().numpy()
            tgt_list = target.detach().cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output).detach().cpu().numpy()])
            tgt_list = np.concatenate([tgt_list, target.detach().cpu().numpy()])
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
    
    pred_labels = (opt_list.squeeze() > throshold).astype(int)  # 确保是1D
    true_labels = tgt_list.squeeze().astype(int)  
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sens = tp / (tp + fn)
    Spec = tn / (tn + fp)
    PPV = safe_divide(tp, tp + fp, default=0.0)   # 未预测正类时返回0
    NPV = safe_divide(tn, tn + fn, default=0.0)   # 未预测负类时返回0
    F1 = 2 * tp / (2 * tp + fp + fn)
    AUC = roc_auc_score(tgt_list, opt_list)
    metric_logger.synchronize_between_processes()
    print('* loss@all {losses.global_avg:.3f}, Acc {Acc:.3f}, Sens {Sens:.3f}, Spec {Spec:.3f}, PPV {PPV:.3f}, NPV {NPV:.3f}, F1 {F1:.3f}, AUC {AUC:.3f}'.format(
        losses=metric_logger.loss,
        Acc=Acc,
        Sens=Sens,
        Spec=Spec,
        PPV=PPV,
        NPV=NPV,
        F1=F1,
        AUC=AUC
    ))
    
            
    saliency_maps = np.concatenate(saliency_maps, axis=0) # [n, 3, 12, 2250]
    sample_list = np.concatenate(sample_list, axis=0)
    sample_ids = np.array(eid_list)
    print('saliency_maps shape:', saliency_maps.shape)
    print('sample_list shape:', sample_list.shape)
    print('sample_ids shape:', sample_ids.shape)
    args.saliency_dir = os.path.join(args.saliency_dir, args.saliency_mode)
    if not os.path.exists(args.saliency_dir):
        os.makedirs(args.saliency_dir)
    # np.save(os.path.join(args.saliency_dir, 'sample_list.npy'), sample_list)
    merge_saliency_maps_crops(ECG_length=args.ECG_length, norm=False, saliency=saliency_maps, sample_ids=sample_ids,
                              saliency_dir=args.saliency_dir)
            
    return saliency_maps, sample_ids
import os
import re
def merge_saliency_maps_crops(ECG_length=10000, norm=True, saliency=None, sample_ids=None,
                              saliency_dir='/home/dingzhengyao/Work/ECG_CMR_TAR/ECG_CMR_Rework/ECG_CMR_CMAI/Explain/results/Bin_saliency_maps'):
    
    # saliency_maps: [n, 3, 12, 2250]
    num_segments, num_leads, crop_length = 3, 12, 2250

    if ECG_length < crop_length:
        raise ValueError(f"ECG_length ({ECG_length}) must be >= crop_length ({crop_length}).")

    step = (ECG_length - crop_length) // (num_segments - 1)
    start_idx = np.arange(
        start=0,
        stop=ECG_length - crop_length + 1,
        step=step
    )

    if len(start_idx) != num_segments:
        raise ValueError(
            f"Expected {num_segments} crop start positions, but got {len(start_idx)}. "
            f"Please check ECG_length={ECG_length} and crop_length={crop_length}."
        )

    print("start_idx:", start_idx)

    
    x = np.zeros((len(saliency), num_leads, ECG_length), dtype=np.float32)

    # map cropped ECG saliency back to the original ECG length
    for i, idx in enumerate(start_idx):
        x[:, :, idx:idx + crop_length] = np.maximum(
            x[:, :, idx:idx + crop_length],
            saliency[:, i]
        )

    # min-max normalize the saliency to [0, 1]
    if norm:
        for lead_idx in range(num_leads):
            min_x = np.min(x[:, lead_idx], axis=1, keepdims=True)
            max_x = np.max(x[:, lead_idx], axis=1, keepdims=True)
            x[:, lead_idx] = (x[:, lead_idx] - min_x) / (max_x - min_x + 1e-9)

    saliency_maps = x

    if norm:
        np.save(os.path.join(saliency_dir, 'saliency_map_original_normalized.npy'), saliency_maps)
        np.save(os.path.join(saliency_dir, 'saliency_map_original_example_normalized.npy'), saliency_maps[0:100])
    else:
        np.save(os.path.join(saliency_dir, 'saliency_map_original.npy'), saliency_maps)
        np.save(os.path.join(saliency_dir, 'saliency_map_original_example.npy'), saliency_maps[0:100])
    np.save(os.path.join(saliency_dir, 'sample_ids.npy'), sample_ids)
    return saliency_maps


import numpy as np
@torch.no_grad()
def test_infer(model: torch.nn.Module,   
             data_loader: Iterable,
             device: torch.device,
             use_amp: bool = True,
             args=None,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    opt_list = np.array([])
    eid_list = []
    for ecg, cmr, eid in metric_logger.log_every(data_loader, 10, header):
        if args.input_modality == 'ECG':
            sample = ecg
        elif args.input_modality == 'CMR':
            sample = cmr
        else:
            raise ValueError(f"Unsupported input modality: {args.input_modality}")
        sample = sample.to(device, non_blocking=True).float()

        # print(f'sample shape: {sample.shape}, target shape: {target.shape}')
        with torch.cuda.amp.autocast(enabled=use_amp):
            if sample.ndim == 4 and not 'CMRmode' in args.output_dir:  # batch_size, n_drops, n_channels, n_frames
                    logits_list = []
                    for i in range(sample.size(1)):
                        logits = model(sample[:, i])
                        logits_list.append(logits)
                    logits_list = torch.stack(logits_list, dim=1)
                    output = logits_list.mean(dim=1)
            else:
                output = model(sample)

        if len(opt_list) == 0:
            opt_list = torch.sigmoid(output).cpu().numpy()
        else:
            opt_list = np.concatenate([opt_list, torch.sigmoid(output).cpu().numpy()])
        
        eid_list.append(eid)
            
    return opt_list, eid_list
