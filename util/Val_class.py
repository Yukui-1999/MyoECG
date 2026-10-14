import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import argparse
import scipy.stats as stats
import os
from tqdm import tqdm

from sklearn.metrics import roc_curve, precision_recall_curve

def find_optimal_threshold(y_true, y_pred, method='youden'):
    """
    寻找最佳分类阈值
    
    参数:
    y_true -- 真实标签
    y_pred -- 预测概率
    method -- 'youden' (医学常用) 或 'f1' (不平衡数据常用)
    
    返回:
    optimal_threshold -- 最佳阈值
    optimal_point -- 对应的 (FPR, TPR) 或 (Recall, Precision) 以便绘图
    """
    if method == 'youden':
        # 1. 获取 ROC 曲线上的所有点
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        # 2. 计算 Youden Index = TPR - FPR (等价于 Sensitivity + Specificity - 1)
        # 注意：sklean 的 roc_curve 返回的 thresholds[0] 可能是 > 1 的数，需处理
        youden_index = tpr - fpr
        
        # 3. 找到最大值对应的索引
        idx = np.argmax(youden_index)
        optimal_threshold = thresholds[idx]
        
        # 打印信息
        print(f"Best Threshold (Youden): {optimal_threshold:.4f}")
        print(f"At this threshold: Sensitivity = {tpr[idx]:.4f}, Specificity = {1-fpr[idx]:.4f}")
        
        return optimal_threshold

    elif method == 'f1':
        # 1. 获取 PR 曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
        # 2. 计算 F1 Scores
        # 注意: precision 和 recall 的长度比 thresholds 多 1，最后那个是 1.0/0.0
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 3. 找到最大 F1 对应的索引 (忽略最后一个点)
        idx = np.argmax(f1_scores[:-1]) 
        optimal_threshold = thresholds[idx]
        
        print(f"Best Threshold (Max F1): {optimal_threshold:.4f}")
        print(f"At this threshold: F1 = {f1_scores[idx]:.4f}, Recall = {recall[idx]:.4f}, Precision = {precision[idx]:.4f}")
        
        return optimal_threshold
        
    else:
        return 0.5


import numpy as np
from sklearn.metrics import roc_auc_score


def _confusion_counts(y_true, y_prob, threshold=0.5):
    """
    y_true: (N,) {0,1}
    y_prob: (N,) predicted probability
    """
    y_pred = (y_prob >= threshold).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN


def _safe_div(num, den):
    return num / den if den != 0 else np.nan


def _compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Return metrics as floats (may contain np.nan).
    """
    TP, FP, TN, FN = _confusion_counts(y_true, y_prob, threshold=threshold)
    total = TP + FP + TN + FN

    sensitivity = _safe_div(TP, TP + FN)
    specificity = _safe_div(TN, TN + FP)
    accuracy = _safe_div(TP + TN, total)
    ppv = _safe_div(TP, TP + FP)
    npv = _safe_div(TN, TN + FN)

    f1 = _safe_div(2 * TP, (2 * TP + FP + FN))

    # Cohen's kappa
    # pe = expected accuracy by chance
    if total == 0:
        kappa = np.nan
    else:
        pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / (total ** 2)
        kappa = _safe_div((accuracy - pe), (1 - pe)) if (1 - pe) != 0 else np.nan

    # AUC
    # if only one class present, roc_auc_score is undefined -> np.nan
    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "PPV": ppv,
        "NPV": npv,
        "F1-Score": f1,
        "AUC": auc,
        "Kappa": kappa,
    }


def _bootstrap_indices_stratified(y_true, rng):
    """
    Stratified bootstrap: resample positives and negatives separately,
    keeping the same counts as original data.
    """
    y_true = np.asarray(y_true)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    # If either class is empty, fall back to plain bootstrap
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        n = len(y_true)
        return rng.integers(0, n, size=n)

    pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)

    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return idx


def _bootstrap_ci(values, ci=0.95):
    """
    values: (B,) array with possible nans
    percentile CI using nanpercentile
    """
    alpha = 1.0 - ci
    low = np.nanpercentile(values, 100 * (alpha / 2.0))
    high = np.nanpercentile(values, 100 * (1.0 - alpha / 2.0))
    return low, high


def calculate_metrics(
    y_true,
    y_pred,
    threshold=0.5,
    n_bootstrap=1000,
    ci=0.95,
    seed=42,
    stratified=True,
):
    """
    Compute point estimates + bootstrap 95% CI for all metrics.

    Returns:
      dict: metric -> (point, (ci_low, ci_high))
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must have same length"

    rng = np.random.default_rng(seed)

    # Point estimates on full data
    point = _compute_all_metrics(y_true, y_pred, threshold=threshold)

    # Bootstrap distributions
    boot = {k: np.empty(n_bootstrap, dtype=np.float64) for k in point.keys()}

    n = len(y_true)
    for b in range(n_bootstrap):
        if stratified:
            idx = _bootstrap_indices_stratified(y_true, rng)
        else:
            idx = rng.integers(0, n, size=n)

        m = _compute_all_metrics(y_true[idx], y_pred[idx], threshold=threshold)
        for k in boot:
            boot[k][b] = m[k]

    # Build output with CI
    out = {}
    for k, v in point.items():
        ci_low, ci_high = _bootstrap_ci(boot[k], ci=ci)
        out[k] = (v, (ci_low, ci_high))

    return out
