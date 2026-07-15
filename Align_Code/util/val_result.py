import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import argparse
import scipy.stats as stats
import os
from util.Val_mutiClass import calculate_multiclass_metrics
from util.Val_class import calculate_metrics
import csv
from tqdm import tqdm


def save_multiclass_metrics_to_csv(metrics, filename, n_classes, digits=4, include_summary=True):
    """
    metrics: output from calculate_multiclass_metrics(...)
    filename: output csv path
    n_classes: number of classes (K)
    digits: float formatting digits
    include_summary: whether to save macro/weighted/overall if present
    """

    def fmt(val):
        if val is None:
            return ""
        try:
            # handle nan
            if isinstance(val, float) and (val != val):
                return "nan"
        except Exception:
            pass
        if isinstance(val, (int,)):
            return str(val)
        if isinstance(val, (float,)):
            return f"{val:.{digits}f}"
        return str(val)

    def fmt_point_ci(tup):
        """
        tup: (point, (low, high))
        """
        if tup is None:
            return ""
        point, (low, high) = tup
        return f"{fmt(point)} ({fmt(low)}, {fmt(high)})"

    per_class = metrics.get("per_class", {})
    metric_order = ["Sensitivity", "Specificity", "Accuracy", "AUC", "PPV", "NPV", "F1-Score", "Kappa"]

    # 1) columns
    columns = []
    for i in range(n_classes):
        for m in metric_order:
            columns.append(f"{m}_{i}")

    # optional summaries
    summary_sections = []
    if include_summary:
        for sec in ["macro", "weighted", "overall"]:
            if sec in metrics and isinstance(metrics[sec], dict):
                summary_sections.append(sec)

        for sec in summary_sections:
            for m in metric_order:
                # overall 里未必有全部指标，比如只含 Accuracy/Kappa/AUC-*
                columns.append(f"{sec}_{m}")

            # overall 的两个 multiclass AUC 名称不同
            if sec == "overall":
                # 只要存在就加列（不强制）
                for m in ["AUC-macro-ovr", "AUC-weighted-ovr"]:
                    if m in metrics["overall"]:
                        columns.append(f"overall_{m}")

    # 2) single row
    row = []
    for i in range(n_classes):
        cls_key = f"Class_{i}"
        cls_dict = per_class.get(cls_key, {})
        for m in metric_order:
            row.append(fmt_point_ci(cls_dict.get(m)))

    if include_summary:
        for sec in summary_sections:
            sec_dict = metrics.get(sec, {})

            if sec in ["macro", "weighted"]:
                for m in metric_order:
                    row.append(fmt_point_ci(sec_dict.get(m)))
            elif sec == "overall":
                # overall: 先按 metric_order 写（没有就空）
                for m in metric_order:
                    row.append(fmt_point_ci(sec_dict.get(m)) if m in sec_dict else "")
                # 再写两个 multiclass auc（如果有）
                for m in ["AUC-macro-ovr", "AUC-weighted-ovr"]:
                    if m in sec_dict:
                        row.append(fmt_point_ci(sec_dict.get(m)))

    # 3) write
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerow(row)

    print(f"Metrics saved to {filename}")


def save_metrics_to_csv(metrics, filename, digits=4):
    def fmt(val):
        if val is None:
            return ""
        try:
            # handle nan
            if isinstance(val, float) and (val != val):
                return "nan"
        except Exception:
            pass
        if isinstance(val, (int,)):
            return str(val)
        if isinstance(val, (float,)):
            return f"{val:.{digits}f}"
        return str(val)
    # 需要的列名
    columns = ["Sensitivity", "Specificity", "Accuracy", "AUC", "PPV", "NPV", "F1-Score", "Kappa"]

    # 整理数据
    rows = []
    
    # 第一行: 平均值
    rows.append([
        f"{fmt(metrics['Sensitivity'][0])} ({fmt(metrics['Sensitivity'][1][0])}, {fmt(metrics['Sensitivity'][1][1])})",
        f"{fmt(metrics['Specificity'][0])} ({fmt(metrics['Specificity'][1][0])}, {fmt(metrics['Specificity'][1][1])})",
        f"{fmt(metrics['Accuracy'][0])} ({fmt(metrics['Accuracy'][1][0])}, {fmt(metrics['Accuracy'][1][1])})",
        f"{fmt(metrics['AUC'][0])} ({fmt(metrics['AUC'][1][0])}, {fmt(metrics['AUC'][1][1])})",  # AUC的特殊处理
        f"{fmt(metrics['PPV'][0])} ({fmt(metrics['PPV'][1][0])}, {fmt(metrics['PPV'][1][1])})",
        f"{fmt(metrics['NPV'][0])} ({fmt(metrics['NPV'][1][0])}, {fmt(metrics['NPV'][1][1])})",
        f"{fmt(metrics['F1-Score'][0])} ({fmt(metrics['F1-Score'][1][0])}, {fmt(metrics['F1-Score'][1][1])})",
        f"{fmt(metrics['Kappa'][0])} ({fmt(metrics['Kappa'][1][0])}, {fmt(metrics['Kappa'][1][1])})"
    ])

    # 写入CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)  # 写入列名
        writer.writerows(rows)    # 写入数据行

    print(f"Metrics saved to {filename}")


def process_val_result(y_true, y_pred, threshold, args):

    if args.downtask_type == 'BCE':
        # save y_true and y_pred
        np.savez_compressed(os.path.join(args.metric_save_path, "y_true_pred.npz"), y_true=y_true, y_pred=y_pred)
        metrics = calculate_metrics(y_true, y_pred, threshold=threshold)

        save_metrics_to_csv(metrics, os.path.join(args.metric_save_path, f"metrics.csv"))


    elif args.downtask_type == 'CE':
       
        # save y_true and y_pred
        np.savez_compressed(os.path.join(args.metric_save_path, "y_true_pred.npz"), y_true=y_true, y_pred=y_pred)
        metrics = calculate_multiclass_metrics(y_true, y_pred,
            decision_rule="argmax",   # 默认就是 argmax，其实可以不写
            n_bootstrap=1000,
            seed=42,
            stratified=True,)
        save_multiclass_metrics_to_csv(metrics, os.path.join(args.metric_save_path, f"metrics.csv"), y_pred.shape[1]
)

