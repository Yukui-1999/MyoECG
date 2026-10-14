import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize


def _safe_div(num, den):
    return num / den if den != 0 else np.nan


def _bootstrap_ci(values, ci=0.95):
    alpha = 1.0 - ci
    low = np.nanpercentile(values, 100 * (alpha / 2.0))
    high = np.nanpercentile(values, 100 * (1.0 - alpha / 2.0))
    return low, high


def _to_label(y_true):
    """
    Accept:
      - (N,) integer labels
      - (N, K) one-hot / probabilities -> argmax
    """
    y_true = np.asarray(y_true)
    if y_true.ndim == 2:
        return np.argmax(y_true, axis=1).astype(int)
    return y_true.astype(int)


def _bootstrap_indices_stratified_multiclass(y_label, rng):
    """
    Multiclass stratified bootstrap: sample within each class with replacement,
    keep same per-class counts as original.
    """
    y_label = np.asarray(y_label)
    classes = np.unique(y_label)

    # if only one class exists, fall back
    if len(classes) < 2:
        n = len(y_label)
        return rng.integers(0, n, size=n)

    chunks = []
    for c in classes:
        idx_c = np.flatnonzero(y_label == c)
        if len(idx_c) == 0:
            continue
        chunks.append(rng.choice(idx_c, size=len(idx_c), replace=True))

    if len(chunks) == 0:
        n = len(y_label)
        return rng.integers(0, n, size=n)

    idx = np.concatenate(chunks)
    rng.shuffle(idx)
    return idx


def _confusion_counts_ovr(y_true_label, y_pred_pos_mask, cls):
    """
    y_true_label: (N,) integer labels
    y_pred_pos_mask: (N,) boolean predicted positive for class cls in OvR
    """
    y_true_pos = (y_true_label == cls)

    TP = np.sum(y_true_pos & y_pred_pos_mask)
    FP = np.sum((~y_true_pos) & y_pred_pos_mask)
    TN = np.sum((~y_true_pos) & (~y_pred_pos_mask))
    FN = np.sum(y_true_pos & (~y_pred_pos_mask))
    return TP, FP, TN, FN


def _compute_ovr_metrics_for_class(y_true_label, y_pred_prob, cls, decision_rule="argmax", threshold=0.5):
    """
    decision_rule:
      - "argmax": predicted positive if argmax==cls  (recommended for softmax multiclass)
      - "threshold": predicted positive if prob[:,cls] >= threshold
    """
    y_true_label = np.asarray(y_true_label).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)

    if decision_rule == "argmax":
        y_pred_label = np.argmax(y_pred_prob, axis=1)
        y_pred_pos = (y_pred_label == cls)
    elif decision_rule == "threshold":
        y_pred_pos = (y_pred_prob[:, cls] >= threshold)
    else:
        raise ValueError("decision_rule must be 'argmax' or 'threshold'.")

    TP, FP, TN, FN = _confusion_counts_ovr(y_true_label, y_pred_pos, cls)
    total = TP + FP + TN + FN

    sensitivity = _safe_div(TP, TP + FN)
    specificity = _safe_div(TN, TN + FP)
    accuracy = _safe_div(TP + TN, total)
    ppv = _safe_div(TP, TP + FP)
    npv = _safe_div(TN, TN + FN)
    f1 = _safe_div(2 * TP, (2 * TP + FP + FN))

    # OvR Cohen's kappa (binary)
    if total == 0 or np.isnan(accuracy):
        kappa_ovr = np.nan
    else:
        pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / (total ** 2)
        kappa_ovr = _safe_div((accuracy - pe), (1 - pe)) if (1 - pe) != 0 else np.nan

    # OvR AUC
    y_true_bin = (y_true_label == cls).astype(int)
    if len(np.unique(y_true_bin)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true_bin, y_pred_prob[:, cls])

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "PPV": ppv,
        "NPV": npv,
        "F1-Score": f1,
        "AUC": auc,
        "Kappa": kappa_ovr,
        "Support": int(np.sum(y_true_bin == 1)),
    }


def calculate_multiclass_metrics(
    y_true,
    y_pred,
    n_classes=None,
    decision_rule="argmax",
    threshold=0.5,
    n_bootstrap=1000,
    ci=0.95,
    seed=42,
    stratified=True,
    return_macro_weighted=True,
    return_overall=True,
):
    """
    Multiclass OvR metrics + bootstrap CI for each class.

    Inputs:
      y_true: (N,) labels or (N,K) one-hot
      y_pred: (N,K) probabilities (softmax recommended)
      n_classes: optional, inferred from y_pred if None

    Returns:
      {
        "per_class": { "Class_0": {metric: (point,(low,high)), ...}, ... },
        "macro": {...},           # optional
        "weighted": {...},        # optional
        "overall": {...},         # optional (multiclass accuracy/kappa + multiclass AUC if possible)
      }
    """
    y_true_label = _to_label(y_true)
    y_pred = np.asarray(y_pred).astype(float)
    assert y_pred.ndim == 2, "y_pred must be shape (N, K)"
    assert len(y_true_label) == y_pred.shape[0], "y_true and y_pred must have same N"

    if n_classes is None:
        n_classes = y_pred.shape[1]

    rng = np.random.default_rng(seed)

    metric_keys = ["Sensitivity", "Specificity", "Accuracy", "PPV", "NPV", "F1-Score", "AUC", "Kappa"]

    # ---------- point estimates ----------
    per_class_point = {}
    for c in range(n_classes):
        per_class_point[c] = _compute_ovr_metrics_for_class(
            y_true_label, y_pred, c, decision_rule=decision_rule, threshold=threshold
        )

    # macro/weighted point
    def _aggregate_point(agg="macro", y_lbl=None, per_cls=None):
        out = {}
        vals = {k: np.array([per_cls[c][k] for c in range(n_classes)], dtype=float) for k in metric_keys}
        supports = np.array([per_cls[c]["Support"] for c in range(n_classes)], dtype=float)
        if agg == "macro":
            w = None
        else:
            ssum = np.sum(supports)
            w = supports / ssum if ssum > 0 else None

        for k in metric_keys:
            if w is None:
                out[k] = np.nanmean(vals[k])
            else:
                out[k] = np.nansum(vals[k] * w)  # weighted nan-safe enough for typical use
        return out

    macro_point = _aggregate_point("macro", per_cls=per_class_point) if return_macro_weighted else None
    weighted_point = _aggregate_point("weighted", per_cls=per_class_point) if return_macro_weighted else None

    overall_point = {}
    if return_overall:
        y_pred_label = np.argmax(y_pred, axis=1)
        overall_point["Accuracy"] = np.mean(y_pred_label == y_true_label) if len(y_true_label) > 0 else np.nan
        overall_point["Kappa"] = (
            cohen_kappa_score(y_true_label, y_pred_label) if len(np.unique(y_true_label)) > 1 else np.nan
        )
        # multiclass AUC (OVR) if feasible
        try:
            overall_point["AUC-macro-ovr"] = macro_point["AUC"]
        except Exception:
            overall_point["AUC-macro-ovr"] = np.nan
        try:
            overall_point["AUC-weighted-ovr"] = weighted_point["AUC"]
        except Exception:
            overall_point["AUC-weighted-ovr"] = np.nan

    # ---------- bootstrap distributions ----------
    boot_per_class = {
        c: {k: np.empty(n_bootstrap, dtype=np.float64) for k in metric_keys}
        for c in range(n_classes)
    }

    boot_macro = {k: np.empty(n_bootstrap, dtype=np.float64) for k in metric_keys} if return_macro_weighted else None
    boot_weighted = {k: np.empty(n_bootstrap, dtype=np.float64) for k in metric_keys} if return_macro_weighted else None

    boot_overall = None
    if return_overall:
        boot_overall = {
            "Accuracy": np.empty(n_bootstrap, dtype=np.float64),
            "Kappa": np.empty(n_bootstrap, dtype=np.float64),
            "AUC-macro-ovr": np.empty(n_bootstrap, dtype=np.float64),
            "AUC-weighted-ovr": np.empty(n_bootstrap, dtype=np.float64),
        }

    n = len(y_true_label)
    for b in range(n_bootstrap):
        if stratified:
            idx = _bootstrap_indices_stratified_multiclass(y_true_label, rng)
        else:
            idx = rng.integers(0, n, size=n)

        yb = y_true_label[idx]
        pb = y_pred[idx]

        # per class
        per_cls_b = {}
        for c in range(n_classes):
            m = _compute_ovr_metrics_for_class(yb, pb, c, decision_rule=decision_rule, threshold=threshold)
            per_cls_b[c] = m
            for k in metric_keys:
                boot_per_class[c][k][b] = m[k]

        # macro/weighted in this bootstrap sample
        if return_macro_weighted:
            macro_b = _aggregate_point("macro", per_cls=per_cls_b)
            weighted_b = _aggregate_point("weighted", per_cls=per_cls_b)
            for k in metric_keys:
                boot_macro[k][b] = macro_b[k]
                boot_weighted[k][b] = weighted_b[k]

        # overall in this bootstrap sample
        if return_overall:
            y_pred_lbl_b = np.argmax(pb, axis=1)
            boot_overall["Accuracy"][b] = np.mean(y_pred_lbl_b == yb) if len(yb) > 0 else np.nan
            boot_overall["Kappa"][b] = cohen_kappa_score(yb, y_pred_lbl_b) if len(np.unique(yb)) > 1 else np.nan
            try:
                boot_overall["AUC-macro-ovr"][b] = macro_b["AUC"]
            except Exception:
                boot_overall["AUC-macro-ovr"][b] = np.nan
            try:
                boot_overall["AUC-weighted-ovr"][b] = weighted_b["AUC"]
            except Exception:
                boot_overall["AUC-weighted-ovr"][b] = np.nan

    # ---------- pack results with CI ----------
    out = {"per_class": {}}

    for c in range(n_classes):
        cls_name = f"Class_{c}"
        out["per_class"][cls_name] = {}
        for k in metric_keys:
            point_val = per_class_point[c][k]
            ci_low, ci_high = _bootstrap_ci(boot_per_class[c][k], ci=ci)
            out["per_class"][cls_name][k] = (point_val, (ci_low, ci_high))

    if return_macro_weighted:
        out["macro"] = {}
        out["weighted"] = {}
        for k in metric_keys:
            ci_low, ci_high = _bootstrap_ci(boot_macro[k], ci=ci)
            out["macro"][k] = (macro_point[k], (ci_low, ci_high))
            ci_low, ci_high = _bootstrap_ci(boot_weighted[k], ci=ci)
            out["weighted"][k] = (weighted_point[k], (ci_low, ci_high))

    if return_overall:
        out["overall"] = {}
        for k, point_val in overall_point.items():
            ci_low, ci_high = _bootstrap_ci(boot_overall[k], ci=ci)
            out["overall"][k] = (point_val, (ci_low, ci_high))

    return out
