"""
Train and evaluate a credit card fraud detection model.

Workflow:
1. Read data -> split into train/validation/holdout, with holdout used only once.
2. Handle class imbalance through LightGBM's scale_pos_weight.
3. Train LightGBM and choose the operating threshold on validation.
4. Report AUC / KS / Precision / Recall on holdout and plot ROC, PR, and KS.
"""
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "reports/figures"

def ks_statistic(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    ks_thr = thr[np.argmax(tpr - fpr)]
    return ks, ks_thr, fpr, tpr

def main(data_path):
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"]); y = df["Class"]

    # Three-way split: 70% train / 15% validation / 15% holdout.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_hold, y_val, y_hold = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    print(f"train {len(y_tr):,} | val {len(y_val):,} | holdout {len(y_hold):,}")
    print(f"Fraud rate train={y_tr.mean()*100:.3f}%  holdout={y_hold.mean()*100:.3f}%")

    # Imbalance handling: use the negative/positive ratio as the class weight.
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              eval_metric="auc", callbacks=[lgb.early_stopping(40, verbose=False)])

    # Choose the threshold on validation by maximizing KS.
    val_score = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_score)
    _, ks_thr, _, _ = ks_statistic(y_val, val_score)

    # Use holdout only once.
    hp = model.predict_proba(X_hold)[:, 1]
    auc = roc_auc_score(y_hold, hp)
    ks, _, fpr, tpr = ks_statistic(y_hold, hp)
    ap = average_precision_score(y_hold, hp)

    # Apply the validation-selected threshold to holdout confusion metrics.
    pred = (hp >= ks_thr).astype(int)
    tp = int(((pred == 1) & (y_hold == 1)).sum())
    fp = int(((pred == 1) & (y_hold == 0)).sum())
    fn = int(((pred == 0) & (y_hold == 1)).sum())
    tn = int(((pred == 0) & (y_hold == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    metrics = {
        "validation_AUC": round(val_auc, 4),
        "holdout_AUC": round(auc, 4),
        "holdout_KS": round(ks, 4),
        "holdout_AveragePrecision_PR_AUC": round(ap, 4),
        "threshold_chosen_on_val": round(float(ks_thr), 6),
        "at_threshold": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "precision": round(precision, 4), "recall": round(recall, 4)},
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- Figure 1: ROC + KS ----
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2F6F6A", lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    gap_idx = np.argmax(tpr - fpr)
    plt.vlines(fpr[gap_idx], fpr[gap_idx], tpr[gap_idx], color="#C2410C", lw=2,
               label=f"KS={ks:.3f}")
    plt.xlabel("FPR (false alarm rate)"); plt.ylabel("TPR (recall)")
    plt.title("ROC curve & KS"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/roc_ks.png", dpi=130); plt.close()

    # ---- Figure 2: Precision-Recall, especially important under imbalance ----
    prec, rec, _ = precision_recall_curve(y_hold, hp)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#2563EB", lw=2, label=f"PR (AP={ap:.3f})")
    plt.axhline(y_hold.mean(), ls="--", color="#888", lw=1, label=f"baseline={y_hold.mean():.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall curve"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/precision_recall.png", dpi=130); plt.close()

    # ---- Figure 3: Feature importance ----
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
    plt.figure(figsize=(6, 5))
    imp.plot(kind="barh", color="#2F6F6A")
    plt.title("Top 12 feature importance"); plt.tight_layout()
    plt.savefig(f"{FIG}/feature_importance.png", dpi=130); plt.close()

    print(f"\nFigures saved to {FIG}/  Metrics saved to reports/metrics.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard_synthetic.csv")
    args = ap.parse_args()
    main(args.data)
