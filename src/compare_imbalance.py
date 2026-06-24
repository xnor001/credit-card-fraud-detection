"""
Compare imbalance-handling strategies.

Using the same data, the same holdout set, and the same LightGBM hyperparameters,
change only how class imbalance is handled and compare AUC / KS / PR-AUC.

Key discipline: resampling is applied to the training set only. Validation and
holdout keep the original distribution; otherwise the evaluation leaks data and
looks artificially strong.
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def ks_stat(y, s):
    fpr, tpr, thr = roc_curve(y, s); i = np.argmax(tpr - fpr)
    return tpr[i]-fpr[i], thr[i]

BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

def fit_eval(X_tr, y_tr, X_val, y_val, X_hold, y_hold, scale_pos_weight=1):
    m = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **BASE)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    # Select the threshold on validation by max KS; holdout is evaluation only.
    _, thr = ks_stat(y_val, m.predict_proba(X_val)[:,1])
    hp = m.predict_proba(X_hold)[:,1]
    auc = roc_auc_score(y_hold, hp); ks,_ = ks_stat(y_hold, hp)
    ap = average_precision_score(y_hold, hp)
    pred = (hp >= thr).astype(int)
    tp=int(((pred==1)&(y_hold==1)).sum()); fp=int(((pred==1)&(y_hold==0)).sum())
    fn=int(((pred==0)&(y_hold==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    return dict(AUC=round(auc,4), KS=round(ks,4), PR_AUC=round(ap,4),
               precision=round(prec,4), recall=round(rec,4), train_pos=int((y_tr==1).sum()),
               train_size=int(len(y_tr)))

def main(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"]); y = df["Class"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_hold, y_val, y_hold = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    spw = (y_tr==0).sum()/max((y_tr==1).sum(),1)

    results = {}
    # 1. No imbalance treatment
    results["None (baseline)"] = fit_eval(X_tr, y_tr, X_val, y_val, X_hold, y_hold)
    # 2. Class weight
    results["Class weight (scale_pos_weight)"] = fit_eval(X_tr, y_tr, X_val, y_val, X_hold, y_hold, scale_pos_weight=spw)
    # 3. Undersampling
    Xu, yu = RandomUnderSampler(random_state=42).fit_resample(X_tr, y_tr)
    results["Random undersampling"] = fit_eval(Xu, yu, X_val, y_val, X_hold, y_hold)
    # 4. Oversampling
    Xo, yo = RandomOverSampler(random_state=42).fit_resample(X_tr, y_tr)
    results["Random oversampling"] = fit_eval(Xo, yo, X_val, y_val, X_hold, y_hold)
    # 5. SMOTE
    Xs, ys = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    results["SMOTE synthetic oversampling"] = fit_eval(Xs, ys, X_val, y_val, X_hold, y_hold)

    tab = pd.DataFrame(results).T[["AUC","KS","PR_AUC","precision","recall","train_size","train_pos"]]
    print(tab.to_string())
    tab.to_csv("reports/imbalance_comparison.csv")
    with open("reports/imbalance_comparison.json","w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot PR-AUC and KS; PR-AUC is the key fraud metric under imbalance.
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    names = list(results.keys())
    labels = ["None","ClassWeight","Undersample","Oversample","SMOTE"]
    ax[0].barh(labels, [results[n]["PR_AUC"] for n in names], color="#2563EB")
    ax[0].set_title("PR-AUC (higher=better, key metric)"); ax[0].invert_yaxis()
    for i,n in enumerate(names): ax[0].text(results[n]["PR_AUC"], i, f" {results[n]['PR_AUC']}", va="center")
    ax[1].barh(labels, [results[n]["KS"] for n in names], color="#2F6F6A")
    ax[1].set_title("KS"); ax[1].invert_yaxis()
    for i,n in enumerate(names): ax[1].text(results[n]["KS"], i, f" {results[n]['KS']}", va="center")
    plt.tight_layout(); plt.savefig("reports/figures/imbalance_comparison.png", dpi=130); plt.close()
    print("\nFigure -> reports/figures/imbalance_comparison.png  Table -> reports/imbalance_comparison.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/creditcard.csv")
    main(ap.parse_args().data)
