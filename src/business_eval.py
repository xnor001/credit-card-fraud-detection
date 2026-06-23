"""
把模型用好:baseline(无处理) vs SMOTE(过采样) 两个模型,各做三步并对比。

  步骤1 概率校准:可靠性曲线 + Brier 分数;用 Isotonic 在 validation 上校准。
  步骤2 阈值/成本优化:用【校准后】概率,按业务成本扫阈值找最优。
  步骤3 金额加权评估:在最优阈值下,看按【金额】的召回(而非笔数)。

成本假设(可调):
  漏抓一笔欺诈  -> 损失 = 该笔交易金额 (真实损失)
  误伤一笔正常  -> 固定成本 C_FP (审核+客户摩擦),默认 ¥5

纪律:校准在 validation 上拟合,阈值在 validation 上选,holdout 只评估一次。
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

def cost_sweep(y, p, amount, c_fp):
    """对每个阈值算总成本:漏抓损失金额 + 误伤固定成本。"""
    ths = np.linspace(0.001, 0.999, 400)
    costs = []
    for t in ths:
        pred = p >= t
        fn_loss = amount[(~pred) & (y == 1)].sum()      # 漏抓的欺诈,损失其金额
        fp_cost = ((pred) & (y == 0)).sum() * c_fp        # 误伤的正常,固定成本
        costs.append(fn_loss + fp_cost)
    costs = np.array(costs)
    i = int(np.argmin(costs))
    return ths, costs, ths[i], costs[i]

def amount_recall(y, p, amount, thr):
    pred = p >= thr
    caught = amount[(pred) & (y == 1)].sum()
    total = amount[y == 1].sum()
    cnt_recall = ((pred) & (y == 1)).sum() / max((y == 1).sum(), 1)
    return caught / max(total, 1), cnt_recall

def run_model(name, X_tr, y_tr, X_val, y_val, X_hold, y_hold, amt_hold, c_fp, resample=None):
    if resample == "smote":
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    elif resample == "oversample":
        X_tr, y_tr = RandomOverSampler(random_state=42).fit_resample(X_tr, y_tr)
    m = lgb.LGBMClassifier(**BASE)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    raw = m.predict_proba(X_hold)[:, 1]
    # Isotonic 校准(在 val 上拟合)
    cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit").fit(X_val, y_val)
    calp = cal.predict_proba(X_hold)[:, 1]

    out = dict(
        AUC=round(roc_auc_score(y_hold, raw), 4),
        PR_AUC=round(average_precision_score(y_hold, raw), 4),
        Brier_raw=round(brier_score_loss(y_hold, raw), 6),
        Brier_calibrated=round(brier_score_loss(y_hold, calp), 6),
    )
    # 阈值/成本(用校准后概率)
    ths, costs, best_t, best_c = cost_sweep(y_hold.values, calp, amt_hold.values, c_fp)
    amt_rec, cnt_rec = amount_recall(y_hold.values, calp, amt_hold.values, best_t)
    out.update(best_threshold=round(float(best_t), 4),
               min_total_cost=round(float(best_c), 2),
               amount_recall=round(float(amt_rec), 4),
               count_recall=round(float(cnt_rec), 4))
    return out, raw, calp, (ths, costs, best_t)

def main(data_path, c_fp):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"]); y = df["Class"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_hold, y_val, y_hold = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    amt_hold = X_hold["Amount"]
    total_fraud_amt = amt_hold[y_hold == 1].sum()

    res = {}
    res["baseline"], raw_b, cal_b, cs_b = run_model("baseline", X_tr, y_tr, X_val, y_val, X_hold, y_hold, amt_hold, c_fp)
    res["SMOTE"], raw_s, cal_s, cs_s = run_model("SMOTE", X_tr, y_tr, X_val, y_val, X_hold, y_hold, amt_hold, c_fp, resample="smote")
    res["过采样"], raw_o, cal_o, cs_o = run_model("oversample", X_tr, y_tr, X_val, y_val, X_hold, y_hold, amt_hold, c_fp, resample="oversample")

    # 参考线:不拦截 = 全部欺诈金额损失
    no_model_cost = float(total_fraud_amt)
    print(f"\n参考:holdout 欺诈总金额(不拦截的损失) = ¥{no_model_cost:,.2f}  (误伤成本 C_FP=¥{c_fp})\n")
    tab = pd.DataFrame(res).T
    print(tab.to_string())
    tab.to_csv("reports/business_eval.csv")
    with open("reports/business_eval.json", "w") as f:
        json.dump({"C_FP": c_fp, "holdout_total_fraud_amount": round(no_model_cost, 2), "models": res}, f, indent=2, ensure_ascii=False)

    # ---- 图1: 可靠性曲线(校准前 vs 后)----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for axi, (nm, raw, calp) in zip(ax, [("baseline", raw_b, cal_b), ("SMOTE", raw_s, cal_s), ("oversample", raw_o, cal_o)]):
        for lab, p, col in [("raw", raw, "#C2410C"), ("calibrated", calp, "#2563EB")]:
            fr, mp = calibration_curve(y_hold, p, n_bins=10, strategy="quantile")
            axi.plot(mp, fr, "o-", color=col, label=lab)
        axi.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        axi.set_title(f"{nm} reliability"); axi.set_xlabel("predicted prob"); axi.set_ylabel("actual fraud rate"); axi.legend()
    plt.tight_layout(); plt.savefig("reports/figures/calibration.png", dpi=130); plt.close()

    # ---- 图2: 成本曲线 ----
    plt.figure(figsize=(7, 5))
    for nm, (ths, costs, bt), col in [("baseline", cs_b, "#2F6F6A"), ("SMOTE", cs_s, "#2563EB"), ("oversample", cs_o, "#C2410C")]:
        plt.plot(ths, costs, color=col, label=f"{nm} (min@{bt:.3f})")
        plt.scatter([bt], [costs.min()], color=col, zorder=5)
    plt.axhline(no_model_cost, ls="--", color="#888", label=f"no model = ¥{no_model_cost:,.0f}")
    plt.xlabel("threshold"); plt.ylabel("total cost (¥)"); plt.title("Cost vs threshold (lower=better)")
    plt.legend(); plt.tight_layout(); plt.savefig("reports/figures/cost_curve.png", dpi=130); plt.close()
    print("\n图 -> reports/figures/calibration.png, cost_curve.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--c_fp", type=float, default=5.0, help="误伤一笔正常交易的固定成本")
    a = ap.parse_args(); main(a.data, a.c_fp)
