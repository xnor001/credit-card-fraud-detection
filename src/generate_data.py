"""
合成信用卡欺诈数据生成器。
模仿业界标准的 ULB 信用卡数据集结构(Time, V1..V28, Amount, Class),
但完全由代码生成 —— 仓库 clone 下来即可运行,无需下载任何外部数据。

设计要点(刻意贴近真实):
- 极度不平衡:欺诈约占 0.3%
- 28 个匿名特征 V1..V28(模拟 PCA 变换后的特征),其中少数几个对欺诈有真实区分力,
  其余为噪声 —— 这正是真实风控数据的样子。
- 金额分布右偏(lognormal),欺诈交易金额结构略有不同。
"""
import numpy as np
import pandas as pd
import argparse, os

def generate(n=284807, fraud_rate=0.0030, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make_features(n_rows, signal_shift):
        # 28 个特征:前 6 个携带可区分信号,其余 22 个是噪声
        X = rng.standard_normal((n_rows, 28))
        signal_cols = [0, 1, 2, 3, 9, 13]   # 模拟少数强特征
        for j, s in zip(signal_cols, signal_shift):
            X[:, j] += s
        return X

    # 合法交易:信号特征居中;欺诈交易:信号特征整体偏移(可分但重叠)
    X_legit = make_features(n_legit, [0, 0, 0, 0, 0, 0])
    X_fraud = make_features(n_fraud, [-2.1, 1.8, -1.5, 1.2, -1.0, 1.4])

    X = np.vstack([X_legit, X_fraud])
    y = np.r_[np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]

    # 金额:右偏;欺诈金额分布略有不同
    amt_legit = np.round(rng.lognormal(3.0, 1.1, n_legit) + 1, 2)
    amt_fraud = np.round(rng.lognormal(3.4, 1.3, n_fraud) + 1, 2)
    amount = np.r_[amt_legit, amt_fraud]

    # 时间:两天内的秒数(与 ULB 一致),欺诈在夜间略多
    t_legit = rng.uniform(0, 172792, n_legit)
    t_fraud = rng.uniform(0, 172792, n_fraud)
    night = rng.random(n_fraud) < 0.35
    t_fraud[night] = (t_fraud[night] % 86400) * 0.25  # 偏向凌晨
    time = np.r_[t_legit, t_fraud]

    cols = {f"V{i+1}": X[:, i] for i in range(28)}
    df = pd.DataFrame(cols)
    df.insert(0, "Time", np.round(time).astype(int))
    df["Amount"] = amount
    df["Class"] = y

    # 打乱顺序
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=284807)
    ap.add_argument("--out", default="data/creditcard_synthetic.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"已生成 {len(df):,} 笔交易 -> {args.out}")
    print(f"欺诈占比: {df.Class.mean()*100:.3f}%  (欺诈 {int(df.Class.sum()):,} 笔)")
