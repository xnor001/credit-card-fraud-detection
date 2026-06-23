"""
错误分析:为什么漏掉大额欺诈?
用过采样模型(校准后),在成本最优阈值下解剖 FN(漏抓欺诈)。

回答三问:
  1. 漏抓的欺诈是"分数差一点没过阈值"(阈值问题)还是"分数很低"(特征盲区)?
  2. 漏抓的是不是集中在大额?
  3. 漏抓的欺诈在哪些特征上"长得像正常交易"?
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"]); y = df["Class"]
X_tr,X_tmp,y_tr,y_tmp = train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
X_val,X_hold,y_val,y_hold = train_test_split(X_tmp,y_tmp,test_size=.50,stratify=y_tmp,random_state=42)
amt = X_hold["Amount"].values; yh = y_hold.values

Xo,yo = RandomOverSampler(random_state=42).fit_resample(X_tr,y_tr)
m = lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.8,
    colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1).fit(
    Xo,yo,eval_set=[(X_val,y_val)],eval_metric="auc",callbacks=[lgb.early_stopping(40,verbose=False)])
cal = CalibratedClassifierCV(m,method="isotonic",cv="prefit").fit(X_val,y_val)
p = cal.predict_proba(X_hold)[:,1]

# 成本最优阈值(漏抓=金额, 误伤=¥5)
ths=np.linspace(.001,.999,400)
costs=[amt[(p<t)&(yh==1)].sum()+((p>=t)&(yh==0)).sum()*5 for t in ths]
thr=ths[int(np.argmin(costs))]
pred = p>=thr

fraud = yh==1
caught = fraud & pred
missed = fraud & ~pred
print(f"阈值={thr:.3f} | 欺诈 {fraud.sum()} 笔: 抓到 {caught.sum()}, 漏掉 {missed.sum()}")
print(f"金额: 欺诈总额 ¥{amt[fraud].sum():,.0f} | 漏掉金额 ¥{amt[missed].sum():,.0f} "
      f"({amt[missed].sum()/amt[fraud].sum()*100:.0f}% 的损失来自漏抓)\n")

# 问1: 漏抓欺诈的分数 vs 阈值
sc = p[missed]
print("【问1 阈值问题 vs 特征盲区】漏抓欺诈的分数分布:")
print(f"  接近阈值(thr/2 ~ thr): {((sc>=thr/2)&(sc<thr)).sum()} 笔  -> 调阈值可救")
print(f"  深度漏判(< thr/2):    {(sc<thr/2).sum()} 笔  -> 特征盲区")
print(f"  其中分数≈0(<0.001):   {(sc<0.001).sum()} 笔\n")

# 问2: 金额对比
print("【问2 是否集中在大额】")
print(f"  抓到欺诈 金额中位/均值: ¥{np.median(amt[caught]):.0f} / ¥{amt[caught].mean():.0f}")
print(f"  漏抓欺诈 金额中位/均值: ¥{np.median(amt[missed]):.0f} / ¥{amt[missed].mean():.0f}")
big = df.loc[X_hold.index][missed].nlargest(8,"Amount")[["Amount"]].copy()
big["score"]=p[missed][np.argsort(-amt[missed])][:0] if False else None
# 重新对齐:取漏抓样本里金额最大的几笔及其分数
mi = np.where(missed)[0]
order = mi[np.argsort(-amt[mi])][:8]
print("\n  漏抓的最大额欺诈(金额, 校准分数):")
for i in order:
    print(f"    ¥{amt[i]:8.0f}   score={p[i]:.4f}")

# 问3: 特征画像(漏抓 vs 抓到 vs 正常)
feats=["V17","V14","V12","V10","Amount"]
Xh = X_hold.reset_index(drop=True)
prof = pd.DataFrame({
  "漏抓欺诈":[Xh.loc[missed,f].mean() for f in feats],
  "抓到欺诈":[Xh.loc[caught,f].mean() for f in feats],
  "正常交易":[Xh.loc[yh==0,f].mean() for f in feats],
}, index=feats).round(2)
print("\n【问3 漏抓欺诈在哪些特征上像正常】各组均值:")
print(prof.to_string())

# 图
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].hist(np.clip(p[caught],0,1),bins=30,alpha=.6,label="caught fraud",color="#2F6F6A")
ax[0].hist(np.clip(p[missed],0,1),bins=30,alpha=.6,label="missed fraud",color="#C2410C")
ax[0].axvline(thr,ls="--",color="k",label=f"threshold={thr:.3f}")
ax[0].set_xlim(0,0.3); ax[0].set_xlabel("calibrated score"); ax[0].set_ylabel("count")
ax[0].set_title("Fraud score: caught vs missed"); ax[0].legend()
ax[1].scatter(amt[caught],p[caught],s=18,alpha=.6,label="caught",color="#2F6F6A")
ax[1].scatter(amt[missed],p[missed],s=28,alpha=.8,label="missed",color="#C2410C",marker="x")
ax[1].axhline(thr,ls="--",color="k"); ax[1].set_xscale("symlog")
ax[1].set_xlabel("Amount (symlog)"); ax[1].set_ylabel("calibrated score")
ax[1].set_title("Amount vs score (frauds)"); ax[1].legend()
plt.tight_layout(); plt.savefig("reports/figures/error_analysis.png",dpi=130); plt.close()
print("\n图 -> reports/figures/error_analysis.png")
