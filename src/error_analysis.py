"""
Error analysis: why are large-value fraud transactions missed?
Use the calibrated oversampling model and inspect false negatives at the
cost-optimal threshold.

Answer three questions:
  1. Are missed frauds close to the threshold (a threshold issue) or scored very
     low (a feature blind spot)?
  2. Are missed frauds concentrated in high amounts?
  3. On which features do missed frauds look like legitimate transactions?
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

# Cost-optimal threshold: missed fraud loses amount; false positives cost ¥5.
ths=np.linspace(.001,.999,400)
costs=[amt[(p<t)&(yh==1)].sum()+((p>=t)&(yh==0)).sum()*5 for t in ths]
thr=ths[int(np.argmin(costs))]
pred = p>=thr

fraud = yh==1
caught = fraud & pred
missed = fraud & ~pred
print(f"Threshold={thr:.3f} | Fraud {fraud.sum()} transactions: caught {caught.sum()}, missed {missed.sum()}")
print(f"Amount: total fraud ¥{amt[fraud].sum():,.0f} | missed amount ¥{amt[missed].sum():,.0f} "
      f"({amt[missed].sum()/amt[fraud].sum()*100:.0f}% of loss comes from misses)\n")

# Question 1: missed-fraud scores vs threshold.
sc = p[missed]
print("[Q1 threshold issue vs feature blind spot] score distribution of missed fraud:")
print(f"  Near threshold (thr/2 ~ thr): {((sc>=thr/2)&(sc<thr)).sum()} transactions -> threshold tuning may recover them")
print(f"  Deep misses (< thr/2):        {(sc<thr/2).sum()} transactions -> feature blind spot")
print(f"  Score near zero (<0.001):     {(sc<0.001).sum()} transactions\n")

# Question 2: amount comparison.
print("[Q2 are misses concentrated in high amounts?]")
print(f"  Caught fraud amount median/mean: ¥{np.median(amt[caught]):.0f} / ¥{amt[caught].mean():.0f}")
print(f"  Missed fraud amount median/mean: ¥{np.median(amt[missed]):.0f} / ¥{amt[missed].mean():.0f}")
big = df.loc[X_hold.index][missed].nlargest(8,"Amount")[["Amount"]].copy()
big["score"]=p[missed][np.argsort(-amt[missed])][:0] if False else None
# Re-align and print the largest missed fraud transactions with their scores.
mi = np.where(missed)[0]
order = mi[np.argsort(-amt[mi])][:8]
print("\n  Largest missed fraud transactions (amount, calibrated score):")
for i in order:
    print(f"    ¥{amt[i]:8.0f}   score={p[i]:.4f}")

# Question 3: feature profile for missed vs caught vs legitimate transactions.
feats=["V17","V14","V12","V10","Amount"]
Xh = X_hold.reset_index(drop=True)
prof = pd.DataFrame({
  "missed_fraud":[Xh.loc[missed,f].mean() for f in feats],
  "caught_fraud":[Xh.loc[caught,f].mean() for f in feats],
  "legitimate":[Xh.loc[yh==0,f].mean() for f in feats],
}, index=feats).round(2)
print("\n[Q3 where do missed frauds look legitimate?] group means:")
print(prof.to_string())

# Plot.
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
print("\nFigure -> reports/figures/error_analysis.png")
