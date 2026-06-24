"""
Interpretability analysis for the best model: XGBoost + no resampling.
1) TreeSHAP: global importance (beeswarm), dependence plot, and single-case
   explanations for caught and missed high-value fraud.
2) Permutation Importance on holdout, scored by PR-AUC, as a cross-check.
SHAP values are in log-odds / margin space; additivity means the sum of feature
SHAP values equals the sample margin minus the baseline margin.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import shap, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df=pd.read_csv("data/creditcard.csv"); X=df.drop(columns=["Class"]); y=df["Class"]
X_tr,X_tmp,y_tr,y_tmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
X_val,X_hold,y_val,y_hold=train_test_split(X_tmp,y_tmp,test_size=.50,stratify=y_tmp,random_state=42)

m=XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,
  tree_method="hist",eval_metric="auc",random_state=42,n_jobs=-1,early_stopping_rounds=40,base_score=0.5)
m.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],verbose=False)

# ---- SHAP (TreeSHAP) ----
expl=shap.TreeExplainer(m)
Xs=X_hold.sample(n=5000,random_state=42)          # sample for faster global plots
sv=expl(Xs)
mean_abs=pd.Series(np.abs(sv.values).mean(0),index=X_hold.columns).sort_values(ascending=False)
print("== SHAP global importance Top 8 (mean|SHAP|) ==")
print(mean_abs.head(8).round(4).to_string())

plt.figure(); shap.plots.beeswarm(sv,max_display=12,show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_beeswarm.png",dpi=130,bbox_inches="tight"); plt.close()
plt.figure(); shap.plots.bar(sv,max_display=12,show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_bar.png",dpi=130,bbox_inches="tight"); plt.close()
top=mean_abs.index[0]
plt.figure(); shap.plots.scatter(sv[:,top],show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_dependence.png",dpi=130,bbox_inches="tight"); plt.close()

# ---- Single-case explanations: caught fraud vs largest fraud ----
ph=m.predict_proba(X_hold)[:,1]; yh=y_hold.values; amt=X_hold["Amount"].values
fraud_idx=np.where(yh==1)[0]
caught_i=fraud_idx[np.argmax(ph[fraud_idx])]      # highest-scored fraud
biggest_i=fraud_idx[np.argmax(amt[fraud_idx])]    # largest-amount fraud
for tag,i in [("caught",caught_i),("biggest_fraud",biggest_i)]:
    e=expl(X_hold.iloc[[i]])
    plt.figure(); shap.plots.waterfall(e[0],max_display=10,show=False)
    plt.tight_layout(); plt.savefig(f"reports/figures/shap_waterfall_{tag}.png",dpi=130,bbox_inches="tight"); plt.close()
    print(f"\nSingle case [{tag}] amount=¥{amt[i]:.0f} predicted fraud probability={ph[i]:.4f}")

# ---- Permutation Importance (PR-AUC) ----
print("\n== Permutation Importance Top 8 (PR-AUC drop) ==")
pi=permutation_importance(m,X_hold,y_hold,scoring="average_precision",n_repeats=5,random_state=42,n_jobs=-1)
pis=pd.Series(pi.importances_mean,index=X_hold.columns).sort_values(ascending=False)
print(pis.head(8).round(4).to_string())
plt.figure(figsize=(7,5))
pis.head(12)[::-1].plot(kind="barh",color="#2F6F6A")
plt.xlabel("PR-AUC drop when permuted"); plt.title("Permutation importance (XGBoost, holdout)")
plt.tight_layout(); plt.savefig("reports/figures/permutation_importance.png",dpi=130); plt.close()

# Cross-check rankings.
cmp=pd.DataFrame({"SHAP_rank":mean_abs.rank(ascending=False).astype(int),
                  "Perm_rank":pis.rank(ascending=False).astype(int)}).sort_values("SHAP_rank")
print("\n== SHAP vs Permutation ranking (Top 8) ==")
print(cmp.head(8).to_string())
print("\nFigures saved: shap_beeswarm/shap_bar/shap_dependence/shap_waterfall_*/permutation_importance")
