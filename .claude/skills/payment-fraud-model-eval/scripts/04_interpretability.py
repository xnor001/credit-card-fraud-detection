"""
可解释性分析:对最优模型 XGBoost + 无处理。
1) TreeSHAP:全局重要性(beeswarm)、依赖图、单笔解释(抓到的 / 漏掉的大额)
2) Permutation Importance(holdout, scoring=PR-AUC)做交叉验证
SHAP 值在 log-odds(margin)空间;可加性:各特征 SHAP 之和 = 该样本 margin − 基线 margin。
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
Xs=X_hold.sample(n=5000,random_state=42)          # 全局图用抽样,快
sv=expl(Xs)
mean_abs=pd.Series(np.abs(sv.values).mean(0),index=X_hold.columns).sort_values(ascending=False)
print("== SHAP 全局重要性 Top8 (mean|SHAP|) ==")
print(mean_abs.head(8).round(4).to_string())

plt.figure(); shap.plots.beeswarm(sv,max_display=12,show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_beeswarm.png",dpi=130,bbox_inches="tight"); plt.close()
plt.figure(); shap.plots.bar(sv,max_display=12,show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_bar.png",dpi=130,bbox_inches="tight"); plt.close()
top=mean_abs.index[0]
plt.figure(); shap.plots.scatter(sv[:,top],show=False)
plt.tight_layout(); plt.savefig("reports/figures/shap_dependence.png",dpi=130,bbox_inches="tight"); plt.close()

# ---- 单笔解释:抓到的欺诈 vs 漏掉的大额欺诈 ----
ph=m.predict_proba(X_hold)[:,1]; yh=y_hold.values; amt=X_hold["Amount"].values
fraud_idx=np.where(yh==1)[0]
caught_i=fraud_idx[np.argmax(ph[fraud_idx])]                    # 分数最高的欺诈
biggest_i=fraud_idx[np.argmax(amt[fraud_idx])]   # 金额最大的欺诈
for tag,i in [("caught",caught_i),("biggest_fraud",biggest_i)]:
    e=expl(X_hold.iloc[[i]])
    plt.figure(); shap.plots.waterfall(e[0],max_display=10,show=False)
    plt.tight_layout(); plt.savefig(f"reports/figures/shap_waterfall_{tag}.png",dpi=130,bbox_inches="tight"); plt.close()
    print(f"\n单笔 [{tag}] 金额=¥{amt[i]:.0f} 预测欺诈概率={ph[i]:.4f}")

# ---- Permutation Importance (PR-AUC) ----
print("\n== Permutation Importance Top8 (PR-AUC 下降) ==")
pi=permutation_importance(m,X_hold,y_hold,scoring="average_precision",n_repeats=5,random_state=42,n_jobs=-1)
pis=pd.Series(pi.importances_mean,index=X_hold.columns).sort_values(ascending=False)
print(pis.head(8).round(4).to_string())
plt.figure(figsize=(7,5))
pis.head(12)[::-1].plot(kind="barh",color="#2F6F6A")
plt.xlabel("PR-AUC drop when permuted"); plt.title("Permutation importance (XGBoost, holdout)")
plt.tight_layout(); plt.savefig("reports/figures/permutation_importance.png",dpi=130); plt.close()

# 交叉对比
cmp=pd.DataFrame({"SHAP_rank":mean_abs.rank(ascending=False).astype(int),
                  "Perm_rank":pis.rank(ascending=False).astype(int)}).sort_values("SHAP_rank")
print("\n== SHAP vs Permutation 排名(Top8) ==")
print(cmp.head(8).to_string())
print("\n图已保存: shap_beeswarm/shap_bar/shap_dependence/shap_waterfall_*/permutation_importance")
