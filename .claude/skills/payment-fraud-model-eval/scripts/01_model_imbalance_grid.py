"""
模型 × 不平衡方法 网格(按 --method 分批)。
两个操作点:KS 最优阈值 与 F1 最优阈值(都在 validation 上选,holdout 评估)。
输出:train_size, AUC, KS, PR_AUC, 以及 KS阈值下 P/R/F1 和 F1阈值下 P/R/F1。
"""
import numpy as np, pandas as pd, time, argparse, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

ap=argparse.ArgumentParser(); ap.add_argument("--method",required=True); method=ap.parse_args().method
OUT="reports/model_imbalance_grid.csv"; t0=time.time()
df=pd.read_csv("data/creditcard.csv"); X=df.drop(columns=["Class"]); y=df["Class"]
X_tr,X_tmp,y_tr,y_tmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
X_val,X_hold,y_val,y_hold=train_test_split(X_tmp,y_tmp,test_size=.50,stratify=y_tmp,random_state=42)
yv=y_val.values; yh=y_hold.values

def ks_thr(yy,s):
    f,t,th=roc_curve(yy,s); i=np.argmax(t-f); return float(t[i]-f[i]), float(th[i])
def f1_opt_thr(yy,s):
    p,r,th=precision_recall_curve(yy,s); f1=2*p*r/(p+r+1e-12)
    i=int(np.argmax(f1[:-1])) if len(th)>0 else 0
    return float(th[i]) if len(th)>0 else 0.5
def prf(yy,s,thr):
    pred=(s>=thr).astype(int)
    tp=int(((pred==1)&(yy==1)).sum()); fp=int(((pred==1)&(yy==0)).sum()); fn=int(((pred==0)&(yy==1)).sum())
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1); F=2*P*R/(P+R) if P+R>0 else 0
    return round(P,4),round(R,4),round(F,4)

def build(model,balanced,spw):
    if model=="LogReg": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,class_weight="balanced" if balanced else None))
    if model=="RandomForest": return RandomForestClassifier(n_estimators=80,max_depth=14,max_samples=0.5,n_jobs=-1,random_state=42,class_weight="balanced" if balanced else None)
    if model=="LightGBM": return lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1,scale_pos_weight=spw if balanced else 1)
    if model=="XGBoost": return XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,tree_method="hist",eval_metric="auc",random_state=42,n_jobs=-1,early_stopping_rounds=40,scale_pos_weight=spw if balanced else 1)

if method=="欠采样":   Xt,yt=RandomUnderSampler(random_state=42).fit_resample(X_tr,y_tr); bal=False
elif method=="过采样": Xt,yt=RandomOverSampler(random_state=42).fit_resample(X_tr,y_tr); bal=False
elif method=="SMOTE":  Xt,yt=SMOTE(random_state=42).fit_resample(X_tr,y_tr); bal=False
elif method=="类别权重": Xt,yt,bal=X_tr,y_tr,True
else:                  Xt,yt,bal=X_tr,y_tr,False
spw=(yt==0).sum()/max((yt==1).sum(),1); n=len(yt)

rows=[]
for model in ["LogReg","RandomForest","LightGBM","XGBoost"]:
    clf=build(model,bal,spw)
    if model=="LightGBM": clf.fit(Xt,yt,eval_set=[(X_val,y_val)],eval_metric="auc",callbacks=[lgb.early_stopping(40,verbose=False)])
    elif model=="XGBoost": clf.fit(Xt,yt,eval_set=[(X_val,y_val)],verbose=False)
    else: clf.fit(Xt,yt)
    pv=clf.predict_proba(X_val)[:,1]; ph=clf.predict_proba(X_hold)[:,1]
    _,t_ks=ks_thr(yv,pv); t_f1=f1_opt_thr(yv,pv)
    Pks,Rks,Fks=prf(yh,ph,t_ks); Pf,Rf,Ff=prf(yh,ph,t_f1)
    ksv,_=ks_thr(yh,ph)
    rows.append(dict(model=model,method=method,train_size=n,
        AUC=round(roc_auc_score(yh,ph),4),KS=round(ksv,4),PR_AUC=round(average_precision_score(yh,ph),4),
        P_ks=Pks,R_ks=Rks,F1_ks=Fks,P_f1=Pf,R_f1=Rf,F1_f1=Ff))
    print(f"[{time.time()-t0:4.0f}s] {model:13s} PR={rows[-1]['PR_AUC']:.3f} | KS阈值 P/R/F1={Pks:.2f}/{Rks:.2f}/{Fks:.2f} | F1阈值 P/R/F1={Pf:.2f}/{Rf:.2f}/{Ff:.2f}",flush=True)

cols=["model","method","train_size","AUC","KS","PR_AUC","P_ks","R_ks","F1_ks","P_f1","R_f1","F1_f1"]
pd.DataFrame(rows)[cols].to_csv(OUT,mode="a",header=os.path.getsize(OUT)==0 if os.path.exists(OUT) else True,index=False)
print("appended",method)
