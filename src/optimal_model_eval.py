"""
Business evaluation and error analysis for the best candidate models.
Compare XGBoost+None (best PR-AUC) and RandomForest+None (best F1 threshold).
Metrics include AUC/KS/PR-AUC, Brier before/after calibration, and P/R/F1 plus
amount recall at the cost-optimal threshold.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df=pd.read_csv("data/creditcard.csv"); X=df.drop(columns=["Class"]); y=df["Class"]
X_tr,X_tmp,y_tr,y_tmp=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
X_val,X_hold,y_val,y_hold=train_test_split(X_tmp,y_tmp,test_size=.50,stratify=y_tmp,random_state=42)
amt=X_hold["Amount"].values; yh=y_hold.values; C_FP=5.0
no_model=amt[yh==1].sum()
def ks(yy,s): f,t,_=roc_curve(yy,s); return float(np.max(t-f))

def evaluate(name, clf, fit_kw):
    clf.fit(X_tr,y_tr,**fit_kw) if fit_kw else clf.fit(X_tr,y_tr)
    raw=clf.predict_proba(X_hold)[:,1]
    cal=CalibratedClassifierCV(clf,method="isotonic",cv="prefit").fit(X_val,y_val)
    p=cal.predict_proba(X_hold)[:,1]
    ths=np.linspace(.001,.999,400)
    costs=np.array([amt[(p<t)&(yh==1)].sum()+((p>=t)&(yh==0)).sum()*C_FP for t in ths])
    bi=int(np.argmin(costs)); thr=ths[bi]; pred=p>=thr
    tp=int((pred&(yh==1)).sum());fp=int((pred&(yh==0)).sum());fn=int((~pred&(yh==1)).sum())
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1); F1=2*P*R/(P+R) if P+R>0 else 0
    amt_rec=amt[pred&(yh==1)].sum()/no_model
    return dict(model=name, AUC=round(roc_auc_score(yh,raw),4), KS=round(ks(yh,raw),4),
        PR_AUC=round(average_precision_score(yh,raw),4),
        Brier_raw=round(brier_score_loss(yh,raw),6), Brier_cal=round(brier_score_loss(yh,p),6),
        best_thr=round(float(thr),3), min_cost=round(float(costs[bi]),0),
        P=round(P,3), R=round(R,3), F1=round(F1,3), amt_recall=round(float(amt_rec),3),
        TP=tp, FP=fp, FN=fn), (raw,p,thr)

xgb=XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,
    tree_method="hist",eval_metric="auc",random_state=42,n_jobs=-1,early_stopping_rounds=40)
rf=RandomForestClassifier(n_estimators=80,max_depth=14,max_samples=0.5,n_jobs=-1,random_state=42)

r1,_=evaluate("XGBoost+None", xgb, dict(eval_set=[(X_val,y_val)],verbose=False))
r2,_=evaluate("RandomForest+None", rf, None)
res=pd.DataFrame([r1,r2]).set_index("model").T
print(f"Loss if blocking nothing=¥{no_model:,.0f}  (false-positive cost C_FP=¥{C_FP})\n")
print(res.to_string())
res.to_csv("reports/optimal_compare.csv")
