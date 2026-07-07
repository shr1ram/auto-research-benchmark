import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
D=os.environ["DATA_DIR"]
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
ss=pd.read_csv(f"{D}/sample_submission.csv")
idc=ss.columns[0]; tgt="target" if "target" in tr.columns else [c for c in tr.columns if c not in te.columns][0]
feats=[c for c in tr.columns if c not in [idc,tgt]]
X=tr[feats].select_dtypes("number").fillna(0); y=tr[tgt].astype("category").cat.codes
cats=tr[tgt].astype("category").cat.categories
Xte=te[feats].select_dtypes("number").fillna(0)[X.columns]
import lightgbm as lgb
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=0.1,random_state=0)
dtr=lgb.Dataset(Xtr,ytr); dv=lgb.Dataset(Xv,yv)
params=dict(objective="multiclass",num_class=len(cats),learning_rate=0.1,num_leaves=63,metric="multi_logloss",verbose=-1,n_jobs=-1)
m=lgb.train(params,dtr,num_boost_round=200,valid_sets=[dv],callbacks=[lgb.early_stopping(20),lgb.log_evaluation(0)])
P=m.predict(Xte)
# submission format: sample_submission has target column (single label) or per-class? check
subcols=list(ss.columns)[1:]
if len(subcols)==1:
    ss[subcols[0]]=[cats[i] for i in P.argmax(1)]
else:
    for j,c in enumerate(subcols): ss[c]=P[:,j]
ss.to_csv("submission.csv",index=False); print("done",ss.shape)
