import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
D=os.environ["DATA_DIR"]
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
ss=pd.read_csv(f"{D}/sample_submission.csv")
idc=ss.columns[0]; tgt="target"
feats=[c for c in tr.columns if c not in [idc,tgt]]
X=tr[feats].select_dtypes("number").fillna(0); y=tr[tgt].astype(int)
Xte=te[feats].select_dtypes("number").fillna(0)[X.columns]
import lightgbm as lgb
Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
m=lgb.train(dict(objective="binary",learning_rate=0.05,num_leaves=127,metric="auc",verbose=-1,n_jobs=-1),
            lgb.Dataset(Xtr,ytr),num_boost_round=400,valid_sets=[lgb.Dataset(Xv,yv)],
            callbacks=[lgb.early_stopping(30),lgb.log_evaluation(0)])
ss[ss.columns[1]]=m.predict(Xte)
ss.to_csv("submission.csv",index=False); print("done",ss.shape)
