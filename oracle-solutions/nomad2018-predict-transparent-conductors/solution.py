import os, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
D=os.environ["DATA_DIR"]
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
ss=pd.read_csv(f"{D}/sample_submission.csv")
targets=[c for c in ss.columns if c!="id"]
feats=[c for c in tr.columns if c not in targets+["id"]]
Xtr=tr[feats].select_dtypes("number").fillna(0); Xte=te[feats].select_dtypes("number").fillna(0)
Xte=Xte[Xtr.columns]
for t in targets:
    y=np.log1p(tr[t].clip(lower=0))
    m=GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    m.fit(Xtr,y)
    ss[t]=np.expm1(m.predict(Xte)).clip(min=0)
ss.to_csv("submission.csv", index=False)
print("done", ss.shape)
