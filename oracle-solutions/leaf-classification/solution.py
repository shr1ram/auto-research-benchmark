import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_DIR = os.environ["DATA_DIR"]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

# Species class columns in the EXACT order of sample_submission (id first, then 99 species)
class_cols = list(sample.columns[1:])
assert len(class_cols) == 99, f"expected 99 classes, got {len(class_cols)}"

feature_cols = [c for c in train.columns if c not in ("id", "species")]
assert len(feature_cols) == 192, f"expected 192 features, got {len(feature_cols)}"

X = train[feature_cols].values.astype(np.float64)
y_raw = train["species"].values
X_test = test[feature_cols].values.astype(np.float64)
test_ids = test["id"].values

# Encode labels. Fit on sample_submission ordering so classes_ is a known mapping.
le = LabelEncoder()
le.fit(class_cols)
y = le.transform(y_raw)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xts = scaler.transform(X_test)

clf = LogisticRegression(
    C=1000.0,
    max_iter=5000,
    solver="lbfgs",
    multi_class="multinomial",
)

# Quick CV sanity check (log-loss).
try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xs, y, cv=skf, scoring="neg_log_loss")
    print("CV neg_log_loss:", np.mean(scores), "+/-", np.std(scores), flush=True)
except Exception as e:
    print("CV skipped:", e, flush=True)

clf.fit(Xs, y)
proba = clf.predict_proba(Xts)  # columns ordered by le.classes_

proba_df = pd.DataFrame(proba, columns=list(le.classes_))
sub = pd.DataFrame({"id": test_ids})
for c in class_cols:
    sub[c] = proba_df[c].values

# Row order must match sample_submission ids exactly.
sample_order = sample[["id"]].merge(sub, on="id", how="left")
assert not sample_order.isnull().any().any(), "null values in submission"
sample_order = sample_order[["id"] + class_cols]

sample_order.to_csv("submission.csv", index=False)
print("wrote submission.csv", sample_order.shape, flush=True)
print(sample_order.iloc[:2, :4].to_string(), flush=True)
