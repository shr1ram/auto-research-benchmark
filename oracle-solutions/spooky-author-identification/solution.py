import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

DATA_DIR = os.environ["DATA_DIR"]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

classes = ["EAP", "HPL", "MWS"]
class_to_idx = {c: i for i, c in enumerate(classes)}
y = train["author"].map(class_to_idx).values

# Text features: word ngrams + char ngrams
word_vec = TfidfVectorizer(
    sublinear_tf=True, analyzer="word", token_pattern=r"\w{1,}",
    ngram_range=(1, 2), max_features=60000, min_df=2,
)
char_vec = TfidfVectorizer(
    sublinear_tf=True, analyzer="char_wb",
    ngram_range=(2, 5), max_features=60000, min_df=2,
)

all_text = pd.concat([train["text"], test["text"]], axis=0)
word_vec.fit(all_text)
char_vec.fit(all_text)

Xtr = hstack([word_vec.transform(train["text"]), char_vec.transform(train["text"])]).tocsr()
Xte = hstack([word_vec.transform(test["text"]), char_vec.transform(test["text"])]).tocsr()

def make_model():
    return LogisticRegression(C=1.0, solver="liblinear", multi_class="ovr", max_iter=1000)

# CV out-of-fold to estimate log-loss and produce averaged test preds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros((Xtr.shape[0], 3))
test_pred = np.zeros((Xte.shape[0], 3))

for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y)):
    model = make_model()
    model.fit(Xtr[tr_idx], y[tr_idx])
    # map model.classes_ ordering to our class order
    proba_va = model.predict_proba(Xtr[va_idx])
    proba_te = model.predict_proba(Xte)
    # reorder columns to EAP,HPL,MWS
    col_order = [list(model.classes_).index(i) for i in range(3)]
    oof[va_idx] = proba_va[:, col_order]
    test_pred += proba_te[:, col_order] / skf.n_splits

cv_ll = log_loss(y, oof, labels=[0, 1, 2])
print(f"CV log-loss: {cv_ll:.5f}")

sub = pd.DataFrame({"id": test["id"]})
sub["EAP"] = test_pred[:, 0]
sub["HPL"] = test_pred[:, 1]
sub["MWS"] = test_pred[:, 2]

# Align to sample_submission id order/columns exactly
sub = sample[["id"]].merge(sub, on="id", how="left")
sub = sub[["id", "EAP", "HPL", "MWS"]]
assert sub.isnull().sum().sum() == 0, "null predictions!"
assert len(sub) == len(sample), "row count mismatch"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape)
print(sub.head())
