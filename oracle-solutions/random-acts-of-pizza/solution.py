import os, json
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

DATA_DIR = os.environ["DATA_DIR"]

with open(os.path.join(DATA_DIR, "train.json")) as f:
    train = json.load(f)
with open(os.path.join(DATA_DIR, "test.json")) as f:
    test = json.load(f)

# Text field present in BOTH train and test (request_text is train-only).
TEXT = "request_text_edit_aware"
TITLE = "request_title"

# Numeric metadata available at request time (present in both train & test).
NUM_FEATS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]

def get_text(rec):
    return (rec.get(TEXT) or "") + " " + (rec.get(TITLE) or "")

def get_num(rec):
    row = []
    for k in NUM_FEATS:
        v = rec.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        row.append(v)
    # engineered: text length
    row.append(float(len(rec.get(TEXT) or "")))
    row.append(float(len((rec.get(TEXT) or "").split())))
    return row

y = np.array([1 if r.get("requester_received_pizza") else 0 for r in train])
train_text = [get_text(r) for r in train]
test_text = [get_text(r) for r in test]

Xnum_tr = np.array([get_num(r) for r in train], dtype=float)
Xnum_te = np.array([get_num(r) for r in test], dtype=float)
# log1p on skewed count features (all non-negative except possibly upvotes_minus)
Xnum_tr = np.sign(Xnum_tr) * np.log1p(np.abs(Xnum_tr))
Xnum_te = np.sign(Xnum_te) * np.log1p(np.abs(Xnum_te))

scaler = StandardScaler()
Xnum_tr = scaler.fit_transform(Xnum_tr)
Xnum_te = scaler.transform(Xnum_te)

tfidf = TfidfVectorizer(
    sublinear_tf=True, min_df=2, max_df=0.9,
    ngram_range=(1, 2), stop_words="english", max_features=20000,
)
Xtxt_tr = tfidf.fit_transform(train_text)
Xtxt_te = tfidf.transform(test_text)

Xtr = hstack([Xtxt_tr, csr_matrix(Xnum_tr)]).tocsr()
Xte = hstack([Xtxt_te, csr_matrix(Xnum_te)]).tocsr()

clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="liblinear")

# quick CV AUC estimate for sanity
try:
    cv = cross_val_score(clf, Xtr, y, cv=5, scoring="roc_auc")
    print("CV AUC mean=%.4f std=%.4f" % (cv.mean(), cv.std()))
except Exception as e:
    print("CV skipped:", e)

clf.fit(Xtr, y)
proba = clf.predict_proba(Xte)[:, 1]

ids = [r["request_id"] for r in test]
import csv
with open("submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["request_id", "requester_received_pizza"])
    for rid, p in zip(ids, proba):
        w.writerow([rid, float(p)])

print("wrote submission.csv rows=%d positives_rate=%.3f" % (len(ids), y.mean()))

