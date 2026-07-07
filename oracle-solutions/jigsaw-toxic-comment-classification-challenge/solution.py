import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

t0 = time.time()

DATA_DIR = os.environ["DATA_DIR"]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

train_text = train["comment_text"].fillna("unknown").astype(str)
test_text = test["comment_text"].fillna("unknown").astype(str)
all_text = pd.concat([train_text, test_text])

print(f"train={train.shape} test={test.shape} sample={sample.shape}", flush=True)

# Word-level TF-IDF
word_vec = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
)
word_vec.fit(all_text)
train_word = word_vec.transform(train_text)
test_word = word_vec.transform(test_text)
print(f"word tfidf done t={time.time()-t0:.1f}s", flush=True)

# Char-level TF-IDF
char_vec = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(2, 4),
    max_features=50000,
)
char_vec.fit(all_text)
train_char = char_vec.transform(train_text)
test_char = char_vec.transform(test_text)
print(f"char tfidf done t={time.time()-t0:.1f}s", flush=True)

X_train = hstack([train_word, train_char]).tocsr()
X_test = hstack([test_word, test_char]).tocsr()
print(f"X_train={X_train.shape} X_test={X_test.shape}", flush=True)

# submission frame keyed by test id
sub = pd.DataFrame({"id": test["id"]})
for label in LABELS:
    y = train[label].values
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
    clf.fit(X_train, y)
    sub[label] = clf.predict_proba(X_test)[:, 1]
    print(f"label {label} done t={time.time()-t0:.1f}s", flush=True)

# Align exactly to sample_submission (row order via id, column order fixed)
sub = sample[["id"]].merge(sub, on="id", how="left")
sub = sub[["id"] + LABELS]
sub[LABELS] = sub[LABELS].fillna(0.5)

sub.to_csv("submission.csv", index=False)
print(f"submission written rows={len(sub)} t={time.time()-t0:.1f}s", flush=True)
print(sub.head(3).to_string(), flush=True)
