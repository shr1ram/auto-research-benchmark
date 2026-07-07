import os, csv, glob, sys
import numpy as np

DATA = os.environ["DATA_DIR"]
ESS = os.path.join(DATA, "essential_data")
WAVDIR = os.path.join(ESS, "src_wavs")
N_SPECIES = 19

np.random.seed(0)

# ---- load mappings ----
rec2fn = {}
with open(os.path.join(ESS, "rec_id2filename.txt")) as f:
    r = csv.reader(f); next(r)
    for row in r:
        if not row: continue
        rec2fn[int(row[0])] = row[1]

# labels: rows with '?' are TEST; rows without are TRAIN (blank = no species present)
train_labels = {}   # rec_id -> set(species)
test_recs = []
with open(os.path.join(ESS, "rec_labels_test_hidden.txt")) as f:
    header = f.readline()
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split(",")
        rec = int(parts[0])
        rest = [p for p in parts[1:] if p != ""]
        if "?" in rest:
            test_recs.append(rec)
        else:
            labs = set(int(x) for x in rest)
            train_labels[rec] = labs

train_recs = sorted(train_labels.keys())
test_recs = sorted(test_recs)
print(f"train recs: {len(train_recs)}, test recs: {len(test_recs)}", flush=True)

# ---- feature extraction ----
import librosa

def _stats(M):
    # per-band mean/std/median/max/percentiles across time -> robust summary
    return np.concatenate([
        M.mean(axis=1), M.std(axis=1), np.median(M, axis=1),
        M.max(axis=1), np.percentile(M, 25, axis=1), np.percentile(M, 75, axis=1),
    ])

def features(rec_id):
    fn = rec2fn[rec_id]
    path = os.path.join(WAVDIR, fn + ".wav")
    y, sr = librosa.load(path, sr=16000, mono=True)
    # light pre-emphasis to boost high-freq bird calls
    y = librosa.effects.preemphasis(y)
    feats = []
    n_fft, hop = 512, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=40)
    logmel = librosa.power_to_db(mel + 1e-10)
    # MFCCs from logmel (24) + delta + delta2
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=24)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    for M in (mfcc, d1, d2):
        feats.append(M.mean(axis=1)); feats.append(M.std(axis=1))
    # log-mel band summary (captures which freq bands are active)
    feats.append(_stats(logmel))
    # spectral contrast
    sc = librosa.feature.spectral_contrast(S=S, sr=sr)
    feats.append(sc.mean(axis=1)); feats.append(sc.std(axis=1))
    # chroma
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    feats.append(chroma.mean(axis=1)); feats.append(chroma.std(axis=1))
    # scalar spectral descriptors
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr)
    flat = librosa.feature.spectral_flatness(S=S)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(S=S, frame_length=n_fft)
    for A in (cent, bw, roll, flat, zcr, rms):
        feats.append(np.array([A.mean(), A.std(), np.median(A), A.max()]))
    out = np.concatenate([np.asarray(x).ravel() for x in feats])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

all_recs = sorted(set(train_recs) | set(test_recs))
feat_cache = {}
for i, rec in enumerate(all_recs):
    feat_cache[rec] = features(rec)
    if (i+1) % 50 == 0:
        print(f"  extracted {i+1}/{len(all_recs)}", flush=True)

Xtr = np.vstack([feat_cache[r] for r in train_recs])
Xte = np.vstack([feat_cache[r] for r in test_recs])

# label matrix for train
Ytr = np.zeros((len(train_recs), N_SPECIES), dtype=int)
for i, r in enumerate(train_recs):
    for s in train_labels[r]:
        if 0 <= s < N_SPECIES:
            Ytr[i, s] = 1

# ---- standardize ----
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# ---- per-species classifier: ensemble of trees + calibrated logistic ----
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def species_proba(yc):
    pos = int(yc.sum())
    n = len(yc)
    if pos == 0:
        return np.zeros(len(Xte_s))
    if pos == n:
        return np.ones(len(Xte_s))
    probs = []
    # tree ensembles (handle nonlinearity, robust on raw scaled feats)
    et = ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                              class_weight="balanced_subsample",
                              min_samples_leaf=1, random_state=0, n_jobs=-1)
    et.fit(Xtr_s, yc); probs.append(et.predict_proba(Xte_s)[:, 1])
    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                class_weight="balanced_subsample",
                                min_samples_leaf=1, random_state=1, n_jobs=-1)
    rf.fit(Xtr_s, yc); probs.append(rf.predict_proba(Xte_s)[:, 1])
    # linear model (helps when signal is roughly linear in features)
    lr = LogisticRegression(max_iter=5000, C=0.3, class_weight="balanced")
    lr.fit(Xtr_s, yc); probs.append(lr.predict_proba(Xte_s)[:, 1])
    return np.mean(probs, axis=0)

# ---- internal CV AUC estimate (train-only, using CVfolds) ----
try:
    from sklearn.metrics import roc_auc_score
    fold = {}
    with open(os.path.join(ESS, "CVfolds_2.txt")) as f:
        rr = csv.reader(f); next(rr)
        for row in rr:
            if row: fold[int(row[0])] = int(row[1])
    tr_fold = np.array([fold.get(r, 0) for r in train_recs])
    cv_scores = []
    for fv in sorted(set(tr_fold)):
        tri = tr_fold != fv; tei = tr_fold == fv
        if tei.sum() == 0: continue
        # temporarily swap the global test matrix to eval on held-out fold
        _Xte_bak = Xte_s.copy()
        globals()['Xte_s'] = Xtr_s[tei]
        P = np.zeros((tei.sum(), N_SPECIES))
        for s in range(N_SPECIES):
            yc = Ytr[tri, s]
            _Xtr_bak = Xtr_s.copy()
            globals()['Xtr_s'] = _Xtr_bak[tri]
            try: P[:, s] = species_proba(yc)
            except Exception: P[:, s] = yc.mean() if len(yc) else 0.0
            globals()['Xtr_s'] = _Xtr_bak
        globals()['Xte_s'] = _Xte_bak
        yt = Ytr[tei]
        cols = [c for c in range(N_SPECIES) if len(set(yt[:, c])) > 1]
        if cols:
            cv_scores.append(roc_auc_score(yt[:, cols], P[:, cols], average="macro"))
    if cv_scores:
        print(f"CV macro-AUC: {np.mean(cv_scores):.4f}", flush=True)
except Exception as e:
    print(f"CV skip: {e}", flush=True)

preds = np.zeros((len(test_recs), N_SPECIES))
for s in range(N_SPECIES):
    try:
        preds[:, s] = species_proba(Ytr[:, s])
    except Exception as e:
        print(f"species {s} fallback: {e}", flush=True)
        preds[:, s] = Ytr[:, s].mean()

# ---- write submission matching sample order EXACTLY ----
sample = os.path.join(DATA, "sample_submission.csv")
rec_index = {r: i for i, r in enumerate(test_recs)}
rows_out = []
with open(sample) as f:
    r = csv.reader(f); next(r)
    for row in r:
        if not row: continue
        Id = int(row[0])
        rec = Id // 100
        sp = Id % 100
        p = float(preds[rec_index[rec], sp]) if rec in rec_index and sp < N_SPECIES else 0.0
        rows_out.append((Id, p))

with open("submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Id", "Probability"])
    for Id, p in rows_out:
        w.writerow([Id, f"{p:.6f}"])

print(f"wrote submission.csv with {len(rows_out)} rows", flush=True)
