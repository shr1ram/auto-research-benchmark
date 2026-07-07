import os, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

t0 = time.time()
DATA_DIR = os.environ["DATA_DIR"]
SMALL = os.environ.get("SMALL", "0") == "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, "DATA_DIR", DATA_DIR, flush=True)

LABELS = ["ETT - Abnormal","ETT - Borderline","ETT - Normal","NGT - Abnormal",
          "NGT - Borderline","NGT - Incompletely Imaged","NGT - Normal",
          "CVC - Abnormal","CVC - Borderline","CVC - Normal","Swan Ganz Catheter Present"]

train_dir = os.path.join(DATA_DIR, "train")
test_dir  = os.path.join(DATA_DIR, "test")
train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
samp = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sub_cols = [c for c in samp.columns if c != "StudyInstanceUID"]
print("sub_cols match LABELS:", sub_cols == LABELS, flush=True)

# subsample train for speed
N_SUB = 200 if SMALL else 5500
if len(train_csv) > N_SUB:
    train_csv = train_csv.sample(n=N_SUB, random_state=42).reset_index(drop=True)
print("train rows used:", len(train_csv), flush=True)

IMG = 128
norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMG,IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), norm])
eval_tf = transforms.Compose([
    transforms.Resize((IMG,IMG)),
    transforms.ToTensor(), norm])

class DS(Dataset):
    def __init__(self, ids, ddir, labels=None, tf=None):
        self.ids=ids; self.ddir=ddir; self.labels=labels; self.tf=tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        p = os.path.join(self.ddir, self.ids[i] + ".jpg")
        img = Image.open(p).convert("RGB")
        img = self.tf(img)
        if self.labels is not None:
            return img, torch.tensor(self.labels[i], dtype=torch.float32)
        return img, self.ids[i]

y = train_csv[LABELS].values.astype("float32")
tr_ds = DS(train_csv["StudyInstanceUID"].tolist(), train_dir, y, train_tf)
tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)

test_ids = samp["StudyInstanceUID"].tolist()
te_ds = DS(test_ids, test_dir, None, eval_tf)
te_ld = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model = model.to(device)

crit = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
EPOCHS = 1 if SMALL else 4

model.train()
for ep in range(EPOCHS):
    tl=0.0; nb=0
    for x,yb in tr_ld:
        x=x.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            out=model(x); loss=crit(out,yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl+=loss.item(); nb+=1
    print(f"epoch {ep} loss {tl/max(nb,1):.4f} t={time.time()-t0:.0f}s", flush=True)

# predict
model.eval()
preds=[]; order=[]
with torch.no_grad():
    for x,ids in te_ld:
        x=x.to(device,non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            out=model(x)
        p=torch.sigmoid(out).float().cpu().numpy()
        preds.append(p); order.extend(list(ids))
preds=np.concatenate(preds,0)

out_df = pd.DataFrame(preds, columns=LABELS)
out_df.insert(0, "StudyInstanceUID", order)
# reindex to sample_submission row order exactly
out_df = out_df.set_index("StudyInstanceUID").loc[test_ids].reset_index()
out_df = out_df[["StudyInstanceUID"]+LABELS]
out_df.to_csv("submission.csv", index=False)
print("wrote submission.csv shape", out_df.shape, "t=", time.time()-t0, flush=True)
