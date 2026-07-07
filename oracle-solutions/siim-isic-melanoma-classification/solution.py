import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

t0 = time.time()
DATA = os.environ["DATA_DIR"]
TRAIN_CSV = os.path.join(DATA, "train.csv")
TEST_CSV = os.path.join(DATA, "test.csv")
JPEG_TRAIN = os.path.join(DATA, "jpeg", "train")
JPEG_TEST = os.path.join(DATA, "jpeg", "test")

SUBSAMPLE = os.environ.get("SUBSAMPLE", "0") == "1"
IMG = 128
EPOCHS = 4
BS = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Subsample: keep ALL positives, sample negatives ~1:3
pos = train_df[train_df.target == 1]
neg = train_df[train_df.target == 0]
n_neg = min(len(neg), len(pos) * 3)  # ~1:3
neg_s = neg.sample(n=n_neg, random_state=42)
sub = pd.concat([pos, neg_s]).sample(frac=1.0, random_state=42).reset_index(drop=True)
if SUBSAMPLE:
    sub = sub.sample(n=min(400, len(sub)), random_state=1).reset_index(drop=True)
    test_df = test_df.head(200).copy()
    EPOCHS = 1
print("train subsample", len(sub), "pos", int(sub.target.sum()), "neg", int((sub.target==0).sum()), flush=True)
print("test", len(test_df), flush=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class DS(Dataset):
    def __init__(self, df, img_dir, tf, has_target):
        self.names = df.image_name.tolist()
        self.tf = tf
        self.img_dir = img_dir
        self.has_target = has_target
        self.targets = df.target.tolist() if has_target else None
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        p = os.path.join(self.img_dir, self.names[i] + ".jpg")
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        if self.has_target:
            return x, torch.tensor(self.targets[i], dtype=torch.float32)
        return x, self.names[i]

train_ds = DS(sub, JPEG_TRAIN, train_tf, True)
test_ds = DS(test_df, JPEG_TEST, eval_tf, False)
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

# pos_weight to handle imbalance within the ~1:3 subsample
n_pos = int(sub.target.sum()); n_neg = len(sub) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")

model.train()
for ep in range(EPOCHS):
    running = 0.0
    for x, y in train_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item() * x.size(0)
    print(f"epoch {ep} loss {running/len(train_ds):.4f} elapsed {time.time()-t0:.0f}s", flush=True)

# Predict
model.eval()
preds = {}
with torch.no_grad():
    for x, names in test_dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
        p = torch.sigmoid(out).squeeze(1).float().cpu().numpy()
        for nm, pv in zip(names, p):
            preds[nm] = float(pv)

# Build submission matching sample_submission order/columns
samp = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
samp["target"] = samp["image_name"].map(preds).fillna(0.0)
samp.to_csv("submission.csv", index=False)
print("wrote submission.csv rows", len(samp), "cols", list(samp.columns), flush=True)
print("pred stats min/mean/max", samp.target.min(), samp.target.mean(), samp.target.max(), flush=True)
print("total elapsed", time.time()-t0, flush=True)
