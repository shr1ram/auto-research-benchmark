import os, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

t0 = time.time()
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = os.environ["DATA_DIR"]
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

# ---- subsample balanced ~10000 ----
labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
N_PER_CLASS = 15000
pos = labels[labels.label == 1]
neg = labels[labels.label == 0]
pos = pos.sample(n=min(N_PER_CLASS, len(pos)), random_state=SEED)
neg = neg.sample(n=min(N_PER_CLASS, len(neg)), random_state=SEED)
train_df = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print("train subsample:", len(train_df), "pos:", int(train_df.label.sum()), flush=True)

IMG = 96
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    norm,
])

class TrainDS(Dataset):
    def __init__(self, df, tf):
        self.ids = df.id.values
        self.y = df.label.values.astype(np.float32)
        self.tf = tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(os.path.join(TRAIN_DIR, self.ids[i] + ".tif")).convert("RGB")
        return self.tf(img), self.y[i]

class TestDS(Dataset):
    def __init__(self, ids, tf):
        self.ids = ids; self.tf = tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(os.path.join(TEST_DIR, self.ids[i] + ".tif")).convert("RGB")
        return self.tf(img), i

train_loader = DataLoader(TrainDS(train_df, train_tf), batch_size=128,
                          shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

EPOCHS = 4
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(train_loader))

model.train()
for ep in range(EPOCHS):
    tl = 0.0; nb = 0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tl += loss.item(); nb += 1
    print(f"epoch {ep} loss {tl/max(nb,1):.4f} t {time.time()-t0:.0f}s", flush=True)

# ---- inference on ALL test rows in sample_submission order ----
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
test_ids = sub.id.values
test_loader = DataLoader(TestDS(test_ids, eval_tf), batch_size=256,
                         shuffle=False, num_workers=8, pin_memory=True)

model.eval()
preds = np.zeros(len(test_ids), dtype=np.float32)
with torch.no_grad():
    for x, idx in test_loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(x).squeeze(1)
        p = torch.sigmoid(out).float().cpu().numpy()
        preds[idx.numpy()] = p

sub["label"] = preds
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv rows", len(sub), "t", time.time()-t0, flush=True)
print(sub.head())

