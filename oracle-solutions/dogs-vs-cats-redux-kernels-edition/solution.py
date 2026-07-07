import os, re, time, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

t0 = time.time()
DATA_DIR = os.environ["DATA_DIR"]
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE = os.path.join(DATA_DIR, "sample_submission.csv")

# Allow env override for quick smoke test
TRAIN_SUBSAMPLE = int(os.environ.get("TRAIN_SUBSAMPLE", "5000"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
IMG = 128
BS = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, "subsample", TRAIN_SUBSAMPLE, "epochs", EPOCHS, flush=True)

# --- build train file list (label from filename: cat->0, dog->1) ---
all_train = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.jpg")))
cats = [f for f in all_train if os.path.basename(f).startswith("cat.")]
dogs = [f for f in all_train if os.path.basename(f).startswith("dog.")]
rng = np.random.RandomState(42)
rng.shuffle(cats); rng.shuffle(dogs)
per = TRAIN_SUBSAMPLE // 2
sel = cats[:per] + dogs[:per]
rng.shuffle(sel)
labels = [0 if os.path.basename(f).startswith("cat.") else 1 for f in sel]
print("train files", len(sel), "cats", sum(1 for l in labels if l==0), "dogs", sum(labels), flush=True)

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class ImgDS(Dataset):
    def __init__(self, files, labels=None, tf=None):
        self.files = files; self.labels = labels; self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        x = self.tf(img)
        if self.labels is not None:
            return x, self.labels[i]
        return x, i

train_ds = ImgDS(sel, labels, train_tf)
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")

model.train()
for ep in range(EPOCHS):
    running = 0.0; n = 0
    for x, y in train_dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x); loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        running += loss.item()*x.size(0); n += x.size(0)
    sched.step()
    print(f"epoch {ep} loss {running/n:.4f} t {time.time()-t0:.0f}s", flush=True)

# --- inference on test ---
test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
test_ids = [int(re.match(r"(\d+)\.jpg", os.path.basename(f)).group(1)) for f in test_files]
test_ds = ImgDS(test_files, None, eval_tf)
test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=6, pin_memory=True)

model.eval()
probs = np.zeros(len(test_files), dtype=np.float64)
with torch.no_grad():
    for x, idx in test_dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
        p = torch.softmax(out.float(), dim=1)[:,1].cpu().numpy()
        probs[idx.numpy()] = p

# clip to avoid infinite log-loss
probs = np.clip(probs, 1e-4, 1-1e-4)

import pandas as pd
sub = pd.DataFrame({"id": test_ids, "label": probs})
# order to match sample_submission
samp = pd.read_csv(SAMPLE)
sub = samp[["id"]].merge(sub, on="id", how="left")
sub["label"] = sub["label"].fillna(0.5)
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv", len(sub), "rows; total", f"{time.time()-t0:.0f}s", flush=True)
