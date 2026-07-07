import os, glob, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# avoid shared-memory (/dev/shm) exhaustion segfaults in DataLoader workers
try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

t0 = time.time()
DATA = os.environ["DATA_DIR"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DATA_DIR", DATA, "device", device, flush=True)

torch.manual_seed(0)
np.random.seed(0)

IMG = 224
BATCH = 48
EPOCHS = 12
NUM_WORKERS = 4
TIME_BUDGET = 1600  # stop training if we approach this; leave room for inference+grade

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
sub_cols = list(sample.columns)  # id_code, diagnosis
print("train", train_df.shape, "test", sample.shape, "cols", sub_cols, flush=True)

TRAIN_DIR = os.path.join(DATA, "train_images")
TEST_DIR = os.path.join(DATA, "test_images")

def find_img(d, idc):
    p = os.path.join(d, idc + ".png")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(d, idc + ".*"))
    return hits[0] if hits else None

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DS(Dataset):
    def __init__(self, ids, dirp, tf, labels=None):
        self.ids = list(ids); self.dirp = dirp; self.tf = tf; self.labels = labels
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        idc = self.ids[i]
        p = find_img(self.dirp, idc)
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        if self.labels is not None:
            return x, torch.tensor(float(self.labels[i]), dtype=torch.float32)
        return x, idc

train_ds = DS(train_df["id_code"].values, TRAIN_DIR, train_tf, train_df["diagnosis"].values)
train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                      pin_memory=True, drop_last=False, persistent_workers=True,
                      prefetch_factor=2)

# pretrained resnet18 (weights cached on box)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)  # ordinal regression
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.SmoothL1Loss()
scaler = torch.amp.GradScaler("cuda")

model.train()
done_epochs = 0
for ep in range(EPOCHS):
    if time.time() - t0 > TIME_BUDGET:
        print("time budget reached, stopping training early", flush=True)
        break
    tot = 0.0; n = 0
    for x, y in train_ld:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x).squeeze(1)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot += loss.item() * x.size(0); n += x.size(0)
    sched.step()
    done_epochs += 1
    print(f"epoch {ep+1}/{EPOCHS} loss {tot/n:.4f} t={time.time()-t0:.0f}s", flush=True)

# inference
test_ds = DS(sample["id_code"].values, TEST_DIR, test_tf)
test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                     pin_memory=True)
model.eval()
preds = {}
with torch.no_grad():
    for x, ids in test_ld:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x).squeeze(1)
        out = out.float().cpu().numpy()
        for idc, v in zip(ids, out):
            preds[idc] = v

vals = np.array([preds[i] for i in sample["id_code"].values])
diag = np.clip(np.rint(vals), 0, 4).astype(int)
out = pd.DataFrame({sub_cols[0]: sample["id_code"].values, sub_cols[1]: diag})
out.to_csv("submission.csv", index=False)
print("epochs_done", done_epochs, "wrote submission.csv rows", len(out),
      "dist", np.bincount(diag, minlength=5).tolist(), flush=True)
print("total time", f"{time.time()-t0:.0f}s", flush=True)
