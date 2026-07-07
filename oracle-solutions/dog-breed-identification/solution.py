import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

t0 = time.time()
DATA = os.environ["DATA_DIR"]
TRAIN_DIR = os.path.join(DATA, "train")
TEST_DIR = os.path.join(DATA, "test")
IMG = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
breed_cols = list(sample.columns[1:])          # 120 breeds in exact submission order
assert len(breed_cols) == 120, len(breed_cols)
breed_to_idx = {b: i for i, b in enumerate(breed_cols)}

labels = pd.read_csv(os.path.join(DATA, "labels.csv"))
labels["target"] = labels["breed"].map(breed_to_idx)
assert labels["target"].notna().all(), "unmapped breed"
labels["target"] = labels["target"].astype(int)
print("train", labels.shape, "test", sample.shape, flush=True)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize(int(IMG * 256 / 224)),
    transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DogDS(Dataset):
    def __init__(self, ids, img_dir, targets, tf):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.targets = targets
        self.tf = tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.ids[i] + ".jpg")).convert("RGB")
        x = self.tf(img)
        if self.targets is not None:
            return x, torch.tensor(self.targets[i], dtype=torch.long)
        return x, i


train_ds = DogDS(labels["id"].values, TRAIN_DIR, labels["target"].values, train_tf)
test_ids = sample["id"].tolist()
test_ds = DogDS(test_ids, TEST_DIR, None, eval_tf)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

try:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("loaded pretrained resnet18", flush=True)
except Exception as e:
    print("pretrained unavailable, from scratch:", e, flush=True)
    model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 120)
model = model.to(device)

# Discriminative LR: gentle on pretrained backbone, faster on the new head.
head_params = list(model.fc.parameters())
head_ids = {id(p) for p in head_params}
backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
opt = torch.optim.AdamW([
    {"params": backbone_params, "lr": 5e-4},
    {"params": head_params, "lr": 5e-3},
], weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")
EPOCHS = 18
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-4, 5e-3],
        steps_per_epoch=len(train_dl), epochs=EPOCHS, pct_start=0.15)

model.train()
for ep in range(EPOCHS):
    tot, n = 0.0, 0
    for x, y in train_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item() * x.size(0)
        n += x.size(0)
    print(f"epoch {ep+1}/{EPOCHS} loss={tot/n:.4f} t={time.time()-t0:.0f}s", flush=True)

model.eval()
probs = np.zeros((len(test_ids), 120), dtype=np.float32)
with torch.no_grad():
    for x, idx in test_dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
            out_f = model(torch.flip(x, dims=[3]))  # horizontal-flip TTA
        p = 0.5 * (torch.softmax(out.float(), dim=1) + torch.softmax(out_f.float(), dim=1))
        probs[idx.numpy()] = p.cpu().numpy()
print("inference done t=", int(time.time()-t0), flush=True)

sub = sample.copy()
sub[breed_cols] = probs
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv", sub.shape, "total t=", int(time.time()-t0), flush=True)
print(sub.iloc[:2, :4].to_string(), flush=True)
