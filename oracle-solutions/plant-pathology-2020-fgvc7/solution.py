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
IMG_DIR = os.path.join(DATA, "images")
LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
IMG = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
print("train", train_df.shape, "test", sample.shape)

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PPDataset(Dataset):
    def __init__(self, ids, labels, tf):
        self.ids = list(ids)
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(IMG_DIR, self.ids[i] + ".jpg")).convert("RGB")
        x = self.tf(img)
        if self.labels is not None:
            y = torch.tensor(self.labels[i], dtype=torch.long)
            return x, y
        return x, 0


y_int = train_df[LABELS].values.argmax(1)
train_ds = PPDataset(train_df["image_id"].values, y_int, train_tf)
test_ds = PPDataset(sample["image_id"].values, None, eval_tf)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

try:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("loaded pretrained resnet18")
except Exception as e:
    print("pretrained unavailable, from scratch:", e)
    model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")
EPOCHS = 8
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

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
        tot += loss.item() * x.size(0)
        n += x.size(0)
    sched.step()
    print(f"epoch {ep+1}/{EPOCHS} loss={tot/n:.4f} t={time.time()-t0:.0f}s")

model.eval()
probs = []
with torch.no_grad():
    for x, _ in test_dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(x)
        p = torch.softmax(out.float(), dim=1).cpu().numpy()
        probs.append(p)
probs = np.concatenate(probs, 0)

sub = sample.copy()
sub[LABELS] = probs
sub.to_csv("submission.csv", index=False)
print("wrote submission.csv", sub.shape, "total t=", f"{time.time()-t0:.0f}s")
print(sub.head(3).to_string())
