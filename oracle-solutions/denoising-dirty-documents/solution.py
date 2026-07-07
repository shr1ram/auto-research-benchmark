import os
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

DATA_DIR = os.environ["DATA_DIR"]
test_dir = os.path.join(DATA_DIR, "test")

# Image ordering must match sampleSubmission.csv exactly: image ids are sorted
# as STRINGS (e.g. 110,111,...,216,26,...,6,62,...,8,80,95). Rows/cols are
# 1-indexed, iteration is row-major (col varies fastest within a row).
names = [f for f in os.listdir(test_dir) if f.lower().endswith(".png")]
img_ids = sorted(n[:-4] for n in names)  # strip .png, string sort

dims = {}
total = 0
for iid in img_ids:
    im = Image.open(os.path.join(test_dir, iid + ".png"))
    w, h = im.size
    dims[iid] = (w, h)
    total += w * h

# Tuned on train_cleaned (all 115 imgs): background-division illumination
# correction. NO pre-median (it blurs thin text strokes and hurt badly).
BG_WIN = 21   # large median => background/illumination estimate
WHITE_THR = 0.85  # push near-white pixels fully white (clean paper)

out_path = os.path.join(os.getcwd(), "submission.csv")
with open(out_path, "w", buffering=1024 * 1024) as f:
    f.write("id,value\n")
    for iid in img_ids:
        w, h = dims[iid]
        im = Image.open(os.path.join(test_dir, iid + ".png")).convert("L")
        arr = np.asarray(im, dtype=np.float64) / 255.0  # (h, w) in [0,1]

        bg = median_filter(arr, size=BG_WIN)
        bg = np.clip(bg, 1e-3, None)
        cleaned = np.clip(arr / bg, 0.0, 1.0)
        cleaned = np.where(cleaned > WHITE_THR, 1.0, cleaned)

        rows_idx = np.repeat(np.arange(1, h + 1), w)
        cols_idx = np.tile(np.arange(1, w + 1), h)
        vals = cleaned.reshape(-1)
        lines = [
            "%s_%d_%d,%.6g\n" % (iid, r, c, v)
            for r, c, v in zip(rows_idx.tolist(), cols_idx.tolist(), vals.tolist())
        ]
        f.write("".join(lines))

print("WROTE", out_path, "images", len(img_ids), "rows", total)
