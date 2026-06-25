"""The MLE-Bench *Lite* task set.

MLE-bench's official "Lite" = the Low-complexity split (22 Kaggle competitions,
~158GB). We start with a curated, smallest-first ordering so a single competition
can be prepared and run end-to-end without pulling the whole 158GB split.

`LITE_COMPETITIONS` mirrors mlebench's experiments/splits/low.txt. `SMALL_FIRST`
is our recommended ordering by dataset footprint (smallest first) so the first
end-to-end run is cheap. Both are plain data — the benchmark validates ids
against the installed mlebench registry at load time.
"""
from __future__ import annotations

# The Low-complexity ("Lite") split, per mlebench experiments/splits/low.txt.
LITE_COMPETITIONS: list[str] = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "us-patent-phrase-to-phrase-matching",
]

# Smallest-footprint-first ordering for cheap smoke runs. These are tabular/text
# competitions with tiny datasets — ideal for the first end-to-end validation.
SMALL_FIRST: list[str] = [
    "random-acts-of-pizza",                 # ~tiny JSON, text classification
    "detecting-insults-in-social-commentary",
    "spooky-author-identification",
    "nomad2018-predict-transparent-conductors",
    "leaf-classification",
]
