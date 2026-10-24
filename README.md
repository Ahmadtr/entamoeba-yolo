# Deep Learning-Based Entamoeba Detection in Brightfield Microscopy Images Using Lightweight YOLO Variants

Code, data and results for PNTD-D-26-00436.
Every table and figure in the manuscript can be regenerated from this package.

---

## Quick check

```
python verify_package.py            # data + tables, ~20 s, no GPU
python verify_package.py --weights  # also re-runs inference (needs the weights)
```

25 checks: file counts and naming, md5 of every image, annotation and class
counts, fold integrity (no patient in two folds), and that the published tables
regenerate from the per-fold results. Exit code 0 means everything passed.

## Contents

```
data/
  internal/   90 images (image1..image90), annotations.csv, groups.csv, manifest.csv
  external/   80 images (image1..image80), labels/, annotations.csv, manifest.csv
  control/    13 images (image1..image13), manifest.csv
code/
  train_kfold.py                      three-stage progressive training, 5-fold
  run_ablation.py                     single-stage ablation at matched epochs
  evaluate_external_and_control.py    external validation + control false positives
  calibration_sweep.py                recall vs false positives across thresholds
  make_data_figures.py                Figs 3, 4, 5, 6, S4
  make_qual_figures.py                Figs 1, 7, 8, 9, S1-S3
results/      per-architecture JSON, CSV outputs, tables/, figures/
env/          requirements.txt
verify_package.py
```

`manifest.csv` lists every released image with its md5 and size, so you can
confirm the files are the ones the paper used.

## The trained models

The 45 stage-3 models (154 MB) are **attached to the GitHub release**, not
committed to the repository. Download `entamoeba-weights.zip` from the
Releases page and unpack it into the package root so that `weights/` sits
beside `code/` and `data/`.

```
entamoeba-weights.zip   161,385,315 bytes
md5     fc4d661e472a3de80dcc08b31518e02f
sha256  fd6eb5fd53286951a38e09dfde0fa3f89c9acff1c8e4de4764739f86810acf85
```

The repository's own files need no checksum file: git hashes every object, so
`git clone` verifies them for you. The per-image md5 values in each
`manifest.csv` serve a different purpose - they attest that a released image is
byte-identical to the file that produced the published results.

They are needed to:

- regenerate Figs 1, 7, 8, 9 and S1-S3, which run inference
- reproduce Tables 5 and 6 and the control-set counts without retraining
- run `verify_package.py --weights`

Tables 2, 3, 4 and Figs 3, 4, 5, 6 and S4 regenerate from `results/` alone and
need no weights and no GPU.

## Data

`image1 ... imageN` in all three sets, contiguous, no duplicates.

`data/internal/groups.csv` gives the patient identifier for each image.
Cross-validation is grouped on this column: all images from one patient fall in
the same fold. 90 images span 87 patients (two patients contributed more than
one image).

`data/internal/annotations.csv` holds the ground truth (188 boxes). It is the
second of two independent annotation passes; agreement between the passes was
Cohen's kappa 0.897.

The external set is 80 annotated images (207 parasites) from an independent
cohort collected at a different hospital with a different microscope and
smartphone. The control set is 13 parasite-free images.

Images are released exactly as they were used for training and evaluation. They
are not cropped or otherwise adjusted, so the package reproduces the published
numbers rather than a cleaned-up variant of them.

## Environment

```
conda create -p ./envs/amoeba python=3.11
./envs/amoeba/python -m pip install -r env/requirements.txt
```

Pinned versions matter: numpy must stay below 2.0 for this OpenCV build.

## Reproducing the results

Without retraining, using the released weights:

```
python code/make_data_figures.py --results results/ --out results/figures
python code/make_qual_figures.py --package .  --results results/ --out results/figures
```

From scratch (about 13 hours on an RTX A4000):

```
python code/train_kfold.py
python code/evaluate_external_and_control.py
python code/calibration_sweep.py
python code/run_ablation.py
```

Training is stochastic. Retraining will not reproduce the released weights
bit-for-bit, and fold-level metrics typically vary by a few thousandths. The
released weights reproduce the published tables exactly; retraining reproduces
them approximately.

## Which output corresponds to which table

| Manuscript | File | Weights needed |
|---|---|---|
| Table 2 | `results/tables/Table2_macro_average.csv` | no |
| Table 3 | `results/tables/Table3_per_fold.csv` | no |
| Table 4 | `results/tables/Table4_control_false_positives.csv` | no |
| Table 5 | `results/tables/Table5_external.csv` | no |
| Ablation | `results/tables/Table6_ablation.csv` | no |
| Figs 3, 4, 5, 6, S4 | `results/figures/` | no |
| Figs 1, 7, 8, 9, S1-S3 | `results/figures/` | yes |

## Fold assignment

Fold membership is fixed in `results/true_folds.json` and read from that file
rather than recomputed, so every script uses identical splits.

## Third-party images

27 of the 90 internal images come from public sources (CDC Public Health Image
Library, parasite.org, medical-labs.net and published figures).
`data/internal/manifest.csv` records the source of each.

13 of these are CDC Public Health Image Library images, which are US federal
government works in the public domain. The remainder are reproduced here under
the terms of their original sources; where redistribution rights could not be
confirmed, consult the source column rather than redistributing the file.

## Licence

Code: MIT. Clinical images: CC BY 4.0. Third-party images remain under their
original terms.
