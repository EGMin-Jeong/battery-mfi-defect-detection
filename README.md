# Battery MFI Defect Detection
End-to-end pipeline for **automatic battery defect diagnosis** from high-cost, low-throughput Magnetic Field Imaging (MFI) data.

## Overview
- **Problem:** MFI enables non-contact, in-situ defect diagnosis by measuring induced magnetic fields (x/y/z), but data collection is slow/expensive.  
- **Approach:** augment tiny MFI datasets (Rotation/Inversion/Synthesis/Mix-up) and benchmark multiple classical classifiers to pick the most reliable detector.
- **Key result:** from **16 images (8 normal / 8 defective)** to **39,168 images**, improving average performance from **90.78% Acc / 0.90 AUC** to **97.77% Acc / 0.99 AUC** with **RI**, and achieving **100.0% Acc / 1.00 AUC** (SVM on RI).

---

## What
We build an automated battery defect detection pipeline by training ML classifiers on **augmented MFI images**, then systematically comparing models/augmentations to identify a strong and stable detector under extreme low-data constraints.

## Why MFI + why augmentation
MFI measures magnetic field intensity in x/y/z directions and can detect abnormal current flow non-destructively; however, acquiring MFI images is time-intensive, motivating augmentation to increase diversity and reduce data acquisition burden.

## My role
Led end-to-end execution: augmentation design, model benchmarking (DT/LR/LogR/RF/SVM/kNN), and evaluation/reporting for poster/paper deliverables.

---

## Method

### Pipeline
1) Acquire MFI images for normal vs defect-simulated batteries  
2) Generate augmented datasets  
3) Train classical classifiers  
4) Evaluate with Accuracy/AUC and select the best setting

### Augmentations (what each dataset name means)
We create multiple training sets by combining these operations:

- **R (Rotation):** rotate in **0.2° increments from -1° to 1°** to mitigate scan-angle variation.
- **I (Inversion):** horizontal/vertical flips to reduce orientation sensitivity.
- **RI:** Rotation + Inversion combined. (R then I)
- **SYN (Synthesis):** generate new images by averaging pixel values of two images.
- **SYN-RI:** apply RI to synthesized images (SYN + RI).
- **MIX (Mix-up):** create mixed samples by combining normal/defect images at an **8:2 ratio** (and inversely for defect-based mixing).
- **MIX-RI:** apply RI on top of Mix-up images (MIX + RI).
- **ORIG:** original dataset without augmentation.

### Method overview figure
<img width="1596" height="898" alt="overview" src="https://github.com/user-attachments/assets/3a23a7a7-1a73-4753-9c40-ff200bec3eed" />

---

## Demo / Results

<img width="527" height="548" alt="results_overview" src="https://github.com/user-attachments/assets/8f3a5933-3066-499b-9017-b6cccd060102" />

*Accuracy/AUC across augmentation sets (ORIG, RI, SYN, SYN-RI, MIX, MIX-RI) and classical classifiers (DT, kNN, LR, LogR, RF, SVM).*

### Key results (copy-paste friendly)
- **Data scaling:** **16 images → 39,168 images** via augmentation.
- **Average improvement with RI:** baseline (ORIG) **90.78% Acc / 0.90 AUC** → **97.77% Acc / 0.99 AUC** when RI is incorporated across datasets (avg over 3 experiments).
- **Best setting:** **SVM on RI** achieves **100.0% accuracy / 1.00 AUC**.
- **Interpretation:** RI reduces nuisance sensitivity (scan-angle/orientation), and strong margin/ensemble models remain robust across augmented distributions; MIX-RI can underperform MIX due to diluted defect signals/noise from cross-class mixing.

---

## How to run
```bash
pip install -r requirements.txt
python src/train.py --config configs/exp.yaml
python src/eval.py  --config configs/exp.yaml
