# Battery MFI Defect Detection
End-to-end pipeline for **automatic battery defect diagnosis** from **high-cost, low-throughput Magnetic Field Imaging (MFI)** data.

## Overview
- **Problem:** MFI enables non-contact defect diagnosis by measuring induced magnetic fields (x/y/z), but data collection is slow/expensive.
- **Approach:** Augment tiny MFI datasets (Rotation/Inversion/Synthesis/Mix-up) and benchmark classical classifiers to select a reliable detector.
- **Key result:** **16 images (8 normal / 8 defect) → 39,168 images**, improving average performance from **90.78% Acc / 0.90 AUC** to **97.77% Acc / 0.99 AUC** with **RI**, and reaching **100.0% Acc / 1.00 AUC** (**SVM on RI**).

---

## What
Train ML classifiers on **augmented MFI images**, then systematically compare **augmentation × model** settings to identify a strong and stable detector under extreme low-data constraints.

## Why MFI + why augmentation
MFI can detect abnormal current-flow patterns non-destructively, but MFI acquisition is time-intensive; augmentation increases dataset diversity and reduces the need for additional scans.

## My role
Led end-to-end execution: augmentation design, model benchmarking (DT/LR/LogR/RF/SVM/kNN), and evaluation/reporting for poster/paper deliverables.

---

## Method

### Pipeline
1) Acquire MFI images (normal vs defect-simulated)  
2) Generate augmented datasets  
3) Train classical classifiers  
4) Evaluate with Accuracy/AUC and select the best setting

### Dataset abbreviations
- **ORIG:** original dataset (no augmentation)  
- **R:** Rotation only  
- **I:** Inversion (flip) only  
- **RI:** Rotation + Inversion  
- **SYN:** Synthesis (pixel-average of two images)  
- **SYN-RI:** Synthesis + RI  
- **MIX:** Mix-up (normal/defect mixing at 8:2 and inverse)  
- **MIX-RI:** Mix-up + RI

### Augmentations
- **Rotation:** 0.2° increments from -1° to 1° to mitigate scan-angle variation.
- **Inversion:** horizontal/vertical flips to reduce orientation sensitivity.
- **Synthesis:** generate new images by averaging pixel values of two images.
- **Mix-up:** synthesize images by mixing normal/defect images at an 8:2 ratio (and vice versa).

<img width="1596" height="898" alt="overview" src="https://github.com/user-attachments/assets/3a23a7a7-1a73-4753-9c40-ff200bec3eed" />

---

## Demo / Results

<img width="527" height="548" alt="results_overview" src="https://github.com/user-attachments/assets/8f3a5933-3066-499b-9017-b6cccd060102" />

*Accuracy/AUC across augmentation sets (ORIG, RI, SYN, SYN-RI, MIX, MIX-RI) and classical classifiers (DT, kNN, LR, LogR, RF, SVM).*

### Key takeaways
- **Data scaling:** **16 → 39,168 images** via augmentation.
- **Consistent winner:** **RI** is strong across models → suggests robustness to scan-angle/orientation nuisance factors. 
- **Best setting:** **SVM + RI = 100.0% Acc / 1.00 AUC** (perfect separation in this experiment).
- **Strong runner-ups (examples):**
  - **LR + RI:** 99.31% Acc / 1.00 AUC
  - **kNN + RI:** 98.39% Acc / 0.98 AUC
  - **RF + SYN-RI:** 98.85% Acc / 1.00 AUC
  - **LogR + SYN-RI:** 97.24% Acc / 1.00 AUC  
  
---

## How to run
```bash
pip install -r requirements.txt
python src/train.py --config configs/exp.yaml
python src/eval.py  --config configs/exp.yaml
