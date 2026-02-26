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


# Robust Battery Defect Detection via Magnetic Field Imaging (MFI)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classical_Classifiers-FF6F00.svg)]()
[![Data](https://img.shields.io/badge/Data-Augmentation-brightgreen.svg)]()

> **End-to-end pipeline for automatic battery defect diagnosis using high-cost, low-throughput Magnetic Field Imaging (MFI) data.**

This repository contains the implementation, data augmentation strategies, and benchmarking suite for robust battery defect detection under extreme low-data constraints.

## 📖 Abstract
**Magnetic Field Imaging (MFI)** enables non-destructive defect diagnosis by measuring induced magnetic fields (x/y/z) to detect abnormal current-flow patterns. However, MFI data collection is mechanically slow and highly expensive. 

This project tackles the **extreme low-data problem** by introducing a domain-specific data augmentation pipeline. By simulating mechanical scan-angle variations and orientation sensitivities, we successfully scaled a tiny dataset of **16 core images** (8 normal / 8 defect) into a robust training set of **39,168 images**. This approach significantly stabilized classifier performance, identifying a highly reliable and computationally efficient detector for MFI-based quality assurance.

### ✨ Key Contributions
- **Domain-Specific Augmentation:** Designed physical-world-informed augmentations (Rotation, Inversion, Synthesis, Mix-up) to overcome data scarcity.
- **Extensive Benchmarking:** Systematically evaluated 6 classical ML classifiers (DT, LR, LogR, RF, SVM, kNN) across various augmentation sets.
- **State-of-the-Art Results:** Improved baseline average performance from 90.78% Acc / 0.90 AUC to **97.77% Acc / 0.99 AUC**, achieving perfect separation (**100.0% Acc / 1.00 AUC**) with SVM on the Rotation+Inversion (RI) dataset.

---

## ⚙️ Methodology

### 1. Augmentation Pipeline
To build a robust dataset and mitigate nuisance factors (e.g., scan-angle variance), combinations of the following techniques were applied:

* **ORIG:** Baseline dataset (no augmentation).
* **Rotation (R):** Rotated in 0.2° increments from -1° to 1° to mitigate mechanical scan-angle variations.
* **Inversion (I):** Horizontal/vertical flips to reduce spatial orientation sensitivity.
* **Synthesis (SYN):** Pixel-wise averaging of two images to generate intermediate states.
* **Mix-up (MIX):** Synthesizing normal and defect images at an 8:2 ratio (and vice versa).
* *Combined Sets:* **RI** (Rotation + Inversion), **SYN-RI**, **MIX-RI**.

<p align="center">
  <img width="800" alt="augmentation_overview" src="https://github.com/user-attachments/assets/3a23a7a7-1a73-4753-9c40-ff200bec3eed">
  <br>
  <em>Figure 1: Overview of the MFI data augmentation and classification pipeline.</em>
</p>

### 2. Experimental Setup
1. **Acquisition:** Collect raw MFI images (Normal vs. Defect-simulated).
2. **Generation:** Execute the augmentation pipeline (16 images → 39,168 images).
3. **Training & Evaluation:** Train classical classifiers and benchmark using Accuracy and AUC metrics to select the optimal model-dataset pair.

---

## 📊 Benchmark Results

The Rotation + Inversion (**RI**) dataset consistently demonstrated the highest robustness across nearly all models, suggesting that resolving mechanical scan-angle and orientation variances is the most critical factor for accurate MFI data classification.

<p align="center">
  <img width="500" alt="results_table" src="https://github.com/user-attachments/assets/8f3a5933-3066-499b-9017-b6cccd060102">
  <br>
  <em>Table 1: Accuracy and AUC across augmentation sets and classical classifiers.</em>
</p>

### 🏆 Top Performing Configurations
| Model | Augmentation Set | Accuracy | AUC |
|:---:|:---:|:---:|:---:|
| **SVM** | **RI** | **100.0%** | **1.00** |
| LR | RI | 99.31% | 1.00 |
| RF | SYN-RI | 98.85% | 1.00 |
| kNN | RI | 98.39% | 0.98 |

---

## 🚀 Getting Started

### Prerequisites
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YourUsername/YourRepositoryName.git](https://github.com/YourUsername/YourRepositoryName.git)
cd YourRepositoryName
pip install -r requirements.txt
