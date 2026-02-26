# Battery MFI Defect Detection
Research project delivering an **end-to-end pipeline for automatic battery defect diagnosis** from **high-cost, low-throughput MFI** data.

**Goal:** Non-destructive battery defect detection using Magnetic Field Imaging (MFI) with **data augmentation + optimized classification**.

> MFI enables non-contact, in-situ diagnosis by measuring induced magnetic fields (x/y/z). Since acquiring MFI images is time-intensive, we use augmentation to expand limited data and train an automated defect detector.
---

## What
We propose an automated battery defect detection method by training ML classifiers on **augmented MFI images** (rotation, inversion, synthesis, mix-up), then benchmarking multiple models to find the most suitable detector.

## My role
- Led end-to-end research execution: augmentation design, model benchmarking (Decision Tree/Linear Regression/Logistic Regression/Random Forest/SVM/kNN), and evaluation/reporting for poster/paper deliverables.

## Key result
- Expanded an original dataset of **16 images (8 normal / 8 defective)** up to **39,168 images** via augmentation.
- Across 3 runs, baseline achieved **90.78% accuracy / 0.90 AUC**, while incorporating **Rotation+Inversion (RI)** improved to **97.77% accuracy / 0.99 AUC**.
---

## Method (Augmentations)
<img width="1596" height="898" alt="overview" src="https://github.com/user-attachments/assets/3a23a7a7-1a73-4753-9c40-ff200bec3eed" />

- **Rotation:** rotate images in **0.2° increments from -1° to 1°** to mitigate scan-angle variation.
- **Inversion:** vertical/horizontal flips to reduce orientation sensitivity.
- **Synthesis:** generate new images by averaging pixel values from two images.
- **Mix-up:** synthesize images by mixing normal/defective at an **8:2 ratio** (and vice versa).

---

## Demo / Results
<img width="527" height="548" alt="results_overview" src="https://github.com/user-attachments/assets/8f3a5933-3066-499b-9017-b6cccd060102" />

*Accuracy/AUC across augmentation sets (ORIG, RI, SYN, SYN-RI, MIX, MIX-RI) and classical classifiers (DT, kNN, LR, LogR, RF, SVM).*

**Key takeaways (60-sec scan):**
- **RI (Rotation + Inversion)** is consistently strong across models, suggesting robustness to scan-angle/orientation nuisance factors.
- **Best overall:** **SVM + RI** achieves **100.0% Acc / 1.00 AUC** (perfect separation in this setting).
- **Runner-ups:**  
  - **LR + RI:** **99.31% Acc / 1.00 AUC**  
  - **kNN + RI:** **98.39% Acc / 0.98 AUC**  
  - **RF + SYN-RI:** **98.85% Acc / 1.00 AUC**  
  - **LogR + SYN-RI:** **97.24% Acc / 1.00 AUC**
- **Model sensitivity:** Decision Tree shows unstable performance across augmentations, while margin/ensemble models (SVM/RF/LogR) remain strong, indicating the task benefits from more stable decision boundaries under augmented distributions.

---

## How to run
```bash
pip install -r requirements.txt
python src/train.py --config configs/exp.yaml
python src/eval.py  --config configs/exp.yaml
