# Battery MFI Defect Detection
Research project delivering an **end-to-end pipeline for automatic battery defect diagnosis** from high-cost, low-throughput MFI data

**Goal:** Non-destructive battery defect detection using Magnetic Field Imaging (MFI) with a data augmentation + optimized classification pipeline.

> In battery manufacturing, defects and degradation can cause safety incidents; MFI enables non-contact, in-situ diagnosis by measuring induced magnetic fields (x/y/z). Because MFI data acquisition is time-intensive, we use data augmentation to expand limited datasets and train an automated defect detector.

---

## What
We propose an automated battery defect detection method by training ML classifiers on **augmented MFI images** (rotation, inversion, synthesis, mix-up), then benchmarking multiple models to find the most suitable detector.

## My role
- Led end-to-end research execution: augmentation design, model benchmarking (DT/LR/LogR/RF/SVM/kNN), and evaluation/reporting for poster/paper deliverables.

## Key result
- Expanded an original dataset of **16 images (8 normal / 8 defective)** up to **39,168 images** via augmentation. :contentReference
- Across 3 runs, baseline achieved **90.78% accuracy / 0.90 AUC**, while incorporating **Rotation+Inversion (RI)** improved to **97.77% accuracy / 0.99 AUC**.
- On the RI dataset, **SVM reached 100.0% accuracy / 1.00 AUC** (best-performing setting).
---

## Method (Augmentations)
<img width="1596" height="898" alt="overview" src="https://github.com/user-attachments/assets/3a23a7a7-1a73-4753-9c40-ff200bec3eed" />
- **Rotation:** rotate images in **0.2° increments from -1° to 1°** to mitigate scan-angle variation.
- **Inversion:** vertical/horizontal flips to reduce orientation sensitivity.
- **Synthesis:** generate new images by averaging pixel values from two images.
- **Mix-up:** synthesize images by mixing normal/defective at an **8:2 ratio** (and vice versa).

---

## Demo / Results
![Results](assets/results_overview.png)
*Augmentation + classifier benchmarking for defect detection; RI improves performance and SVM achieves the best AUC and perfect accuracy in the best setting.* 

---

## How to run
```bash
pip install -r requirements.txt
python src/train.py --config configs/exp.yaml
python src/eval.py  --config configs/exp.yaml
