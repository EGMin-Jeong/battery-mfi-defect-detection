# Battery Defect Detection via Magnetic Field Imaging and Data Augmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classical_Classifiers-FF6F00.svg)]()
[![Data](https://img.shields.io/badge/Data-Augmentation-brightgreen.svg)]()

This repository documents a year-long group research project conducted at DGIST, exploring how data augmentation and machine learning classification models can be combined to automate battery defect detection using Magnetic Field Imaging (MFI). The project evolved from an initial proof-of-concept presented at KICS 2024 to a refined, optimized system presented as a poster at ISEPD 2025.

## Objectives

- To address the critical shortage of labeled training data in MFI-based battery inspection by designing effective augmentation pipelines
- To identify the optimal combination of augmentation technique and classification model for automated defect detection
- To demonstrate that non-destructive, in-operando MFI can replace counter-current methods for detecting battery defects with near-perfect accuracy

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Models](#models)
- [Results](#results)
- [Evolution of the Project](#evolution-of-the-project)
- [Future Work](#future-work)
- [Prerequisites](#prerequisites)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

---

## Background

As lithium-ion batteries scale up in energy density, internal manufacturing defects — tab tears, anode folds, and misalignments — have become a growing cause of fire and explosion incidents. Magnetic Field Imaging (MFI) offers a non-contact, non-destructive solution: by applying current to a battery and measuring the induced magnetic field (in x, y, and z directions, per the Biot-Savart law), anomalous current flow patterns can be identified in real time.

The fundamental challenge is data scarcity. Fabricating defect-simulated batteries is time-intensive, which means labeled datasets are small by nature. This project directly tackles that problem through systematic augmentation, allowing powerful classifiers to be trained on a realistically limited dataset.

## Dataset

The raw dataset consists of MFI scans of normal and defect-simulated lithium-ion batteries (Tab Torn Defect). All images capture magnetic field intensity across spatial coordinates.

| Split | Normal | Defective | Total |
|-------|--------|-----------|-------|
| Training | 4 | 7 | 11 |
| Test | 4 | 7 | 11 |
| **Total** | **8** | **14** | **22** |

> Raw data and augmented datasets are available in the `data/` directory. Identifiable or proprietary data has been excluded. Final datasets will be archived on [Dryad](https://datadryad.org/) upon publication.

## Data Augmentation

Starting from as few as 16 images (8 normal, 8 defective) in the initial study, the augmentation pipeline expanded the training set to up to **39,168 images** in the refined version. Six augmentation strategies were evaluated:

### Base Techniques

- **Rotation (R):** Original images rotated in 0.2° increments from −1° to +1°, mitigating angular variation introduced during scanning.
- **Inversion (I):** Horizontal and vertical flipping to account for differences in battery orientation.
- **Rotation + Inversion (RI):** Combined application of both techniques above.
- **Saturation Intensity (INT):** Adjusting pixel saturation to simulate varying current magnitudes and strengthen magnetic field signals.
- **Synthesis (SYN):** Generating new images by averaging pixel values from two same-class images, preserving class-specific features while increasing diversity.

### Key Innovation: Mix-up

- **Mix-up:** Blending normal and defective images at an 8:2 ratio (and vice versa for defective synthesis). For the KICS 2024 study, mix ratios of 10%, 20%, and 30% were explored.

All augmented images are stored under `data/augmented/`. Source code for each augmentation technique is in `code/augmentation/`.

## Models

Six classical machine learning classifiers were evaluated across all augmented datasets. Model selection prioritized generalizability and resistance to overfitting given the small dataset size.

| Model | Abbreviation |
|-------|-------------|
| Decision Tree | DT |
| Linear Regression | LR |
| Logistic Regression | LogR |
| Random Forest | RF |
| Support Vector Machine | SVM |
| k-Nearest Neighbors | kNN |

Each model was evaluated using 100 decision thresholds (0.00–1.00), with performance reported as accuracy and AUC (Area Under the ROC Curve). Training and evaluation scripts are in `code/classification/`.

## Results

### ISEPD 2025 (Poster — Refined Study)

The optimized pipeline used RI augmentation on top of synthesis and mix-up datasets, with 4,000 images randomly sampled per dataset for training balance.

**Key findings:**

- Applying the RI technique improved average accuracy by up to **11%** over non-RI baselines.
- The **SVM + RI dataset** achieved the best overall performance: **100.0% accuracy and AUC = 1.00**.
- SYN-RI outperformed MIX-RI because same-class synthesis preserves discriminative features, whereas cross-class mixing in MIX-RI dilutes defect signals and introduces noise.

| Model | Dataset | Accuracy (%) | AUC |
|-------|---------|-------------|-----|
| SVM | RI | **100.0** | **1.00** |
| LogR | RI | 99.31 | 1.00 |
| kNN | RI | 98.39 | 0.98 |
| RF | MIX | 98.85 | 1.00 |

### KICS 2024 (Conference Paper — Initial Study)

The initial study used a slightly different dataset split and a smaller augmentation scope. GBM was included as an additional model; SVM was not yet evaluated.

**Key findings:**

- Mix-up augmentation improved accuracy and AUC by approximately **10%** on average over non-Mix-up baselines.
- **RF and kNN** were the top-performing models across all augmentation strategies.
- **MIX-ALL** (Mix-up combined with all base techniques) achieved the strongest average performance: **85.61% accuracy, AUC = 0.88**.

| Model | Dataset | Accuracy (%) | AUC |
|-------|---------|-------------|-----|
| kNN | MIX-ALL | 81.82 | 0.93 |
| GBM | MIX-ALL | 90.91 | 0.93 |
| RF | MIX-ALL | **87.88** | **0.89** |

Full performance tables are available in `manuscript/tables/`.

## Evolution of the Project

This repository reflects the full arc of the research across two publications:

1. **KICS 2024** — Initial investigation into Mix-up augmentation with four base classifiers (GBM, kNN, LR, RF). Dataset: 22 images total (8 normal, 14 defective). Core finding: Mix-up consistently improves performance; RF and kNN lead overall.

2. **ISEPD 2025** — Refined methodology adding SVM and expanding the augmentation dataset to 39,168 images. RI technique introduced as a key driver of performance gains. Core finding: SVM + RI achieves perfect defect classification. Mechanistic explanation provided for why SYN-RI outperforms MIX-RI (noise dilution vs. feature preservation).

Figures and tables from both publications are in `manuscript/src/figures/` and `manuscript/src/tables/`.

## Future Work

- Extending the approach to **semi-supervised and unsupervised** learning frameworks, reducing reliance on labeled defective samples.
- Simulating and classifying additional defect types: **Anode Folded Defect** and **Misalign Defect**, in addition to the current Tab Torn Defect.
- Scaling the pipeline to industrial inspection settings with larger and more varied battery form factors.

## Prerequisites

### Software

- **Python 3.8+** — data augmentation, preprocessing, and ML model training
- Recommended IDE: PyCharm or VS Code
- Key libraries: `numpy`, `scikit-learn`, `matplotlib`, `Pillow`

### Environment Setup

```bash
git clone https://github.com/<your-username>/battery-defect-mfi.git
cd battery-defect-mfi
pip install -r requirements.txt
```

### Credentials

If connecting to any external database or server, store credentials in a local `secret.py` file. This file is listed in `.gitignore` and will not be synced to the repository. Import it at the top of your main script:

```python
from secret import DB_USER, DB_PASSWORD
```

## Repository Structure

```
battery-defect-mfi/
├── code/
│   ├── augmentation/       # Rotation, inversion, synthesis, mix-up scripts
│   └── classification/     # Model training and evaluation (SVM, RF, kNN, etc.)
├── data/
│   ├── raw/                # Original MFI scans (16–22 images)
│   └── augmented/          # Expanded datasets up to 39,168 images
├── manuscript/
│   └── src/
│       ├── figures/        # All figures used in KICS 2024 and ISEPD 2025
│       └── tables/         # Performance evaluation tables
└── README.md
```

## Authors

- **Nahye Kim** — DGIST School of Undergraduate Studies (equal contribution)
- **Eugene Jeong** — DGIST School of Undergraduate Studies (equal contribution)
- **Moonyoung Choi** — DGIST School of Undergraduate Studies
- **Jiyeol Kim** — DGIST Division of Nanotechnology
- **Sangchul Lee** *(Corresponding Author)* — DGIST Division of Nanotechnology

## Acknowledgements

This research was supported by the DGIST Institutional R&D Program (24-ET-02), the DGIST Undergraduate Group Research Program (UGRP, 2024020031), and the National Research Foundation of Korea ERC Program (RS-2023-00222166).

## License

This project is licensed under the MIT License — see the `LICENSE.md` file for details.

## References

1. Lee, M., et al. (2023). Diagnosis of Current Flow Patterns Inside Fault-Simulated Li-Ion Batteries via Non-Invasive, In Operando Magnetic Field Imaging. *Small Methods*, 7(11), 2300748.
2. Kim N., et al. Performance Comparison of Battery Defect Detection According to Data Augmentation Techniques and Classification Models. *The Korean Institute of Communications and Information Sciences (KICS)*, pp. 274–275, 2024.
3. Burkov, A. *The Hundred-Page Machine Learning Book*, 2019.
