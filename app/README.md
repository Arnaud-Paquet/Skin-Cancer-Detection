---
title: NaeviScan
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Skin lesion classifier — EfficientNetV2-S on HAM10000
---

# NaeviScan

NaeviScan (from *naevi*, the Latin plural of *nevus* — "mole") classifies
dermatoscopic images into 7 skin lesion types using a fine-tuned
EfficientNetV2-S model trained on the HAM10000 dataset.

## Lesion types

| Code   | Full name                              | Risk            |
|--------|----------------------------------------|-----------------|
| nv     | Melanocytic nevus                      | Benign          |
| mel    | Melanoma                               | Malignant       |
| bkl    | Benign keratosis                       | Benign          |
| bcc    | Basal cell carcinoma                   | Malignant       |
| akiec  | Actinic keratosis / Bowen's disease    | Pre-cancerous   |
| vasc   | Vascular lesion                        | Benign          |
| df     | Dermatofibroma                         | Benign          |

## Model

- Architecture: EfficientNetV2-S, fine-tuned ConvNet
- Dataset: HAM10000 — 10,015 dermatoscopic images
- Top-1 balanced accuracy: 75.2% on held-out test set
- Melanoma F1: 0.62 | Melanoma precision: 55.3%
- Inference: 9-pass TTA (1 clean + 8 augmented) + temperature scaling (T=0.8722)
- Explainability: Grad-CAM heatmap overlay on every prediction

## Disclaimer

This tool is for research and educational purposes only. It is not a medical
device and must not be used for clinical diagnosis or treatment decisions.
Always consult a qualified dermatologist.
