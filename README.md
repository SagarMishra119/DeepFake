# 🎭 DeepFake Detection with EfficientNet

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Computer Vision and Deep Learning system engineered to detect facial manipulation and classify images as **Real** or **Fake (DeepFake)** using an **EfficientNetB0** transfer learning and progressive fine-tuning pipeline.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Model Architecture & Pipeline](#-model-architecture--pipeline)
- [Dataset & Processing](#-dataset--processing)
- [Training Strategy](#-training-strategy)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Desktop Inference](#-desktop-inference)
- [Configuration](#-configuration)
- [License](#-license)

---

## 🔍 Project Overview

With the rapid advancement of generative adversarial networks (GANs) and diffusion-based face synthesis, identifying manipulated media has become crucial for digital forensics and identity verification.

This repository provides a modular, production-ready framework to detect facial tampering:
- **Transfer Learning Backbone:** Utilizes `EfficientNetB0` pre-trained on ImageNet for high-accuracy feature representation with low parameter overhead.
- **Two-Phase Training:** Implements frozen feature extraction followed by deeper fine-tuning from higher convolutional blocks.
- **Embedded Augmentations:** GPU-accelerated spatial and color transformations directly inside the model graph.
- **Interactive Inference:** Desktop GUI utility with instant confidence diagnostics.

---

## 🧠 Model Architecture & Pipeline

The neural pipeline integrates data augmentation, normalization, feature extraction, and binary classification:

```text
               ┌───────────────────────────────┐
               │    Input Image (224x224x3)    │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │    Data Augmentation Layer    │
               │ (Flip, Rotation, Zoom, Cont.) │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │   EfficientNet Preprocessing  │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     EfficientNetB0 Backbone   │
               │   (ImageNet Pretrained Base)  │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │    GlobalAveragePooling2D     │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │       Dropout Layer (0.2)     │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │      Dense Output Layer       │
               │     (1 Unit, Sigmoid Act.)    │
               └───────────────┬───────────────┘
                               │
                               ▼
            Probability: [0.0 = Fake | 1.0 = Real]
```

---

## 📊 Dataset & Processing

The project includes pre-processed, balanced subsets organized under `processed/`:

- **Split Breakdown:**
  - **Train:** 2,800 Real / 2,800 Fake (5,600 images)
  - **Validation:** 600 Real / 600 Fake (1,200 images)
  - **Test:** 600 Real / 600 Fake (1,200 images)
  - **Total:** 8,000 balanced images
- **Augmentation Pipeline:**
  - Random Horizontal Flip
  - Random Rotation (`factor=0.1`)
  - Random Zoom (`factor=0.1`)
  - Random Contrast Adjustment (`factor=0.1`)
- **Performance:** Streamed via `tf.data.Dataset` pipelines with `AUTOTUNE`, memory caching, and background prefetching.

---

## 🏋️ Training Strategy

To balance convergence speed with feature retention, a two-phase schedule is employed:

| Parameter | Phase 1: Feature Extraction | Phase 2: Fine-Tuning |
| :--- | :--- | :--- |
| **Backbone State** | Frozen (`trainable=False`) | Unfrozen from Layer 100 onwards |
| **Optimizer** | Adam | Adam |
| **Learning Rate** | `0.001` (`1e-3`) | `0.00001` (`1e-5`) |
| **Max Epochs** | 10 | 30 |
| **Callbacks** | `ModelCheckpoint`, `EarlyStopping` (patience=3) | `ModelCheckpoint`, `EarlyStopping` (patience=3) |
| **Monitored Metric** | `val_loss` / `val_accuracy` | `val_loss` / `val_accuracy` |

---

## 📁 Repository Structure

```text
DeepFake/
├── app/
│   ├── __init__.py
│   └── inference.py              # Desktop UI with file picker & prediction engine
├── evaluation/
│   ├── __init__.py
│   ├── confusion_matrix.png      # Generated test confusion matrix heatmap
│   └── evaluate.py               # Evaluates metrics: Accuracy, Precision, Recall, F1
├── models/
│   ├── __init__.py
│   └── efficientnet_model.py     # EfficientNetB0 architecture and compilation
├── preprocessing/
│   ├── __init__.py
│   ├── augmentation.py           # Keras augmentation layer definitions
│   ├── data_load.py              # tf.data loading, batching, and caching pipeline
│   └── vil_augmentation.py       # Visualizer script for augmented samples
├── processed/                    # Balanced dataset splits (Train, Valid, Test)
│   ├── test/
│   ├── train/
│   └── valid/
├── training/
│   ├── __init__.py
│   └── train.py                  # Two-phase training script with curve plotting
├── utils/
│   ├── __init__.py
│   ├── config.py                 # Central configurations & portable relative paths
│   └── visualization.py          # Training curve visualization utilities
├── sampling.py                   # Automated dataset subset extraction script
├── requirements.txt              # Python project dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SagarMishra119/DeepFake.git
cd DeepFake
pip install -r requirements.txt
```

### 2. Dataset Sampling (Optional)

If generating a fresh split from raw dataset archives:

```bash
python sampling.py
```

### 3. Model Training

Run the two-phase training script:

```bash
python -m training.train
```

Trained checkpoints and loss curves are saved under `saved_models/`.

---

## 📈 Evaluation & Metrics

Evaluate the trained checkpoint against the independent test set:

```bash
python -m evaluation.evaluate
```

This generates:
- **Test Metrics:** Accuracy, Precision, Recall, and F1-Score.
- **Confusion Matrix:** Saved to `evaluation/confusion_matrix.png`.
- **Classification Report:** Detailed per-class precision and recall summary.

---

## 🖥️ Desktop Inference

Test any facial image locally using the interactive file dialog:

```bash
python -m app.inference
```

- Prompts an image selection dialog (`.jpg`, `.jpeg`, `.png`).
- Returns classification label (`real` / `fake`), confidence percentage, and raw sigmoid logit.

---

## ⚙️ Configuration

Hyperparameters, image dimensions, and training settings are configured in [`utils/config.py`](utils/config.py):

```python
# Image and Batching
Image_Size = (224, 224)
Batch_Size = 64

# Phase 1: Feature Extraction
Phase1_Epochs = 10
Phase1_LR = 0.001

# Phase 2: Fine-Tuning
Phase2_Epochs = 30
Phase2_LR = 0.00001
FINE_TUNE_AT_LAYER = 100
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
