# 🎭 DeepFake Detection with EfficientNet

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Computer Vision and Deep Learning pipeline designed to detect facial manipulation and classify images as **Real** or **Fake (DeepFake)** using **EfficientNetB0** transfer learning and progressive fine-tuning.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture & Pipeline](#-architecture--pipeline)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Training Strategy](#-training-strategy)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Inference & GUI](#-inference--gui)
- [Configuration](#-configuration)
- [Google Colab Support](#-google-colab-support)
- [License](#-license)

---

## 🔍 Project Overview

The proliferation of hyper-realistic generative AI models and face-swapping algorithms poses severe security, misinformation, and identity verification challenges. This project implements a robust deep learning framework to detect subtle visual artifacts and synthetic manipulation patterns across facial images.

### Key Highlights
- **Backbone:** Pre-trained `EfficientNetB0` on ImageNet as a scalable, high-efficiency feature extractor.
- **Two-Phase Transfer Learning:** Combines frozen feature extraction with deep unfreezing of higher-order feature layers.
- **In-Graph Augmentation:** GPU-accelerated data augmentations directly embedded into the model architecture.
- **Production-Ready Inference:** Built-in desktop GUI for point-and-click image testing with confidence scoring.

---

## 🧠 Architecture & Pipeline

The model utilizes an integrated preprocessing and classification pipeline:

```
[ Input Image: 224x224x3 ]
           │
           ▼
[ Data Augmentation Layer ]  ──> (RandomFlip, RandomRotation, RandomZoom, RandomContrast)
           │
           ▼
[ EfficientNet Preprocessing ] ──> (Normalization & Scaling)
           │
           ▼
[ EfficientNetB0 Backbone ]   ──> (Pre-trained on ImageNet)
           │
           ▼
[ GlobalAveragePooling2D ]   ──> (Feature Vector Extraction)
           │
           ▼
[ Dropout Layer (0.2) ]       ──> (Regularization / Anti-Overfitting)
           │
           ▼
[ Dense Layer (1 Unit, Sigmoid) ] ──> Output Probability: [0.0 = Fake, 1.0 = Real]
```

---

## 📊 Dataset & Preprocessing

The system processes balanced image subsets derived from the **Real vs Fake** face dataset:

- **Stratified Split Targets (`sampling.py`):**
  - **Train:** 2,800 images/class (5,600 total)
  - **Validation:** 600 images/class (1,200 total)
  - **Test:** 600 images/class (1,200 total)
  - **Total Dataset Size:** 8,000 balanced images
- **Data Augmentations:**
  - Random Horizontal Flip
  - Random Rotation (`±10%`)
  - Random Zoom (`±10%`)
  - Random Contrast Adjustment (`±10%`)
- **Optimization:** Utilizes `tf.data` with `AUTOTUNE`, memory caching, and prefetching for optimal GPU utilization.

---

## 🏋️ Training Strategy

A **Two-Phase Progressive Training** approach is used to achieve high validation accuracy while preventing catastrophic forgetting:

| Parameter | Phase 1 (Feature Extraction) | Phase 2 (Fine-Tuning) |
| :--- | :--- | :--- |
| **Backbone State** | Fully Frozen (`trainable=False`) | Unfrozen from Layer 100 onwards |
| **Optimizer** | Adam | Adam |
| **Learning Rate** | `1e-3` (`0.001`) | `1e-5` (`0.00001`) |
| **Epochs** | Up to 10 | Up to 30 |
| **Callbacks** | `EarlyStopping` (patience=3), `ModelCheckpoint` | `EarlyStopping` (patience=3), `ModelCheckpoint` |
| **Primary Metric** | Validation Loss / Accuracy | Validation Loss / Accuracy |

---

## 📁 Project Structure

```text
DeepFake/
├── app/
│   ├── __init__.py
│   └── inference.py              # Desktop UI with file picker & prediction engine
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py               # Computes accuracy, precision, recall, F1 & confusion matrix
├── models/
│   ├── __init__.py
│   └── efficientnet_model.py     # Model architecture & build definition
├── preprocessing/
│   ├── __init__.py
│   ├── augmentation.py           # Keras augmentation layers
│   ├── data_load.py              # tf.data loading, batching, and caching pipeline
│   └── vil_augmentation.py       # Augmentation visualization script
├── training/
│   ├── __init__.py
│   └── train.py                  # Two-phase training script with history plotting
├── utils/
│   ├── __init__.py
│   ├── config.py                 # Central configurations, hyperparams & directory paths
│   └── visualization.py          # Loss and accuracy curve generator
├── collabb/                      # Mirrored structure optimized for Google Colab
├── sampling.py                   # Automated dataset splitter and sampling script
└── requirements.txt              # Core project dependencies
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/SagarMishra119/DeepFake.git
cd DeepFake
pip install -r requirements.txt
```

### 2. Dataset Sampling

Extract a balanced subset from your raw dataset folder into `processed/`:

```bash
python sampling.py
```

### 3. Model Training

Run the two-phase training pipeline:

```bash
python -m training.train
```

Outputs will be saved automatically:
- `saved_models/best_model.keras`
- `saved_models/final_model.keras`
- `saved_models/training_curves.png`

---

## 📈 Evaluation & Metrics

Evaluate the best model checkpoint on the independent test set:

```bash
python -m evaluation.evaluate
```

This generates:
- **Metrics Report:** Accuracy, Precision, Recall, and F1-Score.
- **Confusion Matrix:** Saved to `evaluation/confusion_matrix.png`.
- **Classification Report:** Detailed per-class precision and recall breakdown.

---

## 🖥️ Inference & GUI

Test individual face images using the built-in Tkinter desktop interface:

```bash
python -m app.inference
```

- Opens an interactive file selection dialog.
- Preprocesses and resizes the image to `(224, 224, 3)`.
- Outputs the classification decision (`real` / `fake`), confidence percentage, and raw sigmoid score.

---

## ⚙️ Configuration

All hyperparameters and paths are centralized in [`utils/config.py`](utils/config.py):

```python
# Image & Batch Settings
Image_Size = (224, 224)
Batch_Size = 64

# Phase 1 Hyperparameters
Phase1_Epochs = 10
Phase1_LR = 0.001

# Phase 2 Hyperparameters
Phase2_Epochs = 30
Phase2_LR = 0.00001
FINE_TUNE_AT_LAYER = 100
```

---

## ☁️ Google Colab Support

To train on Google Colab with GPU acceleration:
1. Upload the `collabb/` directory to Google Drive.
2. In `collabb/utils/config.py`, set:
   ```python
   Collab = True
   ```
3. Mount Google Drive and run `collabb/training/train.py`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
