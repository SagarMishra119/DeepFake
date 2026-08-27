# DeepFake Detection System

An end-to-end Deep Learning pipeline built with **TensorFlow / Keras** to detect and classify facial images as **Real** or **Fake (DeepFake)** using **EfficientNetB0** transfer learning and fine-tuning.

---

## 📌 Project Overview

This project provides a complete workflow for binary deepfake classification:
- **Architecture:** Pre-trained **EfficientNetB0** feature extractor with custom augmentation layers, Global Average Pooling, Dropout, and a Sigmoid classification head.
- **Two-Phase Training:**
  - **Phase 1 (Frozen Backbone):** Trains only the classification head with a higher learning rate (`1e-3`).
  - **Phase 2 (Fine-Tuning):** Unfreezes layers from layer 100 onwards and fine-tunes the network with a smaller learning rate (`1e-5`).
- **Inference App:** Simple desktop GUI file picker using Tkinter for instant image predictions with confidence scoring.

---

## 📁 Repository Structure

```text
DeepFake/
├── app/
│   └── inference.py          # Desktop inference script with file picker
├── evaluation/
│   └── evaluate.py           # Model evaluation (Accuracy, F1, Confusion Matrix)
├── models/
│   └── efficientnet_model.py # EfficientNetB0 architecture definition
├── preprocessing/
│   ├── augmentation.py       # Data augmentation layers
│   ├── data_load.py          # tf.data pipeline loading and prefetching
│   └── vil_augmentation.py   # Visualizing augmentation samples
├── training/
│   └── train.py              # Two-phase model training script
├── utils/
│   ├── config.py             # Global configurations and hyperparameters
│   └── visualization.py      # Training curves plotting utility
├── sampling.py               # Dataset sampler for balanced splits
└── requirements.txt          # Project dependencies
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/SagarMishra119/DeepFake.git
cd DeepFake
pip install -r requirements.txt
```

### 2. Prepare Dataset

Run the sampling script to prepare balanced train, validation, and test subsets from the dataset:

```bash
python sampling.py
```

### 3. Train the Model

Start the two-phase training process:

```bash
python -m training.train
```
Trained weights (`best_model.keras`, `final_model.keras`) and training curves will be saved in `saved_models/`.

### 4. Evaluate the Model

Evaluate performance metrics (Accuracy, Precision, Recall, F1-Score) and generate the confusion matrix:

```bash
python -m evaluation.evaluate
```

### 5. Run Inference

Launch the GUI inference tool to select an image from your system and classify it:

```bash
python -m app.inference
```

---

## ⚙️ Configuration

Hyperparameters and paths can be configured in `utils/config.py`:
- `Image_Size`: `(224, 224)`
- `Batch_Size`: `64`
- `Phase1_Epochs`: `10` | `Phase1_LR`: `0.001`
- `Phase2_Epochs`: `30` | `Phase2_LR`: `0.00001`
- `FINE_TUNE_AT_LAYER`: `100`

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
