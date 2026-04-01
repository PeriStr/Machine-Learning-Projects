# 🖼️ Image Classification with MLP & kNN
### CIFAR-10 / CIFAR-100 with Color & Edge Histogram Features

---

## 📌 Description

A classic image classification pipeline that intentionally avoids raw pixel input.
Instead, hand-crafted **color histograms** and **edge histograms** are extracted from each image and used as feature vectors for two classifiers:

- **MLP (Multi-Layer Perceptron)** — a fully connected neural network trained with Adam + Early Stopping
- **kNN (k-Nearest Neighbors)** — a distance-based classifier tested across multiple values of k

All experiments run on both **CIFAR-10** (10 classes) and **CIFAR-100** (100 classes) to compare how classifier performance scales with dataset difficulty.

---

## 📂 Project Structure

```
project/
│
├── main.py      # Full pipeline: feature extraction + MLP + kNN + evaluation
└── README.md
```

No external dataset files needed — both CIFAR datasets are downloaded automatically via Keras.

---

## 🗄️ Datasets

| Dataset | Images | Classes | Image Size |
|---|---|---|---|
| CIFAR-10 | 60,000 | 10 | 32×32 RGB |
| CIFAR-100 | 60,000 | 100 (fine labels) | 32×32 RGB |

Both datasets are split into:
- **Train (80%)** — used for feature extraction and model training
- **Validation (20% of train)** — used for early stopping in MLP
- **Test (fixed)** — used for final evaluation only

---

## ⚙️ Installation

```bash
pip install numpy opencv-python matplotlib seaborn tensorflow scikit-learn
```

---

## 🔄 Full Pipeline

```
Load Dataset (CIFAR-10 or CIFAR-100)
        │
        ▼
Normalize pixels [0,255] → [0.0,1.0]
        │
        ▼
Train/Validation split (80/20, stratified)
        │
        ▼
Feature Extraction (per image)
   ├── Color Histogram  → 48-dim vector  (16 bins × 3 RGB channels)
   └── Edge Histogram   → 16-dim vector  (Sobel gradient magnitudes)
        │
        ▼
   ┌────────────────────────────┐
   │          MLP               │
   │  Dense(256) → Dropout(0.5) │
   │  Dense(128) → Dense(n_cls) │
   │  Adam + EarlyStopping      │
   └────────────────────────────┘
        │
        ▼
   ┌────────────────────────────┐
   │    kNN (k = 1,3,5,7,11)    │
   │  Euclidean distance        │
   └────────────────────────────┘
        │
        ▼
Evaluation: Accuracy, Classification Report, Confusion Matrix
```

---

## 🧩 Feature Extraction

### 1. Color Histogram

Each image is described by the **distribution of pixel intensities** across its 3 color channels.

```
For each channel (R, G, B):
  → compute histogram with 16 bins over [0, 255]
  → normalize to sum = 1
  → concatenate all 3 channels

Output: 16 × 3 = 48-dimensional feature vector
```

**Optional:** HSV color space is also supported (`color_space="HSV"`).

**Limitation:** Ignores spatial layout — two images with the same colors but different objects look identical to this descriptor.

---

### 2. Edge Histogram

Each image is described by the **distribution of edge intensities** using Sobel gradients.

```
Convert image to grayscale
  → Apply Sobel operator on X axis (horizontal edges)
  → Apply Sobel operator on Y axis (vertical edges)
  → Compute gradient magnitude = √(Gx² + Gy²)
  → Build histogram (16 bins) of magnitude values
  → Normalize to sum = 1

Output: 16-dimensional feature vector
```

**Why edges?** Edges capture object boundaries and textures, which are more invariant to color changes than raw pixels.

---

## 🏗️ MLP Architecture

```
Input (48-dim color or 16-dim edge)
        │
  Dense(256, ReLU)
        │
  Dropout(0.5)          ← prevents overfitting
        │
  Dense(128, ReLU)
        │
  Dense(num_classes, Softmax)   ← 10 for CIFAR-10, 100 for CIFAR-100
```

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr = 0.001) |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping | patience = 5 on val_loss |

---

## 🔵 kNN Classifier

| Parameter | Value |
|---|---|
| Distance metric | Euclidean |
| k values tested | 1, 3, 5, 7, 11 |
| Training | None (lazy learner) |

kNN does not train a model — it stores all training vectors and classifies new samples by finding the k nearest neighbors at query time.

---

## 📊 Experiments

4 experiments are run automatically:

| # | Dataset | Features | Expected Accuracy (MLP) |
|---|---|---|---|
| 1 | CIFAR-10 | Color Histogram | ~35 – 45% |
| 2 | CIFAR-10 | Edge Histogram | ~25 – 35% |
| 3 | CIFAR-100 | Color Histogram | ~10 – 18% |
| 4 | CIFAR-100 | Edge Histogram | ~8 – 13% |

> Accuracy is lower than CNN-based models because hand-crafted histograms discard spatial structure. The goal of this project is to compare feature types and classifiers, not to maximize accuracy.

---

## 📈 Outputs per Experiment

- ✅ MLP test accuracy
- ✅ Classification report (Precision / Recall / F1 per class)
- ✅ Confusion matrix heatmap
- ✅ MLP learning curve (train vs val accuracy per epoch)
- ✅ kNN accuracy for each value of k

---

## ▶️ How to Run

```bash
python main.py
```

All 4 experiments run sequentially. CIFAR datasets are downloaded automatically on first run (~170 MB total).

To run a single experiment:
```python
run_experiment("cifar10",  feature_type="color")
run_experiment("cifar10",  feature_type="edge")
run_experiment("cifar100", feature_type="color")
run_experiment("cifar100", feature_type="edge")
```

---

## 🔑 Key Design Decisions

| Decision | Reason |
|---|---|
| Histogram features instead of raw pixels | Reduces input dimensionality, tests classical feature engineering |
| Stratified train/val split | Preserves class balance across splits |
| Dropout(0.5) in MLP | Regularization — prevents overfitting on small feature vectors |
| Early stopping (patience=5) | Avoids wasting time on epochs that don't improve validation loss |
| Multiple k values for kNN | Shows the bias-variance tradeoff as k increases |
| `1e-6` epsilon in edge normalization | Prevents division by zero on blank/uniform images |

---

## 📦 Requirements

```
numpy
opencv-python
matplotlib
seaborn
tensorflow
scikit-learn
```
