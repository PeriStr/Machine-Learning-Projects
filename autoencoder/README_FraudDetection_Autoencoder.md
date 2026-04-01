# 💳 Credit Card Fraud Detection
### Anomaly Detection with Autoencoder (Semi-Supervised)

---

## 📌 Description

An **unsupervised anomaly detection** system that identifies fraudulent credit card transactions using an **Autoencoder** neural network.

The key idea: the model is trained **only on normal transactions**. Because it has never seen fraud, it struggles to reconstruct fraudulent transactions accurately — producing a high reconstruction error. Any transaction with an error above a chosen **threshold** is flagged as fraud.

> Source code adapted and rewritten from TF1 to **TensorFlow 2.0** based on:
> [github.com/phansieeex3/Credit-Card-Fraud-Detection-Autoencoder](https://github.com/phansieeex3/Credit-Card-Fraud-Detection-Autoencoder)

---

## 📂 Project Structure

```
project/
│
├── main.py           # Full pipeline: preprocessing + training + evaluation
├── creditcard.csv    # Input dataset
└── README.md
```

---

## 🗄️ Dataset

**Credit Card Fraud Detection** — anonymized European credit card transactions from September 2013.

> Download: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Column | Description |
|---|---|
| `V1` – `V28` | PCA-transformed features (anonymized) |
| `Amount` | Transaction amount → normalized to `normAmount` |
| `Time` | Seconds since first transaction → **dropped** |
| `Class` | **Target** — 0 = Normal, 1 = Fraud |

**Class imbalance:**

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |

---

## ⚙️ Installation

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

---

## 🧠 Why an Autoencoder for Fraud Detection?

```
Normal transaction  →  Autoencoder  →  Reconstruction  →  Low MSE  →  Normal ✅
Fraud transaction   →  Autoencoder  →  Reconstruction  →  High MSE →  Fraud ⚠️
```

The autoencoder learns a **compressed representation** of normal behavior. It cannot reconstruct fraud well because those patterns were never seen during training — making the reconstruction error a natural anomaly score.

This is a **semi-supervised** approach: labels are only used at evaluation time, not during training.

---

## 🔄 Preprocessing Pipeline

```
Load creditcard.csv
   │
   ▼
Normalize Amount → StandardScaler → normAmount
Drop Time and Amount columns
   │
   ▼
Shuffle dataset (random_state=42)
   │
   ▼
80/20 split → train_set | test_set
   │
   ▼
X_train = train_set WHERE Class == 0 (normal only, no fraud)
X_test  = full test_set (normal + fraud)
y_test  = Class labels (for evaluation only)
```

**Why train only on Class 0?**
The autoencoder must learn what "normal" looks like. If fraud examples were included in training, the model would also learn to reconstruct them — defeating the purpose.

---

## 🏗️ Autoencoder Architecture

```
Input (29 features)
        │
  Dense(16, ReLU)     ← Encoder layer 1
  Dropout(0.2)
  Dense(8,  ReLU)     ← Bottleneck (compressed representation)
        │
  Dense(16, ReLU)     ← Decoder layer 1
  Dropout(0.2)
  Dense(29, Linear)   ← Reconstructed output
        │
   MSE Loss
```

| Component | Value |
|---|---|
| Input dimension | 29 |
| Hidden layer 1 | 16 neurons |
| Bottleneck | 8 neurons |
| Output dimension | 29 (reconstruction) |
| Activation (hidden) | ReLU |
| Activation (output) | Linear |
| Dropout | 0.2 |
| Loss | MSE |
| Optimizer | Adam (lr = 0.001) |

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_hidden_1` | 16 | First compression layer |
| `n_hidden_2` | 8 | Bottleneck size |
| `learning_rate` | 0.001 | Adam optimizer |
| `num_epochs` | 100 | Max training epochs |
| `batch_size` | 2048 (GPU) / 256 (CPU) | Samples per update |
| `patience` | 10 | Early stopping patience |
| `dropout` | 0.2 | Regularization |

---

## 🔄 Training Loop

```
For each epoch (up to 100):
   │
   ├── Shuffle X_train
   ├── For each batch:
   │     └── partial_fit (train_on_batch)
   │
   ├── Compute val_loss on X_test
   └── Early Stopping: if val_loss didn't improve for 10 epochs → stop
```

**Note:** `X_test` is used here only as a validation signal for early stopping — labels are never exposed during training.

---

## 🎯 Threshold Selection

After training, reconstruction error (MSE) is computed on all normal training samples:

```python
train_mse = mean((X_train - reconstruct(X_train))², axis=1)
threshold  = np.percentile(train_mse, 90)   # active threshold
```

Three percentiles were tested:

| Threshold | Effect |
|---|---|
| 95th percentile | Balanced — fewer false positives |
| 99th percentile | Strict — very few false positives, may miss fraud |
| **90th percentile** | **Active** — higher recall, catches more fraud |

Any test transaction with MSE > threshold is classified as **fraud**.

---

## 📊 Actual Results (90th percentile threshold)

```
Confusion Matrix:
[[54014  2843]       ← TN: 54014  |  FP: 2843
 [   11    94]]      ← FN: 11     |  TP: 94
```

| Metric | Normal (0) | Fraud (1) |
|---|---|---|
| Precision | 1.00 | 0.03 |
| Recall | 0.95 | **0.90** |
| F1-Score | 0.97 | 0.06 |
| Accuracy | **95%** | — |

**Key takeaway:**
- **Recall = 0.90** — the model catches 90% of all real fraud cases ✅
- **Precision = 0.03** — many false positives (2,843 normal transactions flagged)
- This trade-off is intentional: in fraud detection, **missing a fraud (FN) is more costly** than a false alarm (FP)

---

## 📈 Output

- Classification report (Precision / Recall / F1 per class)
- Confusion matrix
- Reconstruction Error Distribution plot — shows the MSE separation between normal and fraud transactions with the threshold line

---

## ▶️ How to Run

```bash
python main.py
```

GPU is detected automatically. If available, batch size increases to 2048 for faster training.

---

## 📦 Requirements

```
numpy
pandas
tensorflow
scikit-learn
matplotlib
```
