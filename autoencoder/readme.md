# Credit Card Fraud Detection using Autoencoder

## 📌 Project Overview

This project implements an **Autoencoder neural network** for detecting fraudulent credit card transactions using an **unsupervised anomaly detection approach**. Instead of directly classifying transactions as fraud or normal, the model learns the patterns of **normal transactions** and identifies anomalies based on **reconstruction error**.

The main idea is that fraudulent transactions differ significantly from normal behavior, causing the Autoencoder to reconstruct them poorly, resulting in a **high error value**.

---

## 📊 Dataset

* Dataset Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* The dataset contains credit card transactions made by European cardholders.
* It is **highly imbalanced**, where fraudulent transactions represent a very small percentage of the data.

### Key Characteristics:

* Features **V1–V28**: Result of PCA transformation (already scaled & numerical)
* **Amount**: Transaction amount (normalized in this project)
* **Class**:

  * `0` → Normal transaction
  * `1` → Fraudulent transaction

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* Removal of unnecessary columns (`Time`, raw `Amount`)
* Scaling of the `Amount` feature using **StandardScaler**
* No need for:

  * Missing value handling (dataset is clean)
  * One-hot encoding (no categorical features)

### Important Design Choice:

The dataset is already transformed using PCA → this simplifies preprocessing and improves training stability.

---

## 🧠 Model Architecture

The model is a **fully connected Autoencoder** consisting of:

### Encoder:

* Dense layer → reduces dimensionality
* Dropout → prevents overfitting
* Dense layer → compressed representation (latent space)

### Decoder:

* Dense layer → reconstructs data
* Dropout
* Output layer → reconstruct original input

### Architecture Flow:

```
Input (29 features)
   ↓
Dense (16 neurons)
   ↓
Dense (8 neurons)   ← Latent Space (Compression)
   ↓
Dense (16 neurons)
   ↓
Output (29 features)
```

### Key Hyperparameters:

* `n_hidden_1 = 16` → first compression level
* `n_hidden_2 = 8` → bottleneck (core representation)
* `learning_rate = 0.001` → standard for Adam optimizer
* `epochs = 100` → allows convergence with early stopping
* `batch_size`:

  * GPU → 2048 (faster training)
  * CPU → 256 (memory-safe)

---

## 🚀 Training Strategy

### Important Concept:

The model is trained **ONLY on normal transactions (Class = 0)**.

👉 This is crucial because:

* The model learns only "normal behavior"
* Fraud becomes **anomaly (outlier detection)**

### Training Features:

* Batch training using `train_on_batch`
* Data shuffling every epoch
* **Early stopping** (patience = 10) to prevent overfitting
* Validation loss monitored on test set

---

## 🔍 Anomaly Detection Method

After training:

1. The model reconstructs input data

2. Compute **Reconstruction Error (MSE)**:

   ```
   MSE = (Original - Reconstructed)^2
   ```

3. Define threshold:

   * Based on percentile of training error
   * Typically:

     * 90% → more sensitive
     * 95% → balanced
     * 99% → stricter

4. Classification:

   ```
   if MSE > threshold → Fraud
   else → Normal
   ```

---

## 📈 Evaluation Metrics

The model is evaluated using:

* **Confusion Matrix**
* **Precision**
* **Recall**
* **F1-score**

### Why these metrics?

Because the dataset is **imbalanced**, accuracy alone is misleading.

---

## 📊 Results Interpretation

Example output:

* High recall for fraud → detects most fraud cases
* Low precision → many false positives

👉 This is expected in anomaly detection:

* Better to **detect fraud aggressively** than miss it

---

## 📉 Visualization

A histogram is used to visualize:

* Distribution of reconstruction error
* Separation between normal and fraud transactions
* Threshold line for classification

This helps in:

* Understanding model behavior
* Choosing better threshold

---

## ⚡ Hardware Utilization

The code automatically detects GPU:

* If GPU available → uses large batch size (2048)
* If CPU → smaller batch size (256)

This improves:

* Training speed
* Resource efficiency

---

## 🧩 Key Concepts Used

* Autoencoders
* Unsupervised learning
* Anomaly detection
* Reconstruction error
* PCA-transformed features
* Imbalanced datasets handling

---

## 📌 Conclusion

This project demonstrates how Autoencoders can be effectively used for **fraud detection without supervised labels**.

### Key Takeaways:

* Training only on normal data enables anomaly detection
* Reconstruction error is a powerful indicator of fraud
* Threshold selection is critical for performance
* Model prioritizes **recall over precision** due to problem nature

---

## 📎 References

* Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* Code inspiration: https://github.com/phansieeex3/Credit-Card-Fraud-Detection-Autoencoder

---

## 🧑‍💻 Author Notes

This implementation adapts and modernizes existing approaches using **TensorFlow 2.x**, improving readability, modularity, and training efficiency while maintaining the core anomaly detection methodology.
