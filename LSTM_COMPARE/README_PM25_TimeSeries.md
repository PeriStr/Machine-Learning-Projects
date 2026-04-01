# 🌫️ PM2.5 Air Pollution Forecasting
### Time Series Prediction: LSTM vs BiLSTM vs GRU vs ConvLSTM vs AttentionLSTM

---

## 📌 Description

A time series forecasting project that predicts **PM2.5 air pollution levels** in Beijing using five different recurrent neural network architectures.

All five models are trained on the same dataset and evaluated on the same test set, producing a **side-by-side comparison table** of RMSE, MAE, and R² Score.

---

## 📂 Project Structure

```
project/
│
├── main.py                                        # Full pipeline: preprocessing + training + comparison
├── PRSA_data_2010.1.1-2014.12.31.csv              # Input dataset
└── README.md
```

---

## 🗄️ Dataset

**Beijing PM2.5 Dataset** — hourly air quality and weather measurements from 2010 to 2014.

> Download: [PM2.5 Air Pollution Dataset on Kaggle](https://www.kaggle.com/datasets/ineubytes/pm25-airpolution-dataset)

| Column | Description |
|---|---|
| `pm2.5` | **Target** — PM2.5 concentration (μg/m³) |
| `DEWP` | Dew point temperature |
| `TEMP` | Temperature |
| `PRES` | Atmospheric pressure |
| `Iws` | Cumulated wind speed |
| `Is` | Cumulated hours of snow |
| `Ir` | Cumulated hours of rain |
| `cbwd` | Wind direction (categorical → One-Hot encoded) |

After preprocessing the dataset has **11 columns** and ~41,000 hourly records.

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

## 🔄 Preprocessing Pipeline

```
Load CSV
   │
   ▼
Build datetime index from year/month/day/hour columns
   │
   ▼
Drop rows with missing pm2.5 values
   │
   ▼
One-Hot Encode wind direction (cbwd → wind_NE, wind_NW, wind_SE, wind_cv)
   │
   ▼
Train/Test split  →  80% train | 20% test  (chronological, no shuffle)
   │
   ▼
MinMaxScaler fit on train only → transform both train and test
   │
   ▼
Sliding window (look_back = 12)
   ├── X shape: (samples, 12, 1)   ← 12 past hours as input
   └── y shape: (samples, 1)       ← next hour as target
```

**Why MinMaxScaler fit only on train?**
Fitting on test data would cause data leakage — the model would indirectly "see" future values during training.

**Why `shuffle=False`?**
Time series data has temporal order. Shuffling would break the sequence and cause the model to learn nonsensical patterns.

---

## 🏗️ Model Architectures

### 1. LSTM
```
LSTM(50, return_sequences=True, relu)
LSTM(50)
Dense(1)
```

### 2. BiLSTM (Bidirectional LSTM)
```
Bidirectional(LSTM(50, return_sequences=True, relu))
Bidirectional(LSTM(50, return_sequences=True))
Bidirectional(LSTM(50, return_sequences=False))
Dense(1)
```
Processes the sequence both **forward and backward** — captures past and future context within the training window.

### 3. GRU (Gated Recurrent Unit)
```
GRU(50, return_sequences=True, relu)
GRU(50, return_sequences=True)
GRU(50, return_sequences=False)
Dropout(0.3)
Dense(1)
```
Lighter than LSTM — fewer parameters, faster training, often comparable accuracy.

### 4. ConvLSTM (CNN + LSTM)
```
Conv1D(64 filters, kernel=2, relu)
MaxPooling1D(pool=2)
LSTM(50, relu)
Dense(1)
```
The Conv1D layer extracts **local temporal patterns** before passing them to the LSTM.

### 5. AttentionLSTM
```
LSTM(50, return_sequences=True, relu)
Attention([lstm_out, lstm_out])    ← self-attention
GlobalAveragePooling1D()
Dense(1)
```
The attention mechanism lets the model **focus on the most relevant time steps** rather than weighting all steps equally.

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `LOOK_BACK` | 12 | Hours of history used as input |
| `UNITS` | 50 | Hidden units per recurrent layer |
| `BATCH_SIZE` | 64 | Samples per gradient update |
| `EPOCHS` | 20 | Maximum training epochs |
| `Early Stopping` | patience = 10 | Stops if val_loss doesn't improve |
| `Validation split` | 20% of train | Used during training only |
| `Loss function` | MAPE | Mean Absolute Percentage Error |
| `Optimizer` | Adam | Adaptive learning rate |

---

## 📊 Actual Results

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| LSTM | 265.64 | 191.67 | -1.02 |
| BiLSTM | 1017.68 | 1000.06 | -28.67 |
| **GRU** | **208.11** | **151.50** | **-0.24** |
| ConvLSTM | 68,509,477 | 22,001,607 | -1.34×10¹¹ |
| AttentionLSTM | 277.59 | 204.78 | -1.21 |

**Winner: GRU** — best on all three metrics.

**Notes on results:**
- All R² scores are negative, meaning no model reliably outperforms a simple mean baseline. This is expected given that only **pm2.5 as a univariate input** is used (the other columns are not passed as features to the sequence).
- The **ConvLSTM diverged** completely — the Conv1D + MaxPooling combination likely collapsed too much temporal information for the short look-back window of 12 steps.
- **BiLSTM underperformed** despite its larger architecture — it may need more epochs and tuning to converge on this task.

---

## 📈 Evaluation Metrics

| Metric | Why it was chosen |
|---|---|
| **RMSE** | Penalizes large errors heavily — important when extreme pollution spikes matter |
| **MAE** | Average error in the original unit (μg/m³) — directly interpretable |
| **R² Score** | Measures how well the model captures the overall trend (0 = mean baseline, 1 = perfect) |

> Sources: [GeeksForGeeks Regression Metrics](https://www.geeksforgeeks.org/machine-learning/regression-metrics/) · [ML Mastery Regression Metrics](https://machinelearningmastery.com/regression-metrics-for-machine-learning/)

---

## ▶️ How to Run

```bash
python main.py
```

All 5 models train sequentially and print the final comparison table automatically.

---

## 💡 Potential Improvements

- Use **all weather features** (TEMP, DEWP, PRES, etc.) as multivariate inputs instead of pm2.5 only
- Increase `LOOK_BACK` to 24 or 48 hours for longer context
- Tune `UNITS` and add more layers for BiLSTM and AttentionLSTM
- Replace MAPE with MSE or Huber loss (MAPE is unstable near zero values)

---

## 📦 Requirements

```
pandas
numpy
matplotlib
scikit-learn
tensorflow
```
