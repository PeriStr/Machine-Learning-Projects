# LSTM Model Comparison for Time Series Prediction

## 📌 Project Overview

This project focuses on the **implementation, training, and comparative evaluation** of multiple deep learning architectures designed for **time series forecasting**. Specifically, it analyzes different variants of Recurrent Neural Networks (RNNs), including **LSTM-based models**, to predict air pollution levels (**PM2.5**).

The main objective is to determine **which architecture performs best** when modeling temporal dependencies in real-world environmental data.

---

## 📊 Dataset

* Dataset: https://www.kaggle.com/datasets/ineubytes/pm25-airpolution-dataset
* Type: Multivariate Time Series
* Frequency: Hourly measurements

### Description:

The dataset contains:

* Air pollution measurements (**PM2.5 – target variable**)
* Meteorological features:

  * Temperature
  * Pressure
  * Dew Point
  * Wind speed
  * Wind direction (categorical)
  * Rain/Snow indicators

This makes it ideal for **sequence modeling and forecasting tasks**.

---

## 📚 Theoretical Background

The project is based on different variants of recurrent architectures explained in:

* https://medium.com/data-science-data-engineering/time-series-prediction-lstm-bi-lstm-gru-99334fc16d75

These architectures are designed to:

* Capture **temporal dependencies**
* Handle **sequential data**
* Overcome issues like **vanishing gradients**

---

## 🧠 Models Implemented

The following models were implemented and compared:

* **LSTM** → Baseline sequential model
* **Bidirectional LSTM (BiLSTM)** → Processes sequences forward & backward
* **GRU** → Simplified and faster alternative to LSTM
* **Conv1D + LSTM (ConvLSTM)** → Extracts local temporal patterns before sequence modeling
* **Attention-based LSTM** → Learns which time steps are more important

---

## 🔄 Workflow

### 1. Data Preprocessing

* Conversion to **datetime index**
* Removal of unnecessary columns
* Handling missing values (NaN removal)
* **One-hot encoding** for categorical feature (wind direction)
* Feature scaling using **MinMaxScaler**

---

### 2. Time Series Transformation

* Creation of sequences using a **sliding window approach**
* Parameter:

  * `look_back = 12` → model uses previous 12 time steps to predict next value

---

### 3. Model Training

Each model is trained using:

* **Validation split (20%)**
* **Early stopping (patience = 10)** → prevents overfitting
* **Batch training**
* No shuffling → preserves time order

---

### 4. Prediction

* Models generate predictions on test data
* Sequential forecasting approach is applied

---

## 📈 Evaluation Metrics

The models are evaluated using:

* **RMSE (Root Mean Squared Error)**
  → Penalizes large errors more heavily

* **MAE (Mean Absolute Error)**
  → Measures average prediction error

* **R² Score (Coefficient of Determination)**
  → Indicates how well the model explains variance

### Why these metrics?

They provide a **balanced evaluation**:

* Accuracy (MAE)
* Sensitivity to large errors (RMSE)
* Overall fit (R²)

---

## 🏆 Results

All models are evaluated and compared in a **final results table**.

### Key Finding:

👉 The **GRU model** achieved the best overall performance:

* Lowest RMSE
* Lowest MAE
* Best R² score (relative to others)

### Interpretation:

GRU performs better because:

* Simpler architecture → less overfitting
* Faster convergence
* Efficient handling of temporal dependencies

---

## 📊 Output Example

The script prints:

* Model comparison table
* Best model selection based on metrics
* Performance summary

---

## ▶️ How to Run

1. Download the dataset from Kaggle
2. Update the dataset path inside the script
3. Run the Python file

---

## ⚙️ Requirements

* Python 3.x
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📌 Notes

* Training can run on **CPU or GPU**
* GPU improves training speed significantly
* Early stopping ensures efficient training without overfitting
* The dataset is **multivariate**, but the model predicts only **PM2.5**

---

## 🧩 Key Concepts Demonstrated

* Time series forecasting
* Recurrent Neural Networks (RNNs)
* LSTM, GRU, BiLSTM architectures
* Attention mechanism
* Feature scaling & sequence generation
* Model evaluation and comparison

---

## 📎 Conclusion

This project demonstrates a complete pipeline for **time series prediction using deep learning**, from preprocessing to evaluation.

### Main takeaway:

👉 While LSTM-based models are powerful, **GRU can outperform them** in certain cases due to its **simpler and more efficient design**.

---

## 👨‍💻 Author Note

This project was developed for educational purposes to explore and compare different deep learning architectures for time series forecasting and understand their strengths and limitations in real-world data scenarios.

