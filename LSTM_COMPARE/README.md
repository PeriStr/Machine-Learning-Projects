LSTM Model Comparison for Time Series Prediction
📌 Project Overview

This project focuses on the implementation and comparison of multiple recurrent neural network architectures for time series forecasting. Specifically, different variants of Long Short-Term Memory (LSTM) networks are trained to predict air pollution levels (PM2.5).

The goal is to evaluate and compare the performance of these models based on standard regression metrics.

📊 Dataset
Dataset used:
https://www.kaggle.com/datasets/ineubytes/pm25-airpolution-dataset
Description:
The dataset contains hourly air pollution measurements along with meteorological data such as temperature, pressure, wind direction, and more.
📚 Reference / Theory
Source for LSTM architectures:
https://medium.com/data-science-data-engineering/time-series-prediction-lstm-bi-lstm-gru-99334fc16d75
⚙️ Models Implemented

The following models were implemented and compared:

LSTM
Bidirectional LSTM (BiLSTM)
GRU
Conv1D + LSTM (ConvLSTM)
Attention-based LSTM
🔄 Workflow
Data preprocessing:
Datetime conversion
Handling missing values
One-hot encoding
Normalization (MinMaxScaler)
Sequence creation using sliding window (look-back)
Model training with:
Early stopping
Validation split
Prediction and evaluation
📈 Evaluation Metrics

The models are compared using:

RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
R² Score
🏆 Results

The comparison results are printed in a table, and the best model is selected based on performance metrics.

In this implementation, the GRU model achieved the best performance across most metrics.

▶️ How to Run
Download the dataset from Kaggle
Update the dataset path in the code
Run the script
📌 Notes
The project uses TensorFlow/Keras for model implementation
Training is done on CPU (GPU optional)
Early stopping is used to prevent overfitting
