# DATASET LINK : https://www.kaggle.com/datasets/ineubytes/pm25-airpolution-dataset
# SOURCES LINK : https://medium.com/data-science-data-engineering/time-series-prediction-lstm-bi-lstm-gru-99334fc16d75
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,median_absolute_error, mean_squared_log_error
from tensorflow.keras.preprocessing import sequence # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense ,Dropout, Embedding, LSTM, Bidirectional,GRU # type: ignore
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# DATA PREPROCCESSING

import pandas as pd

# 1. Φόρτωση του dataset
df = pd.read_csv('C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/LSTM_COMPARE/PRSA_data_2010.1.1-2014.12.31.csv')

# 2. Δημιουργία Datetime Index
df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# Βάζουμε τη νέα στήλη 'date' ως ευρετήριο (Index) του πίνακα
df.set_index('date', inplace=True)

# 3. Διαγραφή των στηλών που δεν χρειαζόμαστε πια
df.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)

# 4. Καθαρισμός των κενών τιμών (NaN values)
# Διαγράφουμε τις ώρες που ο σταθμός δεν κατέγραψε τη ρύπανση pm2.5
df.dropna(subset=['pm2.5'], inplace=True)

# 5. Μετατροπή Λεκτικών (Categorical) σε Αριθμούς (One-Hot Encoding)
# Το 'cbwd' (κατεύθυνση ανέμου) γίνεται 4 νέες στήλες με 0 και 1
df = pd.get_dummies(df, columns=['cbwd'], prefix='wind')

# Μετατρέπουμε τα True/False του get_dummies σε 1/0 
df = df.astype(float)

# 6. Εμφάνιση των πρώτων 5 γραμμών για επιβεβαίωση
print(df.head())

# Data Validation

train_size = int(len(df) * 0.80) # Βρίσκουμε το 80%

train_df = df.iloc[:train_size] # Κρατάμε το 80% για Train
test_df = df.iloc[train_size:]  # Κρατάμε το υπόλοιπο 20% για Test

y_train = train_df[['pm2.5']] # Απομονώνουμε μόνο τον στόχο (pm2.5)
y_test = test_df[['pm2.5']]


#========= Scalling Data ============"
# Data Normalaization
def Scale (y_train,y_test):  
    #AttributeError προσθεσα τα 2 if επειδη μου εβγαζε attribute error λογω του οτι ο κωδικας υποθετει οτι τα data ειναι series 
    #Επομένως χρησημοποιούμε την hasattr για να ειμαστε σιγουροι οτι διαβαζει df και οχι series
    train=y_train.to_frame() if hasattr(y_train, 'to_frame') else y_train
    test= y_test.to_frame() if hasattr(y_test, 'to_frame') else y_test
    scalerr = MinMaxScaler(feature_range=(0, 1))
    scaler = scalerr.fit(train)
    y_trainS =scaler.transform(train)
    y_testS = scaler.transform(test)
    return(y_trainS,y_testS,scaler)

y_trainS,y_testS,scaler=Scale (y_train,y_test)

print("Data Preproccessing Completed!\n ")


#============= reshape the input of LSTM model==============#
def Create_Dataset (X, look_back):
    Xs, ys = [], []
    
    for i in range(len(X)-look_back):
        v = X[i:i+look_back]
        Xs.append(v)
        ys.append(X[i+look_back])
        
    return np.array(Xs), np.array(ys)
LOOK_BACK = 12
X_trainn, y_trainn = Create_Dataset(y_trainS,LOOK_BACK)
X_testt, y_testt = Create_Dataset(y_testS,LOOK_BACK)
print('X_trainn.shape',X_trainn.shape)
print('y_trainn.shape',y_trainn.shape)
print('X_testt.shape',X_testt.shape)
print('y_testt.shape',y_testt.shape)


#==Define model architecture
units = 50
model = Sequential()
#===== Add LSTM layers
model.add(LSTM(units = units, return_sequences=True,activation='relu',
                   input_shape=(X_trainn.shape[1], X_trainn.shape[2])))
#===== Hidden layer
model.add(LSTM(units = units))
#=== output layer
model.add(Dense(units = 1))
#==== Compiling the model
model.compile(optimizer='adam', loss='mape') 


def Train_LSTM(X_trainn, y_trainn, units, batch_size, epochs):
    model = Sequential()
    model.add(LSTM(units = units, return_sequences=True, activation='relu',
                   input_shape=(X_trainn.shape[1], X_trainn.shape[2])))
    model.add(LSTM(units = units)) # return_sequences=False (εξ ορισμού)
    model.add(Dense(units = 1))
    model.compile(optimizer='adam', loss='mape') 
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    history = model.fit(X_trainn, y_trainn, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size, shuffle = False, callbacks = [early_stop], verbose=0)
    
    return(history, 'LSTM', model)


def Train_BiLSTM(X_trainn, y_trainn, units, batch_size, epochs):
    model = Sequential()
    model.add(Bidirectional(LSTM(units = units, return_sequences=True, activation='relu'),
                            input_shape=(X_trainn.shape[1], X_trainn.shape[2])))
    model.add(Bidirectional(LSTM(units = units, return_sequences=True)))
    # ΑΛΛΑΓΗ ΕΔΩ: return_sequences=False για να συνδεθεί σωστά με το Dense
    model.add(Bidirectional(LSTM(units = units, return_sequences=False))) 
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mape') 
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    history = model.fit(X_trainn, y_trainn, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size, shuffle = False, callbacks = [early_stop], verbose=0)
    
    return(history, 'BiLSTM', model)


def Train_GRU(X_trainn, y_trainn, units, batch_size, epochs):
    model = Sequential()
    model.add(GRU (units = units, return_sequences = True, activation='relu',
                   input_shape = (X_trainn.shape[1], X_trainn.shape[2])))
    model.add(GRU(units = units, return_sequences = True))  
    # ΑΛΛΑΓΗ ΕΔΩ: return_sequences=False
    model.add(GRU(units = units, return_sequences = False)) 
    model.add(Dropout(0.3))
    model.add(Dense(units = 1)) 
    model.compile(optimizer='adam', loss='mape') 
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    history = model.fit(X_trainn, y_trainn, epochs = epochs, validation_split = 0.2,
                        batch_size = batch_size, shuffle = False, callbacks = [early_stop], verbose=0)
    
    return(history, 'GRU', model)



def Train_ConvLSTM(X_trainn, y_trainn, units, batch_size, epochs):
    model = Sequential()
    
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', 
                                     input_shape=(X_trainn.shape[1], X_trainn.shape[2])))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    model.add(LSTM(units=units, activation='relu'))
    
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mape')
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_trainn, y_trainn, epochs=epochs, validation_split=0.2,
                        batch_size=batch_size, shuffle=False, callbacks=[early_stop], verbose=0)
    
    return history, 'ConvLSTM', model


def Train_AttentionLSTM(X_trainn, y_trainn, units, batch_size, epochs):
    inputs = tf.keras.Input(shape=(X_trainn.shape[1], X_trainn.shape[2]))
    
    lstm_out = LSTM(units, return_sequences=True, activation='relu')(inputs)
    
    attention_out = tf.keras.layers.Attention()([lstm_out, lstm_out])
    
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
    
    outputs = Dense(1)(pooled)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='mape')
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_trainn, y_trainn, epochs=epochs, validation_split=0.2,
                        batch_size=batch_size, shuffle=False, callbacks=[early_stop], verbose=0)
    
    return history, 'AttentionLSTM', model




def make_pred_LSTM(model, scaled_train_data, scaled_test_data, n_input, n_features, scalerfit):
    #=========== Predict train =============#
    lstm_predictions_scaledt = list()
    batcht = scaled_train_data[:n_input]
    current_batcht = batcht.reshape((1, n_input, n_features))
    
    for i in range(min(100, len(scaled_train_data))):   
        lstm_predt = model.predict(current_batcht, verbose=0)[0]
        lstm_predictions_scaledt.append(lstm_predt) 
        
        lstm_predt_reshaped = lstm_predt.reshape((1, 1, n_features))
        current_batcht = np.append(current_batcht[:, 1:, :], lstm_predt_reshaped, axis=1)
        
    lstm_predict_train = abs(scalerfit.inverse_transform(lstm_predictions_scaledt))
    
    lstm_predictions_scaled = list()
    batch = scaled_train_data[-n_input:]
    current_batch = batch.reshape((1, n_input, n_features))
     #============ Predict test ==============#
    for i in range(min(100, len(scaled_test_data))):   
        lstm_pred = model.predict(current_batch, verbose=0)[0]
        lstm_predictions_scaled.append(lstm_pred) 
        
        lstm_pred_reshaped = lstm_pred.reshape((1, 1, n_features))
        current_batch = np.append(current_batch[:, 1:, :], lstm_pred_reshaped, axis=1)
        
    lstm_predict_test = abs(scalerfit.inverse_transform(lstm_predictions_scaled))
    
    return (abs(lstm_predict_train), abs(lstm_predict_test))


def make_Forecast_LSTM(model,scaled_test_data,n_input,n_features,scalerfit,nbr_month):
    lstm_predictions_scaled = list()
    batch = scaled_test_data[-n_input:]
    current_batch = batch.reshape((1, n_input, n_features))
    for i in range(nbr_month+1):   
        lstm_pred = model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred) 
        current_batch = np.append(current_batch[:,1:,:],[lstm_pred],axis=1)
    lstm_forcast = scalerfit.inverse_transform(lstm_predictions_scaled)
    return (abs(lstm_forcast))


UNITS = 50
BATCH_SIZE = 64
EPOCHS = 20
results = []

model_functions = [
    (Train_LSTM, "LSTM"),
    (Train_BiLSTM, "BiLSTM"),
    (Train_GRU, "GRU"),
    (Train_ConvLSTM, "ConvLSTM"),
    (Train_AttentionLSTM, "AttentionLSTM")
]

for train_func, name in model_functions:
    print(f"Εκπαίδευση μοντέλου: {name}...")
    
    history, m_name, trained_model = train_func(X_trainn, y_trainn, UNITS, BATCH_SIZE, EPOCHS)
    
    _, test_predictions = make_pred_LSTM(trained_model, y_trainS, y_testS, LOOK_BACK, 1, scaler)
    
    y_actual = y_test.values[:len(test_predictions)]
    
    rmse = np.sqrt(mean_squared_error(y_actual, test_predictions))
    mae = mean_absolute_error(y_actual, test_predictions)
    r2 = r2_score(y_actual, test_predictions)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    })

comparison_df = pd.DataFrame(results)

print("\n" + "="*60)
print("ΤΕΛΙΚΟΣ ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ")
print("="*60)
print(comparison_df.to_string(index=False))
print("-" * 60)

# Εύρεση καλύτερων μοντέλων ανά μετρική
best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin()]
best_mae = comparison_df.loc[comparison_df['MAE'].idxmin()]
best_r2 = comparison_df.loc[comparison_df['R2 Score'].idxmax()]

print(f" Καλύτερο Μοντέλο βάσει RMSE (Λιγότερο Σφάλμα): {best_rmse['Model']} ({best_rmse['RMSE']})")
print(f" Καλύτερο Μοντέλο βάσει MAE  (Μέση Απόκλιση): {best_mae['Model']} ({best_mae['MAE']})")
print(f" Καλύτερο Μοντέλο βάσει R2   (Προσαρμογή):    {best_r2['Model']} ({best_r2['R2 Score']})")

# Συνολικό συμπέρασμα
if best_rmse['Model'] == best_r2['Model']:
    print(f"\n Το μοντέλο {best_rmse['Model']} το καλήτερο.")
else:
    print(f"\n Το μοντέλο {best_rmse['Model']} υπερτερεί σε ακρίβεια σφάλματος.")
print("="*60)

# METRICS CHOSEN FROM THE LINK : https://www.geeksforgeeks.org/machine-learning/regression-metrics/
# https://machinelearningmastery.com/regression-metrics-for-machine-learning/

# MAE was selected because it measures the average prediction error in the dataset’s exact physical units, making it easily interpretable.
# RMSE was chosen because it squares errors, strictly penalizing large prediction outliers to avoid extreme forecasting failures.
# The R^2 score provides a standardized metric from $0$ to $1$ to evaluate how well the model captures the overall trend.

"""                     pm2.5  DEWP  TEMP    PRES   Iws   Is   Ir  wind_NE  wind_NW  wind_SE  wind_cv
date
2010-01-02 00:00:00  129.0 -16.0  -4.0  1020.0  1.79  0.0  0.0      0.0      0.0      1.0      0.0
2010-01-02 01:00:00  148.0 -15.0  -4.0  1020.0  2.68  0.0  0.0      0.0      0.0      1.0      0.0
2010-01-02 02:00:00  159.0 -11.0  -5.0  1021.0  3.57  0.0  0.0      0.0      0.0      1.0      0.0
2010-01-02 03:00:00  181.0  -7.0  -5.0  1022.0  5.36  1.0  0.0      0.0      0.0      1.0      0.0
2010-01-02 04:00:00  138.0  -7.0  -5.0  1022.0  6.25  2.0  0.0      0.0      0.0      1.0      0.0
Data Preproccessing Completed!

X_trainn.shape (33393, 12, 1)
y_trainn.shape (33393, 1)
X_testt.shape (8340, 12, 1)
y_testt.shape (8340, 1)
2026-03-26 01:41:15.550484: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Εκπαίδευση μοντέλου: LSTM...
Εκπαίδευση μοντέλου: BiLSTM...
Εκπαίδευση μοντέλου: GRU...
Εκπαίδευση μοντέλου: ConvLSTM...
Εκπαίδευση μοντέλου: AttentionLSTM...

============================================================
ΤΕΛΙΚΟΣ ΣΥΓΚΡΙΤΙΚΟΣ ΠΙΝΑΚΑΣ
============================================================
        Model         RMSE          MAE      R2 Score
         LSTM 2.656358e+02 1.916723e+02 -1.021758e+00
       BiLSTM 1.017675e+03 1.000058e+03 -2.867385e+01
          GRU 2.081095e+02 1.514951e+02 -2.409081e-01
     ConvLSTM 6.850948e+07 2.200161e+07 -1.344798e+11
AttentionLSTM 2.775906e+02 2.047802e+02 -1.207828e+00
------------------------------------------------------------
 Καλύτερο Μοντέλο βάσει RMSE (Λιγότερο Σφάλμα): GRU (208.10949550868955)
 Καλύτερο Μοντέλο βάσει MAE  (Μέση Απόκλιση): GRU (151.49505624815822)
 Καλύτερο Μοντέλο βάσει R2   (Προσαρμογή):    GRU (-0.24090806452647562)

 Το μοντέλο GRU το καλήτερο.
    """