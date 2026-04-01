# DATASET LINK : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# CODE SCOURCE LINK : https://github.com/phansieeex3/Credit-Card-Fraud-Detection-Autoencoder

import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# READ CSV 
data = pd.read_csv("C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/autoencoder/creditcard.csv")
'''
 Time        V1        V2        V3        V4        V5        V6        V7        V8  ...       V22       V23       V24       V25       V26       V27       V28  Amount  Class
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  0.098698  ...  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053  149.62      0
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  0.085102  ... -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  0.247676  ...  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752  378.66      0
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  0.377436  ...  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458  123.50      0
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941 -0.270533  ...  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   69.99      0

[5 rows x 31 columns]
'''

# Data preproccessing
from sklearn.preprocessing import StandardScaler

# No further preprocessing such as One-Hot Encoding or Missing Value imputation is required.
# Reasons:
# 1. The V1-V28 features are the result of a PCA (Principal Component Analysis) transformation, 
#    meaning they are already numerical and scaled.
# 2. There are no categorical variables in this dataset that would require encoding.
# 3. The dataset is already clean with zero missing (NaN) values, as verified by data.isnull().values.any().
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
print(data.head())
print(data.isnull().values.any())
print(data.dtypes.value_counts())

'''
     V1        V2        V3        V4        V5        V6        V7        V8        V9  ...       V22       V23       V24       V25       V26       V27       V28  Class  normAmount
0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  0.098698  0.363787  ...  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053      0    0.244964
1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  0.085102 -0.255425  ... -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724      0   -0.342475
2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  0.247676 -1.514654  ...  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752      0    1.160686
3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  0.377436 -1.387024  ...  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458      0    0.140534
4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941 -0.270533  0.817739  ...  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153      0   -0.073403

[5 rows x 30 columns]
False
float64    29
int64       1
Name: count, dtype: int64
'''


from tensorflow.keras import layers, models, optimizers

class Autoencoder:
    def __init__(self, n_hidden_1, n_hidden_2, n_input, learning_rate):
        self.encoder = models.Sequential([
            layers.Dense(n_hidden_1, activation='relu', input_shape=(n_input,)),
            layers.Dropout(0.2), 
            layers.Dense(n_hidden_2, activation='relu')
        ])
        
        self.decoder = models.Sequential([
            layers.Dense(n_hidden_1, activation='relu', input_shape=(n_hidden_2,)),
            layers.Dropout(0.2),
            layers.Dense(n_input, activation='linear') 
        ])
        
        self.model = models.Sequential([self.encoder, self.decoder])
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')

    def calc_total_cost(self, X):
        return self.model.evaluate(X, X, verbose=0)

    def partial_fit(self, X):
        return self.model.train_on_batch(X, X)

    def transform(self, X):
        return self.encoder.predict(X, verbose=0)

    def reconstruct(self, X):
        return self.model.predict(X, verbose=0)


# CODE FROM GITHUB REPO NEEDED TO BE REWRITTEN IN TENSORFLOW 2.0:
'''
def encoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    def decoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.sess.run(self.encoder_op, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoder_op, feed_dict={self.x: X})
'''


"""
1. Shuffle: Randomizes the dataset to ensure no ordering bias.
2. Split: Manually calculates an 80/20 split point.
3. Train Filter: Isolates 'Class 0' (Normal) cases for the Training set, 
   following the semi-supervised anomaly detection paradigm.
4. Feature Extraction: Drops the 'Class' column and converts DataFrames 
   to NumPy arrays for neural network compatibility.
"""
#Validation Stage
# shaffling the data
data_shuffled = data.sample(frac=1, random_state=42)

#Validation 80% train and 20% test
split_index = int(0.8 * len(data_shuffled))
train_set = data_shuffled.iloc[:split_index]
test_set = data_shuffled.iloc[split_index:]

# Keep only class 0 for training 
X_train = train_set[train_set['Class'] == 0].drop('Class', axis=1).values

y_test = test_set['Class'].values
X_test = test_set.drop('Class', axis=1).values


# Set the training on GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Training on GPU")
        batch_size_val = 2048
    except:
        print("ERROR ON GPU.")
else:
    batch_size_val = 256
    print("Training on CPU.")
 
'''
1. Hyperparameter Definition:
    - n_input: 29 features (V1-V28 and scaled Amount).
    - n_hidden_1 & n_hidden_2: Compression architecture (29 -> 16 -> 8).
    - batch_size_val: Optimized at 2048 for GPU utilization.
'''

# Hyperparameters   
n_input = X_train.shape[1]
n_hidden_1 = 16
n_hidden_2 = 8
learning_rate = 0.001
num_epochs = 100
batch_size_val = 2048 if gpus else 256
# Initialize the Autoencoder
autoencoder = Autoencoder(n_hidden_1, n_hidden_2, n_input, learning_rate)

'''
2. Model Training Loop:
    - Uses partial_fit for batch-wise weight updates.
    - Implements Early Stopping by monitoring Validation Loss (patience=10) 
      to prevent overfitting.
'''
# Early Stopping Parameters
best_loss = float('inf')
patience = 10
wait = 0

for epoch in range(num_epochs):
    np.random.shuffle(X_train)
    num_batches = len(X_train) // batch_size_val
    
    for i in range(num_batches):
        batch_X = X_train[i*batch_size_val : (i+1)*batch_size_val]
        autoencoder.partial_fit(batch_X)
    
    val_loss = autoencoder.calc_total_cost(X_test)
    print(f"Epoch: {epoch + 1} | Train Cost: {autoencoder.calc_total_cost(X_train):.6f} | Val Loss: {val_loss:.6f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break  

'''
3. Detection Threshold Selection:
    - Calculation of Mean Squared Error (MSE) on the X_train set.
    - Threshold set at the 95th percentile of normal transaction errors.
'''
# Calculate reconstruction errors
train_reconstructions = autoencoder.reconstruct(X_train)
train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)

threshold = np.percentile(train_mse, 95)
threshold = np.percentile(train_mse, 99)  
threshold = np.percentile(train_mse, 90) 
  
test_reconstructions = autoencoder.reconstruct(X_test)
test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)

print(f"Threshold: {threshold:.6f}")
'''
4. Evaluation and Anomaly Detection:
    - Application of the trained model on the X_test set.
    - Transactions classified as 'Fraud' if MSE exceeds the threshold.
    - Generation of Classification Report, Confusion Matrix, and 
      visualization of the Reconstruction Error Distribution.
'''
# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (test_mse > threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.hist(test_mse[y_test==0], bins=50, alpha=0.5, label='Normal')
plt.hist(test_mse[y_test==1], bins=50, alpha=0.5, label='Fraud')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error Distribution')
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Transactions')
plt.legend()
plt.show()

'''
Epoch 10/50, Cost: 0.5769
Epoch 20/50, Cost: 0.5861
Epoch 30/50, Cost: 0.5836
Epoch 40/50, Cost: 0.5957
Epoch 50/50, Cost: 0.5896
Confusion Matrix:
[[54014  2843]
 [   11    94]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.95      0.97     56857
           1       0.03      0.90      0.06       105

    accuracy                           0.95     56962
   macro avg       0.52      0.92      0.52     56962
weighted avg       1.00      0.95      0.97     56962

Training on CPU.'''



    

    
    
    
    





