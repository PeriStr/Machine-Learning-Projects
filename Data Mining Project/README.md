🧠 Customer Segmentation & Behavioral Analysis with Machine Learning
📌 Project Overview

This project implements a complete data science pipeline for customer behavior analysis in an e-commerce environment.
It combines data preprocessing, unsupervised learning (clustering), supervised learning (classification), and association rule mining to extract meaningful insights from customer data.

The main goal is to:

Clean and prepare raw customer data
Identify customer segments using clustering
Predict customer segments using machine learning
Discover hidden behavioral patterns using association rules
📊 Dataset
Source: Custom e-commerce dataset (ecommerecenice.csv)
Description:
The dataset contains customer behavioral data such as:
Purchase frequency and monetary value
Session activity (clicks, pages viewed, duration)
Campaign response
Wishlist interactions
Returns and cart abandonment rates
Customer segmentation labels
⚙️ Project Pipeline
🔹 1. Data Preprocessing

The dataset undergoes extensive cleaning and transformation:

Removal of missing critical values (Customer_ID)
Imputation of missing numerical values using median
Handling of invalid data (e.g., negative values, unrealistic session duration)
Outlier treatment using IQR-based capping
Standardization of categorical values
Encoding:
Label Encoding for Customer IDs
One-Hot Encoding for customer segments
Duplicate removal
Logical validation of inconsistent entries

👉 Output: Clean dataset saved as
clean_preprocessed_ecommerce.csv

🔹 2. Customer Segmentation (K-Means Clustering)

Customers are grouped into clusters based on behavioral features:

Features used:
Recency, Frequency, Monetary
Session activity & engagement metrics
Data scaling using StandardScaler
Optimal number of clusters determined using:
Elbow Method
Silhouette Score

👉 Final model:

K = 4 clusters

👉 Output:

Cluster assignment for each customer
Cluster profiles (mean behavior per cluster)
🔹 3. Cluster Prediction (XGBoost Classifier)

A supervised model is trained to predict customer clusters:

Model: XGBoost Classifier
Key hyperparameters:
n_estimators = 200
max_depth = 4
learning_rate = 0.1
Train/Test split: 80% / 20%

👉 Evaluation Metrics:

Accuracy
Precision / Recall / F1-score

👉 Additional Output:

Feature Importance visualization
🔹 4. Market Basket Analysis (Apriori Algorithm)

Association rules are extracted to identify hidden behavioral patterns:

Binary transformation of features
Use of:
apriori for frequent itemsets
association_rules for rule extraction

👉 Key metrics:

Support
Confidence
Lift

👉 Output:

Frequent itemsets
Top association rules
Strongest rules based on lift
📈 Technologies Used
Python
Pandas / NumPy
Scikit-learn
XGBoost
MLxtend
Matplotlib
🧪 Key Insights
K-Means clustering successfully identifies distinct customer behavior groups
XGBoost can accurately predict customer segments
Apriori reveals strong relationships between:
Customer segments
Campaign responses
Behavioral patterns (wishlist, returns, cart abandonment)
▶️ How to Run
Place the dataset in the correct path
Update the file path inside the script
Run the script:
python script.py
📌 Notes
The project combines unsupervised + supervised + pattern mining, making it a complete data science workflow
Suitable for:
Customer segmentation
Marketing strategy optimization
Recommendation systems
🚀 Future Improvements
Use Deep Learning (Autoencoders) for anomaly detection
Replace K-Means with DBSCAN or Hierarchical Clustering
Deploy results in a dashboard (Power BI / Streamlit)
