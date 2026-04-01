📊 Customer Segmentation & Machine Learning Project
🔍 Project Overview

This project focuses on analyzing customer behavior in an e-commerce dataset using a complete data science pipeline.

The goal is to:

Clean and preprocess raw data
Identify customer groups (clustering)
Predict these groups using machine learning
Discover hidden patterns in customer behavior
📂 Dataset
File used: ecommerecenice.csv
Contains:
Customer activity (sessions, clicks, pages viewed)
Purchase behavior (frequency, monetary value)
Engagement (wishlist, campaigns)
Returns & cart abandonment
Segment labels
⚙️ Workflow
🔹 Data Preprocessing
Removed missing values (especially Customer_ID)
Filled numeric missing values using median
Handled invalid values (negative numbers, unrealistic durations)
Applied outlier capping (IQR)
Cleaned categorical data (Segment labels)
Encoding:
Label Encoding (Customer_ID)
One-Hot Encoding (Segments)
Removed duplicates

Final dataset saved as:

clean_preprocessed_ecommerce.csv
🔹 K-Means Clustering
Used behavioral features (Recency, Frequency, Monetary, etc.)
Applied StandardScaler
Found optimal clusters using:
Elbow Method
Silhouette Score
Final choice: k = 4 clusters

👉 Output:

Each customer assigned to a cluster
Cluster profiles (average behavior)
🔹 XGBoost Classification
Goal: Predict customer cluster
Model: XGBoost Classifier

Hyperparameters:

n_estimators = 200
max_depth = 4
learning_rate = 0.1

👉 Evaluation:

Accuracy
Precision / Recall / F1-score

👉 Also includes:

Feature importance plot
🔹 Apriori Algorithm (Association Rules)
Converts features into binary (0/1)
Finds relationships between:
Segments
Behavior (wishlist, returns, etc.)

👉 Metrics used:

Support
Confidence
Lift

👉 Output:

Frequent itemsets
Top association rules
📈 Technologies Used
Python
Pandas / NumPy
Scikit-learn
XGBoost
MLxtend
Matplotlib
▶️ How to Run
Put the dataset in the correct path
Update the path in the code
Run the script
📌 Notes
Combines:
Unsupervised Learning (K-Means)
Supervised Learning (XGBoost)
Pattern Mining (Apriori)
This is a complete data analysis pipeline project
🧠 Conclusion
K-Means successfully groups customers into meaningful segments
XGBoost predicts these segments effectively
Apriori reveals useful hidden behavioral patterns
