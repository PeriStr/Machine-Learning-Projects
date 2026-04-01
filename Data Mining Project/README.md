# 🛒 E-Commerce Customer Analytics
### Data Mining & Knowledge Extraction Project

---

## 📌 Description

Customer behavior analysis on an e-commerce dataset using three data mining algorithms:

- **K-Means Clustering** → Group customers based on behavioral patterns
- **XGBoost Classifier** → Predict the cluster of new customers
- **Apriori Algorithm** → Discover association rules between customer attributes

---

## 📂 Project Structure

```
project/
│
├── main.py                          # Main script (preprocessing + ML pipeline)
├── ecommerecenice.csv               # Raw input dataset
├── clean_preprocessed_ecommerce.csv # Cleaned dataset (generated output)
└── README.md
```

---

## 🗄️ Dataset

| Column | Description |
|---|---|
| `Customer_ID` | Unique customer identifier |
| `Recency` | Days since last purchase |
| `Frequency` | Total number of orders |
| `Monetary` | Total amount spent |
| `Avg_Order_Value` | Average value per order |
| `Session_Count` | Number of website visits |
| `Avg_Session_Duration` | Average session duration (minutes) |
| `Pages_Viewed` | Total pages visited |
| `Clicks` | Total clicks |
| `Campaign_Response` | Response to marketing campaigns |
| `Wishlist_Adds` | Items added to wishlist |
| `Cart_Abandon_Rate` | Shopping cart abandonment rate |
| `Returns` | Number of returned orders |
| `Segment_Label` | Original segment (Bronze/Silver/Gold/Platinum) |

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn xgboost mlxtend matplotlib
```

---

## 🔄 Pipeline

```
CSV Input
    │
    ▼
1. Preprocessing
   ├── Missing values  → median imputation
   ├── Noise removal   (negative values, sessions > 300 min)
   ├── Outlier capping (IQR method)
   ├── Label Encoding  (Customer_ID)
   └── One-Hot Encoding (Segment_Label)
    │
    ▼
2. K-Means Clustering
   ├── StandardScaler normalization
   ├── Elbow Method → find optimal k
   ├── Silhouette Score → validate k
   └── k=4 clusters → cluster profiling
    │
    ▼
3. XGBoost Classifier
   ├── Train/Test split (80/20)
   ├── Train on cluster labels
   └── Accuracy + Feature Importance
    │
    ▼
4. Apriori
   ├── Binary encoding (wishlist, cart abandon, returns, campaign)
   ├── min_support = 0.05
   └── Association rules with lift > 1.1
```

---

## 🧩 Step 1 — Preprocessing

Data cleaning pipeline in 6 stages:

1. **Missing Values** — drop rows without `Customer_ID`, fill numeric columns with median
2. **Noise Removal** — remove negative values and session durations above 300 minutes
3. **Outlier Capping** — IQR method (Q1 − 1.5×IQR to Q3 + 1.5×IQR)
4. **Text Standardization** — normalize segment labels (e.g. `" silver "` → `"Silver"`)
5. **Encoding** — LabelEncoder for Customer_ID, One-Hot for Segment_Label
6. **Deduplication** — keep one record per Customer_ID

---

## 🔵 Step 2 — K-Means Clustering

**Features used:**
`Recency`, `Frequency`, `Monetary`, `Avg_Order_Value`, `Session_Count`, `Avg_Session_Duration`, `Pages_Viewed`, `Clicks`, `Wishlist_Adds`, `Cart_Abandon_Rate`, `Returns`

**Choosing k:**
- Elbow Method (WCSS) for k = 2 to 9
- Silhouette Score for validation
- Final choice: **k = 4**

**Cluster Profiles (indicative — update after running):**

| Cluster | Label | Key Characteristics |
|---|---|---|
| 0 | High-Value | High Monetary, low Recency |
| 1 | At-Risk | High Recency, low Frequency |
| 2 | Impulse Buyers | High Cart_Abandon, medium Monetary |
| 3 | Loyal | High Frequency, high Session_Count |

---

## 🟢 Step 3 — XGBoost Classifier

**Hyperparameters:**

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | More trees → stronger model |
| `max_depth` | 4 | Controls complexity, prevents overfitting |
| `learning_rate` | 0.1 | Safe, stable learning rate |
| `subsample` | 0.9 | Regularization via row sampling |
| `colsample_bytree` | 0.9 | Regularization via feature sampling |
| `objective` | `multi:softmax` | Multi-class classification |
| `num_class` | 4 | Number of clusters from K-Means |

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1-Score per cluster
- Feature Importance plot (top 15 features)

---

## 🟡 Step 4 — Apriori

**Binary items used:**
- Segment columns (from One-Hot Encoding)
- `Wishlist_High` — wishlist adds above median
- `CartAbandon_High` — cart abandon rate above median
- `Returns_High` — returns above median
- `Campaign_Response` — response > 0

**Parameters:**
- `min_support = 0.05` (item appears in ≥ 5% of customers)
- `metric = "lift"`, `min_threshold = 1.1`

**Example rule:** `{Segment_Gold, Wishlist_High} → {Campaign_Response}`

---

## ▶️ How to Run

```bash
# Make sure ecommerecenice.csv is in the same folder
python main.py
```

**Generated outputs:**
- `clean_preprocessed_ecommerce.csv` — cleaned dataset
- Elbow Method plot
- Silhouette Score plot
- Cluster profiles table
- XGBoost accuracy + classification report
- Feature Importance plot
- Top 10 association rules

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
mlxtend
matplotlib
```
