# 🏠 Housing Price Predictor
### Linear Regression with California Housing Dataset

---

## 📌 Description

A command-line application that predicts house prices using **Linear Regression**.
The user can input property features and get an estimated price, or compare a past price against the current model prediction to see how much the value has changed.

---

## 📂 Project Structure

```
project/
│
├── main.py          # Full pipeline: training + CLI menu
├── housing.csv      # California Housing dataset (input)
└── README.md
```

---

## 🗄️ Dataset

**California Housing Dataset** — housing data from California census blocks.

| Column | Description |
|---|---|
| `median_income` | Median income of the area (in tens of thousands) |
| `total_rooms` | Total number of rooms in the block |
| `housing_median_age` | Median age of houses in the block |
| `median_house_value` | **Target** — median house value (in USD) |

> Download: [California Housing on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn
```

---

## 🏗️ Model

```
Features (X)                     Target (y)
─────────────────────            ──────────────────────
median_income       ──┐
total_rooms         ──┼──► Linear Regression ──► median_house_value
housing_median_age  ──┘
```

**Linear Regression formula:**

```
price = w1×median_income + w2×total_rooms + w3×housing_median_age + bias
```

The model finds the best weights (`w1`, `w2`, `w3`) by minimizing the Mean Squared Error between predicted and actual house values.

---

## 🖥️ CLI Menu

```
Παρακολούθηση τιμών ακινήτων
1. Πρόβλεψη μελλοντικής τιμής
2. Σύγκριση παρελθοντικής και μελλοντικής τιμής
3. Έξοδος
```

### Option 1 — Predict Price
Enter three property features and get back an estimated house value.

**Example:**
```
Δώστε το μέσο εισόδημα: 5.0
Δώστε τον συνολικό αριθμό δωματίων: 2000
Δώστε τη μέση ηλικία των κατοικιών: 20
→ Εκτιμώμενη τιμή: 187,432.50 ευρώ
```

### Option 2 — Compare Past vs Predicted
Enter a past value and the current features. The model predicts the new value and tells you whether the price went up, down, or stayed the same.

**Example output:**
```
Η τιμή του ακινήτου αυξήθηκε κατά 23,150.00 ευρώ.
Εκτιμώμενη μελλοντική τιμή: 210,582.75 ευρώ
```

---

## ▶️ How to Run

```bash
python main.py
```

The model trains automatically on startup using the full `housing.csv` dataset. No separate training step is needed.

---

## 📊 Notes on Accuracy

Linear Regression is a simple baseline model. It assumes a linear relationship between features and price, which may not capture complex patterns. For better accuracy, consider:

- Adding more features (e.g. `latitude`, `longitude`, `population`)
- Using a more powerful model (Random Forest, XGBoost)
- Scaling features with `StandardScaler` before training

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
```
