
# Machine Learning Projects

A collection of **Machine Learning** and **Deep Learning** projects built with Python, covering the full spectrum from classical algorithms to modern deep learning architectures. Each project is a standalone, end-to-end pipeline with its own dataset, preprocessing, training, and evaluation.

---

## 📁 Projects

---

### 🛒 Customer Segmentation & Behavior Analysis
> *Unsupervised Learning · Classification · Association Rules*

Analyzed e-commerce customer behavior using a three-stage data mining pipeline. Customers are first grouped by behavioral patterns, then a classifier is trained to predict group membership, and finally association rules reveal hidden purchasing patterns.

- **K-Means Clustering** with Elbow Method + Silhouette Score for optimal k selection
- **XGBoost Classifier** trained on cluster labels for supervised prediction
- **Apriori Algorithm** for market basket association rule mining (lift > 1.1)
- Full preprocessing pipeline: IQR outlier capping, median imputation, One-Hot encoding, deduplication

![Cluster Visualization](https://github.com/PeriStr/Machine-Learning-Projects/blob/7534725aaaf2bc9dfd6debda08580c1ba24341a5/Figure_2.png)

---

### 🖼️ Image Captioning — CNN + LSTM
> *Computer Vision · Sequence Generation · Transfer Learning*

End-to-end system that generates natural language descriptions for images. A frozen ResNet50 encoder extracts visual features which are prepended to the word sequence and fed into an LSTM decoder for autoregressive caption generation.

- **ResNet50** (ImageNet pre-trained, frozen) as visual encoder — 2048 → 256 projection
- **LSTM Decoder** with custom Vocabulary class (SpaCy tokenizer, freq_threshold=5)
- Trained on **Flickr8k** with `<SOS>/<EOS>/<PAD>/<UNK>` token handling
- Greedy Search at inference time, Early Stopping on validation loss
- Tkinter GUI for real-time caption generation on user-selected images

![Image Captioning Demo](https://github.com/PeriStr/Machine-Learning-Projects/blob/774d28b4d4587222d64619a7a3190a47822bb9d2/%CE%A3%CF%84%CE%B9%CE%B3%CE%BC%CE%B9%CF%8C%CF%84%CF%85%CF%80%CE%BF%20%CE%BF%CE%B8%CF%8C%CE%BD%CE%B7%CF%82%202026-03-19%20232005.png)

---

### 🚗 Car Recognition System — CNN (Stanford Cars)
> *Fine-Grained Image Classification · Transfer Learning*

197-class vehicle classifier built on **ResNet18** with Transfer Learning on the Stanford Cars dataset. Includes a custom "Not a Car" class to handle out-of-distribution inputs.

- **197 classes**: 196 car make/model/year categories + 1 "No Car" class
- **ResNet18** (ImageNet pre-trained) with custom `Linear(512 → 197)` output head
- Data augmentation: `RandomHorizontalFlip`, ImageNet normalization
- `ConcatDataset` merges Stanford Cars + custom no-car images
- Tkinter GUI with green/red visual feedback for car detected / not detected
- Model saved to disk — subsequent runs skip training and load directly

![Car Classifier Demo](https://github.com/PeriStr/Machine-Learning-Projects/blob/f08c06708c25499bc1cd4eb1bcac7a77f3caa479/%CE%A3%CF%84%CE%B9%CE%B3%CE%BC%CE%B9%CF%8C%CF%84%CF%85%CF%80%CE%BF%20%CE%BF%CE%B8%CF%8C%CE%BD%CE%B7%CF%82%202026-03-20%20012815.png)

---

### 📊 MLP vs kNN — CIFAR-10 / CIFAR-100
> *Feature Engineering · Neural Networks · Classical ML · Comparative Study*

Comparative study of a neural network vs. a distance-based classifier on two benchmark datasets, using hand-crafted feature vectors instead of raw pixels.

- **Color Histograms**: 16 bins × 3 RGB channels → 48-dim feature vector
- **Edge Histograms**: Sobel gradient magnitudes → 16-dim feature vector
- **MLP**: Dense(256) → Dropout(0.5) → Dense(128) → Dense(n_classes), Adam + Early Stopping
- **kNN**: Euclidean distance, evaluated at k ∈ {1, 3, 5, 7, 11}
- 4 experiments: CIFAR-10 + CIFAR-100 × color + edge features
- Outputs: Confusion matrices, learning curves, classification reports per class

---

### 🏠 House Price Prediction — Linear Regression
> *Regression · Predictive Modeling*

CLI application that predicts California housing prices from economic and demographic features using a Linear Regression baseline model.

- Features: `median_income`, `total_rooms`, `housing_median_age`
- Two-mode CLI: predict a price, or compare a past price against the model's current estimate
- Instant training on startup — no separate training step needed

---

### 🚦 Smart Traffic Light Controller — Fuzzy Logic
> *Fuzzy Systems · Simulation · Control Theory*

Adaptive traffic signal controller that dynamically adjusts green light duration based on vehicle queue size and arrival rate — replacing fixed timers with human-like decision logic.

- **skfuzzy** control system with 3 input fuzzy sets per variable (low/medium/high, slow/normal/fast)
- Triangular membership functions (trimf) for all variables
- 4 IF-THEN rules covering standard and edge-case traffic conditions
- Multi-cycle simulation across N, S, E, W directions
- **Poisson distribution** for stochastic vehicle arrivals (realistic traffic modeling)
- Fallback default rule prevents system failure on unhandled input combinations

---

### 🎞️ Image Compression — Rate–Distortion Analysis
> *Deep Learning Codecs · Signal Processing · Benchmarking*

Systematic benchmarking of three state-of-the-art learned image codecs from **CompressAI** on the Kodak dataset, producing Rate–Distortion curves across 432 controlled experiments.

- **3 models**: `bmshj2018-factorized`, `mbt2018`, `cheng2020-attn`
- **6 quality levels** per model (1–6), evaluated on all 24 Kodak images
- Metrics per run: **bpp**, **PSNR**, **MS-SSIM**, encoding time, decoding time, peak GPU memory
- Fully automated: single `run.bat` executes compression + plot generation
- Output: `results/image_rd_kodak.csv` + `plots/rd_psnr.png` + `plots/rd_ms_ssim.png`

---

### 🌫️ Air Pollution Forecasting — LSTM Comparison
> *Time Series · Recurrent Neural Networks · Regression*

Benchmarking of five deep learning architectures for hourly PM2.5 air pollution forecasting on the Beijing dataset (2010–2014). All models are trained and evaluated identically for a fair comparison.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM | 265.6 | 191.7 | -1.02 |
| BiLSTM | 1017.7 | 1000.1 | -28.67 |
| **GRU** | **208.1** | **151.5** | **-0.24** |
| ConvLSTM | diverged | — | — |
| AttentionLSTM | 277.6 | 204.8 | -1.21 |

- Preprocessing: datetime index, One-Hot wind direction, MinMaxScaler (fit on train only)
- Sliding window input: `LOOK_BACK = 12` hours
- All models use Adam optimizer, MAPE loss, Early Stopping (patience=10)
- **GRU** outperforms all other architectures on this task

![Forecasting Output](https://github.com/PeriStr/Machine-Learning-Projects/blob/80d2cad3158a96d526a39499ce2074462eca7608/%CE%A3%CF%84%CE%B9%CE%B3%CE%BC%CE%B9%CF%8C%CF%84%CF%85%CF%80%CE%BF%20%CE%BF%CE%B8%CF%8C%CE%BD%CE%B7%CF%82%202026-03-26%20113849.png)

---

### 💳 Credit Card Fraud Detection — Autoencoder
> *Anomaly Detection · Semi-Supervised Learning · Unsupervised*

Semi-supervised fraud detection system trained **exclusively on normal transactions**. The autoencoder learns to reconstruct normal behavior — fraud is detected as any transaction it cannot reconstruct accurately.

- **Architecture**: 29 → 16 → 8 → 16 → 29 (symmetric hourglass with Dropout 0.2)
- Trained only on Class 0 (normal) — labels never seen during training
- Threshold at **90th percentile** of training reconstruction errors
- **~90% Recall** on fraud class — catches 9 out of 10 fraudulent transactions
- Rewritten from TF1 (session-based) to **TensorFlow 2.0** (Keras API)
- GPU-aware batch sizing: 2048 (GPU) / 256 (CPU)

---

## 🛠️ Technologies

| Category | Tools |
|---|---|
| **Languages** | Python 3.10 – 3.12 |
| **Deep Learning** | TensorFlow / Keras, PyTorch, CompressAI |
| **Classical ML** | Scikit-learn, XGBoost, mlxtend |
| **Computer Vision** | OpenCV, Torchvision, PIL |
| **NLP** | SpaCy |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Fuzzy Systems** | scikit-fuzzy |

---

## 🎯 Current Focus

Actively expanding into:
- **Natural Language Processing (NLP)** and **Large Language Models (LLMs)**
- **Conversational AI** systems
- **Big Data** engineering (Spark, Docker, Kubernetes)

---

## ⚙️ Setup

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow torch torchvision spacy scikit-fuzzy compressai xgboost mlxtend pytorch-msssim
python -m spacy download en_core_web_sm
```

> [!IMPORTANT]
> Most projects use **local dataset paths** hardcoded to the original machine. Before running any script, update the file paths in the code to match your own directory structure. Each project's README contains the exact lines that need to be changed.
>
> Contract
> E-mail : peri.oly123@gmail.com
> 
