# 🚗 Stanford Cars Classifier
### CNN Image Classification with ResNet18 & Transfer Learning

---

## 📌 Description

A deep learning system that classifies car images into **197 categories** — 196 car models from the Stanford Cars dataset plus an extra **"Not a Car"** class for images that don't contain a vehicle.

The model uses **ResNet18** with pre-trained ImageNet weights (Transfer Learning) and includes a **Tkinter GUI** for real-time image classification.

---

## 📂 Project Structure

```
project/
│
├── main.py                      # Full pipeline: training + GUI
├── stanford_cars_model.pth      # Saved model weights (generated after training)
├── car_devkit/
│   └── devkit/
│       ├── cars_train_annos.mat # Training annotations (image names + labels)
│       └── cars_meta.mat        # Class names (car make & model strings)
├── cars_train/
│   └── cars_train/              # Training images
├── no_cars/
│   └── no_cars/*.jpg            # Images without cars (custom "No Car" class)
└── README.md
```

---

## 🗄️ Dataset

**Stanford Cars Dataset** — 196 car categories with fine-grained labels (make, model, year).

| Split | Images | Classes |
|---|---|---|
| Training (80%) | ~6,500 | 197 (196 cars + No Car) |
| Validation (20%) | ~1,600 | 197 |

> Download: [Stanford Cars on Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)

**Custom "No Car" class (label 196):** A folder of non-car images is added to the dataset so the model can detect when no vehicle is present, avoiding false classifications.

---

## ⚙️ Installation

```bash
pip install torch torchvision pillow scipy
```

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────┐
│        ResNet18          │
│                         │
│  Pre-trained on ImageNet │  ← Transfer Learning
│  (all layers kept)      │
│                         │
│  fc: Linear(512 → 197)  │  ← Replaced final layer for 197 classes
└────────────┬────────────┘
             │
             ▼
     Softmax Probabilities
     (197 car models + No Car)
             │
             ▼
    Predicted Class Label
```

**Why ResNet18?**
- Lighter and faster than ResNet50 — good for CPU training
- Residual connections prevent vanishing gradients
- Pre-trained weights provide strong visual feature extraction from day one

---

## 🧩 Components

### 1. StanfordCars Dataset
Custom PyTorch `Dataset` that reads image filenames and labels from the `.mat` annotations file using `scipy.io.loadmat`. Labels are shifted from 1-indexed to 0-indexed (`label - 1`) to match PyTorch conventions.

### 2. SimpleDataset
Minimal dataset class for the "No Car" images. All images in this set receive **label 196**.

### 3. ConcatDataset
Both datasets are merged into a single dataset with `torch.utils.data.ConcatDataset`, then split 80/20 into training and validation subsets.

---

## 🔄 Data Augmentation & Preprocessing

| Transform | Purpose |
|---|---|
| `Resize(224, 224)` | Required input size for ResNet |
| `RandomHorizontalFlip()` | Augmentation — improves generalization |
| `ToTensor()` | Convert PIL image to PyTorch tensor |
| `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` | ImageNet mean/std normalization |

> Normalization values match the ImageNet statistics used during ResNet pre-training.

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Images per training step |
| `learning_rate` | 0.0001 | Adam optimizer learning rate |
| `epochs` | 5 | Training epochs |
| `num_classes` | 197 | 196 car models + 1 No Car |
| `train/val split` | 80/20 | Dataset split ratio |
| `optimizer` | Adam | Adaptive learning rate optimizer |
| `loss function` | CrossEntropyLoss | Standard multi-class classification loss |
| `device` | CPU | GPU support detected but runs on CPU |

---

## 🔄 Training Pipeline

```
For each epoch (5 total):
│
├── TRAINING LOOP
│   ├── Forward pass   → model predicts class for each image
│   ├── Loss           → CrossEntropyLoss(predictions, labels)
│   ├── Backward pass  → backpropagation
│   ├── Optimizer step → Adam updates weights
│   └── Print loss every 20 steps + epoch accuracy
│
└── VALIDATION LOOP
    ├── No gradient updates (torch.no_grad())
    ├── Compute accuracy on unseen data
    └── Print validation accuracy
│
└── Save model → stanford_cars_model.pth
```

If `stanford_cars_model.pth` already exists, **training is skipped** and weights are loaded directly.

---

## 🖥️ GUI

After training (or loading a saved model), a **Tkinter window** opens:

1. Click **"ΕΠΙΛΟΓΗ ΦΩΤΟΓΡΑΦΙΑΣ"** to browse and select any image
2. The image is displayed in the window
3. The model returns either:
   - ✅ **Car detected** → displays make, model, and year in green
   - ❌ **No car detected** → displays warning message in red

Class names are loaded from `cars_meta.mat` for human-readable output (e.g. `"2012 BMW M3 Coupe"`).

---

## ▶️ How to Run

```bash
# First run: trains the model and saves stanford_cars_model.pth
python main.py

# Subsequent runs: loads saved weights and opens the GUI directly
python main.py
```

---

## 📊 Expected Results

| Metric | Typical Value (5 epochs, CPU) |
|---|---|
| Training Accuracy | ~40 – 60% |
| Validation Accuracy | ~35 – 55% |

> Accuracy improves significantly with more epochs and GPU training. The task is challenging due to 197 fine-grained classes.

---

## 📦 Requirements

```
torch
torchvision
pillow
scipy
tkinter    # included with standard Python
```
