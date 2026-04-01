# 🖼️ AI Image Captioning
### CNN + LSTM with ResNet50 & SpaCy

---

## 📌 Description

An end-to-end deep learning system that automatically generates natural language captions for images.
The model uses a **CNN Encoder** (ResNet50) to extract visual features and an **LSTM Decoder** to generate descriptive sentences word by word.

A **Tkinter GUI** is included to load any image and get a real-time caption prediction.

---

## 📂 Project Structure

```
project/
│
├── main.py                  # Full pipeline: training + GUI
├── best_caption_model.pth   # Saved model weights (generated after training)
├── captions.txt             # Dataset captions CSV (image, caption)
├── Images/                  # Folder containing all dataset images
└── README.md
```

---

## 🗄️ Dataset

This project uses the **Flickr8k** dataset.

| File | Description |
|---|---|
| `Images/` | Folder with ~8,000 photographs |
| `captions.txt` | CSV with columns `image` and `caption` (5 captions per image) |

Each image has 5 different human-written captions, giving the model diverse descriptions to learn from.

> Download: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

## ⚙️ Installation

```bash
pip install torch torchvision pillow pandas spacy
python -m spacy download en_core_web_sm
```

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────┐
│    EncoderCNN        │
│                     │
│  ResNet50 (frozen)  │  ← Pre-trained on ImageNet (Transfer Learning)
│  Linear(2048→256)   │  ← Custom projection layer
│  ReLU + Dropout     │
└────────┬────────────┘
         │  Image features (1×256)
         ▼
┌─────────────────────┐
│    DecoderLSTM       │
│                     │
│  Embedding(vocab→256)│  ← Word embeddings
│  [img | w1 | w2 ...] │  ← Image features prepended to word sequence
│  LSTM(256→512)      │  ← Sequence modeling
│  Linear(512→vocab)  │  ← Word prediction
└────────┬────────────┘
         │
         ▼
  Generated Caption
  "a dog running on grass"
```

---

## 🧩 Components

### 1. Vocabulary
Builds a word-to-index mapping from all captions. Words appearing fewer than `freq_threshold` (default: 5) times are treated as `<UNK>`.

Special tokens:

| Token | Index | Meaning |
|---|---|---|
| `<PAD>` | 0 | Padding (fills short sentences) |
| `<SOS>` | 1 | Start of sentence |
| `<EOS>` | 2 | End of sentence |
| `<UNK>` | 3 | Unknown word |

---

### 2. FlickrDataset
Custom PyTorch `Dataset` that:
- Loads images from disk and applies transforms (Resize, ToTensor, Normalize)
- Converts captions to numericalized sequences with `<SOS>` and `<EOS>` tokens

---

### 3. CapsCollate
Custom collate function for the DataLoader. Pads all captions in a batch to the same length using `<PAD>` tokens so they can be stacked into a single tensor.

---

### 4. EncoderCNN
- Loads **ResNet50** with pre-trained ImageNet weights
- **Freezes all ResNet parameters** (no retraining — Transfer Learning)
- Removes the final classification layer
- Adds a `Linear(2048 → embed_size)` + ReLU + Dropout(0.5)

---

### 5. DecoderLSTM
- **Embedding layer**: maps word indices to 256-dim vectors
- **Concatenation**: image features are prepended to the word sequence
- **LSTM**: processes the combined sequence to model temporal dependencies
- **Linear layer**: projects hidden state to vocabulary size for next-word prediction
- At inference time uses **Greedy Search** (picks the highest-probability word at each step)

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `embed_size` | 256 | Embedding & CNN output dimension |
| `hidden_size` | 512 | LSTM hidden state size |
| `num_layers` | 1 | Number of LSTM layers |
| `batch_size` | 64 | Images per training step |
| `learning_rate` | 0.0003 | Adam optimizer learning rate |
| `epochs` | 50 | Maximum training epochs |
| `freq_threshold` | 5 | Minimum word frequency for vocabulary |
| `max_caption_length` | 50 | Max words generated at inference |
| `dropout` | 0.5 | Dropout rate (encoder & decoder) |
| `grad_clip` | 5 | Gradient clipping threshold |
| `num_workers` | 4 | CPU workers for data loading |

---

## 🔄 Training Pipeline

```
For each epoch (up to 50):
│
├── TRAINING LOOP
│   ├── Forward pass  → model predicts next word at each position
│   ├── Loss          → CrossEntropyLoss (ignores <PAD> tokens)
│   ├── Accuracy      → token-level (excluding padding)
│   ├── Backward pass → gradients via backpropagation
│   ├── Grad clipping → clip_grad_norm (max=5) to stabilize LSTM
│   └── Adam step     → update weights
│
└── VALIDATION LOOP
    ├── Compute val loss and val accuracy (no gradient updates)
    └── Save model if val loss improved → best_caption_model.pth
```

**Loss function:** `CrossEntropyLoss` with `ignore_index=pad_idx`
— the model is not penalized for wrong predictions at padding positions.

**Gradient Clipping:** prevents the exploding gradient problem common in LSTM training.

**Model saving:** only the best checkpoint (lowest validation loss) is kept.

---

## 🖥️ GUI

After training (or loading a saved model), a **Tkinter window** opens:

1. Click **"Επιλογή Εικόνας"** to browse and select any image
2. The image is displayed in the window
3. The model generates and displays the caption in real time

---

## ▶️ How to Run

```bash
# First run: trains the model from scratch (saves best_caption_model.pth)
python main.py

# Subsequent runs: loads saved weights and opens the GUI directly
python main.py
```

> If `best_caption_model.pth` exists in the directory, training is skipped automatically.

---

## 📊 Expected Results

| Metric | Typical Value |
|---|---|
| Training Loss | ~1.5 – 2.5 (after 50 epochs) |
| Validation Loss | ~2.0 – 3.0 |
| Token-level Accuracy | ~50 – 65% |

Results vary depending on hardware, batch size, and number of epochs.

---

## 📦 Requirements

```
torch
torchvision
pillow
pandas
spacy
en_core_web_sm   # python -m spacy download en_core_web_sm
tkinter          # included with standard Python
```
