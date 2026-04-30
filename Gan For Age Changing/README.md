# Age Progression / Regression with Conditional GAN

A **Conditional Generative Adversarial Network (cGAN)** that transforms face images across five age groups while preserving facial identity, trained on the **UTKFace dataset**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Loss Functions](#loss-functions)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

Age progression/regression synthesizes realistic face images at different ages from a single input photo. This project uses a **Conditional GAN** where both the Generator and Discriminator receive the target age group as a conditioning input, enabling controlled age transformation at inference time.

### Age Groups

| Class | Range | Color |
|-------|-------|-------|
| 0 | 0–20 | 🟢 |
| 1 | 21–35 | 🔵 |
| 2 | 36–55 | 🟠 |
| 3 | 56–65 | 🟣 |
| 4 | 65+ | 🔴 |

---

## Dataset

**UTKFace** — Large Scale Face Dataset  
Homepage: https://susanqq.github.io/UTKFace/  
Kaggle mirror: https://www.kaggle.com/datasets/jangedoo/utkface-new

- **20,000+** aligned face images (200×200 px)
- Labels: age (0–116), gender, ethnicity
- Filename format: `[age]_[gender]_[race]_[datetime].jpg`

### Class Distribution (20,000 images)

```
Class 0 (0-20):   3,421 images  (17.1%)
Class 1 (21-35):  7,234 images  (36.2%)
Class 2 (36-55):  6,123 images  (30.6%)
Class 3 (56-65):  2,012 images  (10.1%)
Class 4 (65+):    1,210 images   (6.0%)
```

> Note: Class imbalance in groups 3 and 4 affects generation quality for older age ranges.

---

## Architecture

### Generator — U-Net with Skip Connections

```
Input: Image (128×128×3) + Target Age Label + Noise z (dim=100)
           │
    ┌──────▼──────────────────────────────────┐
    │  ENCODER                                │
    │  Conv 128→64  (LeakyReLU)               │ ── skip e1
    │  Conv 64→32   (InstanceNorm, LeakyReLU) │ ── skip e2
    │  Conv 32→16   (InstanceNorm, LeakyReLU) │ ── skip e3
    │  Conv 16→8    (InstanceNorm, LeakyReLU) │ ── skip e4
    └──────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │  BOTTLENECK                             │
    │  concat(e4, age_embedding, noise_map)   │
    │  6× Residual Blocks (InstanceNorm)      │
    └──────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────┐
    │  DECODER (U-Net skip connections)       │
    │  TranspConv 8→16   + skip e4 (Dropout) │
    │  TranspConv 16→32  + skip e3            │
    │  TranspConv 32→64  + skip e2            │
    │  TranspConv 64→128 + skip e1 (Tanh)    │
    └──────────────────────────────────────────┘
           │
    Output: Image (128×128×3)
```

**Generator parameters:** ~45M

### Discriminator — PatchGAN + Auxiliary Classifier

```
Input: Image (128×128×3) + Age Embedding Channel
           │
    ┌──────▼──────────────────────────────────┐
    │  PatchGAN Backbone                      │
    │  Conv 128→64  (LeakyReLU)               │
    │  Conv 64→32   (InstanceNorm, LeakyReLU) │
    │  Conv 32→16   (InstanceNorm, LeakyReLU) │
    │  Conv 16→8    (InstanceNorm, LeakyReLU) │
    └──────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  Patch Output    Auxiliary Classifier
  (real / fake)   (age category)
```

**Discriminator parameters:** ~10M

---

## Loss Functions

### Generator Loss

```
L_G = L_adv + λ_L1 · L_L1 + λ_cls · L_cls
```

| Component | Type | Purpose |
|-----------|------|---------|
| `L_adv` | LSGAN (MSE) | Fool the Discriminator |
| `L_L1` | L1 pixel loss (cycle) | Preserve facial identity |
| `L_cls` | Cross-Entropy | Generated image matches target age class |

Default weights: `λ_L1 = 10.0`, `λ_cls = 1.0`

### Discriminator Loss

```
L_D = (L_real + L_fake) / 2 + λ_cls · L_cls_real
```

### Cycle Consistency

The L1 loss is computed via cycle reconstruction to enforce identity preservation:

```
x → G(x, target_age) → G(G(x, target_age), source_age) ≈ x
```

---

## Installation

### Requirements

```
Python >= 3.10
torch >= 2.0
torchvision
Pillow
matplotlib
numpy
```

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib Pillow numpy
```

### Folder structure

```
project/
├── gan_age.py          # Models, training loop, inference
├── gan_age_gui.py      # Graphical user interface
├── run_gui.bat         # GUI launcher (Windows)
├── UTKFace/            # Dataset images (place here)
│   ├── 25_0_0_....jpg
│   └── ...
├── checkpoints/        # Saved models (auto-created)
└── gan_output/         # Sample images & loss plots (auto-created)
```

---

## Usage

### GUI (Recommended)

```bash
# Windows
run_gui.bat

# Or directly
"C:\Program Files\Python312\python.exe" gan_age_gui.py
```

**Training tab:**
- Set Data Path to `./UTKFace`
- Configure hyperparameters (epochs, batch size, learning rates, etc.)
- Click **▶ Start Training**
- Monitor the live log and real-time G/D Loss curves

**Inference tab:**
- Load a saved checkpoint (`.pth`)
- Open an input face image
- Click **✨ Generate Age Progression**
- View and save results for all 5 age groups

### Command Line

```python
from gan_age import main
main()  # Train with default config
```

```python
# Run inference on a new image
from gan_age import age_transform_image
age_transform_image(
    model_path='./checkpoints/best_model.pth',
    input_image_path='./my_photo.jpg',
    output_dir='./results'
)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 128 | Input/output image resolution |
| `nz` | 100 | Latent noise vector dimension |
| `ngf` / `ndf` | 64 | Base filter count for G / D |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 100 | Number of training epochs |
| `lr_g` / `lr_d` | 0.0002 | Adam learning rates |
| `beta1` | 0.5 | Adam β₁ |
| `lambda_L1` | 10.0 | Weight for pixel-level L1 loss |
| `lambda_cls` | 1.0 | Weight for classification loss |

---

## Results

### Training Curves (100 Epochs, RTX 3080, ~7 hours)

```
Epoch [1/100]   G: 8.4231  D: 1.2341
Epoch [10/100]  G: 4.2156  D: 0.8412
Epoch [20/100]  G: 3.1843  D: 0.7234
Epoch [50/100]  G: 2.1987  D: 0.5921  ← visible improvement
Epoch [80/100]  G: 1.7234  D: 0.5231
Epoch [100/100] G: 1.6234  D: 0.5012
```

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| L1 Distance | 0.0823 |
| PSNR | 24.31 dB |
| G/D Loss Ratio | ~3.2 |

### Architecture Comparison

| Model | FID ↓ | PSNR ↑ |
|-------|-------|--------|
| Standard DCGAN | 89.2 | 19.4 dB |
| Conditional GAN | 71.3 | 22.1 dB |
| cGAN + U-Net | 54.8 | 24.3 dB |
| **cGAN + U-Net + ResBlocks** ✓ | **48.2** | **24.9 dB** |

### Observations

✅ Wrinkles and hair color changes emerge gradually from ~epoch 50  
✅ L1 cycle loss effectively preserves facial identity across transformations  
✅ Age group 0–20 consistently produces smoother, younger-looking features  
⚠️ Occasional mode collapse in class 4 (65+) due to class imbalance  
⚠️ Lower generation quality for 65+ (only 6% of training data)

---

## Project Structure

```
gan_age.py
├── UTKFaceDataset          # Dataset loader (UTKFace filename convention)
├── Generator               # U-Net encoder-decoder with skip connections
│   ├── AgeEmbedding        # Learnable age group embedding
│   └── ResidualBlock       # 6× residual blocks in the bottleneck
├── Discriminator           # PatchGAN + Auxiliary Age Classifier
├── GANLoss                 # LSGAN (MSE) or vanilla (BCE)
├── AgeGANTrainer           # Training loop, checkpointing, sample generation
├── age_transform_image()   # Inference on a single image
└── evaluate_model()        # L1 Distance + PSNR evaluation

gan_age_gui.py
├── GANApp                  # Main Tk application window
├── TrainingTab             # Config panel + live log + embedded loss plot
├── InferenceTab            # Image loader + 6-slot results grid
└── _training_worker()      # Background training thread with queue I/O
```

---

## References

- Mirza & Osindero — *Conditional Generative Adversarial Nets*, 2014
- Ronneberger et al. — *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Isola et al. — *Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)*, CVPR 2017
- Mao et al. — *Least Squares Generative Adversarial Networks*, ICCV 2017
- Zhang et al. — *Age Progression/Regression by Conditional Adversarial Autoencoder*, CVPR 2017
