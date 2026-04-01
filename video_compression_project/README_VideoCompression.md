# 🎞️ Image Rate–Distortion Evaluation
### Deep Learning Image Compression Benchmarking with CompressAI

---

## 📌 Description

A complete image compression experiment that benchmarks **three pre-trained deep learning codecs** from the [CompressAI](https://github.com/InterDigitalInc/CompressAI) library against the standard **Kodak dataset**.

For every image × model × quality level combination, the pipeline measures:
- **bpp** — bits per pixel (compression efficiency)
- **PSNR** — Peak Signal-to-Noise Ratio (reconstruction quality)
- **MS-SSIM** — Multi-Scale Structural Similarity (perceptual quality)
- **Encoding / Decoding time**
- **Peak GPU memory usage**

Results are saved to CSV and two **Rate–Distortion curves** are generated automatically.

---

## 📂 Project Structure

```
video_compression_project/
│
├── data/
│   └── kodak/                    # Kodak dataset images (downloaded separately)
│
├── results/
│   └── image_rd_kodak.csv        # All RD measurements
│
├── plots/
│   ├── rd_psnr.png               # BPP vs PSNR curve
│   └── rd_ms_ssim.png            # BPP vs MS-SSIM curve
│
├── scripts/
│   ├── run_image.py              # Compression + measurement pipeline
│   └── run_plots.py              # RD curve generator
│
├── run.bat                       # Single-command launcher (Windows)
├── report.pdf                    # Project report
└── README.md
```

---

## 🗄️ Dataset

**Kodak Lossless True Color Image Suite** — 24 high-quality PNG images widely used as a standard benchmark for image compression research.

> Download: [https://r0k.us/graphics/kodak/](https://r0k.us/graphics/kodak/)

Place all downloaded images in `data/kodak/` before running.

All images are **center-cropped to 256×256** before processing — consistent across all models.

---

## ⚙️ Installation

```bash
pip install torch compressai numpy pandas matplotlib pytorch-msssim pillow
```

**Tested library versions:**

| Library | Version |
|---|---|
| Python | 3.12.6 |
| torch | 2.6.0 |
| compressai | 1.2.8 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| pytorch-msssim | 1.0.0 |
| pillow | 10.4.0 |

---

## 🏗️ Models Evaluated

| Model | Description |
|---|---|
| `bmshj2018-factorized` | Ballé et al. 2018 — factorized entropy model |
| `mbt2018` | Minnen, Ballé & Toderici 2018 — mean-scale hyperprior |
| `cheng2020-attn` | Cheng et al. 2020 — attention-based, highest quality |

Each model is evaluated at **6 quality levels (1–6)**, where 1 = highest compression / lowest quality and 6 = lowest compression / highest quality.

---

## 🔄 Pipeline

```
data/kodak/ (24 images)
        │
        ▼
Center crop 256×256
        │
        ▼
For each image × model × quality level (24 × 3 × 6 = 432 runs):
   ├── Encode image  → measure bpp, encoding time, peak GPU memory
   ├── Decode image  → measure decoding time
   └── Compare original vs reconstruction → PSNR, MS-SSIM
        │
        ▼
Save all results → results/image_rd_kodak.csv
        │
        ▼
run_plots.py reads CSV → generates:
   ├── plots/rd_psnr.png      (BPP vs PSNR)
   └── plots/rd_ms_ssim.png   (BPP vs MS-SSIM)
```

---

## 📊 CSV Output Format

`results/image_rd_kodak.csv` contains one row per run:

| Column | Description |
|---|---|
| `img` | Image filename |
| `model` | Model name |
| `level` | Quality level (1–6) |
| `bpp` | Bits per pixel |
| `psnr` | PSNR in dB |
| `ms_ssim` | MS-SSIM score (0–1) |
| `enc_time` | Encoding time (seconds) |
| `dec_time` | Decoding time (seconds) |
| `mem` | Peak GPU memory (MB) |

---

## ▶️ How to Run

> ⚠️ **Windows only.** See the Known Issues section for details.

Open **Command Prompt** and run:

```bat
cmd /c "c:\Users\YOUR_USERNAME\Desktop\YOUR_FOLDER\video_compression_project\run.bat"
```

Replace `YOUR_USERNAME` with your Windows username and `YOUR_FOLDER` with the directory where the project is installed.

`run.bat` executes both scripts sequentially:
1. `scripts/run_image.py` — runs all compression experiments and saves the CSV
2. `scripts/run_plots.py` — reads the CSV and generates both RD curve plots

---

## ✅ Reproducibility

Every design decision that affects reproducibility is kept constant:

| Factor | How it's controlled |
|---|---|
| Quality ladder | Fixed: `[1, 2, 3, 4, 5, 6]` for all models |
| Image preprocessing | Identical center crop 256×256 for all models |
| Dataset | Fixed Kodak set (24 images) |
| Execution | Single `run.bat` script — no manual steps |
| Results | Full CSV log — every run is recorded |

---

## ⚠️ Known Issues

**This project runs on Windows only.**

The scripts use hardcoded local paths. To run on a different machine or OS:

1. Open `scripts/run_image.py` and update the paths to `data/kodak/` and `results/`
2. Open `scripts/run_plots.py` and update the path to `results/image_rd_kodak.csv`
3. Update `run.bat` with the new script paths

**The Kodak dataset must be downloaded and placed locally** in `data/kodak/` before running — the scripts do not download it automatically. Download from: [https://r0k.us/graphics/kodak/](https://r0k.us/graphics/kodak/)

---

## 📦 Requirements

```
torch
compressai
numpy
pandas
matplotlib
pytorch-msssim
pillow
```
