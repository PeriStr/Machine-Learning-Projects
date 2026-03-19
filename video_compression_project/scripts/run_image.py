import os
import csv
import time
import math
from PIL import Image
import torch
import numpy as np

from compressai.zoo import bmshj2018_factorized, mbt2018, cheng2020_attn
from pytorch_msssim import ms_ssim


# ============================================
# ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΟΣ
# Εδώ ορίζουμε dataset, output CSV και τα μοντέλα
# ============================================

DATASET_DIR = "C:/Users/perio/Desktop/VideoCompress/video_compression_project/data/kodak"
OUTPUT_CSV = "C:/Users/perio/Desktop/VideoCompress/video_compression_project/results/image_rd_kodak.csv"

# Τα τρία μοντέλα του CompressAI που συγκρίνουμε
MODELS = [
    "bmshj2018-factorized",
    "mbt2018",
    "cheng2020-attn"
]

# Ladder ποιότητας (1–6)
QUALITY_LADDER = [1, 2, 3, 4, 5, 6]

CROP_SIZE = 256   # Κάνουμε όλα τα crops ίδιο μέγεθος για δίκαιη σύγκριση


# ============================================
# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ============================================

def center_crop(img, size):
    # Κόβει την εικόνα στο κέντρο για να έχουν όλες την ίδια ανάλυση
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def image_to_tensor(img):
    # Μετατρέπει την εικόνα σε tensor PyTorch [0,1]
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)


def compute_bpp(bitstream_len_bits, H, W):
    # Υπολογισμός BPP = bits ανά pixel
    return bitstream_len_bits / (H * W)


def compute_psnr(x, x_hat):
    # Κλασικός υπολογισμός PSNR από MSE
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


# ============================================
# MAIN ΠΡΟΓΡΑΜΜΑ
# ============================================

def main():
    # Φάκελος για αποθήκευση αποτελεσμάτων
    os.makedirs("results", exist_ok=True)

    # Φορτώνουμε όλες τις PNG εικόνες από το Kodak
    images = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".png"))])

    # Ανοίγουμε το CSV και γράφουμε το header
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img", "model", "level", "bpp", "psnr", "ms_ssim", "enc_time", "dec_time", "mem"])

        # Επεξεργασία εικόνων μία-μία
        for idx, img_name in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] Processing {img_name}")

            # Φορτώνουμε εικόνα και κάνουμε crop
            img = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")
            img = center_crop(img, CROP_SIZE)

            # Tensor μετατροπή
            x = image_to_tensor(img)
            if torch.cuda.is_available():
                x = x.cuda()

            _, _, H, W = x.shape  # κρατάμε το size για BPP

            # Τρέχουμε όλα τα μοντέλα
            for model_name in MODELS:
                print(f"  Model: {model_name}")

                # Για κάθε επίπεδο ποιότητας
                for q in QUALITY_LADDER:
                    print(f"    Quality level: {q}")

                    # ============================
                    # Φόρτωση σωστού μοντέλου
                    # ============================
                    if model_name == "bmshj2018-factorized":
                        model = bmshj2018_factorized(quality=q, pretrained=True).eval()

                    elif model_name == "mbt2018":
                        model = mbt2018(quality=q, pretrained=True).eval()

                    elif model_name == "cheng2020-attn":
                        model = cheng2020_attn(quality=q, pretrained=True).eval()

                    if torch.cuda.is_available():
                        model = model.cuda()

                    # ============================
                    # ΚΩΔΙΚΟΠΟΙΗΣΗ (ENCODE)
                    # ============================
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t0 = time.time()
                    out_enc = model.compress(x)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    enc_time = time.time() - t0

                    # Υπολογισμός bitstream length
                    bitstream_len_bits = sum(len(v) for v in out_enc["strings"][0]) * 8
                    bpp = compute_bpp(bitstream_len_bits, H, W)

                    # ============================
                    # ΑΠΟΚΩΔΙΚΟΠΟΙΗΣΗ (DECODE)
                    # ============================
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t0 = time.time()
                    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    dec_time = time.time() - t0

                    # Ανακατασκευή εικόνας
                    x_hat = out_dec["x_hat"].clamp(0, 1)

                    # ============================
                    # ΜΕΤΡΙΚΕΣ ΠΟΙΟΤΗΤΑΣ
                    # ============================
                    psnr = compute_psnr(x, x_hat)
                    ms_ssim_value = ms_ssim(x_hat, x, data_range=1.0).item()

                    # ============================
                    # GPU MEMORY METRIC
                    # ============================
                    mem = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # ============================
                    # Γράφουμε τη γραμμή στο CSV
                    # ============================
                    writer.writerow([
                        img_name, model_name, q, bpp, psnr, ms_ssim_value,
                        enc_time, dec_time, mem
                    ])

    print("\nDone! CSV saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
