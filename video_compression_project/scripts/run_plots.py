import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
#  Ρυθμίσεις για paths και φακέλους
# ============================================================

CSV_PATH = "C:/Users/perio/Desktop/VideoCompress/video_compression_project/results/image_rd_kodak.csv"
# Το αρχείο CSV με όλα τα αποτελέσματα από το run_image.py

PLOT_DIR = "C:/Users/perio/Desktop/VideoCompress/video_compression_project/plots"
# Φάκελος όπου θα σωθούν τα διαγράμματα

os.makedirs(PLOT_DIR, exist_ok=True)
# Αν δεν υπάρχει ο φάκελος plots, τον δημιουργεί


# ============================================================
# LOAD CSV – Φόρτωση δεδομένων
# ============================================================

print("Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)  
# Διαβάζουμε το CSV σαν DataFrame (πίνακα)


# MODEL LIST – βρίσκουμε τα μοντέλα που υπάρχουν μέσα στο CSV
models = df["model"].unique()
print("Models found:", models)


# ============================================================
# PLOTS – Συνάρτηση που φτιάχνει τα RD curves
# ============================================================

def plot_rd(metric_str, ylabel):
    plt.figure(figsize=(8, 6))  
    # Νέο γράφημα για κάθε metric

    for model in models:
        df_m = df[df["model"] == model]
        # Κρατάμε μόνο τις γραμμές του συγκεκριμένου μοντέλου

        # Average across images – μέσος όρος σε όλες τις εικόνες
        grouped = df_m.groupby("level").agg({
            "bpp": "mean",       # μέσο BPP
            metric_str: "mean"   # μέσο PSNR ή MS-SSIM
        }).reset_index()

        plt.plot(
            grouped["bpp"],              # άξονας Χ
            grouped[metric_str],         # άξονας Υ
            marker="o",                  # κυκλάκι ανά σημείο
            label=model                  # όνομα μοντέλου στο legend
        )

    plt.xlabel("Bits per Pixel (bpp)")
    plt.ylabel(ylabel)
    plt.title(f"Rate–Distortion Curve ({ylabel})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Αποθήκευση γραφήματος ως PNG
    out_path = os.path.join(PLOT_DIR, f"rd_{metric_str}.png")
    plt.savefig(out_path, dpi=300)
    print("Saved:", out_path)

    plt.close()  # κλείνουμε το figure για να μην συσσωρεύονται στη μνήμη


# ============================================================
# GENERATE PLOTS – Δημιουργία των δύο γραφημάτων
# ============================================================

plot_rd("psnr", "PSNR (dB)")      # Πρώτο RD plot
plot_rd("ms_ssim", "MS-SSIM")     # Δεύτερο RD plot

print("\nAll plots generated successfully!")
