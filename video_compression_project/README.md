# Video Compression Project – Image Rate–Distortion Evaluation

Το project υλοποιεί ένα πλήρες πείραμα συμπίεσης εικόνας χρησιμοποιώντας τρία προεκπαιδευμένα deep-learning από το CompressAI.  
Για κάθε εικόνα του Kodak dataset υπολογίζονται οι μετρικές bpp, PSNR, MS-SSIM, χρόνοι συμπίεσης/αποσυμπίεσης και χρήση GPU.  
Τα αποτελέσματα αποθηκεύονται σε CSV και στη συνέχεια παράγονται οι καμπύλες Rate–Distortion.

Το project εκτελείται με μία μόνο εντολή.
## cmd /c "c:\Users\perio\Desktop\VideoCompress\video_compression_project\run.bat"


##  Τι κάνει η εντολή

Η εντολή run.bat  εκτελεί 2 πράγματα:

### 1️ scripts/run_image.py
- φορτώνει το Kodak dataset  
- κάνει center crop 256×256  
- τρέχει τα 3 μοντέλα:
  - `bmshj2018-factorized`
  - `mbt2018`
  - `cheng2020-attn`
- για 6 επίπεδα ποιότητας (1–6)  
- μετρά:
  - bits-per-pixel  
  - PSNR  
  - MS-SSIM  
  - encoding time  
  - decoding time  
  - GPU peak memory  
- αποθηκεύει τα αποτελέσματα στο results/image_rd_kodak
# Το csv περιέχει τις ακόλουθες τιμές 
# img, model, level, bpp, psnr, ms_ssim, enc_time, dec_time, mem


### 2 `scripts/run_plots.py`
Διαβάζει το CSV και παράγει αυτόματα 2 γραφήματα:

- `plots/rd_psnr.png` BPP vs PSNR  
- `plots/rd_ms_ssim.png`  BPP vs MS-SSIM  

##  Αναπαραγωγισιμότητα
- Σταθερό quality ladder: `[1,2,3,4,5,6]`  
- Ίδιο crop & preprocessing για όλα τα μοντέλα  
- Σταθερό dataset  
- Ένα ενιαίο script εκτέλεσης  
- Πλήρης καταγραφή σε CSV  

## Δομή Φακέλων Project

video_compression_project/
│── data/kodak/                      # Εικόνες Kodak dataset
│── results/image_rd_kodak.csv       # αποτελέσματα RD
│── plots/
│    ├── rd_psnr.png
│    └── rd_ms_ssim.png
│── scripts/
│    ├── run_image.py
│    └── run_plots.py
│── run.bat                          # Ενιαία εντολή Windows
│── README.md                        # Το παρόν αρχείο
│── report.pdf                       # Η αναφορά




## Περιβάλλον
- Python ή 3.12.6 
- PyTorch  
- CompressAI  

Βιβλιοθήκες που χρησιμοποιήθηκαν:
numpy  1.26.4
pandas 2.3.3
pip    25.3
torch  2.6.0
compressai  1.2.8
matplotlib  3.10.7
pytorch-msssim  1.0.0
pillow 10.4.0

## Δεδομένα
Εικόνες από:
https://r0k.us/graphics/kodak/

## Γνωστά Σφάλματα
##### ΠΡΟΣΟΧΗ ΤΟ ΠΡΟΓΡΑΜΜΑ ΤΡΕΧΕΙ ΜΟΝΟ ΣΕ WINDOWS #####
Αυτο το πρόγραμμμα εχει σχεδιαστεί για να αντλεί τα δεδομένα απο φακέλους localy οπότε δεν θα έχει επιτυχία αν εκτελεστεί σε διαφορετικό σύστημα μονο αν αλαχτούν τα paths στα 2 προγραμαματα και στο run.bat επίσης χρειαζεται και εγκατάσταση του kodak dataset localy απο το link παραπανω

## Οδηγίες εκτέλεσης
Σε περιβάλλον windows εκτελώντας το αρχείο run.bat θα εκτελεστεί αυτόματα ο κώδικας συμπίεσης και ο κώδικας δημιουργίας plots ταυτόχρονα
εκτελώντας την παρακάτω εντολή 

## cmd /c "c:\Users\ΟΝΟΜΑ_ΧΡΗΣΤΗ\Desktop\ΦΑΚΕΛΟΣ_ΕΓΚΑΤΆΣΤΑΣΗΣ\video_compression_project\run.bat"

ΟΝΟΜΑ_ΧΡΗΣΤΗ: το username του συστήματός σας
ΦΑΚΕΛΟΣ: το directory όπου βρίσκεται το project

