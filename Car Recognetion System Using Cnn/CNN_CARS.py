# Link: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset 
# Το παραπάνω link δείχνει το dataset που χρησιμοποιείται στο project.
# Το Stanford Cars dataset περιέχει εικόνες αυτοκινήτων από πολλές διαφορετικές μάρκες και μοντέλα.
# Κάθε εικόνα έχει label που αντιστοιχεί σε συγκεκριμένο μοντέλο αυτοκινήτου.
# Συνολικά υπάρχουν 196 διαφορετικές κατηγορίες αυτοκινήτων.


import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms


'''
Dataset

Το σύστημα χρησιμοποιεί το Stanford Cars Dataset, το οποίο περιέχει εικόνες από 196 διαφορετικά μοντέλα αυτοκινήτων.
Οι πληροφορίες για κάθε εικόνα (όνομα αρχείου και κατηγορία) αποθηκεύονται σε αρχείο .mat, το οποίο διαβάζεται με τη βιβλιοθήκη scipy. 
Για να μπορεί το dataset να χρησιμοποιηθεί στο PyTorch, δημιουργείται μια custom κλάση dataset που υλοποιεί τις μεθόδους __len__ και __getitem__. 
Με αυτόν τον τρόπο το πρόγραμμα μπορεί να διαβάζει εικόνες και labels δυναμικά κατά την εκπαίδευση.'''

# 1. Ορίζουμε πώς θα διαβάζεται το dataset

class StanfordCars(Dataset):
# Δημιουργείται custom dataset class που κληρονομεί από Dataset του PyTorch.
# Ο λόγος είναι ότι το dataset του Stanford Cars δεν είναι σε έτοιμη μορφή
# που να μπορεί να χρησιμοποιηθεί απευθείας από το PyTorch.

    def __init__(self, mat_path, img_dir, transform=None):
        # Ο constructor δέχεται:
        # mat_path το αρχείο annotations (.mat) που περιέχει labels
        # img_dir τον φάκελο που περιέχει τις εικόνες
        # transform preprocessing που θα εφαρμοστεί στις εικόνες

        data = loadmat(mat_path)
        # Φορτώνεται το αρχείο .mat που περιέχει metadata για το dataset.

        self.annotations = data['annotations'][0]
        # Το dataset περιέχει μια δομή annotations όπου κάθε στοιχείο
        # αντιστοιχεί σε μια εικόνα και περιέχει πληροφορίες:
        # - όνομα αρχείου
        # - class label

        self.img_dir = img_dir
        # Αποθηκεύεται το path του φακέλου εικόνων.

        self.transform = transform
        # Αποθηκεύεται το transform pipeline που θα εφαρμοστεί στις εικόνες.



    def __len__(self):
        # Η συνάρτηση __len__ πρέπει να υπάρχει σε κάθε PyTorch Dataset.
        # Επιστρέφει το συνολικό πλήθος δειγμάτων του dataset.
        return len(self.annotations)



    def __getitem__(self, idx):
        # Η συνάρτηση αυτή επιστρέφει ένα δείγμα του dataset.
        # Το idx είναι ο δείκτης του δείγματος.

        fname = self.annotations[idx]['fname'][0]
        # Παίρνουμε το όνομα αρχείου της εικόνας.

        label = int(self.annotations[idx]['class'][0][0]) - 1 
        # Παίρνουμε το class label.
        # Το -1 γίνεται γιατί στο dataset οι κλάσεις ξεκινούν από 1
        # ενώ στο PyTorch οι κλάσεις ξεκινούν από 0.

        img_path = os.path.join(self.img_dir, fname)
        # Δημιουργείται το πλήρες path της εικόνας.



        try:

            image = Image.open(img_path).convert("RGB")
            # Η εικόνα ανοίγει με την PIL.
            # convert("RGB") εξασφαλίζει ότι η εικόνα έχει 3 κανάλια χρώματος.

        except FileNotFoundError:

            first_img_path = os.path.join(self.img_dir, self.annotations[0]['fname'][0])
            # Αν για κάποιο λόγο λείπει η εικόνα
            # φορτώνεται μια άλλη εικόνα για να μην σπάσει το πρόγραμμα.

            image = Image.open(first_img_path).convert("RGB")



        if self.transform:
            # Αν έχει δοθεί transform pipeline
            # εφαρμόζεται preprocessing στην εικόνα.
            image = self.transform(image)

        return image, label
        # Επιστρέφεται tuple (image, label)
        # Το image είναι tensor και το label είναι integer.


'''Data Augmentation

Πριν χρησιμοποιηθούν οι εικόνες στο νευρωνικό δίκτυο εφαρμόζεται data augmentation. Η τεχνική αυτή αυξάνει την ποικιλία των δεδομένων μέσω μετασχηματισμών όπως 
resize και οριζόντια αναστροφή εικόνας.
Σκοπός είναι να γίνει το μοντέλο πιο ανθεκτικό σε διαφορετικές γωνίες ή παραλλαγές των αντικειμένων και να μειωθεί το overfitting.'''

'''Προεπεξεργασία Εικόνων

Οι εικόνες μετατρέπονται σε tensors και κανονικοποιούνται. 
Η κανονικοποίηση αλλάζει την κλίμακα των pixel ώστε να έχουν συγκεκριμένο μέσο όρο και τυπική απόκλιση. 
Αυτό βοηθά το μοντέλο να εκπαιδεύεται πιο σταθερά και πιο γρήγορα.'''


# 2. Προετοιμασία εικόνων (Augmentation)

data_transforms = transforms.Compose([
# Το Compose επιτρέπει να συνδυάζουμε πολλά transforms σε pipeline.

    transforms.Resize((224, 224)),
    # Όλες οι εικόνες μετατρέπονται σε μέγεθος 224x224.
    # Αυτό απαιτείται γιατί τα περισσότερα CNN μοντέλα
    # (όπως το ResNet) εκπαιδεύτηκαν με αυτό το input size.

    transforms.RandomHorizontalFlip(),
    # Τυχαία οριζόντια αναστροφή εικόνας.
    # Αυτό είναι τεχνική data augmentation.
    # Αυξάνει τεχνητά το dataset και βοηθά το μοντέλο να γενικεύει.

    transforms.ToTensor(),
    # Μετατρέπει την εικόνα από PIL format σε PyTorch tensor.

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Κανονικοποίηση pixel values.
    # Οι τιμές αυτές είναι τα mean και std του ImageNet dataset.
    # Χρησιμοποιούνται επειδή το ResNet έχει εκπαιδευτεί στο ImageNet.
])



# 3. Φόρτωση Stanford Cars

dataset = StanfordCars(

    mat_path='C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/CNN/car_devkit/devkit/cars_train_annos.mat', 
    # Path προς το αρχείο annotations.

    img_dir='C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/CNN/cars_train/cars_train/', 
    # Φάκελος με τις εικόνες.

    transform=data_transforms
    # Το preprocessing pipeline που ορίσαμε πριν.

)

'''Προσθήκη Κατηγορίας No Car

Εκτός από τις 196 κατηγορίες αυτοκινήτων προστίθεται και μια επιπλέον κατηγορία για εικόνες που δεν περιέχουν αυτοκίνητο. 
Για αυτό δημιουργείται ένα δεύτερο dataset με εικόνες χωρίς αυτοκίνητα. 
Οι εικόνες αυτές λαμβάνουν label 196, δημιουργώντας συνολικά 197 κατηγορίες. Αυτό επιτρέπει στο μοντέλο να αποφεύγει λανθασμένες ταξινομήσεις όταν δεν υπάρχει αυτοκίνητο στην εικόνα.'''

# 4. Φόρτωση No Cars

no_car_path = 'C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/CNN/no_cars/no_cars/*.jpg'
# Path που δείχνει σε εικόνες που δεν περιέχουν αυτοκίνητα.

no_car_paths = glob.glob(no_car_path)
# Η glob επιστρέφει λίστα με όλα τα αρχεία που ταιριάζουν στο pattern.



class SimpleDataset(Dataset):
# Δημιουργείται δεύτερο dataset μόνο για εικόνες που δεν είναι αυτοκίνητα.

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = Image.open(self.paths[idx]).convert("RGB")
        # Φόρτωση εικόνας.

        if self.transform: img = self.transform(img)
        # Εφαρμογή preprocessing.

        return img, 196 
        # Επιστρέφεται label 196.
        # Οι κλάσεις αυτοκινήτων είναι 0-195
        # Η 196 είναι η νέα κλάση "όχι αυτοκίνητο".



no_cars_ds = SimpleDataset(no_car_paths, transform=data_transforms)
# Δημιουργία dataset για τις εικόνες που δεν είναι αυτοκίνητα.


'''Training και Validation Dataset

Το dataset χωρίζεται σε δύο μέρη: training και validation. Το training set χρησιμοποιείται για την εκπαίδευση του μοντέλου, 
ενώ το validation set χρησιμοποιείται για την αξιολόγηση της απόδοσης του σε δεδομένα που δεν έχει δει. 
Αυτό βοηθά στον εντοπισμό τοOverfitting, όπου το μοντέλο απομνημονεύει τα δεδομένα εκπαίδευσης αλλά δεν γενικεύει καλά.'''


# 5. Ένωση και Χωρισμός

full_dataset = ConcatDataset([dataset, no_cars_ds])
# Συνενώνονται τα δύο datasets σε ένα.

train_size = int(0.8 * len(full_dataset))
# Το 80% του dataset χρησιμοποιείται για training.

val_size = len(full_dataset) - train_size
# Το υπόλοιπο 20% για validation.

train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
# Τυχαίος διαχωρισμός dataset.



train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
# DataLoader για training.
# batch_size=32 σημαίνει ότι το μοντέλο βλέπει 32 εικόνες κάθε φορά.
# shuffle=True σημαίνει ότι τα δεδομένα ανακατεύονται.

val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
# DataLoader για validation.



print(f"Stanford Cars: {len(dataset)} images detected.")
print(f"Extra Images (No Cars): {len(no_cars_ds)} images detected.")



# 6. Μοντέλο

print(" ANALYZING GPU CAPABILITIES...")

if torch.cuda.is_available():
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
# Ελέγχει αν υπάρχει διαθέσιμη GPU.



device = torch.device("cpu") 
# Ορίζεται ότι το training θα γίνει στην CPU.

'''Νευρωνικό Δίκτυο

Το μοντέλο που χρησιμοποιείται είναι το ResNet-18,
ένα συνελικτικό νευρωνικό δίκτυο που χρησιμοποιείται ευρέως στην αναγνώριση εικόνων. 
Το μοντέλο έχει εκπαιδευτεί αρχικά στο dataset ImageNet και στη συνέχεια προσαρμόζεται στο νέο dataset. 
Αυτή η τεχνική ονομάζεται Transfer Learning και επιτρέπει καλύτερη απόδοση με λιγότερα δεδομένα.'''


model = models.resnet18(weights='DEFAULT')
# Φόρτωση του ResNet18 pretrained μοντέλου.
# Αυτές βοηθούν στην εκπαίδευση πολύ βαθιών δικτύων.



model.fc = nn.Linear(model.fc.in_features, 197)
# Αλλάζουμε το τελευταίο layer.
# Το αρχικό ResNet έχει 1000 classes.
# Εδώ θέλουμε 197 classes (196 αυτοκίνητα + 1 no car).



model = model.to(device)
# Μεταφορά μοντέλου στην CPU.


'''Loss Function

Για τη μέτρηση του σφάλματος χρησιμοποιείται η συνάρτηση κόστους Cross Entropy Loss. 
Η συνάρτηση αυτή συγκρίνει την πρόβλεψη του μοντέλου με την πραγματική κατηγορία της εικόνας. 
Ο στόχος της εκπαίδευσης είναι η ελαχιστοποίηση της τιμής της loss.'''

criterion = nn.CrossEntropyLoss()
# Loss function για classification προβλήματα.


'''Optimizer

Για την ενημέρωση των βαρών χρησιμοποιείται ο optimizer Adam. 
Ο Adam είναι ένας αλγόριθμος βελτιστοποίησης που προσαρμόζει το learning rate κατά τη διάρκεια της εκπαίδευσης. 
Αυτό βοηθά το μοντέλο να συγκλίνει πιο γρήγορα και σταθερά.'''
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Adam optimizer.
# lr=0.0001 είναι το learning rate.



# ΕΛΕΓΧΟΣ ΥΠΑΡΞΗΣ ΜΟΝΤΕΛΟΥ Ή ΕΚΠΑΙΔΕΥΣΗ

MODEL_PATH = 'stanford_cars_model.pth'
# Path όπου αποθηκεύεται το εκπαιδευμένο μοντέλο.

'''Epochs

Η εκπαίδευση πραγματοποιείται σε επαναλαμβανόμενους κύκλους που ονομάζονται epochs. 
Ένα epoch σημαίνει ότι το μοντέλο έχει επεξεργαστεί όλα τα δεδομένα του training set μία φορά. 
Σε κάθε epoch το μοντέλο υπολογίζει την πρόβλεψη, την απώλεια και ενημερώνει τα βάρη του μέσω της διαδικασίας Backpropagation.'''

if os.path.exists(MODEL_PATH):

    print(f"\n Το εκπαιδευμένο μοντέλο '{MODEL_PATH}' βρέθηκε.")

    print(" Φόρτωση βαρών Έτοιμο για χρήση χωρίς training.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # Αν υπάρχει ήδη εκπαιδευμένο μοντέλο φορτώνονται τα weights.



else:

    print(f"\n Το αρχείο '{MODEL_PATH}' δεν βρέθηκε.")

    print(" Ξεκινάει η εκπαίδευση...")



    epochs = 5
    # Πλήθος περασμάτων από όλο το dataset.



    for epoch in range(epochs):

        model.train()
        # Το μοντέλο μπαίνει σε training.

        correct = 0
        total = 0



        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # Μηδενισμός gradients.

            outputs = model(images)
            # Forward pass.

            loss = criterion(outputs, labels)
            # Υπολογισμός loss.

            loss.backward()
            # Backpropagation.

            optimizer.step()
            # Ενημέρωση weights.



            _, predicted = outputs.max(1)
            # Επιλογή κλάσης με τη μεγαλύτερη πιθανότητα.

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()



            if (i + 1) % 20 == 0:

                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')



        print(f' End of Epoch {epoch+1} - Training Accuracy: {100 * correct / total:.2f}%')



        # Validation
        '''Validation
        Μετά από κάθε epoch γίνεται αξιολόγηση στο validation set.
        Κατά τη φάση αυτή το μοντέλο δεν ενημερώνει τα βάρη του αλλά απλώς υπολογίζει την ακρίβεια των προβλέψεων. 
        Αυτό δείχνει πόσο καλά μπορεί το μοντέλο να γενικεύσει σε νέα δεδομένα.'''
        model.eval()
        # Το μοντέλο μπαίνει σε evaluation mode.



        val_correct = 0
        val_total = 0



        with torch.no_grad():
        # Απενεργοποίηση gradient calculation.

            for images, labels in val_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, predicted = outputs.max(1)

                val_total += labels.size(0)

                val_correct += predicted.eq(labels).sum().item()



        print(f' Validation Accuracy: {100 * val_correct / val_total:.2f}%')



    torch.save(model.state_dict(), MODEL_PATH)
    # Αποθήκευση εκπαιδευμένου μοντέλου.



    print("\n Η εκπαίδευση ολοκληρώθηκε και το μοντέλο αποθηκεύτηκε.")





import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Φόρτωση των ονομάτων των κλάσεων (Μοντέλα αυτοκινήτων)
try:
    meta_path = 'C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/CNN/car_devkit/devkit/cars_meta.mat'
    annos_meta = loadmat(meta_path)
    class_names = [name[0] for name in annos_meta['class_names'][0]]
    # Προσθέτουμε το label για την 197η κλάση 
    class_names.append("Unknown / Not a Car")
except Exception as e:
    print(f"Δεν βρέθηκε το cars_meta.mat. Σφάλμα: {e}")
    class_names = [f"Class {i}" for i in range(196)] + ["No Car Detected"]

def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Επεξεργασία εικόνας για το μοντέλο
    img = Image.open(file_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Πρόβλεψη
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
    
    class_idx = predicted.item()

    # Εμφάνιση αποτελέσματος (Μάρκα/Μοντέλο ή No Car)
    if class_idx < 196:
        model_name = class_names[class_idx]
        res_text = f" {model_name}"
        result_color = "green"
    else:
        res_text = " ΔΕΝ ΕΙΝΑΙ ΑΥΤΟΚΙΝΗΤΟ!"
        result_color = "red"

    result_label.config(text=res_text, fg=result_color)
    
    # Εμφάνιση εικόνας στο GUI 
    display_img = img.copy()
    display_img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(display_img)
    panel.config(image=img_tk)
    panel.image = img_tk

# Δημιουργία παραθύρου
root = tk.Tk()
root.title("Stanford Cars Classifier")
root.geometry("600x700") 

title_label = tk.Label(root, text="Car Recognition System", font=("Arial", 20, "bold"))
title_label.pack(pady=20)

btn = tk.Button(root, text="ΕΠΙΛΟΓΗ ΦΩΤΟΓΡΑΦΙΑΣ", command=classify_image, 
                font=("Arial", 12, "bold"), bg="#2196F3", fg="white", 
                padx=20, pady=10, relief="flat")
btn.pack(pady=10)

panel = tk.Label(root, bg="#f0f0f0") 
panel.pack(pady=10)

# Εδώ θα εμφανίζεται η Μάρκα και το Μοντέλο
result_label = tk.Label(root, text="Περιμένω εικόνα...", font=("Arial", 14, "bold"), 
                        wraplength=500, justify="center")
result_label.pack(pady=20)

footer = tk.Label(root, text=" Compute: CPU Mode (Blackwell Optimized Path)", 
                  font=("Arial", 8), fg="gray")
footer.pack(side="bottom", pady=10)

print("\n Το γραφικό περιβάλλον είναι έτοιμο!")
root.mainloop()