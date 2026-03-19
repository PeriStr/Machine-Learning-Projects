import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pandas as pd
import spacy
from torch.nn.utils.rnn import pad_sequence
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Φορτώνουμε το μικρό μοντέλο Αγγλικών του SpaCy. 
# Θα το χρησιμοποιήσουμε για να "κόβουμε" σωστά τις προτάσεις σε λέξεις (tokenization).
spacy_eng = spacy.load("en_core_web_sm")


# 1. ΚΛΑΣΗ VOCABULARY 

class Vocabulary:
    def __init__(self, freq_threshold):
        # itos: Index To String (Από αριθμό σε λέξη). Ορίζουμε τα 4 βασικά "ειδικά" σύμβολα.
        # <PAD>: Γέμισμα για να έχουν ίδιο μέγεθος οι προτάσεις.
        # <SOS>: Start Of Sentence (Ξεκινάει η πρόταση).
        # <EOS>: End Of Sentence (Τελειώνει η πρόταση).
        # <UNK>: Unknown (Λέξη που δεν ξέρει το μοντέλο).
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # stoi: String To Index (Από λέξη σε αριθμό). Το ανάποδο λεξικό.
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        # Το όριο συχνότητας. Αν μια λέξη εμφανιστεί λιγότερες φορές από αυτό, την αγνοούμε (γίνεται UNK).
        self.freq_threshold = freq_threshold

    def __len__(self):
        # Επιστρέφει το συνολικό μέγεθος του λεξιλογίου (πόσες λέξεις ξέρει το AI).
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Παίρνει μια πρόταση και την κόβει σε λίστα από μικρά γράμματα. π.χ. "A Dog" -> ["a", "dog"]
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        # Χτίζει το λεξικό διαβάζοντας όλες τις λεζάντες του dataset.
        frequencies = {}
        idx = 4 # Ξεκινάμε από το 4, γιατί τα 0, 1, 2, 3 είναι τα PAD, SOS, EOS, UNK.
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                # Μετράμε πόσες φορές εμφανίζεται η κάθε λέξη
                frequencies[word] = frequencies.get(word, 0) + 1
                # Αν η λέξη φτάσει το όριο (π.χ. 5 φορές), μπαίνει επίσημα στο λεξικό μας
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        # Μετατρέπει μια πρόταση με λέξεις σε λίστα με αριθμούς (indices) για να μπει στο νευρωνικό δίκτυο.
        tokenized_text = self.tokenizer_eng(text)
        return [
            # Αν η λέξη υπάρχει στο λεξικό, δώσε τον αριθμό της. Αλλιώς, δώσε το 3 (που είναι το UNK).
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


# 2. ΚΛΑΣΗ FLICKRDATASET 

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir # Ο φάκελος με τις φωτογραφίες
        self.df = pd.read_csv(captions_file) # Το αρχείο με τις λεζάντες
        self.transform = transform # Οι μετατροπές της εικόνας (Resize, Normalize κλπ)
        self.imgs = self.df["image"] # Στήλη με τα ονόματα των εικόνων
        self.captions = self.df["caption"] # Στήλη με τις λεζάντες
        
        # Φτιάχνουμε το λεξικό και το "εκπαιδεύουμε" με όλες τις λεζάντες του CSV
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        # Επιστρέφει το πόσα ζευγάρια (εικόνα-λεζάντα) έχουμε συνολικά.
        return len(self.df)

    def __getitem__(self, index):
        # Αυτή η μέθοδος τρέχει κάθε φορά που ο DataLoader ζητάει μια νέα εικόνα
        caption = self.captions[index]
        img_id = self.imgs[index]
        
        # Ανοίγουμε την εικόνα και σιγουρευόμαστε ότι είναι έγχρωμη (RGB)
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        
        # Εφαρμόζουμε το resize και το normalization
        if self.transform is not None:
            img = self.transform(img)
            
        # Μετατρέπουμε τη λεζάντα σε νούμερα προσθέτοντας το <SOS> στην αρχή
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        # Και το <EOS> στο τέλος
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        # Επιστρέφουμε την εικόνα (ως tensor) και τη λεζάντα (ως tensor)
        return img, torch.tensor(numericalized_caption)


# 3. ΚΛΑΣΗ CAPSCOLLATE 

class CapsCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx # Ο αριθμός που αντιστοιχεί στο <PAD> (συνήθως το 0)
        
    def __call__(self, batch):
        # Βάζει μια έξτρα διάσταση σε κάθε εικόνα και μετά τις ενώνει όλες μαζί σε ένα μεγάλο Tensor
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        # Παίρνει όλες τις λεζάντες του batch
        targets = [item[1] for item in batch]
        # Κάνει padding (γεμίζει με 0) τις μικρές προτάσεις για να έχουν το ίδιο μήκος με τη μεγαλύτερη πρόταση του batch
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets


# 4. ΚΛΑΣΗ ENCODERCNN 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Κατεβάζουμε τον ResNet50, προ-εκπαιδευμένο στο ImageNet
        self.resnet = models.resnet50(weights='DEFAULT')
        
        # Freeze: Δεν εκπαιδεύουμε ξανά τον ResNet, τον αφήνουμε όπως είναι (Transfer Learning)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Αφαιρούμε το τελευταίο layer του ResNet (Classification Layer)
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Προσθέτουμε ένα δικό μας Linear layer για να συμπυκνώσουμε τα 2048 features σε 256 (embed_size)
        self.linear = nn.Linear(2048, embed_size) 
        self.relu = nn.ReLU() # Activation function για να κρατάμε τα θετικά σήματα
        self.dropout = nn.Dropout(0.5) # Απενεργοποιεί τυχαία το 50% των νευρώνων για αποφυγή Overfitting

    def forward(self, images):
        # 1. Η εικόνα περνάει από τον ResNet
        features = self.resnet(images)
        # 2. Κάνουμε Flatten (απλώνουμε τα δεδομένα σε 1D)
        features = features.view(features.size(0), -1)
        # 3. Συμπιέζουμε στα 256 νούμερα
        features = self.linear(features)
        # 4. Εφαρμόζουμε ReLU και Dropout και τα στέλνουμε στον Decoder
        return self.dropout(self.relu(features))


# 5. ΚΛΑΣΗ DECODERLSTM 

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderLSTM, self).__init__()
        # Μετατρέπει τους αριθμούς των λέξεων σε πυκνά διανύσματα (vectors) μεγέθους 256
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Η καρδιά της μνήμης. Παίρνει τα embeddings και δημιουργεί το hidden state (τις "σκέψεις")
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        # Το τελικό layer που παίρνει τις σκέψεις του LSTM και διαλέγει λέξη από το Λεξιλόγιο
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        # 1. Μετατρέπουμε τις λέξεις σε διανύσματα και ρίχνουμε λίγο Dropout
        embeddings = self.dropout(self.embed(captions))
        # 2. Το ΣΗΜΑΝΤΙΚΟΤΕΡΟ: Ενώνουμε τα features της εικόνας ΣΤΗΝ ΑΡΧΗ των λέξεων
        # Έτσι το LSTM βλέπει πρώτα την εικόνα και μετά τις λέξεις.
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # 3. Περνάμε το "σάντουιτς" (εικόνα+λέξεις) μέσα από το LSTM
        hiddens, _ = self.lstm(embeddings)
        # 4. Υπολογίζουμε τις πιθανότητες για την επόμενη λέξη
        return self.linear(hiddens)


# 6. ΚΛΑΣΗ CNNtoRNN 

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderLSTM = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        # Εκπαίδευση: Η εικόνα μπαίνει στον Encoder, και το αποτέλεσμά της μαζί με τη λεζάντα πάνε στον Decoder
        return self.decoderLSTM(self.encoderCNN(images), captions)
        
    def caption_image(self, image, vocabulary, max_length=50):
        # Inference (Παραγωγή νέας λεζάντας): Δεν γίνεται εκπαίδευση εδώ.
        result_caption = []
        with torch.no_grad(): # Κλείνουμε τους υπολογισμούς των gradients για εξοικονόμηση μνήμης
            x = self.encoderCNN(image).unsqueeze(0) # Εξάγουμε τα features της εικόνας
            states = None # Αρχικοποιούμε τη μνήμη του LSTM ως κενή
            
            for _ in range(max_length): # Loop παραγωγής μέχρι 50 λέξεις (Auto-regression)
                # Το LSTM παράγει το hidden state
                hiddens, states = self.decoderLSTM.lstm(x, states)
                # Το Linear Layer βγάζει τις πιθανότητες λέξεων
                output = self.decoderLSTM.linear(hiddens.squeeze(0))
                # Παίρνουμε τη λέξη με τη μεγαλύτερη πιθανότητα (Greedy Search)
                predicted = output.argmax(1)
                
                # Μεταφράζουμε τον αριθμό πίσω σε λέξη
                predicted_word = vocabulary.itos[predicted.item()]
                # Αν η λέξη είναι <EOS>, σταματάμε να γράφουμε.
                if predicted_word == "<EOS>": break
                
                result_caption.append(predicted_word)
                # Η λέξη που μόλις προβλέψαμε γίνεται η "είσοδος" για τον επόμενο γύρο του Loop!
                x = self.decoderLSTM.embed(predicted).unsqueeze(0)
                
        return result_caption


# 7. ΓΡΑΦΙΚΟ ΠΕΡΙΒΑΛΛΟΝ 

def run_gui():
    root = tk.Tk()
    root.title("AI Image Captioning - RTX 5070 Edition")
    root.geometry("600x700")

    img_label = tk.Label(root)
    img_label.pack(pady=20)

    caption_label = tk.Label(root, text="Επίλεξε μια εικόνα για περιγραφή...", 
                               wraplength=500, font=("Helvetica", 14, "bold"), fg="blue")
    caption_label.pack(pady=20)

    def select_image():
        file_path = filedialog.askopenfilename() # Ανοίγει παράθυρο επιλογής αρχείου
        if file_path:
            # 1. Εμφάνιση της εικόνας στο GUI
            display_img = Image.open(file_path)
            display_img.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(display_img)
            img_label.config(image=img_tk)
            img_label.image = img_tk
            
            # 2. Προετοιμασία της εικόνας για το μοντέλο
            raw_img = Image.open(file_path).convert("RGB")
            input_tensor = transform(raw_img).to(device).unsqueeze(0) # Προσθέτουμε batch dimension
            
            model.eval() # Θέτουμε το μοντέλο σε κατάσταση αξιολόγησης (κλείνει το Dropout)
            with torch.no_grad():
                # Ζητάμε από το μοντέλο να παράγει τη λεζάντα
                prediction = model.caption_image(input_tensor, dataset.vocab)
                sentence = " ".join(prediction) # Ενώνουμε τις λέξεις με κενά
                caption_label.config(text=f"Πρόβλεψη: {sentence}")

    # Κουμπί που καλεί τη συνάρτηση select_image όταν πατηθεί
    btn = tk.Button(root, text="Επιλογή Εικόνας", command=select_image, 
                    padx=20, pady=10, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
    btn.pack(pady=10)
    root.mainloop()


# 8. ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ ΚΑΙ ΕΚΠΑΙΔΕΥΣΗ 

if __name__ == '__main__':
    # 1. Ορίζουμε τις μετατροπές (στάνταρ για μοντέλα ImageNet όπως ο ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Όλες οι εικόνες γίνονται 224x224
        transforms.ToTensor(), # Μετατροπή σε PyTorch Tensor (αριθμούς)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Κανονικοποίηση χρωμάτων
    ])

    # 2. Φορτώνουμε το Custom Dataset μας
    dataset = FlickrDataset(
        root_dir="C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/CNN LSTM/Images",
        captions_file="C:/Users/perio_bkgb40g/Desktop/Προχωρημένες Τεχνικές Μηχανικής Μάθησης/Projects/CNN LSTM/captions.txt",
        transform=transform
    )

    # 3. Χωρίζουμε τα δεδομένα: 80% για Training, 20% για Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Βρίσκουμε ποιος αριθμός αντιστοιχεί στο σύμβολο <PAD>
    pad_idx = dataset.vocab.stoi["<PAD>"]

    # 4. Φτιάχνουμε τους DataLoaders (οι "Σερβιτόροι" που στέλνουν τα δεδομένα στην 5070)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=64, # 64 εικόνες τη φορά
        num_workers=4,  # Χρήση 4 πυρήνων του CPU για γρήγορο φόρτωμα
        shuffle=True, # Ανακάτεμα για να μην μαθαίνει τη σειρά
        collate_fn=CapsCollate(pad_idx=pad_idx) # Χρήση του Collate για το Padding
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=64,
        num_workers=4, 
        shuffle=False, # Το Validation δεν χρειάζεται ανακάτεμα
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )

    # 5. Ορίζουμε Hardware, Μοντέλο, Optimizer και Loss Function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # (embed_size=256, hidden_size=512, vocab_size, num_layers=1)
    model = CNNtoRNN(256, 512, len(dataset.vocab), 1).to(device)
    
    # Adam Optimizer: Ο καλύτερος "δάσκαλος" με adaptive learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # Cross Entropy Loss: Μετράει το λάθος. Το ignore_index=pad_idx σημαίνει ότι ΔΕΝ τιμωρούμε το μοντέλο αν κάνει λάθος στα κενά (padding).
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    MODEL_PATH = "best_caption_model.pth"
    
    # Αν το μοντέλο έχει ήδη εκπαιδευτεί και υπάρχει αποθηκευμένο, απλά φόρτωσέ το!
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        best_val_loss = float('inf') # Αρχικά το καλύτερο λάθος είναι το "άπειρο"
        
        # 6. TRAINING LOOP (Η Διαδικασία Εκπαίδευσης)
        for epoch in range(50): # 50 εποχές (Διαβάζουμε το dataset 50 φορές)
            model.train() # Ενεργοποίηση Dropout κλπ
            
            for idx, (imgs, captions) in enumerate(train_loader):
                imgs, captions = imgs.to(device), captions.to(device) # Μεταφορά στην GPU
                outputs = model(imgs, captions) # Το μοντέλο μαντεύει
                
                # Υπολογισμός Λάθους: Κάνουμε reshape για να γίνουν οι πίνακες 1D (απαραίτητο για το CrossEntropy)
                # Αγνοούμε την τελευταία λέξη (outputs[:-1]) γιατί μετά το <EOS> δεν έχει νόημα.
                loss = criterion(outputs[:-1].reshape(-1, outputs.shape[2]), captions.reshape(-1))
                
                # Υπολογισμός Accuracy (Ακρίβειας)
                _, preds = outputs[:-1].max(2) # Βρίσκουμε τις λέξεις που διάλεξε το μοντέλο
                mask = captions != pad_idx # Φίλτρο για να αγνοήσουμε τα PADDING
                correct = (preds == captions) & mask # Μετράμε τα σωστά (εξαιρώντας τα padding)
                accuracy = correct.sum().item() / mask.sum().item() # Σωστά δια Συνολικά

                # BACKPROPAGATION (Η διόρθωση του "μυαλού")
                optimizer.zero_grad() # Σβήνουμε τα προηγούμενα λάθη
                loss.backward() # Υπολογίζουμε τις παραγώγους (ποιος νευρώνας φταίει)
                
                # Gradient Clipping: Αποτρέπει το να "εκραγούν" οι υπολογισμοί του LSTM (σταθεροποιεί την εκπαίδευση)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step() # Ανανεώνουμε τα βάρη
                
                # Εκτύπωση προόδου κάθε 100 βήματα
                if (idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/50], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {accuracy*100:.2f}%")

            # 7. VALIDATION LOOP (Οι "Εξετάσεις" στο τέλος κάθε Εποχής)
            model.eval() # Κλείνουμε το Dropout για να δούμε την καθαρή απόδοση
            val_loss = 0
            val_acc = 0
            val_mask_sum = 0
            
            with torch.no_grad(): # Κλείνουμε τον υπολογισμό gradients (γλιτώνουμε μνήμη και χρόνο)
                for v_imgs, v_captions in val_loader:
                    v_imgs, v_captions = v_imgs.to(device), v_captions.to(device)
                    v_outputs = model(v_imgs, v_captions)
                    v_loss = criterion(v_outputs[:-1].reshape(-1, v_outputs.shape[2]), v_captions.reshape(-1))
                    val_loss += v_loss.item()
                    
                    # Υπολογισμός Validation Accuracy
                    _, v_preds = v_outputs[:-1].max(2)
                    v_mask = v_captions != pad_idx
                    val_acc += (v_preds == v_captions)[v_mask].sum().item()
                    val_mask_sum += v_mask.sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = (val_acc / val_mask_sum) * 100
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.2f}%")

            # 8. EARLY STOPPING / SAVING (Αποθήκευση του καλύτερου μοντέλου)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH) # Σώζει το μοντέλο στο δίσκο
                print(f"Model saved with loss: {best_val_loss:.4f}")

    # Αφού τελειώσει η εκπαίδευση (ή αν φορτωθεί έτοιμο μοντέλο), ξεκινάει το GUI
    model.eval() 
    run_gui()