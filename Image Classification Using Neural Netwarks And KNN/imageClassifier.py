
# Φόρτωση απαραίτητων βιβλιοθηκών

import numpy as np  # Για αριθμητικούς υπολογισμούς, πίνακες και μαθηματικές πράξεις
import cv2  # OpenCV για επεξεργασία εικόνας (π.χ. υπολογισμός histogram, μετατροπή χρωματικού χώρου)
import matplotlib.pyplot as plt  # Για plotting και visualization γραφημάτων
import seaborn as sns  # Για στατιστικά γραφήματα

# TensorFlow/Keras για deep learning
from tensorflow.keras.datasets import cifar10, cifar100 # Για να φορτώσουμε τα CIFAR-10 και CIFAR-100 datasets
from tensorflow.keras.models import Sequential  # Για να φτιάξουμε μοντέλα MLP
from tensorflow.keras.layers import Dense, Dropout  # Dense πλήρως συνδεδεμένο layer, Dropout για regularization
from tensorflow.keras.optimizers import Adam  # Optimizer για gradient descent
from tensorflow.keras.callbacks import EarlyStopping  # Σταματά το training όταν δεν βελτιώνεται


from sklearn.model_selection import train_test_split  # Διαχωρισμός dataset σε train/val/test
from sklearn.neighbors import KNeighborsClassifier  # KNN για classification
from sklearn.metrics import (  # Metrics για αξιολόγηση μοντέλων
    accuracy_score,  # Απλή ακρίβεια
    classification_report,  # Precision/Recall/F1 ανά κλάση
    confusion_matrix  # Πίνακας σύγχυσης
)

""" Φόρτωση των συνόλων δεδομένων

Για την υλοποίηση της εργασίας χρησιμοποιήθηκαν τα σύνολα δεδομένων CIFAR-10 και CIFAR-100, τα οποία παρέχονται έτοιμα μέσω της βιβλιοθήκης Keras.
Το CIFAR-10 περιλαμβάνει 60.000 έγχρωμες εικόνες διαστάσεων 32×32 κατανεμημένες σε 10 κατηγορίες, ενώ το CIFAR-100 περιλαμβάνει τον ίδιο αριθμό εικόνων αλλά κατανεμημένες σε 100 κατηγορίες.
Για το CIFAR-100 χρησιμοποιήθηκαν τα fine labels.
Τα δεδομένα διαχωρίζονται σε σύνολα εκπαίδευσης και ελέγχου, ενώ ο αριθμός των κλάσεων χρησιμοποιείται αργότερα για τον ορισμό της εξόδου του νευρωνικού δικτύου."""
 
# Φόρτωση Dataset

def load_dataset(name="cifar10"):
    # Ανάλογα με το όνομα dataset, φορτώνουμε cifar10 ή cifar100
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Επιστρέφει train/test split
        num_classes = 10  # 10 κλάσεις για CIFAR-10
    elif name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")  # fine = 100 κλάσεις
        num_classes = 100
    else:
        raise ValueError("Unknown dataset")  

    # Επιστρέφουμε τα δεδομένα και τον αριθμό κλάσεων
    return x_train, y_train.flatten(), x_test, y_test.flatten(), num_classes
    # flatten() για να έχουμε 1D array αντί για column vector

"""Προεπεξεργασία δεδομένων

Στο στάδιο της προεπεξεργασίας πραγματοποιείται κανονικοποίηση των τιμών των εικονοστοιχείων από το εύρος [0, 255] στο [0, 1].
Η κανονικοποίηση είναι απαραίτητη ώστε τα χαρακτηριστικά να βρίσκονται στην ίδια κλίμακα και να διευκολύνεται η εκπαίδευση των ταξινομητών.
Στη συνέχεια, το σύνολο εκπαίδευσης χωρίζεται σε σύνολα εκπαίδευσης και επικύρωσης (validation), ώστε να είναι δυνατή η παρακολούθηση της απόδοσης του μοντέλου κατά τη διάρκεια της εκπαίδευσης και να αποφευχθεί το overfitting."""
# Κανονικοποίηση εικόνων

def normalize_images(x):
    # Μετατρέπουμε τύπο σε float32 και κανονικοποιούμε [0,255] -> [0,1]
    return x.astype(np.float32) / 255.0


# Split σε training & validation

def split_train_validation(x, y, val_size=0.2):
    # Χρησιμοποιούμε stratify=y για να κρατήσουμε την αναλογία κλάσεων ίδια
    return train_test_split(
        x, y, test_size=val_size, random_state=42, stratify=y
    )

"""Εξαγωγή χρωματικών χαρακτηριστικών Color Histograms

Για την περιγραφή του χρώματος κάθε εικόνας υλοποιήθηκαν χρωματικά ιστογράμματα στον χρωματικό χώρο RGB.
Για κάθε ένα από τα τρία κανάλια R, G, B υπολογίζεται ένα ιστόγραμμα με προκαθορισμένο αριθμό bins.
Τα ιστογράμματα κανονικοποιούνται και στη συνέχεια συγχωνεύονται σε ένα ενιαίο διάνυσμα χαρακτηριστικών.
Με αυτόν τον τρόπο, κάθε εικόνα αναπαρίσταται από τη συνολική κατανομή των χρωμάτων της, αγνοώντας τη χωρική διάταξη των εικονοστοιχείων."""

# Color histogram feature extraction

def extract_color_histogram(image, bins=16, color_space="RGB"):
    # Μετατροπή σε HSV αν ζητηθεί
    if color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    features = []  # Λίστα για όλα τα κανάλια
    for c in range(3):  # Για κάθε κανάλι (R/G/B ή H/S/V)
        hist = cv2.calcHist([image], [c], None, [bins], [0, 256])  # Υπολογισμός ιστόγραμματος
        hist = cv2.normalize(hist, hist).flatten()  # Normalize και flatten σε 1D
        features.extend(hist)  # Προσθέτουμε στο feature vector

    return np.array(features)  # Επιστρέφουμε ως array

"""Εξαγωγή χαρακτηριστικών ακμών (Edge Histograms)

Για την περιγραφή της δομής και της υφής των εικόνων χρησιμοποιήθηκαν ιστογράμματα ακμών.
Αρχικά, κάθε εικόνα μετατρέπεται σε εικόνα αποχρώσεων του γκρι.
Στη συνέχεια εφαρμόζονται οι τελεστές Sobel στους άξονες x και y, ώστε να υπολογιστούν οι κλίσεις της έντασης.
Από τις κλίσεις υπολογίζεται το gradient magnitude, το οποίο περιγράφει την ένταση των ακμών.
Τέλος, δημιουργείται ιστόγραμμα των τιμών της βαθμίδας, το οποίο αποτελεί το διάνυσμα χαρακτηριστικών της εικόνας."""

# Edge histogram feature extraction

def extract_edge_histogram(image, bins=16):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Μετατροπή σε grayscale

    # Υπολογισμός Sobel gradients 
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    magnitude = cv2.magnitude(gx, gy)  # Μέτρο gradient για ένταση άκρων

    # Υπολογισμός ιστογράμματος των magnitude
    hist, _ = np.histogram(magnitude, bins=bins, range=(0, magnitude.max()))
    hist = hist.astype("float")  # Μετατροπή σε float
    hist /= (hist.sum() + 1e-6)  # Κανονικοποίηση σε άθροισμα 1 για σύγκριση

    return hist  # Επιστρέφουμε το feature vector

"""Δημιουργία διανυσμάτων χαρακτηριστικών

Για κάθε εικόνα του συνόλου δεδομένων εφαρμόζεται η αντίστοιχη μέθοδος εξαγωγής χαρακτηριστικών.
Τα διανύσματα χαρακτηριστικών όλων των εικόνων συγκεντρώνονται σε έναν πίνακα χαρακτηριστικών feature matrix, όπου κάθε γραμμή αντιστοιχεί σε μία εικόνα.
Ο πίνακας αυτός αποτελεί την είσοδο τόσο για το νευρωνικό δίκτυο όσο και για τον ταξινομητή kNN."""

# Build feature matrix για όλα τα images

def build_feature_matrix(images, feature_type="color"):
    features = []  # Λίστα για όλα τα samples

    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)  # Επιστροφή σε uint8 για OpenCV

        if feature_type == "color":
            feat = extract_color_histogram(img_uint8, bins=16)
        elif feature_type == "edge":
            feat = extract_edge_histogram(img_uint8, bins=16)
        else:
            raise ValueError("Unknown feature type")

        features.append(feat)  # Προσθήκη feature vector

    return np.array(features)  # Μετατροπή σε numpy array

"""Ταξινόμηση με Πολυεπίπεδο Perceptron

Για την ταξινόμηση των εικόνων υλοποιήθηκε ένα πολυεπίπεδο νευρωνικό δίκτυο .
Η είσοδος του δικτύου είναι το διάνυσμα χαρακτηριστικών κάθε εικόνας.
Το δίκτυο αποτελείται από δύο κρυφά επίπεδα με συνάρτηση ενεργοποίησης ReLU και ένα επίπεδο εξόδου με συνάρτηση Softmax, το οποίο επιστρέφει πιθανότητες για κάθε κατηγορία.
Η εκπαίδευση πραγματοποιείται με τον αλγόριθμο Adam και χρησιμοποιείται early stopping, ώστε να αποφεύγεται η υπερπροσαρμογή."""

# Build MLP model

def build_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),  # Πρώτο fully connected layer
        Dropout(0.5),  # Dropout για regularization και αποφυγή overfitting
        Dense(128, activation="relu"),  # Δεύτερο hidden layer
        Dense(num_classes, activation="softmax")  # Output layer με softmax για multi-class
    ])

    # Compile με Adam optimizer & sparse categorical crossentropy
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model  # Επιστρέφουμε το μοντέλο

"""Ταξινόμηση με k-Κοντινότερους Γείτονες (kNN)

Παράλληλα με το MLP, υλοποιήθηκε και ο ταξινομητής k-Κοντινότερων Γειτόνων.
Ο kNN δεν απαιτεί εκπαίδευση , αλλά βασίζεται στον υπολογισμό αποστάσεων μεταξύ των διανυσμάτων χαρακτηριστικών.
Δοκιμάστηκαν διαφορετικές τιμές της παραμέτρου k (1, 3, 5, 7, 11), ώστε να μελετηθεί η επίδρασή της στην απόδοση του ταξινομητή.
"""
# Train KNN

def train_knn(x_train, y_train, k=5, metric="euclidean"):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)  # Δημιουργία KNN
    knn.fit(x_train, y_train)  # Training
    return knn

"""Αξιολόγηση των αποτελεσμάτων

Η αξιολόγηση των μοντέλων πραγματοποιείται στο test set.
Για κάθε πείραμα υπολογίζεται το accuracy, καθώς και μετρικές precision, recall και F1-score ανά κατηγορία.
Επιπλέον, υπολογίζεται το confusion matrix, ο οποίος επιτρέπει την ανάλυση των λαθών ταξινόμησης.
Για το MLP παρουσιάζονται επίσης καμπύλες μάθησης, οι οποίες δείχνουν την εξέλιξη του accuracy στο σύνολο εκπαίδευσης."""

# Evaluate model

def evaluate_model(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    print(classification_report(y_true, y_pred))  # Precision,Recall,F1 για κάθε κλάση

    cm = confusion_matrix(y_true, y_pred)  # Υπολογισμός confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues", cbar=False)  # Heatmap με seaborn
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

"""Σύγκριση συνόλων δεδομένων και χαρακτηριστικών

Η ίδια μεθοδολογία εφαρμόζεται τόσο στο CIFAR-10 όσο και στο CIFAR-100, ώστε να είναι δυνατή η άμεση σύγκριση των αποτελεσμάτων.
Μελετάται η επίδραση του τύπου χαρακτηριστικών , καθώς και η διαφορά απόδοσης μεταξύ MLP και kNN.
Ιδιαίτερη έμφαση δίνεται στην αυξημένη δυσκολία του CIFAR-100 λόγω του μεγαλύτερου αριθμού κατηγοριών."""

# Run full experiment

def run_experiment(dataset_name="cifar10", feature_type="color"):
    print(f"\n=== Dataset: {dataset_name} | Features: {feature_type} ===")

    # Φόρτωση dataset
    x_train, y_train, x_test, y_test, num_classes = load_dataset(dataset_name)

    # Κανονικοποίηση
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    # Διαχωρισμός train/validation
    x_train, x_val, y_train, y_val = split_train_validation(x_train, y_train)

    # Δημιουργία feature matrices για train, val και test
    X_train_feat = build_feature_matrix(x_train, feature_type)
    X_val_feat = build_feature_matrix(x_val, feature_type)
    X_test_feat = build_feature_matrix(x_test, feature_type)

    # Δημιουργία MLP
    model = build_mlp(X_train_feat.shape[1], num_classes)

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Training
    history = model.fit(
        X_train_feat, y_train,
        validation_data=(X_val_feat, y_val),
        epochs=50,
        batch_size=128,
        callbacks=[early_stop],
        verbose=2
    )

    # Evaluation στο test set
    test_loss, test_acc = model.evaluate(X_test_feat, y_test, verbose=0)
    print(f"MLP Test Accuracy: {test_acc:.4f}")

    # Πρόβλεψη με MLP
    y_pred_mlp = np.argmax(model.predict(X_test_feat), axis=1)
    evaluate_model(y_test, y_pred_mlp, title="MLP Confusion Matrix")

    # Plot learning curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("MLP Learning Curve")
    plt.show()

    # Εκπαίδευση και αξιολόγηση KNN με διαφορετικά k
    for k in [1, 3, 5, 7, 11]:
        knn = train_knn(X_train_feat, y_train, k=k)
        y_pred_knn = knn.predict(X_test_feat)
        acc = accuracy_score(y_test, y_pred_knn)
        print(f"kNN (k={k}) Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Εκτέλεση πειράματος για CIFAR-10 με color features
    run_experiment("cifar10", feature_type="color")
    # Εκτέλεση πειράματος για CIFAR-10 με edge features
    run_experiment("cifar10", feature_type="edge")

    # Εκτέλεση πειράματος για CIFAR-100 με color features
    run_experiment("cifar100", feature_type="color")
    # Εκτέλεση πειράματος για CIFAR-100 με edge features
    run_experiment("cifar100", feature_type="edge")
