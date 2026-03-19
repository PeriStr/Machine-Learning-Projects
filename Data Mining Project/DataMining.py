import pandas as pd
import pandas as pd                     # Βιβλιοθήκη για επεξεργασία δεδομένων (DataFrames)
import numpy as np                      # Αριθμητικές πράξεις υψηλής απόδοσης
from sklearn.preprocessing import LabelEncoder   # Για μετατροπή κειμένου σε αριθμούς (encoding)

# ----------------------------- ΦΟΡΤΩΣΗ DATASET -----------------------------

df = pd.read_csv("C:/Users/perio/Desktop/Εξόρυξη Δεδομένων και Γνώσης/Project/ecommerecenice.csv")
# Διαβάζει το αρχείο CSV και το αποθηκεύει σε DataFrame.
# Από εδώ και πέρα ό,τι κάνουμε εφαρμόζεται πάνω στο df.

print("Initial shape:", df.shape)  
# Εμφανίζει πόσες γραμμές και στήλες έχει το dataset αρχικά.

print(df.head())
# Εμφανίζει τις 5 πρώτες γραμμές για να δούμε τη μορφή των δεδομένων.

print(df.info())
# Δείχνει τύπους δεδομένων (int, float, object) και πόσα missing values έχει κάθε στήλη.

# ----------------------------- MISSING VALUES -----------------------------

print("\nMissing values per column:")
print(df.isna().sum())  
# Υπολογίζει και εμφανίζει πόσα κενά υπάρχουν σε κάθε στήλη.

# Αφαιρούμε γραμμές όπου λείπει το Customer_ID
df = df.dropna(subset=["Customer_ID"])
# Γιατί; Επειδή χωρίς ID δεν ξέρουμε ποιος πελάτης είναι → άρα η γραμμή είναι άχρηστη.

# ---------------------- ΣΥΜΠΛΗΡΩΣΗ ΜISSING ΑΡΙΘΜΗΤΙΚΩΝ ----------------------

numeric_cols = [
    "Recency","Frequency","Monetary","Avg_Order_Value",
    "Session_Count","Avg_Session_Duration","Pages_Viewed",
    "Clicks","Campaign_Response","Wishlist_Adds",
    "Cart_Abandon_Rate","Returns"
]
# Λίστα με όλες τις αριθμητικές στήλες για να μην γράφουμε τον ίδιο κώδικα 12 φορές.

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
        # Αν η τιμή είναι κενή → την αντικαθιστούμε με τη διάμεσο της στήλης
        # Γιατί διάμεσο; 
        # Γιατί δεν επηρεάζεται από outliers και είναι σταθερότερο μέτρο.

# Συμπλήρωση κατηγορικών missing
df["Segment_Label"] = df["Segment_Label"].fillna("Unknown")
# Αν ο πελάτης δεν έχει segment, του δίνουμε Unknown ώστε να μην χαθούν γραμμές.

# ----------------------------- NOISE HANDLING -----------------------------

# Δεν μπορεί να υπάρχουν αρνητικές τιμές σε καμία αριθμητική μεταβλητή
for col in numeric_cols:
    df = df[df[col] >= 0]
    # Φιλτράρει και κρατά ΜΟΝΟ γραμμές όπου η τιμή >= 0

# Session duration πρέπει να είναι λογική (< 300 λεπτά)
df = df[df["Avg_Session_Duration"] < 300]
# Αν το session διαρκεί πάνω από 5 ώρες → μάλλον είναι λάθος log → το αφαιρούμε.

# ----------------------------- OUTLIER CAPPING -----------------------------

def cap_outliers(series):
    Q1 = series.quantile(0.25)           # 1ο τεταρτημόριο (25%)
    Q3 = series.quantile(0.75)           # 3ο τεταρτημόριο (75%)
    IQR = Q3 - Q1                        # Εύρος IQR
    low = Q1 - 1.5 * IQR                 # Κατώτερο όριο
    high = Q3 + 1.5 * IQR                # Ανώτερο όριο
    return series.clip(lower=low, upper=high)
    # Αντικαθιστά outliers με τα όρια → δεν διαγράφουμε δεδομένα, τα "μαζεύουμε".

for col in numeric_cols:
    df[col] = cap_outliers(df[col])
    # Εφαρμόζουμε το "μάζεμα" για κάθε αριθμητική στήλη.

# ----------------------------- STANDARDIZATION -----------------------------

df["Customer_ID"] = df["Customer_ID"].str.strip()
# Αφαιρεί κενά στην αρχή/τέλος → συχνό πρόβλημα σε αρχεία CSV.

df["Segment_Label"] = df["Segment_Label"].str.strip().str.title()
# Κάνει το segment ομοιόμορφο:
# " silver ", "SILVER", "silver" → όλα γίνονται "Silver"

valid_segments = ["Bronze", "Silver", "Gold", "Platinum", "Iron", "Copper"]
df["Segment_Label"] = df["Segment_Label"].apply(
    lambda x: x if x in valid_segments else "Unknown"
)
# Αν κάποιο segment είναι άκυρο → το αντικαθιστούμε με Unknown.

# ----------------------------- ENCODING CUSTOMER ID -----------------------------

user_encoder = LabelEncoder()  
df["Customer_ID_Encoded"] = user_encoder.fit_transform(df["Customer_ID"])
# Μετατρέπει κάθε Customer_ID σε αριθμό:
# π.χ. "A123" → 0, "B457" → 1
# Χρειάζεται για μοντέλα που δέχονται μόνο αριθμούς.

# ----------------------------- ONE-HOT ENCODING SEGMENTS -----------------------------

df = pd.get_dummies(
    df,
    columns=["Segment_Label"],
    prefix="Segment",
    drop_first=False
)
# Δημιουργεί στήλες: Segment_Bronze, Segment_Gold, Segment_Silver, ...
# Η κάθε στήλη έχει τιμή 1 αν ο χρήστης ανήκει στο segment → αλλιώς 0.

print(df.filter(like="Segment").head())
print(df[["Customer_ID", "Customer_ID_Encoded"]].head())

# ----------------------------- REMOVE DUPLICATES -----------------------------

before = df.shape[0]
df = df.drop_duplicates(subset=["Customer_ID"])
after = df.shape[0]

print(f"\nRemoved duplicates: {before - after}")
# Αν υπάρχουν γραμμές με ίδιο Customer_ID → κρατάμε μόνο 1.
# Επειδή είναι ο ίδιος πελάτης → δεν θέλουμε διπλοεγγραφές.

# ----------------------------- LOGICAL VALIDATION -----------------------------

invalid = df[df["Avg_Order_Value"] > df["Monetary"]]
print("\nEntries with impossible order values:", invalid.shape[0])
# Μέση αξία παραγγελίας ΔΕΝ γίνεται να είναι μεγαλύτερη από το συνολικό ποσό.

invalid2 = df[df["Frequency"] > df["Session_Count"]]
print("Entries with Frequency > Sessions:", invalid2.shape[0])
# Δεν γίνεται να έχεις περισσότερες παραγγελίες από sessions.

# ----------------------------- FINAL STANDARDIZATION -----------------------------

df[numeric_cols] = df[numeric_cols].astype(float)
# Βεβαιωνόμαστε ότι όλα είναι float → απαραίτητο για clustering & ML.

df.to_csv("clean_preprocessed_ecommerce.csv", index=False)
# Αποθηκεύουμε το πλήρως καθαρισμένο dataset για χρήση στα επόμενα βήματα.

print("\n Saved as clean_preprocessed_ecommerce.csv")
print(df.head())





# 🔹 K-MEANS CLUSTERING
# -------------------------------------------------------------
# Σκοπός:
# Ο αλγόριθμος K-Means χωρίζει τους πελάτες σε ομάδες (clusters)
# με βάση τα numerical χαρακτηριστικά συμπεριφοράς τους.
# Έτσι βρίσκουμε "τύπους" πελατών (π.χ. high-value, inactive, impulse buyers κ.λπ.)
# -------------------------------------------------------------


from sklearn.preprocessing import StandardScaler   # Για κανονικοποίηση δεδομένων
from sklearn.cluster import KMeans                # Ο αλγόριθμος K-Means
from sklearn.metrics import silhouette_score       # Μετρά την ποιότητα των clusters
import matplotlib.pyplot as plt                   # Για τα διαγράμματα

# -------------------------------------------------------------
# 🔹 Επιλογή χαρακτηριστικών που θα χρησιμοποιηθούν στο clustering
# -------------------------------------------------------------
# Επιλέγουμε ΜΟΝΟ αριθμητικές μεταβλητές που περιγράφουν συμπεριφορά πελάτη.
# Αυτές είναι πολύ χρήσιμες για clustering γιατί:
# - μετρούν αγορές (Monetary, Frequency)
# - μετρούν δραστηριότητα (Sessions, Clicks)
# - μετρούν ενδιαφέρον (Wishlist_Adds)
# - μετρούν αρνητικές συμπεριφορές (Returns, Cart_Abandon)
# -------------------------------------------------------------

features = [
    "Recency", "Frequency", "Monetary", "Avg_Order_Value",
    "Session_Count", "Avg_Session_Duration", "Pages_Viewed",
    "Clicks", "Wishlist_Adds", "Cart_Abandon_Rate", "Returns"
]

# Δημιουργία πίνακα X με τα features
X = df[features]

# -------------------------------------------------------------
# 🔹 Κανονικοποίηση δεδομένων (Standardization)
# -------------------------------------------------------------
# Το K-Means είναι ΠΟΛΥ ευαίσθητο σε χαρακτηριστικά με μεγάλα μεγέθη.
# Π.χ. Recency=200 και Avg_Session_Duration=3.
# Αν δεν γίνει scaling, τα μεγάλου μεγέθους features κυριαρχούν.
# Το StandardScaler φέρνει όλα τα features στην ίδια κλίμακα (mean=0, std=1)
# -------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------
# 🔹 Εύρεση του κατάλληλου αριθμού clusters (k)
# -------------------------------------------------------------
# Χρησιμοποιούμε:
# 1) Elbow Method → δείχνει πότε σταματάει να μειώνεται απότομα το WCSS
# 2) Silhouette Score → μετρά πόσο "καθαρά" είναι τα clusters
# -------------------------------------------------------------

wcss = []  # Within-Cluster Sum of Squares (μέτρο συνοχής κάθε cluster)
sil = []   # Silhouette Score για κάθε k
K_range = range(2, 10)   # δοκιμάζουμε k=2 μέχρι 9

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)   # ορίζουμε μοντέλο με k clusters
    labels = km.fit_predict(X_scaled)            # εκπαίδευση + πρόβλεψη cluster labels
    
    wcss.append(km.inertia_)                     # αποθήκευση WCSS
    sil.append(silhouette_score(X_scaled, labels))  # αποθήκευση silhouette

# -------------------------------------------------------------
# 🔹 Plot του Elbow Method
# -------------------------------------------------------------
# Στο γράφημα:
# - Το "γόνατο" δείχνει ποιο k είναι καλύτερο
# -------------------------------------------------------------

plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.show()

# -------------------------------------------------------------
# 🔹 Plot Silhouette Score
# -------------------------------------------------------------
# Όσο πιο υψηλό σκορ → τόσο καλύτερη διαχωριστικότητα των clusters.
# -------------------------------------------------------------

plt.plot(K_range, sil, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
plt.show()

# -------------------------------------------------------------
# 🔹 Τρέχουμε το K-Means με το τελικό k (εδώ 4)
# -------------------------------------------------------------
# Υποθέτουμε ότι από το elbow + silhouette το καλύτερο k = 4.
# -------------------------------------------------------------

kmeans = KMeans(n_clusters=4, random_state=42)

# Ο αλγόριθμος υπολογίζει ποιο cluster ταιριάζει σε κάθε χρήστη
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------------------------------------
# 🔹 Πόσοι πελάτες ανήκουν σε κάθε cluster
# -------------------------------------------------------------
print(df["Cluster"].value_counts())

# -------------------------------------------------------------
# 🔹 Υπολογισμός του προφίλ κάθε cluster
# -------------------------------------------------------------
# Για κάθε cluster υπολογίζουμε τον μέσο όρο των χαρακτηριστικών του.
# Αυτό παράγει τον ΠΙΟ ΣΗΜΑΝΤΙΚΟ πίνακα:
# → δείχνει τι τύπος πελατών υπάρχει μέσα σε κάθε cluster.
# -------------------------------------------------------------

cluster_profile = df.groupby("Cluster")[[
    "Recency","Frequency","Monetary","Avg_Order_Value",
    "Session_Count","Avg_Session_Duration","Pages_Viewed",
    "Clicks","Wishlist_Adds","Cart_Abandon_Rate","Returns"
]].mean()

print(cluster_profile)


# ----------------------------------------------------------
# XGBOOST CLASSIFIER – ΠΡΟΒΛΕΨΗ ΤΟΥ CLUSTER ΜΕ SUPERVISED ML
# ----------------------------------------------------------

from xgboost import XGBClassifier          # Εισάγουμε το αλγοριθμικό μοντέλο XGBoost
from sklearn.model_selection import train_test_split   # Για να χωρίσουμε δεδομένα σε train/test
from sklearn.metrics import accuracy_score, classification_report  # Για αξιολόγηση μοντέλου

# -----------------------------------------
# 1. Ορίζουμε τα features και τα labels
# -----------------------------------------

# Χρησιμοποιούμε όλα τα χαρακτηριστικά ΕΚΤΟΣ από Cluster & Customer_ID
# • Το Cluster είναι ο στόχος που θέλουμε να προβλέψουμε
# • Το Customer_ID δεν έχει καμία προγνωστική αξία
X = df.drop(["Cluster", "Customer_ID"], axis=1)

# Το label είναι η μεταβλητή που προσπαθούμε να προβλέψουμε: το cluster του πελάτη
y = df["Cluster"]

# --------------------------------------------------------
# 2. Χωρίζουμε τα δεδομένα σε Train (80%) και Test (20%)
# --------------------------------------------------------
# Αυτό γίνεται για να δούμε αν το μοντέλο μαθαίνει πραγματικά,
# χωρίς να "αποστηθίζει" τα δεδομένα.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 3. Δημιουργία του XGBoost Classifier
# -----------------------------------------
model = XGBClassifier(
    n_estimators=200,      # Πόσα δέντρα θα φτιάξει το μοντέλο – περισσότερα = πιο ισχυρό
    max_depth=4,           # Μέγιστο βάθος δέντρου – ελέγχει την πολυπλοκότητα
    learning_rate=0.1,     # Πόσο γρήγορα μαθαίνει – 0.1 είναι ασφαλής επιλογή
    subsample=0.9,         # Ποσοστό δειγμάτων που χρησιμοποιεί κάθε δέντρο (προστασία από overfitting)
    colsample_bytree=0.9,  # Ποσοστό χαρακτηριστικών που χρησιμοποιεί κάθε δέντρο
    objective="multi:softmax",  # Softmax: επειδή έχουμε πολλές κατηγορίες (0,1,2,3)
    num_class=4            # Ο αριθμός των clusters που βρήκαμε στο K-means
)

# -----------------------------------------
# 4. Εκπαίδευση μοντέλου
# -----------------------------------------
# Το μοντέλο "μαθαίνει" πώς τα features οδηγούν στο cluster
model.fit(X_train, y_train)

# -----------------------------------------
# 5. Πρόβλεψη στο test set
# -----------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------
# 6. Αξιολόγηση ποιότητας μοντέλου
# -----------------------------------------

# Accuracy → πόσα ποσοστιαία σωστά predictions έκανε
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report → precision, recall, f1-score για κάθε cluster
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------
# 7. Feature Importance (ποια χαρακτηριστικά είναι τα σημαντικά)
# -----------------------------------------------------------

from xgboost import plot_importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

# Δείχνει ποια features χρησιμοποιεί περισσότερο ο XGBoost για προβλέψεις
plot_importance(model, max_num_features=15)

plt.show()


# -------------------------------
# APRIORI ALGORITHM
# -------------------------------

# Φορτώνουμε τις συναρτήσεις apriori και association rules
# από την βιβλιοθήκη mlxtend, η οποία είναι ειδική για market basket analysis.
from mlxtend.frequent_patterns import apriori, association_rules


# Παίρνουμε όλες τις στήλες που αρχίζουν με "Segment_"
# Αυτές είναι οι one-hot encoded στήλες (Segment_Gold, Segment_Silver κτλ).
# Θέλουμε να τις συμπεριλάβουμε στο Apriori ως "items".
segment_ohe = df.filter(like="Segment_")


# Δημιουργούμε μία λίστα με τα features που θα χρησιμοποιηθούν στο Apriori.
# Περιλαμβάνει:
# - όλα τα segment columns
# - 4 αριθμητικά πεδία που θα τα μετατρέψουμε σε binary items
apriori_features = list(segment_ohe.columns) + [
    "Wishlist_Adds",
    "Cart_Abandon_Rate",
    "Campaign_Response",
    "Returns"
]


# Φτιάχνουμε αντίγραφο του dataset με ΜΟΝΟ τις απαιτούμενες στήλες.
# Το Apriori χρειάζεται binary (0/1) τιμές, οπότε θα τις επεξεργαστούμε στη συνέχεια.
df_ap = df[apriori_features].copy()


# Ξαναφορτώνουμε τα segment columns (σαν one-hot encoded)
segment_ohe = df.filter(like="Segment_")


# Μετατρέπουμε το Campaign_Response σε 0 ή 1.
# Ορίζουμε ότι response > 0 σημαίνει "αντέδρασε".
df_ap["Campaign_Response"] = df_ap["Campaign_Response"].apply(lambda x: 1 if x > 0 else 0)


# Κατηγοριοποίηση υψηλού Wishlist_Adds
# Αν το wishlist_adds είναι πάνω από τη διάμεσο του dataset → θεωρείται υψηλό.
df_ap["Wishlist_High"] = df["Wishlist_Adds"].apply(lambda x: 1 if x > df["Wishlist_Adds"].median() else 0)


# Κατηγοριοποίηση υψηλού Cart Abandon Rate
df_ap["CartAbandon_High"] = df["Cart_Abandon_Rate"].apply(lambda x: 1 if x > df["Cart_Abandon_Rate"].median() else 0)


# Κατηγοριοποίηση υψηλών Returns
df_ap["Returns_High"] = df["Returns"].apply(lambda x: 1 if x > df["Returns"].median() else 0)


# Ετοιμάζουμε το "market basket".
# Συνενώνουμε:
# - segment_ohe (Segment_Gold κτλ)
# - binary μεταβλητές (Campaign_Response, Wishlist_High, CartAbandon_High, Returns_High)
basket = pd.concat([
    segment_ohe,
    df_ap[["Campaign_Response", "Wishlist_High", "CartAbandon_High", "Returns_High"]]
], axis=1)


# Μετατρέπουμε ΟΠΩΣΔΗΠΟΤΕ το basket σε 0/1 integers.
# Το Apriori *απαιτεί* binary items.
basket = basket.astype(bool).astype(int)


# -------------------------------
# FREQUENT ITEMSETS (APRIORI)
# -------------------------------

# Εκτελούμε το Apriori algorithm:
# - min_support=0.05 σημαίνει ότι ένα item πρέπει να εμφανίζεται στο 5% των πελατών
# - use_colnames=True → τα αποτελέσματα θα κρατήσουν τα ονόματα των στηλών
frequent_items = apriori(basket, min_support=0.05, use_colnames=True)


# -------------------------------
# ASSOCIATION RULES
# -------------------------------

# Δημιουργούμε κανόνες συσχέτισης
# metric="lift": χρησιμοποιούμε το lift ως βασικό κριτήριο
# min_threshold=1.1: κρατάμε μόνο κανόνες που έχουν lift > 1.1 (θετική συσχέτιση)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.1)


# Εμφάνιση των συχνών itemsets
print("Frequent Itemsets:")
print(frequent_items.head())


# Εμφάνιση των 5 πρώτων κανόνων συσχέτισης
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())


# Επιλογή των 10 ισχυρότερων κανόνων (με το μεγαλύτερο lift)
strong_rules = rules.sort_values("lift", ascending=False).head(10)

print("\nTop 10 strongest rules:")
print(strong_rules)



