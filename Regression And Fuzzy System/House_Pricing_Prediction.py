'''
Το παρακατω dataset εχει παρθει απο το kaggle συγκεκριμενα αν πληκτρολογισουμε στο kaggle housing prices ειναι η επιλογη California Housing Prices
το αρχειο csv housing.csv και περιεχει τα πεδια 
latitude
housing_median_age
total_rooms 
total_bedrooms
population
household
median_income
median_house_value
ocean_proximity
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Φόρτωση του dataset
# Το αρχείο housing.csv περιέχει δεδομένα ακινήτων. Χρησιμοποιούμε τη βιβλιοθήκη pandas για να το φορτώσουμε σε έναν DataFrame.
data = pd.read_csv(r'/\housing.csv')

# Επιλογή χαρακτηριστικών και στόχου
# Χρησιμοποιούμε τις στήλες 'median_income', 'total_rooms', και 'housing_median_age' ως χαρακτηριστικά (X) για την πρόβλεψη.
# Ο y είναι η τιμή 'median_house_value', δηλαδή η μέση τιμή των ακινήτων.
X = data[['median_income', 'total_rooms', 'housing_median_age']]
y = data['median_house_value']

# Εκπαίδευση του μοντέλου γραμμικής παλινδρόμησης
# Χρησιμοποιούμε τη γραμμική παλινδρόμηση από τη βιβλιοθήκη scikit-learn για να δημιουργήσουμε ένα μοντέλο.
# Το μοντέλο εκπαιδεύεται με τα χαρακτηριστικά X και τον y.
model = LinearRegression()
model.fit(X, y)

"""
Πρόβλεψη τιμής ακινήτου με βάση τα χαρακτηριστικά που δίνει ο χρήστης.
median_income (float): Το μέσο εισόδημα της περιοχής.
total_rooms (float): Ο συνολικός αριθμός δωματίων.
housing_median_age (float): Η μέση ηλικία των κατοικιών.
float: Η προβλεπόμενη τιμή του ακινήτου.
"""
def predict_price(median_income, total_rooms, housing_median_age):
    # Δημιουργία ενός πίνακα εισόδου για το μοντέλο.
    input_data = np.array([[median_income, total_rooms, housing_median_age]])
    # Πρόβλεψη τιμής μέσω του εκπαιδευμένου μοντέλου.
    predicted_price = model.predict(input_data)
    return predicted_price[0]

"""
Σύγκριση παρελθοντικής και μελλοντικής τιμής ακινήτου.
Parameters:
past_value (float): Η παρελθοντική τιμή του ακινήτου.
predicted_value (float): Η προβλεπόμενη τιμή του ακινήτου.
"""
def compare_prices(past_value, predicted_value):
    # Υπολογισμός της διαφοράς ανάμεσα στην παρελθοντική και την προβλεπόμενη τιμή.
    diff = predicted_value - past_value
    # Επιστροφή μηνύματος για αύξηση, μείωση ή σταθερότητα τιμής.
    if diff > 0:
        return f"Η τιμή του ακινήτου αυξήθηκε κατά {diff:.2f} ευρώ."
    elif diff < 0:
        return f"Η τιμή του ακινήτου μειώθηκε κατά {abs(diff):.2f} ευρώ."
    else:
        return "Η τιμή του ακινήτου παρέμεινε αμετάβλητη."

def main():
    while True:
        # Εμφάνιση μενού στον χρήστη.
        print("\nΠαρακολούθηση τιμών ακινήτων")
        print("1. Πρόβλεψη μελλοντικής τιμής")
        print("2. Σύγκριση παρελθοντικής και μελλοντικής τιμής")
        print("3. Έξοδος")
        choice = input("Επιλέξτε επιλογή (1, 2 ή 3): ")
        if choice == '1':
            # Λήψη εισόδων από τον χρήστη για την πρόβλεψη τιμής.
            median_income = float(input("Δώστε το μέσο εισόδημα : "))
            total_rooms = float(input("Δώστε τον συνολικό αριθμό δωματίων: "))
            housing_median_age = float(input("Δώστε τη μέση ηλικία των κατοικιών: "))
            # Κλήση της συνάρτησης predict_price για την πρόβλεψη τιμής.
            predicted_price = predict_price(median_income, total_rooms, housing_median_age)
            print(f"Εκτιμώμενη τιμή: {predicted_price:.2f} ευρώ")
        elif choice == '2':
            # Λήψη παρελθοντικής τιμής και χαρακτηριστικών από τον χρήστη.
            past_value = float(input("Δώστε την παρελθοντική τιμή του ακινήτου: "))
            median_income = float(input("Δώστε το μέσο εισόδημα : "))
            total_rooms = float(input("Δώστε τον συνολικό αριθμό δωματίων: "))
            housing_median_age = float(input("Δώστε τη μέση ηλικία των κατοικιών: "))
            # Πρόβλεψη τιμής και σύγκριση με την παρελθοντική τιμή.
            predicted_price = predict_price(median_income, total_rooms, housing_median_age)
            comparison = compare_prices(past_value, predicted_price)
            print(f"{comparison}\nΕκτιμώμενη μελλοντική τιμή: {predicted_price:.2f} ευρώ")
        elif choice == '3':
            # Έξοδος από το πρόγραμμα.
            print("Έξοδος από το πρόγραμμα.")
            break

        else:
            # Μήνυμα για μη έγκυρη επιλογή.
            print("Μη έγκυρη επιλογή. Προσπαθήστε ξανά.")

if __name__ == "__main__":
    main()
