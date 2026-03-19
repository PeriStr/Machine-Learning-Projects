import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Δημιουργία ασαφών μεταβλητών:
# 1. 'car_count' αντιπροσωπεύει τον αριθμό των αυτοκινήτων που αναμένουν στο φανάρι.
# 2. 'arrival_rate' αντιπροσωπεύει τον ρυθμό άφιξης νέων αυτοκινήτων.
# 3. 'green_time' αντιπροσωπεύει τον χρόνο που το φανάρι παραμένει πράσινο.
car_count = ctrl.Antecedent(np.arange(0, 101, 1), 'car_count')
arrival_rate = ctrl.Antecedent(np.arange(0, 11, 1), 'arrival_rate')
green_time = ctrl.Consequent(np.arange(0, 121, 1), 'green_time')

# Ορισμός ασαφών συνόλων για την κάθε μεταβλητή:
# Χρησιμοποιούνται τριγωνικές συναρτήσεις συμμετοχής (trimf).
# 'car_count' διαχωρίζεται σε τρία σύνολα: 'low', 'medium', και 'high'.
car_count['low'] = fuzz.trimf(car_count.universe, [0, 0, 50])
car_count['medium'] = fuzz.trimf(car_count.universe, [25, 50, 75])
car_count['high'] = fuzz.trimf(car_count.universe, [50, 100, 100])

# 'arrival_rate' διαχωρίζεται σε τρία σύνολα: 'slow', 'normal', και 'fast'.
arrival_rate['slow'] = fuzz.trimf(arrival_rate.universe, [0, 0, 5])
arrival_rate['normal'] = fuzz.trimf(arrival_rate.universe, [2, 5, 8])
arrival_rate['fast'] = fuzz.trimf(arrival_rate.universe, [5, 10, 10])

# 'green_time' διαχωρίζεται σε τρία σύνολα: 'short', 'medium', και 'long'.
green_time['short'] = fuzz.trimf(green_time.universe, [0, 0, 60])
green_time['medium'] = fuzz.trimf(green_time.universe, [30, 60, 90])
green_time['long'] = fuzz.trimf(green_time.universe, [60, 120, 120])

# Ορισμός κανόνων για την ασαφή λογική:
# - Αν υπάρχουν πολλά αυτοκίνητα ('high') και ο ρυθμός άφιξης είναι υψηλός ('fast'),
#   τότε ο χρόνος πράσινου φωτός πρέπει να είναι μεγάλος ('long').
rule1 = ctrl.Rule(car_count['high'] & arrival_rate['fast'], green_time['long'])

# - Αν υπάρχουν μέτρια αυτοκίνητα ('medium') και ο ρυθμός είναι φυσιολογικός ('normal'),
#   τότε ο χρόνος πράσινου πρέπει να είναι μέτριος ('medium').
rule2 = ctrl.Rule(car_count['medium'] & arrival_rate['normal'], green_time['medium'])

# - Αν υπάρχουν λίγα αυτοκίνητα ('low') και ο ρυθμός άφιξης είναι χαμηλός ('slow'),
#   τότε ο χρόνος πράσινου πρέπει να είναι μικρός ('short').
rule3 = ctrl.Rule(car_count['low'] & arrival_rate['slow'], green_time['short'])

# Προσθήκη κανόνα για να διαχειριστούμε απρόβλεπτες καταστάσεις:
# - Αν υπάρχουν μέτρια αυτοκίνητα ('medium') και ο ρυθμός είναι χαμηλός ('slow'),
#   τότε ο χρόνος πράσινου είναι μέτριος ('medium').
rule_default = ctrl.Rule(car_count['medium'] & arrival_rate['slow'], green_time['medium'])

# Δημιουργία συστήματος ελέγχου που συνδυάζει όλους τους κανόνες.
traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule_default])
traffic_sim = ctrl.ControlSystemSimulation(traffic_ctrl)

# Συνάρτηση για την προσομοίωση της κυκλοφορίας στη διασταύρωση.
def simulate_traffic(initial_counts, arrival_rates, cycles=10):
    """
    Προσομοιώνει την κυκλοφορία για ένα συγκεκριμένο αριθμό κύκλων.
    
    Parameters:
        initial_counts : Το αρχικό πλήθος αυτοκινήτων σε κάθε κατεύθυνση.
        arrival_rates : Ο ρυθμός άφιξης αυτοκινήτων σε κάθε κατεύθυνση.
        cycles : Ο αριθμός κύκλων της προσομοίωσης.
    """
    for cycle in range(cycles):  # Για κάθε κύκλο της προσομοίωσης.
        print(f"Cycle {cycle + 1}:")  # Εμφάνιση του τρέχοντος κύκλου.
        for direction in ['N', 'S', 'E', 'W']:  # Για κάθε κατεύθυνση:
            # Εισαγωγή του πλήθους αυτοκινήτων και του ρυθμού άφιξης στον ελεγκτή.
            traffic_sim.input['car_count'] = initial_counts[direction]
            traffic_sim.input['arrival_rate'] = arrival_rates[direction]
            
            try:
                # Εκτέλεση του συστήματος ελέγχου για υπολογισμό του χρόνου πράσινου.
                traffic_sim.compute()
                green_time_value = traffic_sim.output['green_time']  # Αποθήκευση εξόδου.
            except KeyError:
                # Σε περίπτωση λάθους, δίνεται προεπιλεγμένος χρόνος πράσινου (30 δευτερόλεπτα).
                green_time_value = 30
            print(f"  Direction {direction}: Green Time = {green_time_value:.2f} sec")           
            # Υπολογισμός αυτοκινήτων που πέρασαν και αυτοκινήτων που προστέθηκαν.
            cars_passed = min(initial_counts[direction], int(green_time_value // 2))  # Όσα μπορούν να περάσουν.
            # Χρησημοποείται η κατανομή Poisson γιατί είναι κατάλληλη για τυχαία γεγονότα που συμβαίνουν με σταθερό μέσο ρυθμό.
            initial_counts[direction] = max(0, initial_counts[direction] - cars_passed + np.random.poisson(arrival_rates[direction]))
            print(f"    Remaining Cars: {initial_counts[direction]}")
        print()  

# Συνάρτηση για είσοδο δεδομένων από τον χρήστη.
def get_user_input():
    """
    Λαμβάνει από τον χρήστη τα αρχικά δεδομένα για το πλήθος αυτοκινήτων και τους ρυθμούς άφιξης.
    
    Returns:
        initial_counts : Το αρχικό πλήθος αυτοκινήτων σε κάθε κατεύθυνση.
        arrival_rates : Ο ρυθμός άφιξης σε κάθε κατεύθυνση.
    """
    print("Enter the initial car counts and arrival rates for each direction:")
    initial_counts = {}  # Λεξικό για το πλήθος αυτοκινήτων.
    arrival_rates = {}  # Λεξικό για τους ρυθμούς άφιξης.
    for direction in ['N', 'S', 'E', 'W']:  # Για κάθε κατεύθυνση:
        # Εισαγωγή του πλήθους αυτοκινήτων.
        initial_counts[direction] = int(input(f"Initial car count for direction {direction}: "))
        # Εισαγωγή του ρυθμού άφιξης αυτοκινήτων.
        arrival_rates[direction] = float(input(f"Arrival rate for direction {direction} (cars per time unit): "))
    return initial_counts, arrival_rates  # Επιστροφή των δεδομένων.

# Εκτέλεση: Λήψη δεδομένων από τον χρήστη και προσομοίωση της κυκλοφορίας.
initial_counts, arrival_rates = get_user_input()
simulate_traffic(initial_counts, arrival_rates)
