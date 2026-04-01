# 🚦 Fuzzy Logic Traffic Light Controller
### Adaptive Traffic Signal Timing with scikit-fuzzy

---

## 📌 Description

A fuzzy logic system that dynamically controls **traffic light green time** at a 4-way intersection.
Instead of fixed timers, the system adapts the green light duration based on how many cars are waiting and how fast new cars are arriving — mimicking the decision-making of a human traffic controller.

A **multi-cycle simulation** runs the intersection for N cycles, updating car counts at each step using a **Poisson arrival process**.

---

## 📂 Project Structure

```
project/
│
├── main.py      # Fuzzy system definition + simulation + CLI input
└── README.md
```

---

## ⚙️ Installation

```bash
pip install numpy scikit-fuzzy
```

---

## 🧠 How Fuzzy Logic Works Here

Traditional traffic lights use fixed timers. This system uses **fuzzy logic** to make smarter decisions:

```
Inputs (crisp numbers)
  │
  ├── car_count    → How many cars are waiting (0–100)
  └── arrival_rate → How fast new cars are arriving (0–10)
        │
        ▼
  Fuzzification (map crisp values to fuzzy sets)
        │
        ▼
  Rule Evaluation (IF...THEN rules)
        │
        ▼
  Defuzzification (convert fuzzy output to a crisp number)
        │
        ▼
Output
  └── green_time → How long the light stays green (0–120 sec)
```

---

## 🔷 Fuzzy Variables

### Input: `car_count` (0 – 100 cars)

| Set | Range | Meaning |
|---|---|---|
| `low` | 0 – 50 | Few cars waiting |
| `medium` | 25 – 75 | Moderate queue |
| `high` | 50 – 100 | Heavy queue |

### Input: `arrival_rate` (0 – 10 cars/unit)

| Set | Range | Meaning |
|---|---|---|
| `slow` | 0 – 5 | Light incoming traffic |
| `normal` | 2 – 8 | Moderate incoming traffic |
| `fast` | 5 – 10 | Heavy incoming traffic |

### Output: `green_time` (0 – 120 seconds)

| Set | Range | Meaning |
|---|---|---|
| `short` | 0 – 60 | Brief green phase |
| `medium` | 30 – 90 | Standard green phase |
| `long` | 60 – 120 | Extended green phase |

All membership functions use **triangular functions (trimf)**.

---

## 📋 Fuzzy Rules

| Rule | Condition | Result |
|---|---|---|
| Rule 1 | car_count = HIGH **AND** arrival_rate = FAST | green_time = LONG |
| Rule 2 | car_count = MEDIUM **AND** arrival_rate = NORMAL | green_time = MEDIUM |
| Rule 3 | car_count = LOW **AND** arrival_rate = SLOW | green_time = SHORT |
| Rule 4 (default) | car_count = MEDIUM **AND** arrival_rate = SLOW | green_time = MEDIUM |

---

## 🔄 Simulation Logic

```
For each cycle (default: 10):
│
├── For each direction (N, S, E, W):
│   ├── Feed car_count and arrival_rate into fuzzy controller
│   ├── Compute green_time output
│   ├── Cars that pass = min(waiting, green_time ÷ 2)
│   ├── New arrivals = Poisson(arrival_rate)   ← random, realistic
│   └── Updated count = waiting - passed + new_arrivals
│
└── Print results for each direction
```

**Why Poisson distribution for arrivals?**
Car arrivals at an intersection are random events occurring at a roughly constant average rate — exactly what the Poisson distribution models.

---

## ▶️ How to Run

```bash
python main.py
```

The program prompts for input per direction:

```
Enter the initial car counts and arrival rates for each direction:
Initial car count for direction N: 40
Arrival rate for direction N (cars per time unit): 3.5
Initial car count for direction S: 70
...
```

**Example output:**
```
Cycle 1:
  Direction N: Green Time = 52.40 sec
    Remaining Cars: 28
  Direction S: Green Time = 89.75 sec
    Remaining Cars: 41
  ...
```

---

## 📊 Design Decisions

| Decision | Reason |
|---|---|
| Triangular membership functions | Simple, computationally lightweight, interpretable |
| Poisson arrivals | Realistic model for random traffic flow |
| `green_time ÷ 2` throughput | Conservative estimate: not all cars clear on every green |
| Default rule (Rule 4) | Prevents system crash on edge-case inputs not covered by rules 1–3 |
| `try/except KeyError` fallback | Graceful handling if fuzzy output can't be computed → defaults to 30 sec |

---

## 📦 Requirements

```
numpy
scikit-fuzzy
```
