# Pokemon Legendary Classification


Binary classification of Legendary vs Non-Legendary Pokemon using ensemble machine learning on a severely imbalanced dataset (~8% minority class).

---

## Problem Statement

Legendary Pokemon make up only **65 out of 800** (~8%) of the dataset. A naive model predicting "Non-Legendary" for everything would hit 91.8% accuracy while completely failing the task. This project tackles that imbalance using two parallel strategies — `class_weight='balanced'` and SMOTE oversampling — across four ensemble classifiers.

---

## Dataset

**Source:** [Pokemon Dataset — Kaggle](https://www.kaggle.com/datasets/abcsds/pokemon)  
**File:** `pokemon.csv` (place in the same directory before running)

| Feature | Description |
|---|---|
| Type 1 / Type 2 | Primary and secondary Pokemon type |
| HP, Attack, Defense | Base battle stats |
| Sp. Atk, Sp. Def, Speed | Special stats |
| Generation | Game generation (1–6) |
| Legendary | **Target variable** (True / False) |

> `#`, `Name`, and `Total` were dropped — identifiers and a multicollinear aggregate.

---

## Models Trained

Four ensemble classifiers, each trained under two imbalance-handling strategies = **8 models total**.

| Model | Strategy |
|---|---|
| Random Forest | Baseline + SMOTE |
| Voting Classifier (RF + CNB + SVC + LR) | Baseline + SMOTE |
| AdaBoost | Baseline + SMOTE |
| Stacking Classifier (same base + LR meta) | Baseline + SMOTE |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **AdaBoost (Baseline)** ⭐ | 0.9313 | 0.5455 | **0.9231** | **0.6857** | 0.9320 |
| Stacking (Baseline) | 0.9187 | 0.5000 | 0.9231 | 0.6486 | 0.9623 |
| Voting (Baseline) | 0.9062 | 0.4615 | 0.9231 | 0.6154 | 0.9644 |
| Voting (SMOTE) | 0.9375 | 0.6000 | 0.6923 | 0.6429 | 0.9592 |
| AdaBoost (SMOTE) | 0.9375 | 0.6000 | 0.6923 | 0.6429 | **0.9686** |
| Stacking (SMOTE) | 0.9375 | **0.7143** | 0.3846 | 0.5000 | 0.9602 |
| RF (SMOTE) | 0.9187 | 0.5000 | 0.6154 | 0.5517 | 0.9628 |
| RF (Baseline) | 0.9125 | 0.4737 | 0.6923 | 0.5625 | 0.9608 |

**Best model: AdaBoost (Baseline)** — F1 = 0.6857, caught **12 out of 13** legendary Pokemon (recall = 0.92). Missing a legendary is worse than a false alarm, so high recall is prioritised.

---

## Key Findings

- **Baseline (class_weight + threshold=0.30)** → high recall, lower precision. Best when misses are costly.
- **SMOTE** → consistently raised precision across all models (+0.03 to +0.21), but at the cost of recall.
- **All models scored ROC-AUC > 0.93**, confirming strong class separation regardless of strategy.
- Accuracy is misleading here — the 11:1 class ratio means non-legendary samples dominate it.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place pokemon.csv in the repo root (download from Kaggle link above)

# 4. Open the notebook
jupyter notebook DT_003_DM_SpecialAssignment.ipynb
```

> The notebook was originally developed in Google Colab. If running locally, change the dataset path from `/content/pokemon.csv` to `pokemon.csv` in the second cell.

---

## Project Structure

```
├── DT_003_DM_SpecialAssignment.ipynb   # Main notebook (EDA + modeling + evaluation)
├── Pokemon_Legendary_Classification_Report.pdf  # Full project report
├── pokemon.csv                         # Dataset (download from Kaggle)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Tech Stack

`Python` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `pandas` · `numpy` · `matplotlib` · `seaborn`
