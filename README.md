# Metric–Text Score Prediction (DA5401 End-Semester Challenge)

This repository contains the full machine learning pipeline developed for the  
**DA5401 – Data Analytics Laboratory** end-semester metric-learning challenge.

The goal of the competition is to predict a **fitness score (0–10)** describing how well a  
metric definition aligns with a `(prompt, response)` text pair.  
The dataset is multilingual (Hindi, Tamil, Assamese, Bengali, Bodo, Sindhi, English)  
and heavily **skewed toward high scores**, making the problem challenging.

---

## 🔍 Project Summary

### **1. Text Processing & Embeddings**
- All three text fields (`system`, `user`, `response`) are cleaned:
  - Unicode normalization (NFKC)
  - Removal of control characters
  - Whitespace collapsing
- Combined into:
system_prompt [SEP] user_prompt [SEP] response

markdown
Copy code
- Encoded using **google/embeddinggemma-300m** (768-dim embeddings)  
with `max_seq_length=8192`.

Both train and test embeddings are saved as `.npy` files.

---

## ✨ Feature Engineering

For each row, four embedding blocks are concatenated:

- `Metric embedding`  
- `Text embedding`  
- `|Metric – Text|` (absolute difference)  
- `Metric * Text` (elementwise product)

This gives:  
4 × 768 = 3072 features

arduino
Copy code

Additionally, 12 scalar engineered features were added (cosine similarity, norms, text lengths, etc.), giving a final:

3084-dimensional feature vector

yaml
Copy code

All features are scaled with `StandardScaler`.

---

## ⚠️ Handling Extreme Class Imbalance

The score distribution is:

Score 10 → 1442
Score 9 → 3123
Score 8 → 259
Score 0–7 → 176 total

yaml
Copy code

To address this imbalance:

### **✓ Synthetic Hard Negatives**
High-score rows (≥ 9) are paired with a *different* metric to create artificial “misaligned’’ examples labeled **0**.

### **✓ Soft Negatives**
Random metric/text mismatches to mimic noisy pairs.

These are used **only for classifier training** — regressors use only the true dataset.

---

## 🧠 Two-Stage Model Architecture

### **Stage 1: PyTorch Classifier**
Binary target:
GOOD = score ≥ 8
BAD = score ≤ 7

diff
Copy code

Architecture:
- 3084-dim input
- Hidden layers: 1024 → 512 → 128
- LayerNorm, ReLU, Dropout(0.3)
- BCEWithLogits loss with sample weighting
- 5-fold StratifiedKFold
- Early stopping

Output:
p_good = P(score ≥ 8)

yaml
Copy code

### **Stage 2: Two Specialized LightGBM Regressors**
- **GOOD Regressor:** trained only on real data with scores ≥ 8  
- **BAD Regressor:** trained only on real data with scores ≤ 7  
- Both evaluated with 5-fold CV  
- No synthetic samples used here

---

## 🔀 Hybrid Routing (Final Prediction)

Given:

p = classifier probability
g = GOOD regressor prediction
b = BAD regressor prediction

markdown
Copy code

We evaluate:

- **Soft routing:**  
  `y_soft = p*g + (1-p)*b`
- **Hard routing:**  
  g if p>0.5 else b
- **Hybrid (final):**  
  soft routing generally, but forced hard-switch near confident extremes

This hybrid model achieves the best RMSE.

---

## 📈 Performance Summary

### **Classifier (5-fold OOF)**
- AUC ≈ **0.93**
- Accuracy ≈ **0.86**

### **GOOD Regressor**
- RMSE ≈ **0.49**

### **BAD Regressor**
- RMSE ≈ **1.83** (expected due to low sample size)

### **Final Hybrid Model**
- Produces smooth and stable predictions
- Outperforms both pure hard- and pure soft-routing

---
