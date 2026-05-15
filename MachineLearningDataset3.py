# ==========================================
# FEDERATED DATASET - COMPLETE ML CYCLE
# + GRAPHS ADDED
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("epilepsy_federated_dataset.csv")
data = data.select_dtypes(include=[np.number])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================
# STORAGE FOR GRAPHS
# ==========================================

results = {
    "Model": [],
    "Accuracy": [],
    "F1": []
}

def store(name, acc, f1):
    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["F1"].append(f1)

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return acc, f1

print("\n===== FEDERATED FULL CYCLE =====")

# ==========================================
# PIPELINE A
# ==========================================

scaler_a = MinMaxScaler()
Xa_train = scaler_a.fit_transform(X_train)
Xa_test = scaler_a.transform(X_test)

k = min(10, Xa_train.shape[1])
selector = SelectKBest(f_classif, k=k)

Xa_train = selector.fit_transform(Xa_train, y_train)
Xa_test = selector.transform(Xa_test)

model_a = LogisticRegression(max_iter=500)
model_a.fit(Xa_train, y_train)

acc, f1 = evaluate(model_a, Xa_test, y_test)
print("Pipeline A:", (acc, f1))
store("Pipeline A", acc, f1)

# ==========================================
# PIPELINE B
# ==========================================

scaler_b = StandardScaler()
Xb_train = scaler_b.fit_transform(X_train)
Xb_test = scaler_b.transform(X_test)

n = min(10, Xb_train.shape[1])
pca = PCA(n_components=n)

Xb_train = pca.fit_transform(Xb_train)
Xb_test = pca.transform(Xb_test)

model_b = LogisticRegression(max_iter=500)
model_b.fit(Xb_train, y_train)

acc, f1 = evaluate(model_b, Xb_test, y_test)
print("Pipeline B:", (acc, f1))
store("Pipeline B", acc, f1)

# ==========================================
# BASELINE
# ==========================================

base = LogisticRegression(max_iter=500)
base.fit(X_train, y_train)

acc, f1 = evaluate(base, X_test, y_test)
print("Baseline:", (acc, f1))
store("Baseline", acc, f1)

# ==========================================
# UNDERFITTING
# ==========================================

under = LogisticRegression(C=0.001, max_iter=500)
under.fit(X_train, y_train)

acc, f1 = evaluate(under, X_test, y_test)
print("Underfit:", (acc, f1))
store("Underfit", acc, f1)

# ==========================================
# OVERFITTING
# ==========================================

over = LogisticRegression(C=1000, max_iter=500)
over.fit(X_train, y_train)

acc, f1 = evaluate(over, X_test, y_test)
print("Overfit:", (acc, f1))
store("Overfit", acc, f1)

# ==========================================
# REGULARIZATION
# ==========================================

models = {
    "L1": LogisticRegression(penalty='l1', solver='liblinear', C=1, max_iter=500),
    "L2": LogisticRegression(penalty='l2', C=1, max_iter=500),
    "ElasticNet": LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=1,
        max_iter=1000
    )
}

print("\n===== REGULARIZATION =====")

for name, m in models.items():
    m.fit(X_train, y_train)
    acc, f1 = evaluate(m, X_test, y_test)
    print(name, (acc, f1))
    store(name, acc, f1)

# ==========================================
# CLASS IMBALANCE
# ==========================================

print("\n===== CLASS IMBALANCE =====")

# SMOTE
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

model_sm = LogisticRegression(max_iter=500)
model_sm.fit(X_sm, y_sm)

acc, f1 = evaluate(model_sm, X_test, y_test)
print("SMOTE:", (acc, f1))
store("SMOTE", acc, f1)

# Undersampling
us = RandomUnderSampler(random_state=42)
X_us, y_us = us.fit_resample(X_train, y_train)

model_us = LogisticRegression(max_iter=500)
model_us.fit(X_us, y_us)

acc, f1 = evaluate(model_us, X_test, y_test)
print("Undersampling:", (acc, f1))
store("Undersampling", acc, f1)

# Class Weight
cw = LogisticRegression(class_weight='balanced', max_iter=500)
cw.fit(X_train, y_train)

acc, f1 = evaluate(cw, X_test, y_test)
print("Class Weight:", (acc, f1))
store("Class Weight", acc, f1)

# ==========================================
# GRAPHS
# ==========================================

# Accuracy Graph
plt.figure()
plt.bar(results["Model"], results["Accuracy"])
plt.xticks(rotation=45)
plt.title("Federated Dataset - Accuracy Comparison")
plt.tight_layout()
plt.show()

# F1 Score Graph
plt.figure()
plt.bar(results["Model"], results["F1"])
plt.xticks(rotation=45)
plt.title("Federated Dataset - F1 Score Comparison")
plt.tight_layout()
plt.show()