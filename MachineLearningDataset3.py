# ==========================================
# FEDERATED DATASET - FIXED COMPLETE ML CYCLE
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("epilepsy_federated_dataset.csv")
data = data.select_dtypes(include=[np.number])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ✅ FIX: ensure binary classification for PR curve
if len(np.unique(y)) > 2:
    y = (y == np.unique(y)[0]).astype(int)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================================
# STORAGE
# ==========================================

results = {"Model": [], "Accuracy": [], "F1": []}

def store(name, acc, f1):
    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["F1"].append(f1)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    )

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

model_a = LogisticRegression(max_iter=2000, solver='lbfgs')
model_a.fit(Xa_train, y_train)

result_a = evaluate(model_a, Xa_test, y_test)
store("Pipeline A", *result_a)

print("Pipeline A:", result_a)

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

model_b = LogisticRegression(max_iter=2000, solver='lbfgs')
model_b.fit(Xb_train, y_train)

result_b = evaluate(model_b, Xb_test, y_test)
store("Pipeline B", *result_b)

print("Pipeline B:", result_b)

# ==========================================
# BASELINE
# ==========================================

base = LogisticRegression(max_iter=2000, solver='lbfgs')
base.fit(X_train, y_train)

result_base = evaluate(base, X_test, y_test)
store("Baseline", *result_base)

print("Baseline:", result_base)

# ==========================================
# UNDERFIT / OVERFIT
# ==========================================

under = LogisticRegression(C=0.001, max_iter=2000, solver='lbfgs')
over = LogisticRegression(C=1000, max_iter=2000, solver='lbfgs')

under.fit(X_train, y_train)
over.fit(X_train, y_train)

acc_under, f1_under = evaluate(under, X_test, y_test)
acc_over, f1_over = evaluate(over, X_test, y_test)

# ==========================================
# REGULARIZATION
# ==========================================

models = {
    "L1": LogisticRegression(penalty='l1', solver='liblinear', C=1, max_iter=2000),
    "L2": LogisticRegression(penalty='l2', C=1, max_iter=2000),
    "ElasticNet": LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=1,
        max_iter=2000
    )
}

reg_results = {}

for name, m in models.items():
    m.fit(X_train, y_train)
    reg_results[name] = evaluate(m, X_test, y_test)

# ==========================================
# CLASS IMBALANCE
# ==========================================

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

sm_model = LogisticRegression(max_iter=2000, solver='lbfgs')
sm_model.fit(X_sm, y_sm)
sm_acc, sm_f1 = evaluate(sm_model, X_test, y_test)

us = RandomUnderSampler(random_state=42)
X_us, y_us = us.fit_resample(X_train, y_train)

us_model = LogisticRegression(max_iter=2000, solver='lbfgs')
us_model.fit(X_us, y_us)
us_acc, us_f1 = evaluate(us_model, X_test, y_test)

cw = LogisticRegression(class_weight='balanced', max_iter=2000, solver='lbfgs')
cw.fit(X_train, y_train)
cw_acc, cw_f1 = evaluate(cw, X_test, y_test)

# ==========================================
# GRAPHS (CLEAN BEED STYLE)
# ==========================================

# 1 Accuracy
plt.figure(figsize=(8,5))
plt.bar(results["Model"], results["Accuracy"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# 2 F1
plt.figure(figsize=(8,5))
plt.bar(results["Model"], results["F1"])
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.show()

# 3 Under vs Overfit
plt.figure(figsize=(6,5))
plt.bar(["Underfit", "Overfit"], [acc_under, acc_over])
plt.title("Underfitting vs Overfitting")
plt.ylabel("Accuracy")
plt.show()

# 4 Regularization
plt.figure(figsize=(7,5))
plt.bar(list(reg_results.keys()),
        [reg_results[k][0] for k in reg_results])
plt.title("Regularization Comparison")
plt.ylabel("Accuracy")
plt.show()

# 5 Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=2000),
    X,
    y,
    cv=3,
    scoring='accuracy'
)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation")
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 6 Confusion Matrix
y_pred = base.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix - Baseline")
plt.show()

# 7 Class Imbalance
plt.figure(figsize=(8,5))
plt.bar(
    ["Baseline", "SMOTE", "Under", "Class Weight"],
    [result_base[0], sm_acc, us_acc, cw_acc]
)
plt.title("Class Imbalance Handling")
plt.ylabel("Accuracy")
plt.show()

# ==========================================
# FINAL OUTPUTS
# ==========================================

print("\n===== FINAL RESULTS =====")
print("Pipeline A:", result_a)
print("Pipeline B:", result_b)
print("Baseline:", result_base)
print("SMOTE:", (sm_acc, sm_f1))
print("Undersampling:", (us_acc, us_f1))
print("Class Weight:", (cw_acc, cw_f1))