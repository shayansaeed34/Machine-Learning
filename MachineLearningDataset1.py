# ==========================================
# UCI DATASET - COMPLETE ML CYCLE + GRAPHS
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
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("epileptic seizure recognition.csv")

# Features
X = data.iloc[:, :-1]

# Remove non-numeric columns
X = X.select_dtypes(include=['number'])

# Target
y = data.iloc[:, -1]

# Convert to binary classification
# Class 1 = seizure
# Others = non-seizure
y = y.apply(lambda x: 1 if x != 1 else 0)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    # PR Curve
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    pr_auc = auc(recall, precision)

    return acc, f1, pr_auc


print("\n===== UCI FULL CYCLE =====")

# ==========================================
# PIPELINE A
# Normalization → Feature Selection → LR
# ==========================================

scaler_a = MinMaxScaler()

Xa_train = scaler_a.fit_transform(X_train)
Xa_test = scaler_a.transform(X_test)

selector = SelectKBest(
    f_classif,
    k=min(40, Xa_train.shape[1])
)

Xa_train = selector.fit_transform(Xa_train, y_train)
Xa_test = selector.transform(Xa_test)

model_a = LogisticRegression(max_iter=500)

model_a.fit(Xa_train, y_train)

result_a = evaluate(model_a, Xa_test, y_test)

print("Pipeline A:", result_a)

# ==========================================
# PIPELINE B
# Scaling → PCA → LR
# ==========================================

scaler_b = StandardScaler()

Xb_train = scaler_b.fit_transform(X_train)
Xb_test = scaler_b.transform(X_test)

pca = PCA(
    n_components=min(25, Xb_train.shape[1])
)

Xb_train = pca.fit_transform(Xb_train)
Xb_test = pca.transform(Xb_test)

model_b = LogisticRegression(max_iter=500)

model_b.fit(Xb_train, y_train)

result_b = evaluate(model_b, Xb_test, y_test)

print("Pipeline B:", result_b)

# ==========================================
# BASELINE MODEL
# ==========================================

base = LogisticRegression(max_iter=500)

base.fit(X_train, y_train)

result_base = evaluate(base, X_test, y_test)

print("Baseline:", result_base)

# ==========================================
# OVERFITTING / UNDERFITTING
# ==========================================

under = LogisticRegression(
    C=0.001,
    max_iter=500
)

over = LogisticRegression(
    C=1000,
    max_iter=500
)

under.fit(X_train, y_train)
over.fit(X_train, y_train)

result_under = evaluate(under, X_test, y_test)
result_over = evaluate(over, X_test, y_test)

print("Underfit:", result_under)
print("Overfit:", result_over)

# ==========================================
# REGULARIZATION STUDY
# ==========================================

models = {

    "L1": LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1,
        max_iter=500
    ),

    "L2": LogisticRegression(
        penalty='l2',
        C=1,
        max_iter=500
    ),

    "ElasticNet": LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=1,
        max_iter=1000
    )
}

print("\n===== REGULARIZATION =====")

reg_results = {}

for name, m in models.items():

    m.fit(X_train, y_train)

    result = evaluate(m, X_test, y_test)

    reg_results[name] = result

    print(name, result)

# ==========================================
# CLASS IMBALANCE HANDLING
# ==========================================

print("\n===== CLASS IMBALANCE =====")

cw = LogisticRegression(
    class_weight='balanced',
    max_iter=2000
)

cw.fit(X_train, y_train)

result_cw = evaluate(cw, X_test, y_test)

print("Class Weight:", result_cw)

# ==========================================
# GRAPH 1 - PIPELINE ACCURACY COMPARISON
# ==========================================

labels = [
    "Pipeline A",
    "Pipeline B",
    "Baseline"
]

accuracy_values = [
    result_a[0],
    result_b[0],
    result_base[0]
]

plt.figure(figsize=(8, 5))

plt.bar(labels, accuracy_values)

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

# ==========================================
# GRAPH 2 - F1 SCORE COMPARISON
# ==========================================

f1_values = [
    result_a[1],
    result_b[1],
    result_base[1]
]

plt.figure(figsize=(8, 5))

plt.bar(labels, f1_values)

plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")

plt.show()

# ==========================================
# GRAPH 3 - REGULARIZATION COMPARISON
# ==========================================

reg_labels = list(reg_results.keys())

reg_acc = [
    reg_results["L1"][0],
    reg_results["L2"][0],
    reg_results["ElasticNet"][0]
]

plt.figure(figsize=(8, 5))

plt.bar(reg_labels, reg_acc)

plt.title("Regularization Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

# ==========================================
# GRAPH 4 - LEARNING CURVE
# ==========================================

train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=2000),
    X,
    y,
    cv=3,
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))

plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')

plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

# ==========================================
# GRAPH 5 - CONFUSION MATRIX
# ==========================================

y_pred = base.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix - Baseline Model")

plt.show()

# ==========================================
# GRAPH 6 - PRECISION RECALL CURVE
# ==========================================

y_prob = base.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 5))

plt.plot(recall, precision)

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.show()