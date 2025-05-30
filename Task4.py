# ------------------ 1. IMPORT LIBRARIES ------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, RocCurveDisplay
)

# ------------------ 2. LOAD & CLEAN DATA ------------------
df = pd.read_csv("archive\data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# ------------------ 3. FEATURE SCALING ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ 4. TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------ 5. LOGISTIC REGRESSION MODEL ------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------ 6. PREDICTIONS ------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ------------------ 7. METRICS ------------------
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# ------------------ 8. FEATURE IMPORTANCE ------------------
coefficients = pd.Series(model.coef_[0], index=X.columns)
top_features = coefficients.abs().sort_values(ascending=False).head(10)

# ------------------ 9. SIGMOID FUNCTION ------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z_values)

plt.figure(figsize=(6, 4))
plt.plot(z_values, sigmoid_values, color='green')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ 10. DASHBOARD (No Correlation Matrix) ------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# A. ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axs[0])
axs[0].plot([0, 1], [0, 1], 'k--')
axs[0].set_title(f"ROC Curve (AUC = {roc_auc:.3f})")

# B. Feature Importance
top_features.sort_values().plot(kind='barh', color='teal', ax=axs[1])
axs[1].set_title("Top 10 Feature Importances")

# C. Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axs[2])
axs[2].set_title("Confusion Matrix")
axs[2].set_xlabel("Predicted")
axs[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ------------------ 11. THRESHOLD TUNING ------------------
custom_threshold = 0.3
y_custom = (y_proba >= custom_threshold).astype(int)
custom_precision = precision_score(y_test, y_custom)
custom_recall = recall_score(y_test, y_custom)
print(f"\nCustom Threshold = {custom_threshold}")
print(f"Precision = {custom_precision:.3f}, Recall = {custom_recall:.3f}")

# ------------------ 12. PREDICT ON NEW SAMPLE ------------------
sample = X_test[0].reshape(1, -1)
prob = model.predict_proba(sample)[0][1]
label = model.predict(sample)[0]
print(f"\nPrediction on a new test sample:")
print(f"Probability of Malignant: {prob:.3f}, Predicted Class: {label}")
