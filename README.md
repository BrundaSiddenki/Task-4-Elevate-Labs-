# Task-4-Elevate-Labs-
This project builds a binary classification model using Logistic Regression to predict whether a tumor is malignant or benign. It includes data cleaning, feature scaling, model training, evaluation using accuracy, precision, recall, F1-score, and ROC-AUC, and presents a visual dashboard with confusion matrix, ROC curve, and feature importances.
# 🧠 Logistic Regression Classification Dashboard

This project demonstrates how to build a **binary classification model** using **Logistic Regression** to predict whether a tumor is **malignant (M)** or **benign (B)** using the Breast Cancer Wisconsin Dataset.

---

## 📌 Objective

To apply logistic regression for medical diagnosis and visualize model performance using metrics and dashboard plots.

---

## 📊 Dataset

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Target Variable**: `diagnosis` (Malignant → 1, Benign → 0)
- **Features**: 30 numerical measurements (e.g., radius, texture, symmetry)

---

## 🚀 Project Steps

1. **Load and Clean Data**  
   - Removed irrelevant columns (`id`, `Unnamed: 32`)  
   - Encoded diagnosis (`M` = 1, `B` = 0)

2. **Preprocessing**  
   - Standardized features using `StandardScaler`

3. **Model Training**  
   - Used `LogisticRegression` from `scikit-learn`

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix  
   - ROC Curve and AUC Score  

5. **Visualization Dashboard**  
   - ROC Curve  
   - Top 10 Feature Importances  
   - Confusion Matrix  

6. **Threshold Tuning**  
   - Tested custom threshold values to balance Precision & Recall

7. **Sigmoid Function Explanation**  
   - Visualized how logistic regression converts scores to probabilities

---

## 📈 Output Dashboard

- 📉 ROC Curve  
- 📊 Confusion Matrix  
- 🔍 Feature Importance Chart  
- 🧮 Sigmoid Function Plot

---

## 📦 Requirements

Install required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
