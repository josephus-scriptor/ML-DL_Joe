# ========================
# Tabular Classification with RandomForest & LightGBM (Colab-Ready)
# Based on: Project_ML-DL_Group_01_v6.ipynb + GitHub dataset
# ========================

# SECTION: Data Processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
url = 'https://raw.githubusercontent.com/josephus-scriptor/ML-DL_Joe/main/data.csv'
df = pd.read_csv(url)

# Generate cross-features and binarized flags
for i in range(1, 31):
    for j in range(i + 1, 31):
        df[f'V{i}_x_V{j}'] = df[f'V{i}'] * df[f'V{j}']
    df[f'V{i}_gt_mean'] = (df[f'V{i}'] > df[f'V{i}'].mean()).astype(int)

# Target and features
y = df['OBJ']
X = df.drop(columns=['OBJ'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SECTION: Models Used
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# Define classifiers
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
lgb_clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')

# SECTION: Hyperparameter Configuration
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
}

param_grid_lgb = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.05, 0.1],
}

# Grid search for RF
rf_grid = GridSearchCV(
    rf_clf, param_grid_rf, scoring='balanced_accuracy', cv=5, n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)

# Grid search for LGBM
lgb_grid = GridSearchCV(
    lgb_clf, param_grid_lgb, scoring='balanced_accuracy', cv=5, n_jobs=-1
)
lgb_grid.fit(X_train_scaled, y_train)

# SECTION: Evaluation and Metrics
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             precision_recall_curve, classification_report, auc)

def evaluate_model(name, model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    bal_acc = balanced_accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    print(f"--- {name} ---")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, preds))
    print()

# Run evaluations
evaluate_model("Random Forest", rf_grid.best_estimator_, X_test_scaled, y_test)
evaluate_model("LightGBM", lgb_grid.best_estimator_, X_test_scaled, y_test)

# SECTION: Final Model Selected
# Selected based on: Balanced Accuracy + PR-AUC + ROC-AUC
# If LGBM performs better, you may serialize it:
# import joblib
# joblib.dump(lgb_grid.best_estimator_, 'final_lgb_model.pkl')
