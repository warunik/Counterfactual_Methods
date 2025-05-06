import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
# Replace with your actual dataset path
df = pd.read_csv('Datasets/taiwan.csv')  # Expected to have 96 features and a 'class' column (0 or 1)

# --- Features and target ---
X = df.drop(columns='Bankrupt?')
y = df['Bankrupt?']

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- SMOTE Oversampling ---
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# ===========================
# 1️⃣ RANDOM FOREST MODEL
# ===========================
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

# ===========================
# 2️⃣ XGBOOST MODEL
# ===========================
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=5, random_state=42)
xgb.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = xgb.predict(X_test_scaled)
y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

# --- Evaluation Function ---
def evaluate_model(name, y_test, y_pred, y_proba):
    print(f"\n📊 Evaluation Report: {name}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

# --- Run Evaluations ---
plt.figure(figsize=(8, 6))
evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

feature_cols = X.columns.tolist()

# 1) Extract importances
rf_imps  = rf.feature_importances_
xgb_imps = xgb.feature_importances_

# 2) Build a DataFrame of feature vs. importance
import pandas as pd

rf_imp_df = (
    pd.DataFrame({'feature': feature_cols, 'importance': rf_imps})
      .sort_values('importance', ascending=False)
      .reset_index(drop=True)
)

xgb_imp_df = (
    pd.DataFrame({'feature': feature_cols, 'importance': xgb_imps})
      .sort_values('importance', ascending=False)
      .reset_index(drop=True)
)

# 3) Print the top 10
print("Top 10 features by Random Forest:")
print(rf_imp_df.head(10), "\n")

print("Top 10 features by XGBoost:")
print(xgb_imp_df.head(10), "\n")

# 4) Plot them
def plot_top(df, model_name, top_n=10):
    top = df.head(top_n)
    plt.figure(figsize=(8, 4))
    plt.bar(top['feature'], top['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{model_name} Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

plot_top(rf_imp_df,  'Random Forest')
plot_top(xgb_imp_df, 'XGBoost')

