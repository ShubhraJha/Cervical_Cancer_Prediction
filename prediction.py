import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ================================================
# STEP 1: LOAD THE DATASET
# ================================================
df = pd.read_csv("cervical_cancer.csv")

print("Original Dataset Shape:", df.shape)

# Replace '?' with NaN and convert to numeric
df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric, errors='coerce')

# Handle missing values with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print("Missing values handled.")

# ================================================
# STEP 2: MINIMAL FEATURE ENGINEERING (Only Useful Features)
# ================================================

# 1. Total STDs count (very useful for cervical cancer risk)
std_cols = [col for col in df.columns if col.startswith('STDs:') and 
            col not in ['STDs: Number of diagnosis', 
                        'STDs: Time since first diagnosis', 
                        'STDs: Time since last diagnosis']]
df['Total_STDs'] = df[std_cols].sum(axis=1)

# 2. Behavioral Risk Score (combines key risk factors)
risk_features = ['Number of sexual partners', 'Smokes', 'Hormonal Contraceptives', 
                 'IUD', 'STDs']
df['Risk_Score'] = df[risk_features].sum(axis=1)

print("Minimal Feature Engineering done: Added 'Total_STDs' and 'Risk_Score'")
print("Final Dataset Shape:", df.shape)

# ================================================
# STEP 3: DEFINE FEATURES AND TARGET
# ================================================
target = 'Biopsy'          # Target: Cancer diagnosis (0 or 1)
X = df.drop(columns=[target])
y = df[target]

print(f"\nTarget Distribution (Unbalanced):\n{y.value_counts()}")

# ================================================
# STEP 4: TRAIN-TEST SPLIT (80:20 stratified)
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y          # Important for unbalanced dataset
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ================================================
# STEP 5: DECISION TREE CLASSIFIER
# ================================================
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,              # Limited depth to prevent overfitting on unbalanced data
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced'   # Helps with class imbalance
)

dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)

# ================================================
# STEP 6: EVALUATION
# ================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*55)
print("DECISION TREE RESULTS (80:20 Split)")
print("="*55)
print(f"Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Cancer (0)', 'Cancer (1)']))

# Top Important Features
importances = pd.Series(dt_model.feature_importances_, index=X.columns)
print("\nTop 10 Important Features:")
print(importances.sort_values(ascending=False).head(10))