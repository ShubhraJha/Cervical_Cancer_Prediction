import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

df = pd.read_csv("cervical_cancer.csv")

print("Original Dataset Shape:",df.shape)

df = df.replace('?', np.nan)
df = df.apply(pd.to_numeric, errors='coerce') 
print(df.isnull().sum())
print("\nduplicate rows",df.duplicated().sum()) 
df.drop_duplicates(inplace=True)

print("Missing values handled and duplicates are dropped.") 
for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

std_cols = [col for col in df.columns if col.startswith('STDs:') and col not in ['STDs: Number of diagnosis','STDs: Time since first diagnosis','STDs: Time since last diagnosis']] 

df['Total_STDs'] = df[std_cols].sum(axis=1)
risk_features = ['Number of sexual partners','Smokes','Hormonal Contraceptives','IUD', 'STDs']

df['Risk_Score'] = df[risk_features].sum(axis=1)
print(df.corr()['Biopsy'].sort_values())

print("new features are 'Total_STDs' and 'Risk_Score'")
print("Final Dataset Shape:", df.shape) 

target = 'Biopsy'          
X = df.drop(columns=[target])
y = df[target]

print(f"\nTarget Distribution (Unbalanced):\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42,stratify=y)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

dt_model = DecisionTreeClassifier(random_state=42,max_depth=5,min_samples_split=10,min_samples_leaf=5)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
y_pred_proba = dt_model.predict_proba(X_test)[:,1] 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,zero_division=0)
recall = recall_score(y_test, y_pred,zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred) 
auc = roc_auc_score(y_test, y_pred_proba)


print("DECISION TREE RESULTS (80:20 Split)")
print("\n")
print(f"Accuracy   : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC-AUC Score : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Cancer (0)','Cancer (1)']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC ={auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--',label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve - Decision Tree (Unbalanced Dataset)')
plt.legend()
plt.grid(True)
plt.show()