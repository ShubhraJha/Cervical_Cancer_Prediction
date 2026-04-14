import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df=pd.read_csv("cervical_cancer.csv")
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.shape)
print(df.head())
print(df.describe().T)  