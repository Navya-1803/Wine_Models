# adaboost_model.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print("Dataset shape:", X.shape)
print("Target classes:", np.unique(y))

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'ab_scaler.pkl')

# Train AdaBoost
ab = AdaBoostClassifier(n_estimators=100, random_state=42)
ab.fit(X_train_scaled, y_train)
joblib.dump(ab, 'ab_model.pkl')

# Evaluate
print("\nAdaBoost Validation Accuracy:", accuracy_score(y_val, ab.predict(X_val_scaled)))
print("Validation Report:\n", classification_report(y_val, ab.predict(X_val_scaled)))
print("Test Accuracy:", accuracy_score(y_test, ab.predict(X_test_scaled)))
print("Test Report:\n", classification_report(y_test, ab.predict(X_test_scaled)))
