# random_forest_model.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print("Dataset shape:", X.shape)
print("Target classes:", np.unique(y))

# Split into 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'rf_scaler.pkl')

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
joblib.dump(rf, 'rf_model.pkl')

# Evaluate
print("\nRandom Forest Validation Accuracy:", accuracy_score(y_val, rf.predict(X_val_scaled)))
print("Validation Report:\n", classification_report(y_val, rf.predict(X_val_scaled)))
print("Test Accuracy:", accuracy_score(y_test, rf.predict(X_test_scaled)))
print("Test Report:\n", classification_report(y_test, rf.predict(X_test_scaled)))
