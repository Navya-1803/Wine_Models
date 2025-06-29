# naive_bayes_model.py

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print("Dataset shape:", X.shape)
print("Target classes:", np.unique(y))

# 2. Split: 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'nb_scaler.pkl')

# 4. Train Naive Bayes model
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
joblib.dump(nb, 'nb_model.pkl')

# 5. Evaluate model
y_val_pred = nb.predict(X_val_scaled)
print("\nNaive Bayes Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Report:\n", classification_report(y_val, y_val_pred))

y_test_pred = nb.predict(X_test_scaled)
print("Naive Bayes Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Report:\n", classification_report(y_test, y_test_pred))
