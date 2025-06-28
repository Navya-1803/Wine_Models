# predict_knn_user_input.py

import numpy as np
import pandas as pd
import joblib

# Define the feature names in exact order
feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# Collect user input
print(" Enter values for the following 13 wine features:")
user_input = []
for feature in feature_names:
    value = float(input(f"{feature.replace('_', ' ').title()}: "))
    user_input.append(value)

# Convert input to DataFrame with proper column names
user_data = np.array([user_input])  # shape (1, 13)
user_data_df = pd.DataFrame(user_data, columns=feature_names)

# Load scaler and model
scaler = joblib.load('knn_scaler.pkl')
model = joblib.load('knn_model.pkl')

# Scale the input and predict
user_data_scaled = scaler.transform(user_data_df)
prediction = model.predict(user_data_scaled)

# Show prediction
print(f"\n Predicted Wine Class: {prediction[0]} (Sklearn-mapped class)")
