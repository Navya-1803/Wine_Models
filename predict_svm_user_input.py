# predict_svm_user_input.py

import numpy as np
import pandas as pd
import joblib

# Feature names
feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# Take user input
print("🔢 Enter values for the following 13 wine features:")
user_input = []
for feature in feature_names:
    value = float(input(f"{feature.replace('_', ' ').title()}: "))
    user_input.append(value)

# Prepare input
user_data = np.array([user_input])
user_data_df = pd.DataFrame(user_data, columns=feature_names)

# Load model and scaler
scaler = joblib.load('svm_scaler.pkl')
svm = joblib.load('svm_model.pkl')

# Predict
user_data_scaled = scaler.transform(user_data_df)
prediction = svm.predict(user_data_scaled)

print(f"\n✅ Predicted Wine Class: {prediction[0]} (Sklearn-mapped class)")
