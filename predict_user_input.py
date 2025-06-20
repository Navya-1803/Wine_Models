# predict_user_input.py
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('wine_logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# List of wine features (must match training order)
feature_names = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]


print("\n🔢 Enter values for the following 13 wine features:")

user_input = []
for feature in feature_names:
    while True:
        try:
            val = float(input(f"{feature}: "))
            user_input.append(val)
            break
        except ValueError:
            print("❌ Please enter a valid number.")

# Prepare and scale the input
user_data = np.array(user_input).reshape(1, -1)

user_data_df = pd.DataFrame(user_data, columns=feature_names)
user_data_scaled = scaler.transform(user_data_df)

# Predict the class
prediction = model.predict(user_data_scaled)
print("\n✅ Predicted Wine Class:", prediction[0])
