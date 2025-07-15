# rfr.py

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
import socket

# Load Fare Data
df_jeep = pd.read_csv("jeep_fare.csv")
df_bus = pd.read_csv("bus_fare.csv")

# Train Models
models = {
    'Jeep': {
        'Regular': RandomForestRegressor(n_estimators=100, random_state=42),
        'Discounted': RandomForestRegressor(n_estimators=100, random_state=42),
    },
    'Bus': {
        'Regular': RandomForestRegressor(n_estimators=100, random_state=42),
        'Discounted': RandomForestRegressor(n_estimators=100, random_state=42),
    }
}

# Jeep
X_jeep = df_jeep[['Distance (km)']].values
y_jeep_regular = df_jeep['Regular Fare (₱)'].to_numpy()
y_jeep_discounted = df_jeep['Discounted Fare (₱)'].to_numpy()
models['Jeep']['Regular'].fit(X_jeep, y_jeep_regular)
models['Jeep']['Discounted'].fit(X_jeep, y_jeep_discounted)

# Bus
X_bus = df_bus[['Distance (km)']].values
y_bus_regular = df_bus['Regular Fare (₱)'].to_numpy()
y_bus_discounted = df_bus['Discounted Fare (₱)'].to_numpy()
models['Bus']['Regular'].fit(X_bus, y_bus_regular)
models['Bus']['Discounted'].fit(X_bus, y_bus_discounted)

# Threshold Calculation
def calculate_threshold(model, X, y):
    predictions = model.predict(X)
    errors = abs(y - predictions)
    return np.mean(errors) + 3 * np.std(errors)

thresholds = {
    'Jeep': {
        'Regular': calculate_threshold(models['Jeep']['Regular'], X_jeep, y_jeep_regular),
        'Discounted': calculate_threshold(models['Jeep']['Discounted'], X_jeep, y_jeep_discounted),
    },
    'Bus': {
        'Regular': calculate_threshold(models['Bus']['Regular'], X_bus, y_bus_regular),
        'Discounted': calculate_threshold(models['Bus']['Discounted'], X_bus, y_bus_discounted),
    }
}

# Anomaly Check Function
def check_fare_anomaly(vehicle_type, distance_km, charged_fare, discounted):
    fare_type = 'Discounted' if discounted else 'Regular'
    model = models[vehicle_type][fare_type]
    threshold = thresholds[vehicle_type][fare_type]

    predicted_fare = model.predict([[distance_km]])[0]
    difference = abs(charged_fare - predicted_fare)
    is_anomalous = difference > threshold

    return {
        'vehicle_type': vehicle_type,
        'fare_type': fare_type,
        'predicted_fare': round(predicted_fare, 2),
        'charged_fare': round(charged_fare, 2),
        'difference': round(difference, 2),
        'threshold': round(threshold, 2),
        'is_anomalous': bool(is_anomalous),
    }

# Initialize App Hi
rfr_bp = Blueprint('rfr_bp', __name__)

@rfr_bp.route('/predict_fare', methods=['POST'])
def predict_fare():
    if request.json is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.json
    vehicle_type = data.get('vehicle_type')
    distance_km = float(data.get('distance_km', 0))
    charged_fare = float(data.get('charged_fare', 0))
    discounted = bool(data.get('discounted', False))

    if not vehicle_type or not distance_km or not charged_fare:
        return jsonify({"error": "Missing required fields"}), 400

    result = check_fare_anomaly(vehicle_type, distance_km, charged_fare, discounted)
    return jsonify(result)

# This part is no longer needed as app is run from app.py
# if __name__ == '__main__':
#     print(f"\n🧮 RFR backend running at: http://0.0.0.0:5002\n")
#     app.run(host='0.0.0.0', port=5001)
