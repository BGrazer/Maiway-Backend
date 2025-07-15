import numpy as np
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

def analyze_route_with_reference_model():
    print("🔍 Running crowd anomaly analysis using RFR and grouping by route...")

    # 🚨 Load official fare matrix (jeep_fare.csv)
    try:
        df_ref = pd.read_csv("jeep_fare.csv")
        df_ref.columns = df_ref.columns.str.strip()
    except Exception as e:
        print(f"❌ Failed to load jeep_fare.csv: {e}")
        return

    # 🧠 Train Random Forest Regressor (RFR)
    X_train = df_ref[['Distance (km)']].values
    y_train = np.array(df_ref['Regular Fare (₱)'].values)  # Ensure y_train is a NumPy array

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 🔎 Calculate anomaly threshold
    ref_predictions = model.predict(X_train)
    ref_errors = abs(ref_predictions - y_train)
    threshold = np.mean(ref_errors) + 3 * np.std(ref_errors)

    print(f"✅ Anomaly threshold based on fare matrix: ₱{threshold:.2f}")

    # 🧩 Fetch survey documents from local JSON file
    try:
        with open('data/crowd_analysisjson.json', 'r') as f:
            surveys = json.load(f)
    except Exception as e:
        print(f"❌ Failed to fetch surveys from local file: {e}")
        return

    # 📦 Group entries by route
    route_data = defaultdict(list)

    for data in surveys:
        try:
            route = data.get("route")
            distance = float(data.get("distance"))
            fare_given = float(data.get("fare_given"))

            if route:
                route_data[route].append({
                    "distance": distance,
                    "fare_given": fare_given,
                    "timestamp": data.get("timestamp", datetime.now().isoformat())
                })
        except Exception as e:
            print(f"⚠️ Skipping invalid document: {e}")
            continue

    if not route_data:
        print("⚠️ No valid entries found.")
        return

    # 📊 Analyze each route
    print("\n📊 ROUTE-BASED OVERCHARGE ANALYSIS")

    for route, entries in route_data.items():
        distances = [entry['distance'] for entry in entries]
        fares = [entry['fare_given'] for entry in entries]

        X = np.array(distances).reshape(-1, 1)
        predicted = model.predict(X)
        errors = abs(predicted - fares)

        overcharged = sum(e > threshold for e in errors)
        total = len(entries)
        ratio = overcharged / total if total > 0 else 0

        # 🏷️ Label
        if total < 5:
            tag = "⚪ Not enough data"
        elif ratio >= 0.5:
            tag = "🟥 OVERCHARGE ALERT"
        elif ratio >= 0.3:
            tag = "🟠 Warning"
        else:
            tag = "🟢 Normal"

        # 📋 Print route summary
        print(f"\n🚏 Route: {route}")
        print(f"📦 Reports: {total}")
        print(f"❗ Overcharged: {overcharged}")
        print(f"📈 Overcharge Ratio: {ratio:.2%}")
        print(f"💰 Avg Reported Fare: ₱{np.mean(fares):.2f}")
        print(f"📏 Avg Distance: {np.mean(distances):.2f} km")
        print(f"🏷️ Status: {tag}")

# 🚀 Run it
if __name__ == "__main__":
    analyze_route_with_reference_model()