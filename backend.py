from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# Update the static folder to point to /Users/rathnamsundaram/frontend3
app = Flask(__name__, static_folder='/Users/rathnamsundaram/frontend3')

# Enable CORS for the entire app
CORS(app)

def load_data(file_name):
    base_path = '/Users/rathnamsundaram/Downloads/archive'
    file_path = os.path.join(base_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    return pd.read_csv(file_path)

# Route for the homepage to serve startpredicting.html
@app.route('/')
def home():
    static_folder = app.static_folder or '/Users/rathnamsundaram/frontend3'
    return send_from_directory(directory=static_folder, path='startpredicting.html')

# Route for other HTML pages
@app.route('/<page>')
def serve_page(page):
    try:
        static_folder = app.static_folder or '/Users/rathnamsundaram/frontend3'
        return send_from_directory(directory=static_folder, path=f'{page}.html')
    except Exception as e:
        return f"Page {page} not found.", 404

# Route to serve drivers data
@app.route('/drivers', methods=['GET'])
def get_drivers():
    try:
        drivers = load_data('drivers.csv')
        driver_names = drivers.loc[:, ['driverId', 'name']].to_dict(orient='records')
        return jsonify(driver_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve circuits data
@app.route('/circuits', methods=['GET'])
def get_circuits():
    try:
        circuits = load_data('circuits.csv')
        circuit_names = circuits.loc[:, ['circuitId', 'name']].to_dict(orient='records')
        return jsonify(circuit_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve races data
@app.route('/races', methods=['GET'])
def get_races():
    try:
        races = load_data('races.csv')
        race_names = races.loc[:, ['raceId', 'name']].to_dict(orient='records')
        return jsonify(race_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve pit stops data
@app.route('/pit_stops', methods=['GET'])
def get_pit_stops():
    try:
        pit_stops = load_data('pit_stops.csv')
        pit_stop_data = pit_stops.loc[:, ['raceId', 'driverId', 'stop']].to_dict(orient='records')
        return jsonify(pit_stop_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve lap times data
@app.route('/lap_times', methods=['GET'])
def get_lap_times():
    try:
        lap_times = load_data('lap_times.csv')
        lap_time_data = lap_times.loc[:, ['raceId', 'driverId', 'lap', 'milliseconds']].to_dict(orient='records')
        return jsonify(lap_time_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve driver standings data
@app.route('/driver_standings', methods=['GET'])
def get_driver_standings():
    try:
        driver_standings = load_data('driver_standings.csv')
        standings_data = driver_standings.loc[:, ['driverId', 'points', 'position']].to_dict(orient='records')
        return jsonify(standings_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to train the model
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Load CSVs
        circuits = load_data('circuits.csv')
        races = load_data('races.csv')
        drivers = load_data('drivers.csv')
        lap_times = load_data('lap_times.csv')

        # Merge datasets
        lap_race = lap_times.merge(races, on='raceId', how='left')
        lap_driver = lap_race.merge(drivers, on='driverId', how='left')
        lap_full = lap_driver.merge(circuits, on='circuitId', how='left')

        # Filter for years >= 2015
        lap_full = lap_full[lap_full['year'] >= 2015]

        # Calculate driver age
        lap_full['driver_age'] = lap_full['year'] - pd.Series(lap_full['dob']).astype(str).str[:4].astype(int)

        # Convert milliseconds to seconds
        lap_full['lap_time_sec'] = lap_full['milliseconds'] / 1000.0

        # Select features and target
        features = lap_full[['driverId', 'raceId', 'circuitId', 'driver_age', 'lap', 'position']]
        target = lap_full['lap_time_sec']

        # One-hot encoding
        features = pd.get_dummies(features, columns=['driverId', 'raceId', 'circuitId'], drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Model training
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        # Return MAE in the response
        return jsonify({"message": "Model trained successfully", "mean_absolute_error": mae})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)