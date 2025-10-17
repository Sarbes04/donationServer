from flask import Flask, request, jsonify
from flask_cors import CORS # <--- NEW: Import CORS to resolve 403 error
from datetime import datetime
from geopy.distance import geodesic
import numpy as np
import joblib
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app) # <--- NEW: Initialize CORS to allow all cross-origin requests

# MongoDB setup
MONGODB_URL = os.getenv("MONGODB_URL")
if not MONGODB_URL:
    # In a deployed environment, this should log and fail gracefully
    raise ValueError("MONGODB_URL not found in environment variables") 

# It's safer to define the database name explicitly if possible, or use a default
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "HungerBridge")
client = MongoClient(MONGODB_URL)
db = client[MONGO_DB_NAME]
receivers_collection = db["receivers"]

# Encoding for food type
food_type_encoding = {
    "cooked": 0,
    "fresh": 1,
    "dry": 2
}

# Load urgency model
model_path = "urgency_model.pkl"
if not os.path.exists(model_path):
    # This check ensures the deployed file system is correct
    raise FileNotFoundError("❌ Trained model file 'urgency_model.pkl' not found.")

model = joblib.load(model_path)

# Get NGO list from MongoDB
def get_ngos_from_db():
    ngos = []
    # Note: Using .find() can be slow for large collections. Consider using a cursor and batching if performance is an issue.
    for ngo in receivers_collection.find():
        try:
            # Handling potential slight variations in field names
            lat = float(ngo["location"].get("latitude") or ngo["location"].get("lattitude") or 0)
            lon = float(ngo["location"].get("longitude") or 0)
            
            if lat and lon:
                ngos.append({
                    "id": str(ngo["_id"]),
                    "name": ngo["name"],
                    "location": (lat, lon),
                    # Ensure default active_load is 0
                    "active_load": ngo.get("active_load", 0) 
                })
        except Exception as e:
            print(f"Skipping NGO due to data error in DB: {e}")
            continue
    return ngos

# Predict urgency score using model
def predict_urgency(data):
    try:
        # Use datetime.fromisoformat for robust time parsing
        expiry_time = datetime.fromisoformat(data["expiry_time"])
        now = datetime.utcnow()
        # Calculate hours remaining, ensure it's not negative (use max(..., 0.1) to avoid division by zero later if needed)
        hours_left = max((expiry_time - now).total_seconds() / 3600.0, 0.1) 

        food_type = data["food_type"].lower()
        if food_type not in food_type_encoding:
            raise ValueError(f"Invalid food type provided: '{food_type}'. Must be Cooked, Fresh, or Dry.")

        food_type_val = food_type_encoding[food_type]
        quantity = float(data["quantity"])

        # Prepare features for the ML model
        features = np.array([[food_type_val, quantity, hours_left]])
        predicted_score = model.predict(features)[0]

        # The score should not exceed 100
        return round(min(predicted_score, 100), 2)
    except Exception as e:
        # Re-raise as ValueError to be caught by the route handler and return 400
        raise ValueError(f"Urgency prediction failed due to input issue: {e}")

def match_ngos(donor_location, urgency_score):
    ngos = get_ngos_from_db()
    matches = []
    for ngo in ngos:
        distance_km = geodesic(donor_location, ngo["location"]).km
        
        # Filter by distance
        if distance_km <= 20: 
            # Match Score Formula: Proximity + Urgency - Load
            # The +1 in the denominator prevents division by zero if distance_km is 0
            match_score = (100 / (distance_km + 1)) + urgency_score - (10 * ngo["active_load"])
            matches.append({
                "id": ngo["id"],
                "name": ngo["name"],
                "distance_km": round(distance_km, 2),
                "match_score": round(match_score, 2)
            })
    return sorted(matches, key=lambda x: x["match_score"], reverse=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ML Service Operational",
        "endpoints": ["POST /predict-urgency"],
        "message": "Use /predict-urgency (POST) to get predictions."
    })

@app.route("/predict-urgency", methods=["POST"])
def predict():
    try:
        # Use silent=True to avoid automatic 400 if JSON is invalid, allowing custom error handling
        data = request.get_json(silent=True)
        
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload in request."}), 400

        # Validate required fields
        required_keys = ["food_type", "quantity", "expiry_time"]
        if not all(key in data for key in required_keys):
            return jsonify({"error": f"Missing required fields: {', '.join(required_keys)}"}), 400
            
        # Validate location structure
        if 'location' not in data or 'lat' not in data['location'] or 'lon' not in data['location']:
            return jsonify({"error": "Missing location data (required keys: location.lat, location.lon)."}), 400

        # Extract and convert location
        donor_location = (
            float(data["location"]["lat"]),
            float(data["location"]["lon"])
        )

        urgency = predict_urgency(data)
        ngo_matches = match_ngos(donor_location, urgency)

        return jsonify({
            "urgency_score": urgency,
            "matched_ngos": ngo_matches
        })
    
    except ValueError as e:
        # Catches validation errors from predict_urgency and missing keys
        print(f"Input Validation Error: {e}")
        return jsonify({"error": str(e)}), 400
        
    except Exception as e:
        # Catches uncaught server errors (e.g., MongoDB connection issues, unexpected model load failure)
        print(f"Unhandled Server Error during prediction: {e}")
        return jsonify({"error": "Internal Server Error during prediction.", "details": str(e)}), 500

@app.errorhandler(404)
def resource_not_found(e):
    # Custom 404 handler
    return jsonify({"error": "Resource not found. Check the endpoint URL."}), 404
    
if __name__ == '__main__':
    # Use environment PORT provided by Render or default to 5001 (Render suggests using $PORT)
    port = int(os.environ.get('PORT', 5001)) 
    app.run(host='0.0.0.0', port=port)
