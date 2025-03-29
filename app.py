from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["features"]
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)  # Apply scaling
    prediction = model.predict(data)[0]
    return jsonify({"churn_prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
