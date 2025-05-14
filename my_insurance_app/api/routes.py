from flask import Blueprint, request, jsonify
from preprocessing.data_engineering import preprocess_input
import joblib
from config import MODEL_PATH

bp = Blueprint('api', __name__)
# load model once
dt_model = joblib.load(MODEL_PATH)

@bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({ 'error': 'No JSON payload provided' }), 400
    try:
        X = preprocess_input(data)
        pred = dt_model.predict(X)[0]
        return jsonify({ 'estimated_premium': float(pred) })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 422