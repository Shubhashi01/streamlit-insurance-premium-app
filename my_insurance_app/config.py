import os

# Paths for artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
MODEL_PATH  = os.path.join(MODELS_DIR, 'tuned_dt.joblib')
