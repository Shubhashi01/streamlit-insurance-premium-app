import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from config import SCALER_PATH

FEATURE_ORDER = [
    'Age', 'Diabetes', 'BloodPressureProblems',
    'AnyTransplants', 'AnyChronicDiseases',
    'KnownAllergies', 'HistoryOfCancerInFamily',
    'NumberOfMajorSurgeries', 'BMI'
]

# Data preparation for inference

def preprocess_input(raw_json: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_json])
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df = df.drop(columns=['Height', 'Weight'])

    scaler = joblib.load(SCALER_PATH)
    df[['Age','BMI','NumberOfMajorSurgeries']] = scaler.transform(
        df[['Age','BMI','NumberOfMajorSurgeries']]
    )
    return df[FEATURE_ORDER]