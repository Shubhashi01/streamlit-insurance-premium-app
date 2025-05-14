import streamlit as st
import pandas as pd
import joblib

# Load artifacts
scaler = joblib.load('models/scaler.joblib')
model  = joblib.load('models/tuned_dt.joblib')

st.title("🛡️ Health Insurance Premium Estimator")

# Build the form
with st.form("input_form"):
    age    = st.slider("Age", 18, 66, 30)
    height = st.number_input("Height (cm)", 145, 188, 168)
    weight = st.number_input("Weight (kg)", 50, 132, 75)
    diabetes    = st.checkbox("Diabetes")
    bp          = st.checkbox("Blood Pressure Problems")
    transplant  = st.checkbox("Any Transplants")
    chronic     = st.checkbox("Any Chronic Diseases")
    allergies   = st.checkbox("Known Allergies")
    fam_cancer  = st.checkbox("Family History of Cancer")
    surgeries   = st.slider("Number of Major Surgeries", 0, 3, 0)
    submitted   = st.form_submit_button("Estimate Premium")

if submitted:
    # prepare DataFrame
    df = pd.DataFrame([{
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Diabetes": int(diabetes),
        "BloodPressureProblems": int(bp),
        "AnyTransplants": int(transplant),
        "AnyChronicDiseases": int(chronic),
        "KnownAllergies": int(allergies),
        "HistoryOfCancerInFamily": int(fam_cancer),
        "NumberOfMajorSurgeries": surgeries
    }])
    # compute BMI
    df['BMI'] = df['Weight'] / ((df['Height']/100)**2)
    df = df.drop(columns=['Height','Weight'])
    # scale
    df[['Age','BMI','NumberOfMajorSurgeries']] = scaler.transform(
        df[['Age','BMI','NumberOfMajorSurgeries']])
    # predict
    premium = model.predict(df)[0]
    st.success(f"Estimated Annual Premium: ₹{premium:,.2f}")
