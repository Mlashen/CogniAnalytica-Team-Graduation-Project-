import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model, encoders, and feature names
rf_model = joblib.load("best_heart_model.pkl")
label_encoders = joblib.load("label_encoders_heart_attack.joblib")
model_features = joblib.load("model_features.pkl")

def preprocess_input(inputs: dict):
    input_df = pd.DataFrame([inputs])

    # Process categorical data using label encoders
    for col in model_features:
        if col in label_encoders:
            le = label_encoders[col]
            val = input_df.at[0, col]
            if val in le.classes_:
                input_df.at[0, col] = le.transform([val])[0]
            else:
                input_df.at[0, col] = le.transform([le.classes_[0]])[0]
        else:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    input_df = input_df[model_features]
    return input_df

def main():
    st.markdown("<h1 style='color:#a83279; text-align:center;'>Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#a83279; text-align:center;'>Based on health indicators and lifestyle data</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar Inputs
    st.sidebar.header("📋 Input Data")

    # Options for categorical fields
    Sex_options = list(label_encoders['Sex'].classes_)
    AgeCategory_options = list(label_encoders['AgeCategory'].classes_)
    RaceEthnicityCategory_options = list(label_encoders['RaceEthnicityCategory'].classes_)
    LastCheckupTime_options = list(label_encoders['LastCheckupTime'].classes_)
    RemovedTeeth_options = list(label_encoders['RemovedTeeth'].classes_)
    ECigaretteUsage_options = list(label_encoders['ECigaretteUsage'].classes_)
    TetanusLast10Tdap_options = list(label_encoders['TetanusLast10Tdap'].classes_)

    # Input fields
    inputs = {
        "Sex": st.sidebar.selectbox("Sex", Sex_options),
        "GeneralHealth": st.sidebar.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']),
        "PhysicalHealthDays": st.sidebar.slider("Physical health days (last 30)", 0, 30, 5),
        "MentalHealthDays": st.sidebar.slider("Mental health days (last 30)", 0, 30, 5),
        "LastCheckupTime": st.sidebar.selectbox("Last Medical Checkup", LastCheckupTime_options),
        "PhysicalActivities": st.sidebar.selectbox("Do you engage in physical activity?", ['Yes', 'No']),
        "SleepHours": st.sidebar.slider("Sleep Hours", 0.0, 24.0, 7.0),
        "RemovedTeeth": st.sidebar.selectbox("Have teeth been removed?", RemovedTeeth_options),
        "HadAngina": st.sidebar.selectbox("Had Angina?", ['Yes', 'No']),
        "HadStroke": st.sidebar.selectbox("Had Stroke?", ['Yes', 'No']),
        "HadAsthma": st.sidebar.selectbox("Had Asthma?", ['Yes', 'No']),
        "HadSkinCancer": st.sidebar.selectbox("Had Skin Cancer?", ['Yes', 'No']),
        "HadCOPD": st.sidebar.selectbox("Had COPD?", ['Yes', 'No']),
        "HadDepressiveDisorder": st.sidebar.selectbox("Had Depressive Disorder?", ['Yes', 'No']),
        "HadKidneyDisease": st.sidebar.selectbox("Had Kidney Disease?", ['Yes', 'No']),
        "HadArthritis": st.sidebar.selectbox("Had Arthritis?", ['Yes', 'No']),
        "HadDiabetes": st.sidebar.selectbox("Had Diabetes?", ['Yes', 'No']),
        "DeafOrHardOfHearing": st.sidebar.selectbox("Deaf or Hard of Hearing?", ['Yes', 'No']),
        "BlindOrVisionDifficulty": st.sidebar.selectbox("Blind or Vision Difficulty?", ['Yes', 'No']),
        "DifficultyConcentrating": st.sidebar.selectbox("Difficulty Concentrating?", ['Yes', 'No']),
        "DifficultyWalking": st.sidebar.selectbox("Difficulty Walking?", ['Yes', 'No']),
        "DifficultyDressingBathing": st.sidebar.selectbox("Difficulty Dressing or Bathing?", ['Yes', 'No']),
        "DifficultyErrands": st.sidebar.selectbox("Difficulty with Errands?", ['Yes', 'No']),
        "SmokerStatus": st.sidebar.selectbox("Smoking Status", ['Never smoked', 'Former smoker', 'Current smoker - daily', 'Current smoker - some days']),
        "ECigaretteUsage": st.sidebar.selectbox("Use E-Cigarettes?", ECigaretteUsage_options),
        "ChestScan": st.sidebar.selectbox("Had Chest Scan?", ['Yes', 'No']),
        "RaceEthnicityCategory": st.sidebar.selectbox("Race/Ethnicity", RaceEthnicityCategory_options),
        "AgeCategory": st.sidebar.selectbox("Age Category", AgeCategory_options),
        "HeightInMeters": st.sidebar.number_input("Height (in meters)", min_value=1.0, max_value=2.5, value=1.70),
        "WeightInKilograms": st.sidebar.number_input("Weight (in kg)", min_value=30.0, max_value=250.0, value=70.0),
        "BMI": st.sidebar.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0),
        "AlcoholDrinkers": st.sidebar.selectbox("Do you drink alcohol?", ['Yes', 'No']),
        "HIVTesting": st.sidebar.selectbox("Tested for HIV?", ['Yes', 'No']),
        "FluVaxLast12": st.sidebar.selectbox("Received flu vaccine in the last 12 months?", ['Yes', 'No']),
        "PneumoVaxEver": st.sidebar.selectbox("Ever received pneumonia vaccine?", ['Yes', 'No']),
        "TetanusLast10Tdap": st.sidebar.selectbox("Received tetanus shot in the last 10 years?", TetanusLast10Tdap_options),
        "HighRiskLastYear": st.sidebar.selectbox("Were you at high risk last year?", ['Yes', 'No']),
        "CovidPos": st.sidebar.selectbox("Ever tested positive for COVID-19?", ['Yes', 'No'])
    }

    # Convert 'Yes'/'No' responses to 1/0
    yes_no_cols = [
        "PhysicalActivities", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer", "HadCOPD", 
        "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis", "HadDiabetes", "DeafOrHardOfHearing",
        "BlindOrVisionDifficulty", "DifficultyConcentrating", "DifficultyWalking", "DifficultyDressingBathing",
        "DifficultyErrands", "ChestScan", "AlcoholDrinkers", "HIVTesting", "FluVaxLast12", "PneumoVaxEver",
        "HighRiskLastYear", "CovidPos"
    ]

    for col in yes_no_cols:
        inputs[col] = 1 if inputs[col] == 'Yes' else 0

    if st.button("Predict"):
        input_df = preprocess_input(inputs)
        prediction = rf_model.predict(input_df)[0]
        result = "High risk of heart attack" if prediction == 1 else "Low risk of heart attack"
        st.markdown("---")
        st.markdown("### 💓 Prediction Result:") 
        if prediction == 1:
            st.error(result)
        else:
            st.success(result)
        st.markdown("---")

    st.markdown("<small style='color:gray; text-align:center;'>This tool helps predict the risk of a heart attack based on user health data.</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
