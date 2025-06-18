import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(
    page_title="CogniAnalytica Team",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.heart.org',
        'Report a bug': "mailto:support@heartguard.ai",
        'About': "### HeartGuard AI\nAdvanced Heart Attack Risk Assessment Tool"
    }
)

# Load model and preprocessor with caching
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_assets()

# --- Modern CSS Styling with Glassmorphism Effect ---
st.markdown("""
    <style>
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --danger: #ff6b6b;
            --success: #51cf66;
            --warning: #fcc419;
            --info: #22b8cf;
            --light: #f8f9fa;
            --dark: #343a40;
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
        }
        .main {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.15);
            padding: 3rem;
            margin: 2rem 0;
        }
        h1 {
            color: #2a3f5f;
            font-family: 'Inter', sans-serif;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        h2 {
            color: #2a3f5f;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            margin-top: 2rem;
            border-bottom: 2px solid rgba(42, 63, 95, 0.1);
            padding-bottom: 0.5rem;
        }
        h3 {
            color: #2a3f5f;
            font-weight: 600;
        }
        label, .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #2a3f5f !important;
            font-weight: 500;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            font-weight: 600;
            border-radius: 14px;
            padding: 0.75rem 2rem;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 15px rgba(102, 126, 234, 0.3);
        }
        .stButton > button:active {
            transform: translateY(1px);
        }
        .metric-box {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .prediction-box-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 24px rgba(255, 107, 107, 0.3);
            animation: pulse 2s infinite;
        }
        .prediction-box-low {
            background: linear-gradient(135deg, #51cf66 0%, #8ce99a 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 24px rgba(81, 207, 102, 0.3);
        }
        .prediction-box-medium {
            background: linear-gradient(135deg, #fcc419 0%, #ffd43b 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 24px rgba(252, 196, 25, 0.3);
        }
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(0,0,0,0.1);
        }
        .stProgress > div > div > div {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        }
        .stMarkdown {
            line-height: 1.7;
        }
        .risk-factors {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            padding: 1.75rem;
            margin: 1.75rem 0;
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        }
        .tab-content {
            padding: 1.25rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white !important;
            font-weight: 600;
        }
        .stTabs [aria-selected="false"] {
            background: rgba(255, 255, 255, 0.8);
            color: #6c757d;
        }
        .stForm {
            border-radius: 20px;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.7);
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        }
        .stExpander {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            border: 1px solid rgba(0,0,0,0.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        }
        .stExpander .streamlit-expanderHeader {
            font-weight: 600;
            color: #2a3f5f;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .tooltip-icon {
            color: var(--primary);
            margin-left: 5px;
            cursor: pointer;
        }
        .feature-importance-plot {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .risk-meter {
            width: 100%;
            height: 30px;
            background: linear-gradient(90deg, #51cf66 0%, #fcc419 50%, #ff6b6b 100%);
            border-radius: 15px;
            margin: 1rem 0;
            position: relative;
        }
        .risk-meter-indicator {
            position: absolute;
            height: 40px;
            width: 4px;
            background: #2a3f5f;
            top: -5px;
            transform: translateX(-50%);
        }
        .risk-meter-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: #6c757d;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            border-left: 4px solid var(--primary);
        }
        .feature-card-title {
            font-weight: 600;
            color: #2a3f5f;
            margin-bottom: 0.5rem;
        }
        .feature-card-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--secondary);
        }
        .feature-card-impact {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .impact-high {
            background-color: #ffebee;
            color: #c62828;
        }
        .impact-medium {
            background-color: #fff8e1;
            color: #f57f17;
        }
        .impact-low {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .recommendation-card {
            background: rgba(255, 255, 255, 0.9);
            color: #2d3748;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            border-left: 4px solid var(--info);
        }
        .recommendation-card-title {
            font-weight: 600;
            color: var(--info);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
        }
        .recommendation-card-title svg {
            margin-right: 0.5rem;
        }
        .heart-animation {
            animation: heartbeat 1.5s ease-in-out infinite;
        }
        @keyframes heartbeat {
            0% { transform: scale(1); }
            25% { transform: scale(1.1); }
            50% { transform: scale(1); }
            75% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# --- App Container ---
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    # Header with logo and animation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <h1>
            <span style="color: #667eea">Heart</span>
            <span style="color: #764ba2">Guard</span> 
            <span style="color: #ff6b6b" class="heart-animation">❤</span> 
            AI
        </h1>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center; color: #6c757d; margin-bottom: 2rem; font-size: 1.1rem;">
            Advanced Heart Attack Risk Assessment and Prevention Tool
        </p>
        """, unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📊 Risk Assessment", "💡 Health Insights"])

    with tab1:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 👤 Personal Information")
                sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                age_category = st.selectbox("Age Category", [
                    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
                ], help="Select your age range")
                
                # Enhanced state selector with region grouping
                states_by_region = {
                    "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", 
                                  "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"],
                    "Midwest": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", 
                               "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", 
                               "North Dakota", "South Dakota"],
                    "South": ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", 
                             "South Carolina", "Virginia", "West Virginia", "Alabama", 
                             "Kentucky", "Mississippi", "Tennessee", "Arkansas", 
                             "Louisiana", "Oklahoma", "Texas"],
                    "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", 
                            "New Mexico", "Utah", "Wyoming", "Alaska", "California", 
                            "Hawaii", "Oregon", "Washington"]
                }
                
                region = st.selectbox("Region", list(states_by_region.keys()))
                state = st.selectbox("State", states_by_region[region])

                st.markdown("### ⚖ Physical Metrics")
                weight = st.number_input("Weight (kg)", 30.0, 200.0, step=0.5, value=70.0, 
                                       help="Enter your weight in kilograms")
                height = st.number_input("Height (m)", 1.0, 2.5, step=0.01, value=1.75, 
                                      help="Enter your height in meters")
                
                # Enhanced BMI Calculation with visual indicator
                bmi = round(weight / (height ** 2), 2) if height > 0 else 0
                bmi_status = ""
                bmi_color = ""
                if bmi < 18.5:
                    bmi_status = "Underweight"
                    bmi_color = "#2196f3"  # Blue
                elif 18.5 <= bmi < 25:
                    bmi_status = "Normal"
                    bmi_color = "#4caf50"  # Green
                elif 25 <= bmi < 30:
                    bmi_status = "Overweight"
                    bmi_color = "#ff9800"  # Orange
                else:
                    bmi_status = "Obese"
                    bmi_color = "#f44336"  # Red
                
                st.markdown(f"""
                <div class="metric-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <div>
                            <div style="font-size: 0.9rem; color: #6c757d;">Body Mass Index</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: {bmi_color};">{bmi}</div>
                        </div>
                        <div style="background: {bmi_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">
                            {bmi_status}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <div style="height: 10px; background: #e0e0e0; border-radius: 5px; overflow: hidden; position: relative;">
                            <div style="height: 100%; width: {min(100, max(0, (bmi - 15) / 30 * 100))}%; 
                                background: linear-gradient(90deg, #2196f3 0%, #4caf50 18.5%, #ff9800 25%, #f44336 100%); 
                                border-radius: 5px;"></div>
                            <div style="position: absolute; top: -5px; left: 18.5%; width: 1px; height: 20px; background: #2a3f5f;"></div>
                            <div style="position: absolute; top: -5px; left: 25%; width: 1px; height: 20px; background: #2a3f5f;"></div>
                            <div style="position: absolute; top: -5px; left: 30%; width: 1px; height: 20px; background: #2a3f5f;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: #6c757d;">
                            <span>15</span>
                            <span>18.5</span>
                            <span>25</span>
                            <span>30</span>
                            <span>45</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 🏥 Health History")
                general_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"],
                                            help="How would you rate your general health?")
                had_stroke = st.selectbox("Had Stroke", ["Yes", "No"], 
                                         help="Have you ever been told you had a stroke?")
                had_angina = st.selectbox("Had Angina", ["Yes", "No"], 
                                         help="Have you ever been told you had angina or coronary artery disease?")
                smoker = st.selectbox("Smoker Status", ["Never smoked", "Former smoker", "Current smoker"],
                                     help="Select your smoking status")
                removed_teeth = st.selectbox("Removed Teeth", ["None", "1 to 5", "6 or more but not all", "All"],
                                           help="How many permanent teeth have been removed due to tooth decay or gum disease?")
                tetanus = st.selectbox("Tetanus Vaccine Last 10 Years", ["Yes", "No"],
                                     help="Have you had a tetanus shot in the last 10 years?")

                st.markdown("### 🏃 Lifestyle Factors")
                sleep_hours = st.slider("Average Sleep Hours per Day", 0, 24, 7, 
                                      help="Recommended 7-9 hours for adults")
                
                # Enhanced sliders with visual indicators
                physical_days = st.slider("Days with Physical Health Issues (Last 30 Days)", 0, 30, 5,
                                        help="How many days during the past 30 days was your physical health not good?")
                if physical_days > 10:
                    st.warning("Frequent physical health issues may indicate underlying conditions")
                
                mental_days = st.slider("Days with Mental Health Issues (Last 30 Days)", 0, 30, 3,
                                      help="How many days during the past 30 days was your mental health not good?")
                if mental_days > 10:
                    st.warning("Frequent mental health issues may impact cardiovascular health")

            # Form submit button with icon
            submitted = st.form_submit_button("🔍 Assess My Heart Attack Risk", 
                                            help="Click to analyze your risk factors")
            
            if submitted:
                with st.spinner("🔍 Analyzing your risk factors..."):
                    # Simulate processing time with better progress bar
                    import time
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for percent_complete in range(101):
                        time.sleep(0.02)
                        progress_bar.progress(percent_complete)
                        status_text.text(f"Processing... {percent_complete}%")
                        
                        if percent_complete == 30:
                            status_text.text("Analyzing personal information...")
                        elif percent_complete == 60:
                            status_text.text("Evaluating health history...")
                        elif percent_complete == 90:
                            status_text.text("Calculating final risk assessment...")
                    
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    # Prepare input data
                    input_dict = {
                        "HadAngina": had_angina,
                        "BMI": bmi,
                        "WeightInKilograms": weight,
                        "HeightInMeters": height,
                        "AgeCategory": age_category,
                        "SleepHours": sleep_hours,
                        "PhysicalHealthDays": physical_days,
                        "TetanusLast10Tdap": tetanus,
                        "GeneralHealth": general_health,
                        "MentalHealthDays": mental_days,
                        "RemovedTeeth": removed_teeth,
                        "SmokerStatus": smoker,
                        "HadStroke": had_stroke,
                        "Sex": sex,
                        "State": state
                    }

                    df_input = pd.DataFrame([input_dict])
                    df_transformed = preprocessor.transform(df_input)
                    prediction = model.predict(df_transformed)[0]
                    
                    # Get prediction probabilities if available
                    try:
                        probabilities = model.predict_proba(df_transformed)[0]
                        risk_score = round(probabilities[1] * 100, 1)
                    except:
                        risk_score = 75 if prediction == "Yes" else 25

                    st.markdown("---")
                    
                    # Enhanced risk display with meter
                    st.markdown(f"""
                    <div style="margin-bottom: 2rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #2a3f5f;">Your Heart Attack Risk Score</span>
                            <span style="font-weight: 700; color: {'#ff6b6b' if risk_score > 50 else '#51cf66'}">{risk_score}%</span>
                        </div>
                        <div class="risk-meter">
                            <div class="risk-meter-indicator" style="left: {risk_score}%;"></div>
                        </div>
                        <div class="risk-meter-labels">
                            <span>Low Risk</span>
                            <span>Medium Risk</span>
                            <span>High Risk</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced prediction box with more categories
                    if risk_score > 70:
                        st.markdown(f"""
                        <div class="prediction-box-high">
                            <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">⚠ High Risk of Heart Attack</div>
                            <div style="font-size: 1.1rem;">Based on your profile, you have an elevated risk of cardiovascular events</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_score > 30:
                        st.markdown(f"""
                        <div class="prediction-box-medium">
                            <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">🔍 Moderate Risk of Heart Attack</div>
                            <div style="font-size: 1.1rem;">Some risk factors present that may benefit from lifestyle changes</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box-low">
                            <div style="font-size: 1.75rem; margin-bottom: 0.5rem;">✅ Low Risk of Heart Attack</div>
                            <div style="font-size: 1.1rem;">Your current profile suggests low cardiovascular risk</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk factors analysis with cards
                    st.markdown("### 📌 Key Risk Factors")
                    
                    # Create a list of risk factors with their impact
                    risk_factors = [
                        {"factor": "Age", "value": age_category, "impact": "High" if age_category in ["50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"] else "Medium" if age_category in ["35-39", "40-44", "45-49"] else "Low", "description": "Risk increases with age"},
                        {"factor": "BMI", "value": f"{bmi} ({bmi_status})", "impact": "High" if bmi_status in ["Obese", "Overweight"] else "Low", "description": "Higher BMI increases cardiovascular strain"},
                        {"factor": "General Health", "value": general_health, "impact": "High" if general_health in ["Poor", "Fair"] else "Low", "description": "Self-reported health is a strong predictor"},
                        {"factor": "Smoking Status", "value": smoker, "impact": "High" if smoker == "Current smoker" else "Medium" if smoker == "Former smoker" else "Low", "description": "Smoking damages blood vessels"},
                        {"factor": "History of Stroke", "value": had_stroke, "impact": "High" if had_stroke == "Yes" else "Low", "description": "Previous stroke indicates vascular issues"},
                        {"factor": "History of Angina", "value": had_angina, "impact": "High" if had_angina == "Yes" else "Low", "description": "Angina indicates existing heart disease"},
                        {"factor": "Physical Health Days", "value": physical_days, "impact": "High" if physical_days > 10 else "Medium" if physical_days > 5 else "Low", "description": "Frequent health issues may indicate problems"},
                        {"factor": "Sleep Hours", "value": sleep_hours, "impact": "High" if sleep_hours < 6 or sleep_hours > 9 else "Low", "description": "Poor sleep impacts cardiovascular health"},
                    ]
                    
                    # Display risk factors in cards
                    cols = st.columns(2)
                    for i, factor in enumerate(risk_factors):
                        with cols[i % 2]:
                            impact_class = "impact-high" if factor["impact"] == "High" else "impact-medium" if factor["impact"] == "Medium" else "impact-low"
                            st.markdown(f"""
                            <div class="feature-card">
                                <div class="feature-card-title">{factor["factor"]}</div>
                                <div class="feature-card-value">{factor["value"]}</div>
                                <div style="font-size: 0.85rem; color: #6c757d; margin: 0.5rem 0;">{factor["description"]}</div>
                                <div class="feature-card-impact {impact_class}">
                                    {factor["impact"]} Impact
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Feature importance visualization
                    try:
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("### 📈 Feature Importance")
                            feature_names = preprocessor.get_feature_names_out()
                            importances = model.feature_importances_
                            indices = np.argsort(importances)[-10:]  # Top 10 features
                            
                            fig = px.bar(
                                x=importances[indices],
                                y=feature_names[indices],
                                orientation='h',
                                title="Top Influencing Factors in Your Assessment",
                                labels={'x': 'Importance', 'y': 'Factor'},
                                color=importances[indices],
                                color_continuous_scale='Bluered'
                            )
                            fig.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        pass
                    
                    # Enhanced recommendations with cards
                    st.markdown("### 💡 Personalized Recommendations")
                    
                    recommendations = []
                    if bmi_status in ["Obese", "Overweight"]:
                        recommendations.append({
                            "title": "Weight Management",
                            "content": "Consider a weight management program to reach a healthier BMI range. Even 5-10% weight loss can significantly improve cardiovascular health.",
                            "priority": "High"
                        })
                    if smoker == "Current smoker":
                        recommendations.append({
                            "title": "Smoking Cessation",
                            "content": "Quitting smoking can reduce your heart disease risk by 50% within 1 year. Consider nicotine replacement therapy or counseling.",
                            "priority": "High"
                        })
                    if sleep_hours < 6 or sleep_hours > 9:
                        recommendations.append({
                            "title": "Sleep Hygiene",
                            "content": "Aim for 7-9 hours of quality sleep each night. Maintain a consistent sleep schedule and create a restful environment.",
                            "priority": "Medium"
                        })
                    if general_health in ["Poor", "Fair"]:
                        recommendations.append({
                            "title": "Health Check-ups",
                            "content": "Schedule regular check-ups with your healthcare provider to monitor blood pressure, cholesterol, and other key indicators.",
                            "priority": "High"
                        })
                    if had_stroke == "Yes" or had_angina == "Yes":
                        recommendations.append({
                            "title": "Cardiac Monitoring",
                            "content": "Given your medical history, regular cardiac monitoring and specialist consultations are strongly recommended.",
                            "priority": "High"
                        })
                    if physical_days > 5:
                        recommendations.append({
                            "title": "Physical Health",
                            "content": "Addressing your physical health issues may reduce cardiovascular strain. Consult with a healthcare provider about persistent symptoms.",
                            "priority": "Medium"
                        })
                    if mental_days > 5:
                        recommendations.append({
                            "title": "Mental Wellbeing",
                            "content": "Chronic stress and mental health issues can impact heart health. Consider stress management techniques or professional support.",
                            "priority": "Medium"
                        })
                    
                    if not recommendations:
                       st.markdown(
    """
    <div style="color: #1E40AF; background: #DBEAFE; padding: 1rem; border-radius: 8px; border-left: 4px solid #1E40AF;">
        🌟 <strong>You're doing great!</strong> Maintain your healthy lifestyle habits.
    </div>
    """,
    unsafe_allow_html=True
)
                    else:
                        # Sort by priority
                        recommendations.sort(key=lambda x: 0 if x["priority"] == "High" else 1 if x["priority"] == "Medium" else 2)
                        
                        for rec in recommendations:
                            priority_icon = "🔴" if rec["priority"] == "High" else "🟠" if rec["priority"] == "Medium" else "🔵"
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="recommendation-card-title">
                                    {priority_icon} {rec["title"]} ({rec["priority"]} Priority)
                                </div>
                                <div>{rec["content"]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div style="margin-top: 2rem; background:#00008B ; padding: 1.5rem; border-radius: 12px;">
                            <div style="font-weight: 600; color: #FFD700 ; margin-bottom: 0.5rem;">Next Steps</div>
                            <div>Consider discussing these results with your healthcare provider for personalized medical advice. 
                            Small, consistent changes can significantly improve your cardiovascular health over time.</div>
                        </div>
                        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("## 💡 Heart Health Insights & Education")
        
        # Interactive BMI Calculator with more details
        with st.expander("📊 BMI Calculator & Analysis", expanded=True):
            bmi_col1, bmi_col2 = st.columns(2)
            with bmi_col1:
                calc_weight = st.number_input("Your Weight (kg)", 30.0, 200.0, step=0.5, value=70.0, key="bmi_weight")
            with bmi_col2:
                calc_height = st.number_input("Your Height (m)", 1.0, 2.5, step=0.01, value=1.75, key="bmi_height")
            
            calc_bmi = round(calc_weight / (calc_height ** 2), 2) if calc_height > 0 else 0
            bmi_category = ""
            bmi_color = ""
            if calc_bmi < 18.5:
                bmi_category = "Underweight"
                bmi_color = "#2196f3"
            elif 18.5 <= calc_bmi < 25:
                bmi_category = "Normal weight"
                bmi_color = "#4caf50"
            elif 25 <= calc_bmi < 30:
                bmi_category = "Overweight"
                bmi_color = "#ff9800"
            else:
                bmi_category = "Obese"
                bmi_color = "#f44336"
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 16px; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                    <div>
                        <div style="font-size: 0.9rem; color: #6c757d;">Your Body Mass Index</div>
                        <div style="font-size: 1.75rem; font-weight: 700; color: {bmi_color};">{calc_bmi}</div>
                    </div>
                    <div style="background: {bmi_color}; color: white; padding: 0.5rem 1rem; border-radius: 16px; font-size: 1rem; font-weight: 600;">
                        {bmi_category}
                    </div>
                </div>
                <div style="height: 16px; background: linear-gradient(90deg, #2196f3 0%, #4caf50 18.5%, #ff9800 25%, #f44336 100%); border-radius: 8px;"></div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.75rem; font-size: 0.8rem; color: #6c757d;">
                    <span>Underweight</span>
                    <span>Normal</span>
                    <span>Overweight</span>
                    <span>Obese</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI health implications
            if calc_bmi < 18.5:
                st.markdown("""
<div style="
    background: #FFF7ED;
    color: #9A3412;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #F97316;
    margin: 1rem 0;
">
    <div style="font-weight: 600;">Underweight Implications:</div>
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li>May indicate nutritional deficiencies</li>
        <li>Can lead to weakened immune system</li>
        <li>May be associated with osteoporosis</li>
    </ul>
    <div style="font-weight: 600; margin-top: 0.75rem;">Recommendations:</div>
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li>Consult with a nutritionist for healthy weight gain</li>
        <li>Focus on nutrient-dense foods</li>
        <li>Rule out underlying medical conditions</li>
    </ul>
</div>
""", unsafe_allow_html=True)
            elif 18.5 <= calc_bmi < 25:
                st.markdown("""
<div style="
    background: #F0FDF4;
    color: #166534;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #10B981;
    margin: 1rem 0;
">
    <div style="font-weight: 600; margin-bottom: 0.5rem;">Healthy Weight Benefits:</div>
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li>Lower risk of chronic diseases</li>
        <li>Better energy levels and mobility</li>
        <li>Improved metabolic health</li>
    </ul>
    <div style="font-weight: 600; margin: 0.75rem 0 0.5rem 0;">Recommendations:</div>
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li>Maintain current healthy habits</li>
        <li>Continue regular physical activity</li>
        <li>Monitor weight periodically</li>
    </ul>
</div>
""", unsafe_allow_html=True)
            elif 25 <= calc_bmi < 30:
                st.markdown("""
<div style="
    background: #FEFCE8;
    color: #854D0E;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #EAB308;
    margin: 1rem 0;
">
    <div style="font-weight: 600;">Overweight Considerations:</div>
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li>Increased risk of hypertension</li>
        <li>Higher likelihood of developing diabetes</li>
        <li>Potential joint problems</li>
    </ul>
    <div style="font-weight: 600; margin-top: 0.75rem;">Recommendations:</div>
    <ul style="margin: 0; padding-left: 1.2rem;">
        <li>Aim for 5-10% weight loss</li>
        <li>Increase physical activity gradually</li>
        <li>Focus on whole, unprocessed foods</li>
    </ul>
</div>
""", unsafe_allow_html=True)
            else:
                st.error("""
                Obesity Health Risks:  
                - Significantly increased cardiovascular risk  
                - Higher chance of sleep apnea  
                - Greater risk of certain cancers  
                Recommendations:  
                - Seek medical advice for weight management  
                - Consider comprehensive lifestyle changes  
                - Explore supervised weight loss programs
                """)
        
       

        # Enhanced Risk Factors Visualization with interactive elements
        with st.expander("📈 Understanding Risk Factors", expanded=True):
            st.markdown("""
            ### How Different Factors Affect Heart Health
            
            Adjust the sliders below to see how modifying risk factors can impact your cardiovascular risk:
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                bp_control = st.select_slider("Blood Pressure Control", 
                                            options=["Uncontrolled", "Borderline", "Controlled"],
                                            value="Borderline")
                cholesterol = st.select_slider("Cholesterol Levels", 
                                            options=["High (>240)", "Borderline (200-239)", "Optimal (<200)"],
                                            value="Borderline (200-239)")
                activity = st.select_slider("Physical Activity", 
                                         options=["Sedentary", "Moderate", "Active"],
                                         value="Moderate")
                
            with col2:
                diet = st.select_slider("Diet Quality", 
                                      options=["Poor", "Average", "Excellent"],
                                      value="Average")
                stress = st.select_slider("Stress Levels", 
                                        options=["High", "Moderate", "Low"],
                                        value="Moderate")
                smoking = st.select_slider("Smoking Status", 
                                         options=["Current Smoker", "Recent Quit", "Never Smoked"],
                                         value="Recent Quit")
            
            # Simulate risk calculation based on inputs
            risk_score = 50  # Baseline
            if bp_control == "Uncontrolled": risk_score += 20
            elif bp_control == "Controlled": risk_score -= 15
            
            if cholesterol.startswith("High"): risk_score += 15
            elif cholesterol.startswith("Optimal"): risk_score -= 10
            
            if activity == "Sedentary": risk_score += 15
            elif activity == "Active": risk_score -= 10
            
            if diet == "Poor": risk_score += 10
            elif diet == "Excellent": risk_score -= 10
            
            if stress == "High": risk_score += 10
            elif stress == "Low": risk_score -= 5
            
            if smoking == "Current Smoker": risk_score += 20
            elif smoking == "Never Smoked": risk_score -= 5
            
            risk_score = max(5, min(95, risk_score))  # Keep within bounds
            
            # Create radar chart of risk factors
            categories = ['Blood Pressure', 'Cholesterol', 'Activity', 'Diet', 'Stress', 'Smoking']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[100 if bp_control=="Uncontrolled" else 60 if bp_control=="Borderline" else 20,
                   100 if cholesterol.startswith("High") else 60 if cholesterol.startswith("Borderline") else 20,
                   100 if activity=="Sedentary" else 60 if activity=="Moderate" else 20,
                   100 if diet=="Poor" else 60 if diet=="Average" else 20,
                   100 if stress=="High" else 60 if stress=="Moderate" else 20,
                   100 if smoking=="Current Smoker" else 60 if smoking=="Recent Quit" else 20],
                theta=categories,
                fill='toself',
                name='Your Risk Profile',
                line_color='#667eea'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Your Cardiovascular Risk Profile",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show simulated risk impact
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 16px; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #2a3f5f;">Estimated Risk Impact from Current Profile</div>
                    <div style="font-weight: 700; color: {'#ff6b6b' if risk_score > 50 else '#51cf66'}">{risk_score}%</div>
                </div>
                <div class="risk-meter">
                    <div class="risk-meter-indicator" style="left: {risk_score}%;"></div>
                </div>
                <div class="risk-meter-labels">
                    <span>Low Risk</span>
                    <span>Medium Risk</span>
                    <span>High Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("""
            Note: This is a simplified simulation for educational purposes only. 
            Actual risk assessment requires comprehensive medical evaluation.
            """)

    # Enhanced Footer with social links
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 1rem;">
            <a href="#" style="color: #6c757d; text-decoration: none;">Terms of Service</a>
            <a href="#" style="color: #6c757d; text-decoration: none;">Privacy Policy</a>
            <a href="#" style="color: #6c757d; text-decoration: none;">Research</a>
            <a href="#" style="color: #6c757d; text-decoration: none;">Careers</a>
        </div>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
            <a href="#"><img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="24"></a>
            <a href="#"><img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" width="24"></a>
            <a href="#"><img src="https://cdn-icons-png.flaticon.com/512/2111/2111370.png" width="24"></a>
            <a href="#"><img src="https://cdn-icons-png.flaticon.com/512/2111/2111432.png" width="24"></a>
        </div>
        <p>© 2025 HeartGuard AI | Created with ❤ by CogniAnalytica Team Innovations</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">This tool is for educational and informational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)