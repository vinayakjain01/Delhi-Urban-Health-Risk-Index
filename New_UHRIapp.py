# UHRI_app_map_version.py

import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np # New import for random variation

# --- Page Configuration ---
st.set_page_config(
    page_title="Delhi Urban Health Risk Index",
    page_icon="🏙️",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads and returns the trained machine learning models."""
    try:
        aqi_model = joblib.load('aqi_model.joblib')
        traffic_model = joblib.load('final_traffic_model.joblib')
        return aqi_model, traffic_model
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Make sure 'aqi_model.joblib' and 'final_traffic_model.joblib' are in the same directory as the app.")
        return None, None

aqi_model, traffic_model = load_models()

# --- Helper Functions ---
def get_pollution_risk(pm25_value):
    if 0 <= pm25_value <= 50: return 'Good', 1
    elif 51 <= pm25_value <= 100: return 'Moderate', 2
    elif 101 <= pm25_value <= 250: return 'Poor', 3
    elif 251 <= pm25_value <= 350: return 'Unhealthy', 4
    elif 351 <= pm25_value <= 450: return 'Severe', 5
    else: return 'Hazardous', 6

def get_traffic_risk_and_multiplier(traffic_prediction_value):
    if traffic_prediction_value < 0.25: return "🟩 Low Traffic", 1, 1.0
    elif traffic_prediction_value < 0.5: return "🟨 Moderate Traffic", 2, 1.2
    elif traffic_prediction_value < 0.75: return "🟧 High Traffic", 3, 1.5
    else: return "🟥 Very High Traffic", 4, 2.0

def calculate_uhri(pollution_score, risk_multiplier):
    return pollution_score * risk_multiplier

def interpret_uhri(uhri_score):
    if uhri_score <= 2: return "Low Risk", "#4CAF50"
    elif uhri_score <= 4: return "Moderate Risk", "#FFEB3B"
    elif uhri_score <= 7: return "High Risk", "#FF9800"
    elif uhri_score <= 10: return "Very High Risk", "#F44336"
    else: return "Severe Risk", "#B71C1C"

def get_actionable_advice(uhri_category):
    advice = {
        "Low Risk": {"🏃‍♀️ Outdoor Activity": "It's a great day for outdoor activities.", "😷 Masks": "No masks required.", "🏠 Windows": "Feel free to open windows for ventilation."},
        "Moderate Risk": {"🏃‍♀️ Outdoor Activity": "Unusually sensitive people should consider reducing prolonged or heavy exertion.", "😷 Masks": "Not necessary for most people.", "🏠 Windows": "Good day to ventilate your home."},
        "High Risk": {"🏃‍♀️ Outdoor Activity": "Reduce prolonged or heavy outdoor exertion. Take more breaks.", "😷 Masks": "Sensitive individuals may benefit from wearing a mask outdoors.", "🏠 Windows": "Consider using an air purifier if you have one. Limit opening windows."},
        "Very High Risk": {"🏃‍♀️ Outdoor Activity": "Avoid prolonged or heavy outdoor exertion. Consider moving activities indoors.", "😷 Masks": "Wearing an N95 or FFP2 mask is recommended for outdoor exposure.", "🏠 Windows": "Keep windows closed and run an air purifier."},
        "Severe Risk": {"🏃‍♀️ Outdoor Activity": "Avoid all outdoor physical activity. Stay indoors as much as possible.", "😷 Masks": "An N95 or FFP2 mask is essential if you must go outside.", "🏠 Windows": "Keep windows and doors closed. Use air purifiers on high."}
    }
    return advice.get(uhri_category, {})

# --- User Interface ---
st.title('🏙️ Delhi Urban Health Risk Index (UHRI) Predictor')
st.markdown("This tool combines predictions for **Air Pollution (PM2.5)** and **Traffic Congestion** to provide a holistic **Urban Health Risk Index**. This index represents the *actual* health risk to a person exposed to these conditions at a specific time.")
st.markdown("---")

if not aqi_model or not traffic_model:
    st.stop()

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header('Enter Conditions for Prediction')
    prediction_date = st.date_input("Select Date", datetime.date.today())
    prediction_time = st.time_input("Select Time", datetime.datetime.now().time())
    combined_datetime = datetime.datetime.combine(prediction_date, prediction_time)

    st.subheader("Air Quality Inputs")
    co = st.slider('CO (Carbon Monoxide)', 0.0, 15.0, 2.0, 0.1)
    no = st.slider('NO (Nitric Oxide)', 0.0, 150.0, 25.0, 0.5)
    no2 = st.slider('NO2 (Nitrogen Dioxide)', 0.0, 150.0, 40.0, 0.5)
    o3 = st.slider('O3 (Ozone)', 0.0, 200.0, 35.0, 1.0)
    so2 = st.slider('SO2 (Sulphur Dioxide)', 0.0, 100.0, 15.0, 0.5)
    pm10 = st.slider('PM10', 0.0, 800.0, 150.0, 5.0)
    nh3 = st.slider('NH3 (Ammonia)', 0.0, 100.0, 20.0, 0.5)
    recent_pm25 = st.slider('Recent PM2.5 Level (µg/m³)', 0, 500, 150)

    st.subheader("Traffic Inputs")
    recent_density = st.slider('Current Traffic Density', 0.0, 1.0, 0.4, 0.05)
    
    predict_button = st.button('Calculate UHRI', type="primary", use_container_width=True)

# --- Main Panel for Displaying Results ---
if predict_button:
    # Define the exact feature order
    aq_feature_order = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3', 'hour', 'day_of_week', 'month', 'day_of_year', 'pm2_5_lag_1', 'pm2_5_lag_2', 'pm2_5_lag_3', 'pm2_5_lag_24', 'pm2_5_rolling_mean_3', 'pm2_5_rolling_mean_24']
    traffic_feature_order = ['hour', 'day_of_week', 'is_weekend', 'lag_5_sec', 'lag_15_sec', 'lag_30_sec']

    # --- Main Prediction for the selected time ---
    aq_input_data = pd.DataFrame({
        'co': [co], 'no': [no], 'no2': [no2], 'o3': [o3], 'so2': [so2], 'pm10': [pm10], 'nh3': [nh3],
        'hour': [combined_datetime.hour], 'day_of_week': [combined_datetime.weekday()], 'month': [combined_datetime.month], 'day_of_year': [combined_datetime.timetuple().tm_yday],
        'pm2_5_lag_1': [recent_pm25], 'pm2_5_lag_2': [recent_pm25], 'pm2_5_lag_3': [recent_pm25],
        'pm2_5_lag_24': [recent_pm25], 'pm2_5_rolling_mean_3': [recent_pm25], 'pm2_5_rolling_mean_24': [recent_pm25]
    })[aq_feature_order]
    
    traffic_input_data = pd.DataFrame({
        'hour': [combined_datetime.hour], 'day_of_week': [combined_datetime.weekday()], 'is_weekend': [1 if combined_datetime.weekday() >= 5 else 0],
        'lag_5_sec': [recent_density], 'lag_15_sec': [recent_density], 'lag_30_sec': [recent_density]
    })[traffic_feature_order]

    predicted_pm25 = aqi_model.predict(aq_input_data)[0]
    air_category, pollution_score = get_pollution_risk(predicted_pm25)
    
    predicted_traffic_value = traffic_model.predict(traffic_input_data)[0]
    traffic_text, traffic_score, risk_multiplier = get_traffic_risk_and_multiplier(predicted_traffic_value)
    
    final_uhri_score = calculate_uhri(pollution_score, risk_multiplier)
    uhri_category, uhri_color = interpret_uhri(final_uhri_score)

    st.subheader(f"Prediction for {prediction_date.strftime('%A, %B %d, %Y')} at {prediction_time.strftime('%I:%M %p')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Predicted PM2.5 Level", value=f"{predicted_pm25:.2f} µg/m³", delta=air_category)
        st.markdown(f"**Pollution Risk Score:** `{pollution_score}`")
    with col2:
        st.metric(label="Predicted Traffic Condition", value=traffic_text)
        st.markdown(f"**Causal Risk Multiplier:** `{risk_multiplier}x`")
    with col3:
        st.markdown(f"""<div style="padding: 20px; border-radius: 10px; background-color: {uhri_color}; text-align: center; color: #000;">
            <h3 style="margin: 0; color: #000;">Urban Health Risk Index (UHRI)</h3>
            <h1 style="font-size: 3.5em; margin: 0; color: #000;">{final_uhri_score:.1f}</h1>
            <h2 style="margin: 0; color: #000;">{uhri_category}</h2></div>""", unsafe_allow_html=True)
    
    st.info(f"**Explanation:** The base pollution risk score of **{pollution_score}** was multiplied by **{risk_multiplier}x** due to the predicted traffic congestion, resulting in a final, more accurate health risk index of **{final_uhri_score:.1f}**.", icon="ℹ️")
    st.markdown("---")

    # --- NEW: Risk Zone Map Section ---
    st.subheader("Simulated Risk Zones Across Delhi")
    st.caption("This map shows simulated variations in the UHRI based on the overall prediction. Red zones indicate higher potential risk.")

    # Define key locations in Delhi
    locations = {
        "Connaught Place": (28.6330, 77.2194), "India Gate": (28.6129, 77.2295),
        "Chandni Chowk": (28.6562, 77.2410), "Nehru Place": (28.5493, 77.2509),
        "Anand Vihar": (28.6473, 77.3154), "Dwarka": (28.5921, 77.0460),
        "Rohini": (28.7041, 77.1025), "IGI Airport": (28.5562, 77.1000)
    }
    
    map_data = []
    for name, (lat, lon) in locations.items():
        # Create a small random variation for each zone to simulate local conditions
        local_factor = np.random.uniform(0.95, 1.15)
        zone_uhri = final_uhri_score * local_factor
        zone_category, zone_color_hex = interpret_uhri(zone_uhri)
        
        # Convert hex to RGBA for the map
        hex_color = zone_color_hex.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        map_data.append({
            "name": name, "lat": lat, "lon": lon, 
            "risk_score": zone_uhri, "color": (*rgb_color, 200) # Add alpha for transparency
        })
    
    map_df = pd.DataFrame(map_data)
    
    st.map(map_df, latitude=28.6139, longitude=77.2090, zoom=10, color='color', size='risk_score')

    # --- Actionable Advice Section ---
    st.subheader("Health Recommendations Based on Your Risk")
    advice_items = get_actionable_advice(uhri_category)
    
    cols = st.columns(len(advice_items))
    for i, (topic, text) in enumerate(advice_items.items()):
        with cols[i]:
            st.markdown(f"**{topic}**")
            st.markdown(text)

else:
    st.info('Please enter the conditions in the sidebar and click "Calculate UHRI" to see the prediction.')
