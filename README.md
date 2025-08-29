# 🏙️ Delhi Urban Health Risk Index (UHRI) Predictor

A predictive web application that calculates a novel, holistic Urban Health Risk Index (UHRI) for Delhi by fusing real-time predictions for PM2.5 air pollution and traffic congestion.

📖 Table of Contents
The Problem

The Solution: UHRI

✨ Key Features

⚙️ How It Works

🚀 Local Setup and Usage

🗂️ Repository Structure

🤖 Models and Features

🔭 Future Work

🌍 The Problem
Delhi faces a dual crisis of severe air pollution and chronic traffic congestion. Standard metrics like the Air Quality Index (AQI) only measure ambient pollution, failing to account for the dramatically increased health risk when citizens are exposed to concentrated, ground-level pollutants trapped by heavy traffic.

A person's true health risk is a function of both the air quality and their immediate environment. This project addresses the need for a more accurate, contextual, and actionable health metric.

💡 The Solution: Urban Health Risk Index (UHRI)
This project introduces the Urban Health Risk Index (UHRI), a novel metric that quantifies the synergistic risk of air pollution and traffic.

An AQI says: "The air is unhealthy today."

The UHRI says: "The air is unhealthy, and high traffic will multiply your personal risk, making it severe. Here is what you should do."

By fusing predictions from two distinct machine learning models, the UHRI provides a more realistic and life-saving measure of urban environmental danger.

✨ Key Features
Dual-Model Prediction: Utilizes two XGBoost models to independently predict PM2.5 levels and traffic density.

Holistic UHRI Score: Combines the two predictions using a Causal Risk Multiplier to generate a single, easy-to-understand health risk score.

Simulated Risk Map: Displays a map of Delhi with color-coded zones to visualize how the risk might vary across different areas.

Actionable Health Advice: Provides clear, icon-based recommendations tailored to the predicted UHRI level, empowering users to take immediate protective action.

Interactive UI: Built with Streamlit for a clean, intuitive, and responsive user experience.

⚙️ How It Works
The application operates on a three-step pipeline upon user input:

Predict Air Quality: The first XGBoost model (aqi_model.joblib) takes user inputs for current pollutant levels (CO, NO, PM10, etc.) and time-based features to predict the PM2.5 value for a specific time.

Predict Traffic Congestion: The second XGBoost model (final_traffic_model.joblib) uses time-based features and recent traffic density to predict the level of congestion.

Calculate UHRI:

The raw predictions are converted into standardized Risk Scores.

The traffic prediction determines a Causal Risk Multiplier (from 1.0x for low traffic to 2.0x for very high traffic), quantifying the pollutant-trapping effect.

The final UHRI is calculated: UHRI = Pollution Risk Score * Causal Risk Multiplier.

This final score is then displayed with its interpretation, a risk map, and actionable advice.

🚀 Local Setup and Usage
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/vinayakjain01/Delhi-Urban-Health-Risk-Index.git
cd UHRI

Set up a virtual environment (recommended):

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run New_UHRIapp.py

The application should now be open and running in your web browser.

🗂️ Repository Structure
├── New_UHRIapp.py              # Main Streamlit application script

├── aqi_model.joblib            # Trained XGBoost model for PM2.5 prediction

├── final_traffic_model.joblib  # Trained XGBoost model for traffic prediction

├── notebooks/

│   ├── DelhiAQI.ipynb          # Jupyter Notebook for AQI model training

│   └── Delhi_Traffic_prediction.ipynb # Jupyter Notebook for traffic model training

├── assets/
│   └── app screenshot & Video Recording  
├── requirements.txt            # Required Python libraries

└── README.md                   # Project description

🤖 Models and Features
Both models were built using the XGBoost Regressor algorithm due to its high performance and efficiency.

Air Quality Model
Target: pm2_5

Key Features: pm10, co, no2 (other pollutant levels), hour, month, day_of_year (time-based features), and lag/rolling mean features derived from recent PM2.5 levels.

Traffic Model
Target: Average Queue Density

Key Features: hour, day_of_week, is_weekend (strong temporal predictors), and lag_5_sec, lag_15_sec (short-term momentum features).

🔭 Future Work
Automate Live Data: Integrate a live Air Quality API (like OpenAQ or CPCB's official API) to remove the need for manual slider inputs and provide true real-time predictions.

Data-Driven Multiplier: Acquire and fuse historical health data (e.g., hospital admissions for respiratory issues) to develop a statistically validated, data-driven Causal Risk Multiplier.

Granular Predictions: Train location-specific models if zonal data becomes available to move from a simulated map to a map of true zonal predictions.
