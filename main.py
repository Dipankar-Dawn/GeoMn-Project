from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="GeoMn Mining Risk API",
    description="AI/ML based Manganese Mining Risk Prediction System"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "final_manganese_ml_dataset.csv"
)

MODELS_DIR = BASE_DIR

EQUIPMENT_MODEL_PATH = (
    MODELS_DIR
    / "equipment_risk_model.pkl"
)

SHORTFALL_MODEL_PATH = (
    MODELS_DIR
    / "production_shortfall_xgboost_model.json"
)

METADATA_PATH = (
    MODELS_DIR
    / "production_shortfall_model_metadata.pkl"
)

HTML_FILE_PATH = BASE_DIR / "my_gpt.html"


# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
    print("✓ Historical dataset loaded successfully")
except Exception as e:
    df = None
    print("✗ DATASET ERROR:")
    print(e)


# =========================================================
# LOAD EQUIPMENT MODEL
# =========================================================

try:
    equipment_model = joblib.load(
        EQUIPMENT_MODEL_PATH
    )
    print("✓ Equipment risk model loaded successfully")
except Exception as e:
    equipment_model = None
    print("✗ EQUIPMENT MODEL ERROR:")
    print(e)


# =========================================================
# LOAD SHORTFALL XGBOOST MODEL
# =========================================================

try:
    shortfall_model = xgb.XGBClassifier()
    shortfall_model.load_model(
        SHORTFALL_MODEL_PATH
    )
    print("✓ Production shortfall XGBoost model loaded successfully")
except Exception as e:
    shortfall_model = None
    print("✗ SHORTFALL MODEL ERROR:")
    print(e)


# =========================================================
# LOAD MODEL METADATA
# =========================================================

try:
    metadata = joblib.load(
        METADATA_PATH
    )
    print("✓ Model metadata loaded successfully")
except Exception as e:
    metadata = None
    print("✗ METADATA ERROR:")
    print(e)


# =========================================================
# REQUEST MODEL
# =========================================================

class PredictionInput(BaseModel):
    state: str
    district: str
    weather_condition: str
    # Optional for future simulation mode
    production_tonnes: float | None = None


# =========================================================
# WEATHER SCENARIO FUNCTION
# =========================================================

def get_weather_values(
    district_data,
    weather_condition
):
    weather_condition = (
        weather_condition.lower().strip()
    )

    # Dataset average values
    avg_temperature = float(
        district_data["Avg_Temperature_C"].mean()
    )

    avg_rainfall = float(
        district_data["Total_Rainfall_mm"].mean()
    )

    avg_humidity = float(
        district_data["Avg_Humidity_pct"].mean()
    )

    # GOOD WEATHER
    if weather_condition == "good":
        temperature = float(
            district_data["Avg_Temperature_C"].quantile(0.25)
        )
        rainfall = float(
            district_data["Total_Rainfall_mm"].quantile(0.25)
        )
        humidity = float(
            district_data["Avg_Humidity_pct"].quantile(0.25)
        )

    # BAD WEATHER
    elif weather_condition == "bad":
        temperature = float(
            district_data["Avg_Temperature_C"].quantile(0.75)
        )
        rainfall = float(
            district_data["Total_Rainfall_mm"].quantile(0.75)
        )
        humidity = float(
            district_data["Avg_Humidity_pct"].quantile(0.75)
        )

    # WORST WEATHER
    elif weather_condition == "worst":
        temperature = float(
            district_data["Avg_Temperature_C"].max()
        )
        rainfall = float(
            district_data["Total_Rainfall_mm"].max()
        )
        humidity = float(
            district_data["Avg_Humidity_pct"].max()
        )

    # AUTO / NORMAL WEATHER
    else:
        temperature = avg_temperature
        rainfall = avg_rainfall
        humidity = avg_humidity

    return (
        temperature,
        rainfall,
        humidity
    )


# =========================================================
# CALCULATE STRESS VALUES
# =========================================================

def calculate_stress(
    district_data,
    production,
    temperature,
    rainfall,
    humidity
):
    # Production Stress
    district_avg_production = float(
        district_data["Production_Tonnes"].mean()
    )
    production_stress = (
        district_avg_production - production
    ) / district_avg_production
    production_stress = float(
        np.clip(
            production_stress,
            0,
            1
        )
    )

    # Temperature Stress
    temp_mean = float(
        df["Avg_Temperature_C"].mean()
    )
    temp_std = float(
        df["Avg_Temperature_C"].std()
    )
    temperature_stress = abs(
        temperature - temp_mean
    ) / (
        2 * temp_std
    )
    temperature_stress = float(
        np.clip(
            temperature_stress,
            0,
            1
        )
    )

    # Rainfall Stress
    rain_min = float(
        df["Total_Rainfall_mm"].min()
    )
    rain_max = float(
        df["Total_Rainfall_mm"].max()
    )
    rainfall_stress = (
        rainfall - rain_min
    ) / (
        rain_max - rain_min
    )
    rainfall_stress = float(
        np.clip(
            rainfall_stress,
            0,
            1
        )
    )

    # Humidity Stress
    humidity_min = float(
        df["Avg_Humidity_pct"].min()
    )
    humidity_max = float(
        df["Avg_Humidity_pct"].max()
    )
    humidity_stress = (
        humidity - humidity_min
    ) / (
        humidity_max - humidity_min
    )
    humidity_stress = float(
        np.clip(
            humidity_stress,
            0,
            1
        )
    )

    # Weather Stress
    weather_stress = (
        temperature_stress
        + rainfall_stress
        + humidity_stress
    ) / 3

    return {
        "production_stress": round(production_stress, 3),
        "temperature_stress": round(temperature_stress, 3),
        "rainfall_stress": round(rainfall_stress, 3),
        "humidity_stress": round(humidity_stress, 3),
        "weather_stress": round(weather_stress, 3)
    }


# =========================================================
# LOCAL AI RECOMMENDATION ENGINE
# =========================================================

def generate_ai_recommendations(
    production_stress,
    temperature_stress,
    rainfall_stress,
    humidity_stress,
    weather_stress,
    equipment_risk,
    shortfall_risk
):
    recommendations = []
    risk_factors = []

    # Identify Important Risk Factors
    factors = {
        "Production Performance": production_stress,
        "Temperature Conditions": temperature_stress,
        "Rainfall Conditions": rainfall_stress,
        "Humidity Conditions": humidity_stress,
        "Overall Weather Conditions": weather_stress,
        "Equipment Reliability": equipment_risk / 100
    }

    sorted_factors = sorted(
        factors.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for factor, value in sorted_factors[:3]:
        risk_factors.append({
            "factor": factor,
            "severity_score": round(float(value), 3)
        })

    # Production Recommendation
    if production_stress >= 0.4:
        recommendations.append({
            "priority": "High",
            "issue_detected": "Production Performance",
            "why_it_matters": "Production is significantly below the expected operational level.",
            "recommended_actions": [
                "Identify production delays and bottlenecks.",
                "Improve production scheduling.",
                "Improve transportation and material handling."
            ]
        })
    elif production_stress >= 0.2:
        recommendations.append({
            "priority": "Medium",
            "issue_detected": "Production Performance",
            "why_it_matters": "Production performance is slightly below the expected level.",
            "recommended_actions": [
                "Monitor daily production targets.",
                "Improve shift planning and machine utilization."
            ]
        })

    # Equipment Recommendation
    if equipment_risk >= 20:
        recommendations.append({
            "priority": "High",
            "issue_detected": "Equipment Reliability",
            "why_it_matters": "The system detected an increased chance of equipment-related disruption.",
            "recommended_actions": [
                "Prioritize preventive maintenance.",
                "Inspect critical machines before operations.",
                "Monitor equipment performance regularly."
            ]
        })
    elif equipment_risk >= 10:
        recommendations.append({
            "priority": "Medium",
            "issue_detected": "Equipment Monitoring",
            "why_it_matters": "Some equipment may require additional monitoring to prevent unexpected stoppages.",
            "recommended_actions": [
                "Schedule routine maintenance.",
                "Inspect critical machines regularly."
            ]
        })

    # Temperature Recommendation
    if temperature_stress >= 0.6:
        recommendations.append({
            "priority": "High",
            "issue_detected": "High Temperature Stress",
            "why_it_matters": "High temperature may affect equipment performance and mining operations.",
            "recommended_actions": [
                "Monitor equipment temperature.",
                "Schedule heat-sensitive work during suitable hours.",
                "Ensure proper cooling and thermal protection."
            ]
        })

    # Rainfall Recommendation
    if rainfall_stress >= 0.6:
        recommendations.append({
            "priority": "High",
            "issue_detected": "Heavy Rainfall Risk",
            "why_it_matters": "Heavy rainfall may affect transportation and mining operations.",
            "recommended_actions": [
                "Improve mine drainage.",
                "Monitor haul roads regularly.",
                "Prepare backup plans for heavy rainfall."
            ]
        })

    # Humidity Recommendation
    if humidity_stress >= 0.6:
        recommendations.append({
            "priority": "Medium",
            "issue_detected": "High Humidity",
            "why_it_matters": "High humidity may affect equipment performance and increase maintenance needs.",
            "recommended_actions": [
                "Protect electrical equipment from moisture.",
                "Inspect machines for corrosion.",
                "Increase environmental monitoring."
            ]
        })

    # Overall Weather Recommendation
    if weather_stress >= 0.7:
        recommendations.append({
            "priority": "High",
            "issue_detected": "Adverse Weather Conditions",
            "why_it_matters": "Combined weather conditions may create operational challenges.",
            "recommended_actions": [
                "Use weather-based operational planning.",
                "Prepare alternative production schedules.",
                "Increase monitoring during adverse weather."
            ]
        })

    # Low Risk Status
    if shortfall_risk == "Low":
        recommendations.append({
            "priority": "Low",
            "issue_detected": "Current Operational Status",
            "why_it_matters": "The overall production shortfall risk is currently low.",
            "recommended_actions": [
                "Continue regular monitoring.",
                "Maintain preventive maintenance.",
                "Track production and weather changes."
            ]
        })

    # Generate Simple AI Summary
    top_factor = risk_factors[0]["factor"]

    if shortfall_risk == "High":
        ai_summary = (
            f"The system predicts a HIGH production shortfall risk. "
            f"The main area requiring attention is {top_factor}. "
            f"Immediate preventive action is recommended to reduce "
            f"the possibility of production disruption."
        )
    elif shortfall_risk == "Medium":
        ai_summary = (
            f"The system predicts a MODERATE production shortfall risk. "
            f"The main concern is {top_factor}. Early preventive action "
            f"can help reduce the risk."
        )
    else:
        ai_summary = (
            f"The overall production shortfall risk is currently LOW. "
            f"However, the system detected {top_factor} as an important "
            f"operational factor. Regular monitoring and preventive action "
            f"are recommended to maintain stable production."
        )

    return {
        "current_risk": shortfall_risk,
        "ai_summary": ai_summary,
        "top_risk_factors": risk_factors,
        "recommendations": recommendations
    }


# =========================================================
# ROOT (SERVE HTML)
# =========================================================

@app.get("/")
def read_root():
    if not HTML_FILE_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="my_gpt.html file not found in directory"
        )
    return FileResponse(HTML_FILE_PATH)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "equipment_model_loaded": equipment_model is not None,
        "shortfall_model_loaded": shortfall_model is not None,
        "metadata_loaded": metadata is not None,
        "dataset_loaded": df is not None
    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(
            status_code=500,
            detail="Model metadata not loaded"
        )
    return {"metadata": metadata}


# =========================================================
# EQUIPMENT MODEL INFORMATION
# =========================================================

@app.get("/equipment-info")
def equipment_info():
    return {
        "features": [
            "Production_Tonnes",
            "Avg_Temperature_C",
            "Total_Rainfall_mm",
            "Avg_Humidity_pct",
            "Production_Stress",
            "Temperature_Stress",
            "Rainfall_Stress",
            "Humidity_Stress"
        ]
    }


# =========================================================
# GET AVAILABLE STATES
# =========================================================

@app.get("/states")
def get_states():
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    states = sorted(
        df["State"]
        .dropna()
        .unique()
        .tolist()
    )
    return {"states": states}


# =========================================================
# GET DISTRICTS
# =========================================================

@app.get("/districts/{state}")
def get_districts(state: str):
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    state_data = df[
        df["State"]
        .str.lower()
        .str.strip()
        == state.lower().strip()
    ]

    if state_data.empty:
        raise HTTPException(
            status_code=404,
            detail="State not found"
        )

    districts = sorted(
        state_data["District"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "state": state,
        "districts": districts
    }


# =========================================================
# MAIN PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: PredictionInput):

    # Check Models
    if equipment_model is None:
        raise HTTPException(status_code=500, detail="Equipment model not loaded")
    if shortfall_model is None:
        raise HTTPException(status_code=500, detail="Shortfall model not loaded")
    if metadata is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    # Find Selected District
    district_data = df[
        (df["State"].str.lower().str.strip() == data.state.lower().strip()) &
        (df["District"].str.lower().str.strip() == data.district.lower().strip())
    ].copy()

    if district_data.empty:
        raise HTTPException(
            status_code=404,
            detail="Selected State/District not found in dataset"
        )

    # Get Production Data
    production_source = ""
    selected_year = None

    if data.production_tonnes is not None and data.production_tonnes > 0:
        production = float(data.production_tonnes)
        production_source = "Manual input"
    else:
        latest_year_data = district_data[
            district_data["Year"].astype(str).str.strip() == "2025-26"
        ]

        if not latest_year_data.empty:
            latest_row = latest_year_data.iloc[-1]
            production = float(latest_row["Production_Tonnes"])
            selected_year = "2025-26"
            production_source = "Automatic latest historical data"
        else:
            latest_row = district_data.iloc[-1]
            production = float(latest_row["Production_Tonnes"])
            selected_year = str(latest_row["Year"])
            production_source = "Latest available historical data"

    # Get Weather Values & Calculate Stress
    temperature, rainfall, humidity = get_weather_values(
        district_data,
        data.weather_condition
    )

    stress = calculate_stress(
        district_data,
        production,
        temperature,
        rainfall,
        humidity
    )

    # Equipment Risk Prediction
    equipment_features = pd.DataFrame(
        [[
            production,
            temperature,
            rainfall,
            humidity,
            stress["production_stress"],
            stress["temperature_stress"],
            stress["rainfall_stress"],
            stress["humidity_stress"]
        ]],
        columns=[
            "Production_Tonnes",
            "Avg_Temperature_C",
            "Total_Rainfall_mm",
            "Avg_Humidity_pct",
            "Production_Stress",
            "Temperature_Stress",
            "Rainfall_Stress",
            "Humidity_Stress"
        ]
    )

    equipment_risk = float(equipment_model.predict(equipment_features)[0])
    equipment_risk = float(np.clip(equipment_risk, 0, 100))

    # Shortfall Model Input & Prediction
    shortfall_features = [
        temperature,
        rainfall,
        humidity,
        stress["temperature_stress"],
        stress["rainfall_stress"],
        stress["humidity_stress"],
        stress["weather_stress"],
        stress["production_stress"],
        equipment_risk
    ]

    prediction = int(shortfall_model.predict(np.array([shortfall_features]))[0])

    reverse_label_map = {0: "Low", 1: "Medium", 2: "High"}
    if isinstance(metadata, dict) and "reverse_label_map" in metadata:
        reverse_label_map = {
            int(k): v for k, v in metadata["reverse_label_map"].items()
        }

    shortfall_risk = reverse_label_map.get(prediction, "Unknown")

    # Recommendations
    ai_recommendations = generate_ai_recommendations(
        production_stress=stress["production_stress"],
        temperature_stress=stress["temperature_stress"],
        rainfall_stress=stress["rainfall_stress"],
        humidity_stress=stress["humidity_stress"],
        weather_stress=stress["weather_stress"],
        equipment_risk=equipment_risk,
        shortfall_risk=shortfall_risk
    )

    return {
        "state": data.state,
        "district": data.district,
        "production_used": {
            "production_tonnes": round(production, 2),
            "year": selected_year,
            "source": production_source
        },
        "weather_condition": data.weather_condition,
        "weather_values": {
            "temperature_c": round(temperature, 2),
            "rainfall_mm": round(rainfall, 2),
            "humidity_percent": round(humidity, 2)
        },
        "stress_analysis": stress,
        "equipment_breakdown_risk_percent": round(equipment_risk, 2),
        "production_shortfall_risk": shortfall_risk,
        "ai_recommendations": ai_recommendations
    }
