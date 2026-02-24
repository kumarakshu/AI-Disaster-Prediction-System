from database.db import save_prediction
from database.db import save_prediction, get_predictions
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Create FastAPI instance
app = FastAPI(
    title="AI Disaster Prediction API",
    description="Predict flood disaster risk using ML model",
    version="1.0"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("../model/flood_xgb_model.pkl")

# Define input schema
class FloodFeatures(BaseModel):
    MonsoonIntensity: float
    TopographyDrainage: float
    RiverManagement: float
    Deforestation: float
    Urbanization: float
    ClimateChange: float
    DamsQuality: float
    Siltation: float
    AgriculturalPractices: float
    Encroachments: float
    IneffectiveDisasterPreparedness: float   
    DrainageSystems: float
    CoastalVulnerability: float
    Landslides: float
    Watersheds: float
    DeterioratingInfrastructure: float
    PopulationScore: float
    WetlandLoss: float
    InadequatePlanning: float
    PoliticalFactors: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "AI Disaster Prediction API running successfully"}

# Prediction endpoint
@app.post("/predict")
def predict(data: FloodFeatures):

    input_data = np.array([
        data.MonsoonIntensity,
        data.TopographyDrainage,
        data.RiverManagement,
        data.Deforestation,
        data.Urbanization,
        data.ClimateChange,
        data.DamsQuality,
        data.Siltation,
        data.AgriculturalPractices,
        data.Encroachments,
        data.IneffectiveDisasterPreparedness,  # ADD THIS
        data.DrainageSystems,
        data.CoastalVulnerability,
        data.Landslides,
        data.Watersheds,
        data.DeterioratingInfrastructure,
        data.PopulationScore,
        data.WetlandLoss,
        data.InadequatePlanning,
        data.PoliticalFactors
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Save prediction in database
    save_prediction(
        data.MonsoonIntensity,
        data.Urbanization,
        data.ClimateChange,
        int(prediction),
        float(probability)
    )

    return {
        "FloodRisk": int(prediction),
        "Probability": float(probability)
    }


@app.get("/feature-importance")
def feature_importance():

    importances = model.feature_importances_

    feature_names = [
        "MonsoonIntensity",
        "TopographyDrainage",
        "RiverManagement",
        "Deforestation",
        "Urbanization",
        "ClimateChange",
        "DamsQuality",
        "Siltation",
        "AgriculturalPractices",
        "Encroachments",
        "IneffectiveDisasterPreparedness",
        "DrainageSystems",
        "CoastalVulnerability",
        "Landslides",
        "Watersheds",
        "DeterioratingInfrastructure",
        "PopulationScore",
        "WetlandLoss",
        "InadequatePlanning",
        "PoliticalFactors"
    ]

    result = {}

    for name, importance in zip(feature_names, importances):
        result[name] = float(importance)

    return result

@app.get("/prediction-history")
def prediction_history():

    data = get_predictions()

    return {
        "total_predictions": len(data),
        "predictions": data
    }