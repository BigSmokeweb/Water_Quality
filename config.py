import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

SAFETY_THRESHOLDS: Dict[str, dict] = {
    "NO3": {
        "name": "Nitrate (NO3-N)",
        "unit": "mg/L",
        "caution": 7.0,
        "safe": 10.0,
        "description": "Drinking water & environmental trigger limit",
    },
    "Turbidity": {
        "name": "Turbidity",
        "unit": "NTU",
        "caution": 50.0,
        "safe": 100.0,
        "description": "Nephelometric Turbidity Units environmental trigger",
    },
    "Temp": {
        "name": "Water Temperature",
        "unit": "°C",
        "caution": 25.0,
        "safe": 30.0,
        "description": "Thermal baseline limit",
    },
}

TARGETS: List[str] = ["NO3", "Turbidity"]
FORECAST_STEPS: int = 6
N_CV_FOLDS: int = 5
RANDOM_STATE: int = 42

DATA_PATH: str = os.getenv("DATA_PATH", "data/raw/Johnstone_river_coquette_point_joined.csv")
ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "models/artifacts")
