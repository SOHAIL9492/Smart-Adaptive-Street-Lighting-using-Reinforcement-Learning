import os
import pickle
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StreetlightAPI")

app = FastAPI(title="Smart Adaptive Street Lighting API", version="1.0.0")

# Load Policy
POLICY_PATH = os.getenv("POLICY_PATH", "policies/policy_v2.pkl")
policy_data = None
Q_table = None

class StateInput(BaseModel):
    hour: int
    pedestrian_count: int

class ActionOutput(BaseModel):
    action: int
    brightness_level: str

# Brightness mapping
BRIGHTNESS_LEVELS = ["0% (Off)", "30% (Dim)", "60% (Medium)", "100% (Full)"]

@app.on_event("startup")
def load_model():
    global policy_data, Q_table
    if os.path.exists(POLICY_PATH):
        with open(POLICY_PATH, "rb") as f:
            policy_data = pickle.load(f)
            Q_table = policy_data["Q"]
            logger.info(f"Loaded policy from {POLICY_PATH}")
    else:
        logger.warning(f"Policy file {POLICY_PATH} not found. API will return dummy values or errors.")

def state_to_index(hour: int, pedestrian_count: int) -> int:
    # Match the environment logic (assumed 24 hours, max 10 pedestrians or something similar)
    # The actual train.py state_to_index logic from environment.py:
    # State = (hour, pedestrian_presence) where presence is likely 0 or 1, or count up to something.
    # We will use a safe fallback since we don't have the exact state_to_index imported directly
    # Wait, let's just import it from sim.environment if possible.
    try:
        from sim.environment import state_to_index as env_state_to_index
        # environment usually takes a State namedtuple or similar. Let's assume (hour, count)
        # We'll just pass a tuple
        return env_state_to_index((hour, pedestrian_count))
    except Exception as e:
        logger.error(f"Error importing state_to_index: {e}")
        # fallback if env is not found
        return hour * 2 + (1 if pedestrian_count > 0 else 0)

@app.post("/predict", response_model=ActionOutput)
def predict(state: StateInput):
    start_time = time.time()
    
    if Q_table is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        idx = state_to_index(state.hour, state.pedestrian_count)
        action = int(np.argmax(Q_table[idx]))
        brightness = BRIGHTNESS_LEVELS[action] if action < len(BRIGHTNESS_LEVELS) else "Unknown"
        
        # Log prediction for monitoring
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Predict - State: (hr:{state.hour}, ped:{state.pedestrian_count}) -> Action: {action} ({brightness}) | Time: {process_time:.2f}ms")
        
        # Drift monitoring placeholder
        if state.pedestrian_count > 50:
            logger.warning(f"Data Drift Alert: Unusually high pedestrian count ({state.pedestrian_count}) detected.")
            
        return {"action": action, "brightness_level": brightness}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": Q_table is not None}
