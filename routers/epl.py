from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import sys
import os
import joblib
import pandas as pd

# --- Legacy Integration ---
# Add 'epl' path to find the optimized simulator.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../epl'))
import simulator

router = APIRouter()

# Global Legacy Assets
LEGACY_DF = None
LEGACY_MODEL = None

def load_legacy_assets():
    global LEGACY_DF, LEGACY_MODEL
    if LEGACY_DF is None or LEGACY_MODEL is None:
        try:
            LEGACY_DF, LEGACY_MODEL = simulator.load_assets()
            if LEGACY_DF is None:
                print("Error: Simulator returned None for assets.")
        except Exception as e:
            print(f"Error loading assets via simulator: {e}")

class ForcedMatch(BaseModel):
    Home: str
    Away: str
    HomeGoals: int
    AwayGoals: int

class SimulationRequest(BaseModel):
    forced_matches: list[ForcedMatch] = []
    n: int = 100

@router.post("/simulate")
async def run_simulation(req: SimulationRequest):
    try:
        load_legacy_assets()
        
        forced_dict = {}
        for m in req.forced_matches:
            forced_dict[(m.Home, m.Away)] = (m.HomeGoals, m.AwayGoals)
            
        results = simulator.run_simulation_api(
            LEGACY_DF, 
            LEGACY_MODEL, 
            simulator.MODEL_FEATURES, 
            forced_results=forced_dict, 
            simulations=req.n
        )
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/dashboard-data")
async def get_dashboard_data():
    try:
        load_legacy_assets()
        data = simulator.get_dashboard_initial_data(LEGACY_DF, LEGACY_MODEL, simulator.MODEL_FEATURES)
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/insights")
async def get_simulation_insights(results_list: list[dict]):
    try:
        load_legacy_assets()
        insights = simulator.get_insights(results_list, LEGACY_DF, simulator.MODEL_FEATURES)
        return insights
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh-data")
async def refresh_data():
    # Placeholder for legacy refresh if needed (e.g. re-running n_processing)
    return {"status": "Not implemented for legacy mode used in terminal"}
