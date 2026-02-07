#!/usr/bin/env python3
"""
Price Predictor - User-facing laptop price prediction
======================================================
Takes a natural language laptop description, extracts specs via LLM,
then predicts price using CatBoost (chosen for native NaN handling).

Usage:
    python price_predictor.py "HP laptop with Core i5, 16GB RAM, 512GB SSD"
    python price_predictor.py  # Interactive mode
"""
import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from google import genai

# CatBoost for prediction (handles NaN natively)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("WARNING: CatBoost not installed. Price prediction will use dummy fallback.")

# Path configuration
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models"
CATBOOST_MODEL_PATH = MODEL_DIR / "catboost_optimized_20260204_165950"

# Load API key
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env if present
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Feature configuration matching the trained model
NUMERIC_FEATURES = [
    'screen_size', 'refresh_rate', 'cpu_generation', 'ram_gb', 'storage_gb',
    'cpu_score', 'gpu_score', 'ram_score', 'storage_score', 'screen_score',
    'condition_score', 'laptop_score'
]
BOOLEAN_FEATURES = ['is_shop', 'has_delivery', 'is_new', 'is_touchscreen', 'is_ssd']
CATEGORICAL_FEATURES = [
    'brand', 'model', 'cpu', 'ram', 'storage', 'gpu', 'city',
    'cpu_family', 'gpu_type', 'gpu_family', 'model_num', 'gpu_vram'
]


# ============================================================================
# LLM SPEC EXTRACTION
# ============================================================================

def build_extraction_prompt(user_description: str) -> str:
    """Build prompt for extracting laptop specs from user description."""
    return f"""You are a laptop specification extractor.
Extract ALL specifications you can find from this laptop description.

USER DESCRIPTION:
"{user_description}"

INSTRUCTIONS:
1. Extract any specifications mentioned or strongly implied
2. For each found spec, output in format: field=value
3. Use these field names:
   - brand (e.g., HP, Dell, Lenovo, Apple, ASUS)
   - model (e.g., Pavilion, ThinkPad, MacBook Pro)
   - cpu (full name, e.g., "Core i5-1235U", "Ryzen 5 5600H", "M1 Pro")
   - cpu_family (e.g., "Intel Core i5", "AMD Ryzen 5", "Apple M1")
   - cpu_generation (integer, e.g., 12 for 12th gen, 5000 for Ryzen 5000)
   - ram_gb (integer, e.g., 16)
   - storage_gb (integer in GB, e.g., 512 for 512GB SSD, 1000 for 1TB)
   - is_ssd (True/False)
   - gpu (full name if dedicated, e.g., "RTX 3050", "Radeon RX 6600M")
   - gpu_vram (integer in GB, e.g., 4, 6, 8, 16)
   - gpu_type (Dedicated/Integrated/Unknown)
   - screen_size (float in inches, e.g., 15.6)
   - refresh_rate (integer in Hz, e.g., 144)
   - is_new (True if new, False if used/refurbished)
   - is_touchscreen (True/False)

4. CRITICAL RULES:
   - Do NOT guess values that aren't mentioned or strongly implied
   - Output only fields you can extract
   - Output one field per line: field=value
   - No JSON, no explanation, just field=value lines

OUTPUT:"""


def extract_specs_from_description(user_description: str) -> Dict[str, Any]:
    """Use LLM to extract laptop specs from natural language description."""
    if not GEMINI_API_KEY:
        # Cannot proceed without API key
        print("GEMINI_API_KEY not found.")
        return {}
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = build_extraction_prompt(user_description)
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt
        )
        
        response_text = response.text.strip()
        
        # Parse the response
        specs = {}
        for line in response_text.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    specs[key] = value
        
        return specs
        
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return {}


# ============================================================================
# STRICT SCORING (Imported from parser.py)
# ============================================================================
try:
    from .parser import CPUScorer, GPUScorer, RAMScorer, StorageScorer, ScreenScorer, ConditionScorer
except ImportError:
    # Fallback for direct script execution
    from parser import CPUScorer, GPUScorer, RAMScorer, StorageScorer, ScreenScorer, ConditionScorer

def calculate_scores(specs: Dict) -> Dict[str, float]:
    """
    Calculate scores using the exact same logic as the training data parser.
    """
    scores = {}
    
    # 1. CPU Score
    # Extract needed fields with correct types
    cpu = specs.get('cpu', 'Unknown')
    cpu_family = specs.get('cpu_family', '')
    cpu_gen = None
    try:
        if specs.get('cpu_generation'):
            cpu_gen = int(specs.get('cpu_generation'))
    except:
        pass
        
    scores['cpu_score'] = CPUScorer.get_score(cpu, cpu_family, cpu_gen)
    
    # 2. GPU Score
    gpu = specs.get('gpu', 'Unknown')
    gpu_vram = None
    try:
        if specs.get('gpu_vram'):
            gpu_vram = int(specs.get('gpu_vram'))
    except:
        pass
        
    scores['gpu_score'] = GPUScorer.get_score(gpu, gpu_vram)
    
    # 3. RAM Score
    ram_gb = 0
    try:
        if specs.get('ram_gb'):
            ram_gb = int(specs.get('ram_gb'))
    except:
        pass
        
    scores['ram_score'] = RAMScorer.get_score(ram_gb)
    
    # 4. Storage Score
    storage_gb = 0
    try:
        if specs.get('storage_gb'):
            storage_gb = int(specs.get('storage_gb'))
    except:
        pass
    is_ssd = str(specs.get('is_ssd', '')).lower() == 'true'
    
    scores['storage_score'] = StorageScorer.get_score(storage_gb, is_ssd)
    
    # 5. Screen Score
    screen_size = None
    try:
        if specs.get('screen_size'):
            screen_size = float(specs.get('screen_size'))
    except:
        pass
        
    refresh_rate = None
    try:
        if specs.get('refresh_rate'):
            refresh_rate = int(specs.get('refresh_rate'))
    except:
        pass
    
    is_touch = str(specs.get('is_touchscreen', '')).lower() == 'true'
    
    scores['screen_score'] = ScreenScorer.get_score(screen_size, refresh_rate, is_touch)
    
    # 6. Condition Score
    is_new = str(specs.get('is_new', '')).lower() == 'true'
    brand = specs.get('brand', 'Unknown')
    
    scores['condition_score'] = ConditionScorer.get_score(is_new, brand)
    
    return scores


# ============================================================================
# CATBOOST PREDICTION
# ============================================================================

def load_catboost_model():
    """Load the trained CatBoost model and metadata."""
    if not CATBOOST_AVAILABLE:
        # Return Dummy model if not available
        return None, {'feature_names': NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES}
        
    try:
        model = CatBoostRegressor()
        model.load_model(str(CATBOOST_MODEL_PATH / "model.cbm"))
        
        with open(CATBOOST_MODEL_PATH / "metadata.json") as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        print(f"Model loading error: {e}")
        return None, {'feature_names': []}


def prepare_features_for_catboost(specs: Dict, metadata: Dict) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Prepare feature DataFrame for CatBoost prediction using STRICT scoring.
    """
    feature_names = metadata.get('feature_names', [])
    if not feature_names:
        feature_names = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
    
    # 1. Calculate specific scores using parser logic
    calculated_scores = calculate_scores(specs)
    
    # 2. Calculate composite laptop_score
    scores_available = [s for s in [calculated_scores['cpu_score'], calculated_scores['gpu_score']] if s > 0]
    if scores_available:
        # Include RAM/Storage/Screen/Condition in composite if they exist
        all_scores = list(scores_available)
        if calculated_scores.get('ram_score'): all_scores.append(calculated_scores['ram_score'])
        if calculated_scores.get('storage_score'): all_scores.append(calculated_scores['storage_score'])
        if calculated_scores.get('screen_score'): all_scores.append(calculated_scores['screen_score'])
        
        calculated_scores['laptop_score'] = np.mean(all_scores)
    else:
        calculated_scores['laptop_score'] = np.nan
    
    # Build feature row
    row = {}
    for feat in feature_names:
        # Map calculated scores directly
        if feat in calculated_scores:
            row[feat] = calculated_scores[feat]
            
        elif feat in BOOLEAN_FEATURES:
            val = specs.get(feat, '')
            if val.lower() == 'true':
                row[feat] = 1
            elif val.lower() == 'false':
                row[feat] = 0
            else:
                row[feat] = np.nan
                
        elif feat in CATEGORICAL_FEATURES:
            row[feat] = specs.get(feat, '_NaN_')
            
        else:
            # Numeric features (raw specs)
            val = specs.get(feat)
            if val:
                try:
                    row[feat] = float(val)
                except:
                    row[feat] = np.nan
            else:
                row[feat] = np.nan
    
    return pd.DataFrame([row]), calculated_scores


def predict_price(user_description: str) -> Dict[str, Any]:
    """
    Main prediction function.
    """
    # Step 1: Extract specs from description
    specs = extract_specs_from_description(user_description)
    
    if not specs:
        return {
            'error': 'Could not extract any specifications from description',
            'suggestion': 'Try including more details like brand, CPU, RAM, storage',
        }
    
    # Step 2: Load model
    model, metadata = load_catboost_model()
    
    # Step 3: Prepare features with strict scoring
    X, scores = prepare_features_for_catboost(specs, metadata)
    
    # Output details
    print(f"Input: {user_description}")
    
    print(f"{'SPECIFICATION':<20} {'VALUE':<25} {'SCORE':<10}")
    
    # CPU
    cpu_str = f"{specs.get('cpu_family', '')} {specs.get('cpu', '')} {specs.get('cpu_generation', '')}th".strip()
    print(f"{'CPU':<20} {cpu_str[:24]:<25} {scores.get('cpu_score', 0):.0f}")
    
    # GPU
    gpu_str = f"{specs.get('gpu', 'Integrated')} ({specs.get('gpu_vram', '0')}GB)"
    print(f"{'GPU':<20} {gpu_str[:24]:<25} {scores.get('gpu_score', 0):.0f}")
    
    # RAM
    print(f"{'RAM':<20} {specs.get('ram_gb', '?')} GB {'(DDR' + str(specs.get('ram_type','?')) + ')' if specs.get('ram_type') else ''}    {scores.get('ram_score', 0):.0f}")
    
    # Storage
    print(f"{'Storage':<20} {specs.get('storage_gb', '?')} GB {'SSD' if specs.get('is_ssd')=='True' else 'HDD'}       {scores.get('storage_score', 0):.0f}")
    
    # Screen
    print(f"{'Screen':<20} {specs.get('screen_size', '?')} inch {specs.get('refresh_rate', '?')}Hz     {scores.get('screen_score', 0):.0f}")
    
    # Condition
    cond = "New" if specs.get('is_new')=='True' else "Used"
    print(f"{'Condition':<20} {cond:<25} {scores.get('condition_score', 0):.0f}")
    
    print(f"{'OVERALL RATING':<46} {scores.get('laptop_score', 0):.0f}")

    # Step 4: Predict
    if model is not None:
        nan_count = X.isna().sum().sum()
        total_features = len(metadata['feature_names'])
        available_count = int(total_features - nan_count)
        
        try:
            y_pred_log = model.predict(X)[0]
            predicted_price = np.expm1(y_pred_log)
            predicted_price = float(round(predicted_price / 100) * 100)
        except Exception as e:
            print(f"Prediction failed: {e}")
            predicted_price = 0
            available_count = 0
    else:
        # Fallback heuristic if no model
        score = scores.get('laptop_score', 0)
        predicted_price = score * 50 if score > 0 else 2000
        available_count = 0
        print("Using dummy heuristic price.")
    
    print(f"ESTIMATED MARKET VALUE: {predicted_price:,.0f} DH")
    
    # Confidence
    confidence = "HIGH" if available_count >= 20 else ("MEDIUM" if available_count >= 10 else "LOW")
    print(f"Confidence: {confidence} ({available_count} features used)")
    
    # Ensure scores are native types
    safe_scores = {k: float(v) for k, v in scores.items()}
    
    return {
        'predicted_price': predicted_price,
        'confidence': confidence,
        'features_found': available_count,
        'extracted_specs': specs,
        'scores': safe_scores
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        description = " ".join(sys.argv[1:])
        result = predict_price(description)
    else:
        # Interactive mode
        print("LAPTOP PRICE PREDICTOR")
        print("\nDescribe the laptop you want to price:")
        print("(e.g., 'HP laptop with Core i5, 16GB RAM, 512GB SSD')")
        print()
        
        try:
            description = input("Your description: ").strip()
            if description:
                result = predict_price(description)
                print(f"\nResult: {json.dumps(result, indent=2)}")
            else:
                print("No description provided.")
        except KeyboardInterrupt:
            print("\nExiting...")
