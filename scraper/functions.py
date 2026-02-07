#!/usr/bin/env python3
"""
Functions Module - Core utility functions for laptop data analysis.

This module provides user-facing functions for querying and analyzing
the laptop dataset scraped from Avito.
"""
import csv
from pathlib import Path
from typing import List, Dict, Optional

# Data path
DATA_PATH = Path(__file__).parent / "data" / "laptops.csv"


def load_laptops() -> List[Dict]:
    """Load all laptops from CSV."""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def get_best_laptops(max_price: float, limit: int = 20) -> List[Dict]:
    """
    Get the best laptops ranked by score within a price budget.
    
    Args:
        max_price: Maximum price in DH
        limit: Maximum number of results (default 20)
        
    Returns:
        List of laptop dicts sorted by laptop_score descending
    """
    laptops = load_laptops()
    
    # Filter by price
    affordable = [
        laptop for laptop in laptops
        if float(laptop['price']) <= max_price and float(laptop['price']) > 0
    ]
    
    # Sort by score descending
    ranked = sorted(
        affordable,
        key=lambda x: int(x['laptop_score']),
        reverse=True
    )
    
    return ranked[:limit]


def print_laptop_ranking(max_price: float, limit: int = 20) -> None:
    """
    Print a formatted ranking of best laptops within budget.
    
    Args:
        max_price: Maximum price in DH
        limit: Maximum number of results (default 20)
    """
    laptops = get_best_laptops(max_price, limit)
    
    if not laptops:
        print(f"No laptops found under {max_price:,.0f} DH")
        return
    
    print()
    print("=" * 110)
    print(f"TOP {len(laptops)} LAPTOPS UNDER {max_price:,.0f} DH (Ranked by Score)")
    print("=" * 110)
    print(f"{'#':>2} {'Score':>5} {'Price':>8} {'Brand':8} {'CPU':22} {'GPU':16} {'RAM':>4} {'Storage':>8}")
    print("-" * 110)
    
    for i, laptop in enumerate(laptops, 1):
        score = laptop['laptop_score']
        price = f"{int(float(laptop['price'])):,}"
        brand = laptop['brand'][:8]
        cpu = laptop['cpu'][:22]
        gpu = laptop['gpu'][:16]
        ram = laptop['ram_gb']
        storage = laptop['storage'][:8]
        is_new = "NEW" if laptop['is_new'] == 'True' else ""
        
        print(f"{i:>2} {score:>5} {price:>8} {brand:8} {cpu:22} {gpu:16} {ram:>4}GB {storage:>8} {is_new}")
    
    print("-" * 110)
    print(f"Showing {len(laptops)} laptops under {max_price:,.0f} DH")
    print()


# ============================================================================
# ML PRICE PREDICTION FUNCTIONS
# ============================================================================

def train_price_model(do_tuning: bool = True) -> Dict:
    """
    Train the ML price prediction model.
    
    Args:
        do_tuning: Whether to run hyperparameter tuning (slower but better)
        
    Returns:
        Training results including metrics and model path
    """
    from ml_predictor import run_training_pipeline
    return run_training_pipeline(do_tuning=do_tuning)


def predict_price(laptop_specs: Dict) -> float:
    """
    Predict price for a laptop with given specs.
    
    Args:
        laptop_specs: Dictionary with laptop specifications
        
    Returns:
        Predicted price in DH
    """
    from ml_predictor import load_model, FEATURES
    import numpy as np
    
    model, feature_names, encoders, metadata = load_model()
    
    # Build feature vector
    features = []
    categorical = ['brand', 'gpu_type', 'cpu_family', 'gpu_family']
    boolean = ['is_ssd', 'is_new', 'is_touchscreen']
    
    for feat in feature_names:
        if feat in categorical:
            val = str(laptop_specs.get(feat, 'Unknown'))
            if val in encoders[feat].classes_:
                features.append(encoders[feat].transform([val])[0])
            else:
                features.append(0)  # Unknown category
        elif feat in boolean:
            val = laptop_specs.get(feat, False)
            features.append(1 if val else 0)
        else:
            try:
                features.append(float(laptop_specs.get(feat, 0)))
            except:
                features.append(0)
    
    # Predict
    X = np.array([features])
    price = model.predict(X)[0]
    
    return price


def get_model_info() -> Dict:
    """Get information about the currently saved model."""
    from ml_predictor import load_model
    
    try:
        model, feature_names, encoders, metadata = load_model()
        return {
            'model_name': metadata.get('model_name'),
            'created_at': metadata.get('created_at'),
            'features': feature_names,
            'metrics': metadata.get('metrics', {}),
        }
    except FileNotFoundError:
        return {'error': 'No saved model found. Run train_price_model() first.'}


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            max_price = float(sys.argv[1])
            print_laptop_ranking(max_price)
        except ValueError:
            print("Usage: python functions.py <max_price>")
            print("Example: python functions.py 8000")
    else:
        # Interactive mode
        print("=" * 50)
        print("LAPTOP FINDER - Best Deals by Score")
        print("=" * 50)
        try:
            max_price = float(input("Enter your maximum budget (DH): "))
            print_laptop_ranking(max_price)
        except ValueError:
            print("Invalid price. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
