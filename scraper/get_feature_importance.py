#!/usr/bin/env python3
"""Extract feature importance from saved XGBoost and CatBoost models."""
import json
from pathlib import Path
from catboost import CatBoostRegressor

MODEL_DIR = Path(__file__).parent / "models"

# Load XGBoost metadata
xgb_path = MODEL_DIR / "xgb_optimized_20260204_165950"
with open(xgb_path / "metadata.json") as f:
    xgb_meta = json.load(f)

print("=" * 60)
print("XGBOOST FEATURE IMPORTANCE (Top 15)")
print("=" * 60)

try:
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(xgb_path / "model.json"))
    xgb_importance = booster.get_score(importance_type='gain')
    xgb_features = xgb_meta["feature_names"]
    sorted_imp = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feat_idx, imp) in enumerate(sorted_imp[:15], 1):
        idx = int(feat_idx[1:])
        print(f"{i:2}. {xgb_features[idx]:20}: {imp:.2f}")
except Exception as e:
    print(f"XGBoost loading error: {e}")

# Load CatBoost
print()
print("=" * 60)
print("CATBOOST FEATURE IMPORTANCE (Top 15)")
print("=" * 60)

cat_path = MODEL_DIR / "catboost_optimized_20260204_165950"
cat_model = CatBoostRegressor()
cat_model.load_model(str(cat_path / "model.cbm"))
with open(cat_path / "metadata.json") as f:
    cat_meta = json.load(f)

cat_importance = cat_model.feature_importances_
cat_features = cat_meta["feature_names"]
cat_sorted = sorted(zip(cat_features, cat_importance), key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(cat_sorted[:15], 1):
    print(f"{i:2}. {feat:20}: {imp:.2f}")
