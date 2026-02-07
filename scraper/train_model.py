"""
Model training and comparison pipeline.
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
DATA_PATH = DATA_DIR / "laptops.csv"

RANDOM_STATE = 42

# Columns to EXCLUDE
EXCLUDE_COLUMNS = ['title', 'description', 'link']

# Target
TARGET = 'price'

# Outlier thresholds
MIN_PRICE = 500
MAX_PRICE = 50000

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_analyze_data() -> pd.DataFrame:
    """Load data and print analysis."""
    df = pd.read_csv(DATA_PATH)
    
    print("=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Identify column types
    print(f"\nColumns by type:")
    print(f"  Excluded: {EXCLUDE_COLUMNS}")
    
    # Available features
    available = [c for c in df.columns if c not in EXCLUDE_COLUMNS and c != TARGET]
    print(f"  Available features: {len(available)}")
    
    # NaN analysis
    print(f"\nMissing Value Analysis:")
    for col in available:
        nan_count = df[col].isna().sum()
        nan_pct = nan_count / len(df) * 100
        if nan_count > 0:
            print(f"  {col:20}: {nan_count:>5} ({nan_pct:>5.1f}%)")
    
    return df


def identify_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numeric, boolean, and categorical features."""
    available = [c for c in df.columns if c not in EXCLUDE_COLUMNS and c != TARGET]
    
    # Known types based on column inspection
    boolean_cols = ['is_shop', 'has_delivery', 'is_new', 'is_touchscreen', 'is_ssd']
    categorical_cols = ['brand', 'model', 'cpu', 'ram', 'storage', 'gpu', 
                        'city', 'cpu_family', 'gpu_type', 'gpu_family']
    
    # All score columns are numeric
    score_cols = [c for c in available if c.endswith('_score')]
    
    # Numeric features
    numeric_cols = ['price', 'screen_size', 'refresh_rate', 'cpu_generation',
                    'ram_gb', 'storage_gb'] + score_cols
    
    # Handle any remaining columns
    remaining = set(available) - set(boolean_cols) - set(categorical_cols) - set(numeric_cols)
    if remaining:
        print(f"Unclassified columns (treating as categorical): {remaining}")
        categorical_cols.extend(remaining)
    
    # Filter to only columns that exist
    boolean_cols = [c for c in boolean_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns and c != TARGET]
    
    return {
        'boolean': boolean_cols,
        'categorical': categorical_cols,
        'numeric': numeric_cols,
    }


def preprocess_for_xgboost(df: pd.DataFrame, 
                            feature_types: Dict[str, List[str]],
                            remove_outliers: bool = True,
                            log_transform: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], Dict, Dict]:
    """
    Preprocess data for XGBoost with explicit imputation.
    
    Strategy:
    - Numeric NaN → Median
    - Categorical NaN → 'Unknown'
    - Boolean NaN → False (0)
    """
    df = df.copy()
    
    # Filter valid prices
    df = df[df[TARGET] > 0]
    
    # Remove outliers
    if remove_outliers:
        before = len(df)
        df = df[(df[TARGET] >= MIN_PRICE) & (df[TARGET] <= MAX_PRICE)]
        print(f"XGBoost: Removed {before - len(df)} outliers, {len(df)} remaining")
    
    # Build feature list
    all_features = (feature_types['numeric'] + 
                    feature_types['boolean'] + 
                    feature_types['categorical'])
    all_features = [f for f in all_features if f in df.columns]
    
    # Initialize encoders
    encoders = {}
    
    # Process each column
    X_df = pd.DataFrame()
    
    for col in feature_types['numeric']:
        if col in df.columns:
            # IMPUTATION: Fill NaN with median
            median_val = df[col].median()
            X_df[col] = df[col].fillna(median_val)
    
    for col in feature_types['boolean']:
        if col in df.columns:
            # IMPUTATION: Fill NaN with False (0)
            X_df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    for col in feature_types['categorical']:
        if col in df.columns:
            # IMPUTATION: Fill NaN with 'Unknown', then encode
            series = df[col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            X_df[col] = le.fit_transform(series)
            encoders[col] = le
    
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df[TARGET].values
    
    # Transform info
    transform_info = {
        'log_transform': log_transform,
        'imputation': 'median_for_numeric_unknown_for_categorical',
    }
    
    if log_transform:
        y = np.log1p(y)
    
    return X, y, feature_names, encoders, transform_info


def preprocess_for_catboost(df: pd.DataFrame,
                             feature_types: Dict[str, List[str]],
                             remove_outliers: bool = True,
                             log_transform: bool = True) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[int], Dict]:
    """
    Preprocess data for CatBoost with NATIVE NaN handling.
    
    Strategy:
    - Numeric NaN → Left as NaN (CatBoost handles natively)
    - Categorical NaN → Left as 'nan' string (CatBoost handles natively)
    - Boolean NaN → Left as NaN
    """
    df = df.copy()
    
    # Filter valid prices
    df = df[df[TARGET] > 0]
    
    # Remove outliers
    if remove_outliers:
        before = len(df)
        df = df[(df[TARGET] >= MIN_PRICE) & (df[TARGET] <= MAX_PRICE)]
        print(f"CatBoost: Removed {before - len(df)} outliers, {len(df)} remaining")
    
    # Build feature DataFrame
    X_df = pd.DataFrame()
    
    for col in feature_types['numeric']:
        if col in df.columns:
            # NO IMPUTATION - leave NaN for CatBoost native handling
            X_df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in feature_types['boolean']:
        if col in df.columns:
            # Convert to int, NaN stays as NaN
            X_df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    for col in feature_types['categorical']:
        if col in df.columns:
            # Keep as string for CatBoost native encoding
            X_df[col] = df[col].fillna('_NaN_').astype(str)
    
    feature_names = list(X_df.columns)
    cat_feature_indices = [feature_names.index(c) for c in feature_types['categorical'] if c in feature_names]
    
    y = df[TARGET].values
    
    transform_info = {
        'log_transform': log_transform,
        'imputation': 'native_catboost_handling',
    }
    
    if log_transform:
        y = np.log1p(y)
    
    return X_df, y, feature_names, cat_feature_indices, transform_info


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, 
                 cv: int = 5, n_jobs: int = -1) -> Dict[str, Any]:
    """Full XGBoost hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 60)
    
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2],
    }
    
    grid_size = np.prod([len(v) for v in param_grid.values()])
    print(f"Grid size: {grid_size} combinations")
    print(f"Total fits: {grid_size * cv}")
    
    model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0)
    
    # Use RandomizedSearchCV for faster tuning if grid is too large
    if grid_size > 200:
        from sklearn.model_selection import RandomizedSearchCV
        grid_search = RandomizedSearchCV(
            model, param_grid, n_iter=100, cv=cv,
            scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=1, random_state=RANDOM_STATE
        )
    else:
        grid_search = GridSearchCV(
            model, param_grid, cv=cv,
            scoring='neg_mean_absolute_error', n_jobs=n_jobs, verbose=1
        )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    tune_time = time.time() - start
    
    print(f"\nTuning time: {tune_time:.1f}s")
    print("Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV MAE (log): {-grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': float(-grid_search.best_score_),
        'tune_time': tune_time,
    }


def tune_catboost(X_train: pd.DataFrame, y_train: np.ndarray,
                  cat_features: List[int], cv: int = 5) -> Dict[str, Any]:
    """Full CatBoost hyperparameter tuning."""
    print("\n" + "=" * 60)
    print("CATBOOST HYPERPARAMETER TUNING")
    print("=" * 60)
    
    param_grid = {
        'iterations': [200, 400, 600],
        'depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'border_count': [32, 64, 128],
    }
    
    grid_size = np.prod([len(v) for v in param_grid.values()])
    print(f"Grid size: {grid_size} combinations")
    
    # CatBoost doesn't support sklearn GridSearchCV directly for cat_features
    # Use CatBoost's built-in grid search or manual loop
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    from itertools import product
    
    # Reduced grid for speed
    param_grid_reduced = {
        'iterations': [400, 600],
        'depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'l2_leaf_reg': [1, 3],
    }
    
    total_combos = np.prod([len(v) for v in param_grid_reduced.values()])
    print(f"Using reduced grid: {total_combos} combinations")
    
    start = time.time()
    
    keys = list(param_grid_reduced.keys())
    combo_idx = 0
    
    for values in product(*param_grid_reduced.values()):
        combo_idx += 1
        params = dict(zip(keys, values))
        
        # Cross-validation
        cv_scores = []
        for fold in range(cv):
            # Simple CV split
            fold_size = len(X_train) // cv
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            X_tr = pd.concat([X_train.iloc[:val_start], X_train.iloc[val_end:]])
            y_tr = np.concatenate([y_train[:val_start], y_train[val_end:]])
            X_val = X_train.iloc[val_start:val_end]
            y_val = y_train[val_start:val_end]
            
            model = CatBoostRegressor(
                **params,
                cat_features=cat_features,
                random_state=RANDOM_STATE,
                verbose=0
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_scores.append(mae)
        
        mean_mae = np.mean(cv_scores)
        results.append({'params': params, 'mae': mean_mae})
        
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params
        
        if combo_idx % 5 == 0:
            print(f"  Progress: {combo_idx}/{total_combos}, Best MAE: {best_score:.4f}")
    
    tune_time = time.time() - start
    
    print(f"\nTuning time: {tune_time:.1f}s")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV MAE (log): {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'tune_time': tune_time,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                   model_name: str, log_transform: bool = True) -> Dict[str, Any]:
    """Comprehensive evaluation."""
    
    # Inverse transform
    if log_transform:
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
    else:
        y_true = y_true_log
        y_pred = y_pred_log
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    median_error = np.median(np.abs(y_true - y_pred))
    
    print(f"\n{model_name} Test Metrics:")
    print(f"  R²:           {r2:.4f}")
    print(f"  MAE:          {mae:,.0f} DH")
    print(f"  RMSE:         {rmse:,.0f} DH")
    print(f"  MAPE:         {mape:.2f}%")
    print(f"  Median Error: {median_error:,.0f} DH")
    
    # Accuracy thresholds
    abs_errors = np.abs(y_true - y_pred)
    print(f"\n  Prediction Accuracy:")
    for pct in [10, 15, 20, 25]:
        within = np.mean(abs_errors / y_true < pct / 100) * 100
        print(f"    Within {pct}%: {within:.1f}%")
    
    return {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'median_error': float(median_error),
        'within_10pct': float(np.mean(abs_errors / y_true < 0.10) * 100),
        'within_15pct': float(np.mean(abs_errors / y_true < 0.15) * 100),
        'within_20pct': float(np.mean(abs_errors / y_true < 0.20) * 100),
    }


# ============================================================================
# MAIN COMPARISON PIPELINE
# ============================================================================

def run_comparison(save_models: bool = True) -> Dict[str, Any]:
    """Run full XGBoost vs CatBoost comparison."""
    
    print("=" * 60)
    print("XGBOOST vs CATBOOST - FULL COMPARISON")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and analyze data
    df = load_and_analyze_data()
    feature_types = identify_feature_types(df)
    
    print(f"\nFeature Types:")
    print(f"  Numeric ({len(feature_types['numeric'])}): {feature_types['numeric']}")
    print(f"  Boolean ({len(feature_types['boolean'])}): {feature_types['boolean']}")
    print(f"  Categorical ({len(feature_types['categorical'])}): {feature_types['categorical']}")
    
    results = {}
    
    # ========== XGBOOST ==========
    print("\n" + "=" * 60)
    print("PREPARING XGBOOST DATA")
    print("=" * 60)
    
    X_xgb, y_xgb, feat_names_xgb, encoders_xgb, transform_xgb = preprocess_for_xgboost(
        df, feature_types, remove_outliers=True, log_transform=True
    )
    print(f"XGBoost features: {len(feat_names_xgb)}")
    
    # Split
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_xgb, y_xgb, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Tune XGBoost
    xgb_tuning = tune_xgboost(X_train_xgb, y_train_xgb)
    
    # Train final XGBoost
    print("\nTraining final XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        **xgb_tuning['best_params'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    xgb_metrics = evaluate_model(y_test_xgb, y_pred_xgb, "XGBoost", log_transform=True)
    
    results['XGBoost'] = {
        'metrics': xgb_metrics,
        'tuning': xgb_tuning,
        'feature_names': feat_names_xgb,
        'missing_handling': 'median_imputation_for_numeric_unknown_for_categorical',
    }
    
    # ========== CATBOOST ==========
    if HAS_CATBOOST:
        print("\n" + "=" * 60)
        print("PREPARING CATBOOST DATA")
        print("=" * 60)
        
        X_cat, y_cat, feat_names_cat, cat_indices, transform_cat = preprocess_for_catboost(
            df, feature_types, remove_outliers=True, log_transform=True
        )
        print(f"CatBoost features: {len(feat_names_cat)}")
        print(f"Categorical feature indices: {cat_indices}")
        
        # Split
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            X_cat, y_cat, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Tune CatBoost
        cat_tuning = tune_catboost(X_train_cat, y_train_cat, cat_indices)
        
        # Train final CatBoost
        print("\nTraining final CatBoost model...")
        cat_model = CatBoostRegressor(
            **cat_tuning['best_params'],
            cat_features=cat_indices,
            random_state=RANDOM_STATE,
            verbose=0
        )
        cat_model.fit(X_train_cat, y_train_cat)
        
        # Evaluate CatBoost
        y_pred_cat = cat_model.predict(X_test_cat)
        cat_metrics = evaluate_model(y_test_cat, y_pred_cat, "CatBoost", log_transform=True)
        
        results['CatBoost'] = {
            'metrics': cat_metrics,
            'tuning': cat_tuning,
            'feature_names': feat_names_cat,
            'missing_handling': 'native_catboost_handling',
        }
    else:
        print("\n" + "=" * 60)
        print("SKIPPING CATBOOST (not installed)")
        print("=" * 60)
        cat_metrics = None
    
    # ========== COMPARISON SUMMARY ==========
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'XGBoost':>12} {'CatBoost':>12} {'Winner':>12}")
    print("-" * 60)
    
    comparisons = [
        ('R²', xgb_metrics['r2'], cat_metrics['r2'], 'higher'),
        ('MAE (DH)', xgb_metrics['mae'], cat_metrics['mae'], 'lower'),
        ('RMSE (DH)', xgb_metrics['rmse'], cat_metrics['rmse'], 'lower'),
        ('MAPE (%)', xgb_metrics['mape'], cat_metrics['mape'], 'lower'),
        ('Median Error', xgb_metrics['median_error'], cat_metrics['median_error'], 'lower'),
        ('Within 10%', xgb_metrics['within_10pct'], cat_metrics['within_10pct'], 'higher'),
        ('Within 15%', xgb_metrics['within_15pct'], cat_metrics['within_15pct'], 'higher'),
        ('Within 20%', xgb_metrics['within_20pct'], cat_metrics['within_20pct'], 'higher'),
    ]
    
    xgb_wins = 0
    cat_wins = 0
    
    for metric, xgb_val, cat_val, better in comparisons:
        if better == 'higher':
            winner = 'XGBoost' if xgb_val > cat_val else 'CatBoost'
        else:
            winner = 'XGBoost' if xgb_val < cat_val else 'CatBoost'
        
        if winner == 'XGBoost':
            xgb_wins += 1
        else:
            cat_wins += 1
        
        if 'DH' in metric or metric == 'Median Error':
            print(f"{metric:<20} {xgb_val:>12,.0f} {cat_val:>12,.0f} {winner:>12}")
        elif '%' in metric or 'Within' in metric:
            print(f"{metric:<20} {xgb_val:>11.1f}% {cat_val:>11.1f}% {winner:>12}")
        else:
            print(f"{metric:<20} {xgb_val:>12.4f} {cat_val:>12.4f} {winner:>12}")
    
    print("-" * 60)
    overall_winner = 'XGBoost' if xgb_wins > cat_wins else 'CatBoost'
    print(f"\n🏆 OVERALL WINNER: {overall_winner} ({max(xgb_wins, cat_wins)}/{len(comparisons)} metrics)")
    
    results['winner'] = overall_winner
    results['xgb_wins'] = xgb_wins
    results['cat_wins'] = cat_wins
    
    # Save models
    if save_models:
        MODEL_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save XGBoost
        xgb_path = MODEL_DIR / f"xgb_optimized_{timestamp}"
        xgb_path.mkdir(exist_ok=True)
        xgb_model.save_model(str(xgb_path / "model.json"))
        with open(xgb_path / "encoders.pkl", 'wb') as f:
            pickle.dump(encoders_xgb, f)
        with open(xgb_path / "metadata.json", 'w') as f:
            json.dump({
                'model_type': 'XGBoost',
                'created_at': datetime.now().isoformat(),
                'metrics': xgb_metrics,
                'best_params': xgb_tuning['best_params'],
                'feature_names': feat_names_xgb,
            }, f, indent=2)
        print(f"\nXGBoost saved to: {xgb_path}")
        
        # Save CatBoost
        cat_path = MODEL_DIR / f"catboost_optimized_{timestamp}"
        cat_path.mkdir(exist_ok=True)
        cat_model.save_model(str(cat_path / "model.cbm"))
        with open(cat_path / "metadata.json", 'w') as f:
            json.dump({
                'model_type': 'CatBoost',
                'created_at': datetime.now().isoformat(),
                'metrics': cat_metrics,
                'best_params': cat_tuning['best_params'],
                'feature_names': feat_names_cat,
                'cat_feature_indices': cat_indices,
            }, f, indent=2)
        print(f"CatBoost saved to: {cat_path}")
        
        # Save comparison results
        with open(MODEL_DIR / f"comparison_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Comparison results saved to: {MODEL_DIR / f'comparison_results_{timestamp}.json'}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    results = run_comparison(save_models=True)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
