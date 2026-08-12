"""
Train + tune 3 model families (Ridge, RandomForest, HistGradientBoosting)
via cross-validated search on the California Housing dataset, and save
the best pipeline to models/best_model.joblib.

Run: python -m src.train
https://codeaiflow.cloud/
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import FeatureEngineer, load_dataset, stratified_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def build_pipeline(estimator) -> Pipeline:
    """Feature engineering -> scaling -> estimator. Scaling is included
    even for tree-based models (harmless for them) so the same pipeline
    shape works for every model family being compared."""
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_baseline(X_train, y_train, X_val, y_val) -> float:
    """Milestone 2: plain LinearRegression baseline, no tuning, no
    feature engineering — the floor every tuned model must beat."""
    pipe = Pipeline([("scaler", StandardScaler()), ("estimator", LinearRegression())])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    return rmse(y_val, preds)


def run_search(name: str, estimator, param_grid: dict, X_train, y_train, cv):
    pipe = build_pipeline(estimator)
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    cv_rmse_mean = -search.best_score_
    cv_rmse_std = search.cv_results_["std_test_score"][search.best_index_]
    return {
        "name": name,
        "best_params": search.best_params_,
        "cv_rmse_mean": float(cv_rmse_mean),
        "cv_rmse_std": float(cv_rmse_std),
        "best_estimator": search.best_estimator_,
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    X_train, X_test, y_train, y_test = stratified_split(df)

    # Milestone 2 — baseline (held out from CV, just a quick val split)
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_val, y_tr, y_val = tts(X_train, y_train, test_size=0.2, random_state=42)
    baseline_rmse = evaluate_baseline(X_tr, y_tr, X_val, y_val)
    print(f"[baseline] plain LinearRegression val RMSE: {baseline_rmse:.4f}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    searches = [
        ("ridge", Ridge(), {"estimator__alpha": [0.1, 1.0, 10.0, 50.0]}),
        ("random_forest", RandomForestRegressor(random_state=42, n_jobs=-1), {
            "estimator__n_estimators": [200, 400],
            "estimator__max_depth": [None, 10, 20],
        }),
        ("hist_gb", HistGradientBoostingRegressor(random_state=42), {
            "estimator__max_depth": [None, 6, 10],
            "estimator__learning_rate": [0.05, 0.1],
        }),
    ]

    results = []
    for name, estimator, grid in searches:
        print(f"[search] running {name} ...")
        result = run_search(name, estimator, grid, X_train, y_train, cv)
        print(
            f"[search] {name}: CV RMSE = {result['cv_rmse_mean']:.4f} "
            f"± {result['cv_rmse_std']:.4f}  best_params={result['best_params']}"
        )
        results.append(result)

    best = min(results, key=lambda r: r["cv_rmse_mean"])
    print(f"\n[best model] {best['name']} — CV RMSE {best['cv_rmse_mean']:.4f}")

    joblib.dump(best["best_estimator"], MODELS_DIR / "best_model.joblib")

    summary = {
        "baseline_val_rmse": baseline_rmse,
        "cv_results": [
            {k: v for k, v in r.items() if k != "best_estimator"} for r in results
        ],
        "best_model": best["name"],
    }
    with open(REPORTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved best model to {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved training summary to {REPORTS_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
