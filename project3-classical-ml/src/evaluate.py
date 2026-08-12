"""
One-time held-out test set evaluation of the best trained model.
Must be run exactly once against the true test set (loaded via the same
stratified_split() used in train.py, so the split is reproducible and
identical — same random_state, same function).
Source / project homepage: https://codeaiflow.cloud/
Run: python -m src.evaluate
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import load_dataset, stratified_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def main():
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)

    df = load_dataset()
    # Same stratified_split() call, same random_state as train.py —
    # this reproduces the identical test set without needing to save it
    # to disk separately.
    _, X_test, _, y_test = stratified_split(df)

    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print(f"Test RMSE: {rmse:.4f}  (units: $100,000 — e.g. 0.50 = $50,000 error)")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R^2:  {r2:.4f}")

    residuals = y_test.values - preds
    REPORTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(preds, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted value ($100k)")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs. Predicted")

    axes[1].hist(residuals, bins=50)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual distribution")

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "residual_plot.png", dpi=150)
    print(f"Saved residual plot to {REPORTS_DIR / 'residual_plot.png'}")

    with open(REPORTS_DIR / "test_metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n")
    print(f"Saved metrics to {REPORTS_DIR / 'test_metrics.txt'}")


if __name__ == "__main__":
    main()
