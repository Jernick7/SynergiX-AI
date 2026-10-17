# train.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold

def train_synergix_model():
    print("Loading engineered features...")
    df = pd.read_csv("engineered_features.csv")

    # Target: Gamma synergy score (continuous)
    # Lower Gamma indicates higher synergy (typically < 0.95 or < 1.0)
    target_col = "Gamma"
    binary_col = "Label" if "Label" in df.columns else None

    # Exclude targets from training features
    drop_cols = [target_col]
    if binary_col:
        drop_cols.append(binary_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col].values
    feature_names = list(X.columns)

    print(f"Features: {len(feature_names)} | Samples: {len(X)}")

    # 5-Fold Cross Validation for reproducible proof of work
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        rf_fold = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_fold.fit(X_train, y_train)
        oof_predictions[val_idx] = rf_fold.predict(X_val)

    # Cross-validation metrics
    rmse = float(np.sqrt(mean_squared_error(y, oof_predictions)))
    r2 = float(r2_score(y, oof_predictions))
    print(f"5-Fold CV RMSE: {rmse:.4f}")
    print(f"5-Fold CV R2 Score: {r2:.4f}")

    # Binary classification metric if Label is present
    roc_auc = None
    if binary_col and df[binary_col].nunique() > 1:
        # Inverted Gamma as synergy likelihood (lower gamma = higher synergy probability)
        synergy_prob = -oof_predictions
        roc_auc = float(roc_auc_score(df[binary_col], synergy_prob))
        print(f"5-Fold Synergy ROC-AUC: {roc_auc:.4f}")

    # Train final model on 100% of the dataset
    print("Training final Random Forest model...")
    final_model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
    final_model.fit(X, y)

    # Feature Importance ranking
    importances = dict(zip(feature_names, [float(v) for v in final_model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    # Save artifacts for API and Streamlit UI
    model_payload = {
        "model": final_model,
        "feature_names": feature_names
    }
    joblib.dump(model_payload, "synergy_rf_model.joblib")
    print("Saved trained model to synergy_rf_model.joblib")

    metrics_payload = {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
        "feature_importances": sorted_importances,
        "sample_count": len(X),
        "feature_count": len(feature_names)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("Saved evaluation metrics to metrics.json")

if __name__ == "__main__":
    train_synergix_model()