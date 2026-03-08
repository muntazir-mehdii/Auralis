# models/train_xgb_filter.py

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SIGNALS_PATH = ROOT / "data" / "auralis_signals_xgb.parquet"
MODEL_PATH = ROOT / "models" / "xgb_filter.json"

def main():
    df = pd.read_parquet(SIGNALS_PATH)

    # Target: 1 if result_r > 0, else 0
    df["win"] = (df["result_r"] > 0).astype(int)

    feature_cols = ["asia_range", "hour", "dow", "price_close"]
    for col in ["ema_200", "atr_14", "rsi_14"]:
        if col in df.columns:
            feature_cols.append(col)

    # Ensure we have no NaNs
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Time-based split
    train_df = df[df["year"] <= 2021].reset_index(drop=True)
    val_df   = df[df["year"] == 2022].reset_index(drop=True)
    test_df  = df[df["year"] >= 2023].reset_index(drop=True)

    X_train, y_train = train_df[feature_cols].values, train_df["win"].values
    X_val,   y_val   = val_df[feature_cols].values,   val_df["win"].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df["win"].values

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    def eval_block(name, X, y):
        if len(y) == 0:
            print(f"{name}: no samples")
            return
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, proba)
        print(f"{name}: acc={acc:.3f}, auc={auc:.3f}, n={len(y)}")

    eval_block("Train", X_train, y_train)
    eval_block("Val",   X_val,   y_val)
    eval_block("Test",  X_test,  y_test)

    # Save model and feature cols
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    meta = {
        "feature_cols": feature_cols,
    }
    meta_path = ROOT / "models" / "xgb_filter_meta.parquet"
    pd.DataFrame([meta]).to_parquet(meta_path, index=False)

    print(f"[Auralis] XGB filter saved -> {MODEL_PATH}")
    print(f"[Auralis] Meta saved -> {meta_path}")
    print("Feature columns:", feature_cols)

if __name__ == "__main__":
    main()
