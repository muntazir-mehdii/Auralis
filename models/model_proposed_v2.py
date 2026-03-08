"""
model_proposed_v2.py
--------------------

Upgraded ML Model for Assignment 3 (Version 2)

This uses the improved signals_v2 features to train an XGBoost classifier.

Key points:
- Only numeric feature columns are used (no timestamps / strings).
- All time/date columns are automatically ignored.
- Clean train/val/test 70/15/15 split.
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SIGNALS_PATH = ROOT / "data" / "signals" / "auralis_signals_v2.parquet"
MODEL_PATH = ROOT / "models" / "xgb_filter_v2.json"
META_PATH = ROOT / "models" / "xgb_filter_meta_v2.parquet"


class AuralisXGBFilterV2:
    """Improved Assignment 3 classifier (version 2)."""

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.feature_cols: List[str] = []

    # --------------------------------------------
    # LOAD SIGNAL DATASET
    # --------------------------------------------
    def load_signals(self) -> pd.DataFrame:
        if not SIGNALS_PATH.exists():
            raise FileNotFoundError(
                f"Signals dataset not found at {SIGNALS_PATH}. "
                f"Run signals_v2.py first."
            )
        df = pd.read_parquet(SIGNALS_PATH).copy()
        df = df.dropna()
        return df

    # --------------------------------------------
    # FEATURE COLUMNS (robust)
    # --------------------------------------------
    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Automatically detect feature columns:
        - Drop label columns (win, result/total_R, signal flags)
        - Drop any time/date columns
        - Keep only numeric dtypes
        """
        ignore = {
            "entry_long", "entry_short",
            "confirm_long", "confirm_short",
            "win", "result_r", "total_R"
        }

        # Ignore any time/date-like columns
        for c in df.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl:
                ignore.add(c)

        feature_cols = []
        for c in df.columns:
            if c in ignore:
                continue
            # keep only numeric features
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

        if not feature_cols:
            raise ValueError("No numeric feature columns found for training.")

        return feature_cols

    # --------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # --------------------------------------------
    def random_split(self, df: pd.DataFrame):
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        return train_df, val_df, test_df

    # --------------------------------------------
    # TRAINING LOGIC
    # --------------------------------------------
    def train(self):
        print("[Auralis] Loading signals_v2 dataset...")
        df = self.load_signals()

        print(f"[Auralis] signals_v2 rows: {len(df)}")
        print("[Auralis] Detecting feature columns...")
        self.feature_cols = self.get_feature_cols(df)
        print(f"[Auralis] Using {len(self.feature_cols)} features: {self.feature_cols}")

        train_df, val_df, test_df = self.random_split(df)

        X_train = train_df[self.feature_cols].values
        y_train = train_df["win"].values

        X_val = val_df[self.feature_cols].values
        y_val = val_df["win"].values

        X_test = test_df[self.feature_cols].values
        y_test = test_df["win"].values

        # Ensure train has both classes
        if len(set(y_train)) < 2:
            raise ValueError(
                f"Training set has only one class (y_train unique={set(y_train)}). "
                f"Need both wins and losses for classification."
            )

        print("[Auralis] Training XGBoost model V2...")
        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            reg_lambda=2,
            scale_pos_weight=1.2,
            tree_method="hist",
        )

        # NOTE: no verbose / early_stopping to avoid version issues
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)]
        )

        self.model = model

        # Save model + metadata
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        pd.DataFrame({"feature_cols": [self.feature_cols]}).to_parquet(META_PATH)

        # Evaluate splits
        def eval_set(X, y):
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5)
            return {
                "acc": float(accuracy_score(y, pred)),
                "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
            }

        metrics = {
            "train": eval_set(X_train, y_train),
            "val":   eval_set(X_val,   y_val),
            "test":  eval_set(X_test,  y_test),
        }

        # Confusion matrix on test set
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5)
        cm = confusion_matrix(y_test, test_pred)
        cm_path = ROOT / "results" / "confusion_matrix_v2.csv"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm).to_csv(cm_path, index=False)

        print("[Auralis] Model V2 training complete.")
        print("[Auralis] Metrics:", metrics)

        return metrics

    # --------------------------------------------
    # LOAD TRAINED MODEL
    # --------------------------------------------
    def load_trained(self):
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        meta = pd.read_parquet(META_PATH)
        self.feature_cols = meta["feature_cols"][0]
        self.model = model
        return self

    # --------------------------------------------
    # SCORE A SINGLE TRADE (DICT INPUT)
    # --------------------------------------------
    def score_trade(self, feat: Dict[str, float]) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_trained() first.")

        x = np.array([[feat.get(col, 0.0) for col in self.feature_cols]], dtype=np.float32)
        return float(self.model.predict_proba(x)[0, 1])


if __name__ == "__main__":
    filt = AuralisXGBFilterV2()
    metrics = filt.train()
    print(metrics)
