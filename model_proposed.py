# model_proposed.py
"""
Proposed method for Assignment 3:
XGBoost classifier that predicts probability of a trade being a winner,
and is later used as a filter on top of the rule-based Auralis strategy.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
SIGNALS_PATH = ROOT / "data" / "auralis_signals_xgb.parquet"
MODEL_PATH = ROOT / "models" / "xgb_filter.json"
META_PATH = ROOT / "models" / "xgb_filter_meta.parquet"


class AuralisXGBFilter:
    """
    Wrapper around XGBClassifier used as the proposed method in Assignment 3.
    """

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.feature_cols: List[str] | None = None

    # ---------- DATA LOADING ----------

    def load_signals(self) -> pd.DataFrame:
        if not SIGNALS_PATH.exists():
            raise FileNotFoundError(
                f"Signals dataset not found at {SIGNALS_PATH}. "
                "Build it first with pipelines/build_signals_dataset.py."
            )
        df = pd.read_parquet(SIGNALS_PATH)
        df["win"] = (df["result_r"] > 0).astype(int)
        return df

    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        base_cols = ["asia_range", "hour", "dow", "price_close"]
        extra = []
        for col in ["ema_200", "atr_14", "rsi_14"]:
            if col in df.columns:
                extra.append(col)
        return base_cols + extra

    def time_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Train: <=2021, Val: 2022, Test: >=2023."""
        df["year"] = df["entry_time"].dt.year
        train_df = df[df["year"] <= 2021].reset_index(drop=True)
        val_df = df[df["year"] == 2022].reset_index(drop=True)
        test_df = df[df["year"] >= 2023].reset_index(drop=True)
        return train_df, val_df, test_df

    # ---------- TRAINING ----------

    def train(self) -> Dict[str, Dict[str, float]]:
        """
        Train XGBClassifier on signals dataset using a random 70/15/15 split.
        This avoids the 'Invalid classes' error when one split has only a single class.
        Returns metrics for train/val/test.
        """
        df = self.load_signals()
        feature_cols = self.get_feature_cols(df)
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # ---- RANDOM SPLIT (70% train, 15% val, 15% test) ----
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_df = df.iloc[:n_train].copy()
        val_df   = df.iloc[n_train:n_train + n_val].copy()
        test_df  = df.iloc[n_train + n_val:].copy()

        self.feature_cols = feature_cols

        def xy(d: pd.DataFrame):
            return d[feature_cols].values, d["win"].values

        X_train, y_train = xy(train_df)
        X_val,   y_val   = xy(val_df)
        X_test,  y_test  = xy(test_df)

        # ---- Check that training actually has both classes ----
        unique_train = set(pd.unique(y_train))
        if len(unique_train) < 2:
            raise ValueError(
                f"Training split has only one class: {unique_train}. "
                "Try regenerating the signals dataset from an unfiltered baseline."
            )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        )

        # NOTE: no eval_set with different label classes; we just fit on train.
        model.fit(X_train, y_train, verbose=False)

        self.model = model

        # Save model + meta for later use
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        pd.DataFrame([{"feature_cols": feature_cols}]).to_parquet(META_PATH, index=False)

        # ---- Compute metrics for report ----
        metrics: Dict[str, Dict[str, float]] = {}

        def eval_block(name: str, X, y):
            if len(y) == 0:
                return {"acc": float("nan"), "auc": float("nan")}
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)
            return {
                "acc": float(accuracy_score(y, pred)),
                "auc": float(roc_auc_score(y, proba)),
            }

        metrics["train"] = eval_block("train", X_train, y_train)
        metrics["val"]   = eval_block("val",   X_val,   y_val)
        metrics["test"]  = eval_block("test",  X_test,  y_test)

        # Confusion matrix on test set (for Assignment 3)
        if len(y_test) > 0:
            proba_test = model.predict_proba(X_test)[:, 1]
            pred_test = (proba_test >= 0.5).astype(int)
            cm = confusion_matrix(y_test, pred_test)
            cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
            cm_path = ROOT / "results" / "cm_xgb_test.csv"
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            cm_df.to_csv(cm_path)

        return metrics

    # ---------- INFERENCE ----------

    def load_trained(self):
        """Load model + feature list from disk for backtesting."""
        if self.model is None:
            model = XGBClassifier()
            model.load_model(MODEL_PATH)
            self.model = model
        if self.feature_cols is None:
            meta = pd.read_parquet(META_PATH).iloc[0].to_dict()
            self.feature_cols = meta["feature_cols"]
        return self

    def score_trade(self, features: Dict[str, float]) -> float:
        """
        Given a feature dict for a single trade, return p(win) in [0, 1].
        """
        if self.model is None or self.feature_cols is None:
            self.load_trained()
        x = np.array([[features.get(col, 0.0) for col in self.feature_cols]], dtype=np.float32)
        p = self.model.predict_proba(x)[0, 1]
        return float(p)


if __name__ == "__main__":
    # Quick manual run: train model and print metrics
    filt = AuralisXGBFilter()
    metrics = filt.train()
    print("[Auralis] XGBFilter metrics:")
    for split, m in metrics.items():
        print(f"  {split}: acc={m['acc']:.3f}, auc={m['auc']:.3f}")
    print(f"[Auralis] Model saved to {MODEL_PATH}")
