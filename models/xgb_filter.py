# models/xgb_filter.py

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_filter.json"
META_PATH  = ROOT / "models" / "xgb_filter_meta.parquet"

class XGBTradeFilter:
    def __init__(self):
        self.model = XGBClassifier()
        self.model.load_model(MODEL_PATH)

        meta = pd.read_parquet(META_PATH).iloc[0].to_dict()
        self.feature_cols = meta["feature_cols"]

    def score(self, feat_dict: dict) -> float:
        """Return p(win) for a given trade feature dict."""
        x = np.array([[feat_dict.get(col, 0.0) for col in self.feature_cols]], dtype=np.float32)
        proba = self.model.predict_proba(x)[0, 1]
        return float(proba)
