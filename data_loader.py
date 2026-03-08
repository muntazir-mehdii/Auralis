# data_loader.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "labels" / "auralis_labels.parquet"

def load_price_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load cleaned + labelled XAUUSD M5 data (2018–2025).
    """
    p = path or DATA_PATH
    df = pd.read_parquet(p)
    # normalize time column
    tcol_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if not tcol_candidates:
        raise ValueError("No time column found in labels file.")
    tcol = tcol_candidates[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.sort_values(tcol).reset_index(drop=True)
    return df
