# preprocess.py
import pandas as pd
from data_loader import load_price_data

def preprocess_data() -> pd.DataFrame:
    """
    Load raw labelled data and add helper columns:
    - date
    - hour
    - asia_range (per day)
    """
    df = load_price_data()
    tcol = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()][0]
    df["date"] = df[tcol].dt.date
    df["hour"] = df[tcol].dt.hour

    # optional: precompute daily Asia range for EDA
    asia_ranges = []
    for d, day in df.groupby("date"):
        asia = day[(day[tcol].dt.hour >= 0) & (day[tcol].dt.hour < 6)]
        if asia.empty:
            rng = 0.0
        else:
            rng = float(asia["high"].max() - asia["low"].min())
        asia_ranges.extend([rng] * len(day))
    df["asia_range_day"] = asia_ranges

    return df
