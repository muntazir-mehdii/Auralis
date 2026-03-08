# pipelines/labeler.py
# Auralis: Build Asian session range + sweep labels for XAUUSD M5 (UTC).

import pandas as pd
from pathlib import Path

print("[Auralis] Labeler running...")

ROOT = Path(__file__).resolve().parents[1]

# Raw OHLC data (UTC timestamps, M5)
RAW = ROOT / "data" / "raw" / "xauusd_m5_UTC.csv"

# Output labeled frame
OUT = ROOT / "data" / "labels" / "auralis_labels.parquet"

# Asian session window (UTC)
ASIAN_START = 0   # 00:00
ASIAN_END   = 6   # 06:00 (exclusive)


def main():
    # ---- Load & normalize columns ----
    df = pd.read_csv(RAW)
    df.columns = [c.lower() for c in df.columns]

    # time column handling
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    elif "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError("No time or timestamp column found in raw file.")

    # basic OHLC sanity
    needed_ohlc = {"open", "high", "low", "close"}
    missing = needed_ohlc - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns in raw data: {missing}")

    df = df.sort_values("time").reset_index(drop=True)

    # ---- Asian range per day ----
    df["date"] = df["time"].dt.date

    session_marks = []
    for d, day in df.groupby("date"):
        asia = day[
            (day["time"].dt.hour >= ASIAN_START)
            & (day["time"].dt.hour < ASIAN_END)
        ]
        if asia.empty:
            continue

        asia_high = asia["high"].max()
        asia_low = asia["low"].min()
        session_marks.append((d, asia_high, asia_low))

    asia_df = pd.DataFrame(session_marks, columns=["date", "asia_high", "asia_low"])
    df = df.merge(asia_df, on="date", how="left")

    # ---- Sweep detection ----
    # price trades outside the Asian box
    df["above"] = df["high"] > df["asia_high"]
    df["below"] = df["low"] < df["asia_low"]

    # ---- Entry labeling ----
    df["entry_long"] = False
    df["entry_short"] = False

    # Simple rule:
    #  - Long: previous candle swept below Asia low, current closes back inside.
    #  - Short: previous candle swept above Asia high, current closes back inside.
    for i in range(3, len(df)):
        # long setup
        if (
            bool(df.loc[i - 1, "below"])
            and (df.loc[i, "close"] > df.loc[i, "asia_low"])
        ):
            df.loc[i, "entry_long"] = True

        # short setup
        if (
            bool(df.loc[i - 1, "above"])
            and (df.loc[i, "close"] < df.loc[i, "asia_high"])
        ):
            df.loc[i, "entry_short"] = True

    # ---- Confirmation flags ----
    # For now, confirmations are identical to entries; later we can tighten this.
    df["confirm_long"] = df["entry_long"]
    df["confirm_short"] = df["entry_short"]

    # ---- Save ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[Auralis] labels written -> {OUT}")


if __name__ == "__main__":
    main()
