# pipelines/labeler.py
import pandas as pd
from pathlib import Path

print("[Auralis] Labeler running...")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xauusd_m5_PKT__.csv"
OUT = ROOT / "data" / "labels" / "auralis_labels.parquet"

# parameters
ASIAN_START = 0
ASIAN_END = 6

def main():
    df = pd.read_csv(RAW)
    # make sure columns are clean
    df.columns = [c.lower() for c in df.columns]
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("No time column found.")
    df = df.sort_values('time').reset_index(drop=True)

    # find daily highs/lows during Asian session
    df['date'] = df['time'].dt.date
    session_marks = []
    for d, day in df.groupby('date'):
        asia = day[(day['time'].dt.hour >= ASIAN_START) & (day['time'].dt.hour < ASIAN_END)]
        if asia.empty:
            continue
        high = asia['high'].max()
        low = asia['low'].min()
        session_marks.append((d, high, low))
    asia_df = pd.DataFrame(session_marks, columns=['date', 'asia_high', 'asia_low'])

    df = df.merge(asia_df, on='date', how='left')

    # label fakeouts: break & reclose inside
    df['above'] = (df['high'] > df['asia_high'])
    df['below'] = (df['low'] < df['asia_low'])

    # initialize
    df['entry_long'] = False
    df['entry_short'] = False
    for i in range(3, len(df)):
        # long setup: fakeout below & close back inside
        if df.loc[i-1, 'below'] and (df.loc[i, 'close'] > df.loc[i, 'asia_low']):
            df.loc[i, 'entry_long'] = True
        # short setup: fakeout above & close back inside
        if df.loc[i-1, 'above'] and (df.loc[i, 'close'] < df.loc[i, 'asia_high']):
            df.loc[i, 'entry_short'] = True

    # mock confirmation (we’ll upgrade later)
    df['confirm_long'] = df['entry_long']
    df['confirm_short'] = df['entry_short']

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[Auralis] labels written -> {OUT}")

if __name__ == "__main__":
    main()
