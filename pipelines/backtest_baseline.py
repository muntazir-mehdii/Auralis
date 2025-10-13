# pipelines/backtest_baseline.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "labels" / "auralis_labels.parquet"
OUT = ROOT / "reports" / "baseline_2025.csv"

# ---- config ----
LONDON_START = 7   # 07:00 UTC
LONDON_END   = 12 # entries allowed before 12:00
TIMEOUT_CAP_HOUR = 23 # runner managed no later than 23:00 UTC

RR_TARGET   = 3  # take 100% at exactly 2R (no partials)
PARTIAL_PCT = 1.0   # 100% at target -> early exit, no runner management
SL_BUFFER   = 0.5   # $0.50 beyond wick/asia level

# If a bar could hit both SL and TP, choose order of resolution.
# "tp_first" is slightly optimistic; "sl_first" is conservative.
INTRABAR_PRIORITY = "tp_first"   # "tp_first" | "sl_first"

def in_london(ts_series):
    hrs = pd.to_datetime(ts_series, utc=True).dt.hour
    return (hrs >= LONDON_START) & (hrs < LONDON_END)

def main():
    df = pd.read_parquet(LBL).copy()
    tcol = "time" if "time" in df.columns else [c for c in df.columns if "time" in c][0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.sort_values(tcol).reset_index(drop=True)
    df["date"] = df[tcol].dt.date

    need = {"open","high","low","close","asia_high","asia_low","entry_long","entry_short","confirm_long","confirm_short"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labels: {missing}")

    # session filter + signals (one trade per day, London only)
    df["long_sig"]  = df["entry_long"]  & df["confirm_long"]  & in_london(df[tcol])
    df["short_sig"] = df["entry_short"] & df["confirm_short"] & in_london(df[tcol])

    trades = []
    for d, day in df.groupby("date"):
        day = day.reset_index(drop=True)
        took_trade = False

        for i, row in day.iterrows():
            if took_trade: break
            ts = row[tcol]
            if not (LONDON_START <= ts.hour < LONDON_END):
                continue

            # ---- LONG ----
            if row["long_sig"]:
                entry = float(row["close"])
                sl = min(float(row["low"]), float(row["asia_low"])) - SL_BUFFER
                if sl >= entry:
                    continue
                risk = entry - sl
                tp2  = entry + RR_TARGET * risk

                timeout_hour = min(LONDON_END + 2, TIMEOUT_CAP_HOUR)
                forward = day.iloc[i+1:].copy()
                forward = forward[forward[tcol].dt.hour < timeout_hour]

                hitSL = hit2 = None
                for j, r2 in forward.iterrows():
                    lo = float(r2["low"]); hi = float(r2["high"])

                    if INTRABAR_PRIORITY == "tp_first":
                        if hi >= tp2: hit2 = r2[tcol]
                        if lo <= sl and hit2 is None: hitSL = r2[tcol]
                    else:  # sl_first
                        if lo <= sl: hitSL = r2[tcol]
                        if hi >= tp2 and hitSL is None: hit2 = r2[tcol]

                    if hitSL or hit2: break

                if hit2 is not None:
                    # FULL EXIT at TP: no runner logic when PARTIAL_PCT == 1.0
                    total_r = RR_TARGET * PARTIAL_PCT  # equals RR_TARGET
                    trades.append({
                        "date": d, "time": ts, "side": "long",
                        "entry": entry, "sl": sl, "tp2": tp2,
                        "partial_time": hit2, "runner_exit_time": hit2,
                        "total_R": round(total_r, 3)
                    })
                    took_trade = True

                elif hitSL is not None:
                    trades.append({
                        "date": d, "time": ts, "side": "long",
                        "entry": entry, "sl": sl, "tp2": tp2,
                        "partial_time": None, "runner_exit_time": hitSL,
                        "total_R": -1.0
                    })
                    took_trade = True

            # ---- SHORT ----
            elif row["short_sig"]:
                entry = float(row["close"])
                sl = max(float(row["high"]), float(row["asia_high"])) + SL_BUFFER
                if sl <= entry:
                    continue
                risk = sl - entry
                tp2  = entry - RR_TARGET * risk

                timeout_hour = min(LONDON_END + 2, TIMEOUT_CAP_HOUR)
                forward = day.iloc[i+1:].copy()
                forward = forward[forward[tcol].dt.hour < timeout_hour]

                hitSL = hit2 = None
                for j, r2 in forward.iterrows():
                    lo = float(r2["low"]); hi = float(r2["high"])

                    if INTRABAR_PRIORITY == "tp_first":
                        if lo <= tp2: hit2 = r2[tcol]
                        if hi >= sl and hit2 is None: hitSL = r2[tcol]
                    else:
                        if hi >= sl: hitSL = r2[tcol]
                        if lo <= tp2 and hitSL is None: hit2 = r2[tcol]

                    if hitSL or hit2: break

                if hit2 is not None:
                    total_r = RR_TARGET * PARTIAL_PCT  # equals RR_TARGET
                    trades.append({
                        "date": d, "time": ts, "side": "short",
                        "entry": entry, "sl": sl, "tp2": tp2,
                        "partial_time": hit2, "runner_exit_time": hit2,
                        "total_R": round(total_r, 3)
                    })
                    took_trade = True

                elif hitSL is not None:
                    trades.append({
                        "date": d, "time": ts, "side": "short",
                        "entry": entry, "sl": sl, "tp2": tp2,
                        "partial_time": None, "runner_exit_time": hitSL,
                        "total_R": -1.0
                    })
                    took_trade = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(trades)
    if res.empty:
        print("[Auralis] no trades found — check label conditions.")
        return

    res = res.sort_values("time").reset_index(drop=True)
    res["cum_R"] = res["total_R"].cumsum()
    roll_max = res["cum_R"].cummax()
    dd = (roll_max - res["cum_R"]).max()
    res.to_csv(OUT, index=False)

    wins = (res["total_R"] > 0).sum()
    print(f"[Auralis] baseline trades: {len(res)}")
    print(f"[Auralis] win rate: {wins/len(res):.1%}")
    print(f"[Auralis] total R: {res['total_R'].sum():.2f}")
    print(f"[Auralis] max DD (R): {dd:.2f}")
    print(f"[Auralis] saved -> {OUT}")

if __name__ == "__main__":
    main()

