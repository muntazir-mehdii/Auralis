# pipelines/backtest_baseline.py
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBL = ROOT / "data" / "labels" / "auralis_labels.parquet"
OUT = ROOT / "reports" / "baseline_2025.csv"

# config
LONDON_START = 7   # 07:00 UTC
LONDON_END   = 11  # entries allowed before 11:00
TIMEOUT_CAP_HOUR = 20  # runner managed no later than 20:00 UTC
RR_TARGET = 2.0
PARTIAL_PCT = 0.80
SL_BUFFER = 0.10  # $0.10 beyond wick/asia level

def in_london(ts_series):
    hrs = pd.to_datetime(ts_series, utc=True).dt.hour
    return (hrs >= LONDON_START) & (hrs < LONDON_END)


def main():
    df = pd.read_parquet(LBL).copy()
    # find time column written by labeler
    tcol = "time" if "time" in df.columns else [c for c in df.columns if "time" in c][0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.sort_values(tcol).reset_index(drop=True)
    df["date"] = df[tcol].dt.date

    # guard rails
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

            # LONG
            if row["long_sig"]:
                entry = float(row["close"])
                sl = min(float(row["low"]), float(row["asia_low"])) - SL_BUFFER
                if sl >= entry: 
                    continue
                risk = entry - sl
                tp2 = entry + RR_TARGET * risk

                # manage until min(11+2, 20) = 13 or earlier end of data
                timeout_hour = min(LONDON_END + 2, TIMEOUT_CAP_HOUR)
                forward = day.iloc[i+1:].copy()
                forward = forward[forward[tcol].dt.hour < timeout_hour]

                hitSL = None
                hit2 = None
                for j, r2 in forward.iterrows():
                    if r2["low"] <= sl:
                        hitSL = r2[tcol]; break
                    if r2["high"] >= tp2:
                        hit2 = r2[tcol]
                        # after partial: move SL to BE and let runner go until timeout/BE
                        be = entry
                        fwd2 = forward.loc[j+1:]
                        exit_time = None; exit_price = None
                        for _, r3 in fwd2.iterrows():
                            if r3["low"] <= be:
                                exit_time = r3[tcol]; exit_price = be; break
                        if exit_time is None:
                            if not fwd2.empty:
                                last = fwd2.iloc[-1]
                                exit_time = last[tcol]; exit_price = float(last["close"])
                            else:
                                exit_time = r2[tcol]; exit_price = float(r2["close"])
                        r_closed = PARTIAL_PCT * RR_TARGET  # 80% at 2R
                        runner_rr = (exit_price - entry) / risk
                        r_runner = (1 - PARTIAL_PCT) * max(runner_rr, 0.0)
                        total_r = r_closed + r_runner
                        trades.append({
                            "date": d, "time": ts, "side": "long",
                            "entry": entry, "sl": sl, "tp2": tp2,
                            "partial_time": hit2, "runner_exit_time": exit_time,
                            "total_R": round(total_r, 3)
                        })
                        took_trade = True
                        break
                if not took_trade and hitSL is not None:
                    trades.append({
                        "date": d, "time": ts, "side": "long",
                        "entry": entry, "sl": sl, "tp2": tp2,
                        "partial_time": None, "runner_exit_time": hitSL,
                        "total_R": -1.0
                    })
                    took_trade = True

            # SHORT
            elif row["short_sig"]:
                entry = float(row["close"])
                sl = max(float(row["high"]), float(row["asia_high"])) + SL_BUFFER
                if sl <= entry:
                    continue
                risk = sl - entry
                tp2 = entry - RR_TARGET * risk

                timeout_hour = min(LONDON_END + 2, TIMEOUT_CAP_HOUR)
                forward = day.iloc[i+1:].copy()
                forward = forward[forward[tcol].dt.hour < timeout_hour]

                hitSL = None
                hit2 = None
                for j, r2 in forward.iterrows():
                    if r2["high"] >= sl:
                        hitSL = r2[tcol]; break
                    if r2["low"] <= tp2:
                        hit2 = r2[tcol]
                        be = entry
                        fwd2 = forward.loc[j+1:]
                        exit_time = None; exit_price = None
                        for _, r3 in fwd2.iterrows():
                            if r3["high"] >= be:
                                exit_time = r3[tcol]; exit_price = be; break
                        if exit_time is None:
                            if not fwd2.empty:
                                last = fwd2.iloc[-1]
                                exit_time = last[tcol]; exit_price = float(last["close"])
                            else:
                                exit_time = r2[tcol]; exit_price = float(r2["close"])
                        r_closed = PARTIAL_PCT * RR_TARGET
                        runner_rr = (entry - exit_price) / risk
                        r_runner = (1 - PARTIAL_PCT) * max(runner_rr, 0.0)
                        total_r = r_closed + r_runner
                        trades.append({
                            "date": d, "time": ts, "side": "short",
                            "entry": entry, "sl": sl, "tp2": tp2,
                            "partial_time": hit2, "runner_exit_time": exit_time,
                            "total_R": round(total_r, 3)
                        })
                        took_trade = True
                        break
                if not took_trade and hitSL is not None:
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
    res["cum_R"] = res["total_R"].cumsum()
    # max drawdown in R
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
