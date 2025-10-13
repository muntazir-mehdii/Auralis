# pipelines/compare_variants.py
# Auralis variant sweep — RR targets, confirm on/off, session filter
# Uses your labeled parquet produced earlier (auralis_labels.parquet)

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBL  = ROOT / "data" / "labels" / "auralis_labels.parquet"
REP  = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

# ---- Session config (matches your baseline) ----
LONDON = (7, 11)   # entries allowed before 11:00
NEWYORK = (12, 16) # reasonable NY window; adjust if needed
TIMEOUT_CAP_HOUR = 20
SL_BUFFER = 0.10   # beyond wick/asia level
YEAR = 2025        # change to 5y later

def in_session(ts_hour: int, session_key: str) -> bool:
    s = LONDON if session_key == "london" else NEWYORK
    return s[0] <= ts_hour < s[1]

def simulate_day(day_df: pd.DataFrame, rr_target: float, use_confirmation: bool, session_key: str):
    """
    Quick variant tester (no partials): either hits TP=rr_target*R or -1R.
    One trade max per day/session.
    """
    rows = []
    took_trade = False
    day_df = day_df.reset_index(drop=True)

    for i, row in day_df.iterrows():
        if took_trade:
            break
        ts = row["time"]
        hour = ts.hour
        if not in_session(hour, session_key):
            continue

        # base entry candidates from your labels
        long_sig  = bool(row["entry_long"])  and (bool(row["confirm_long"])  if use_confirmation else True)
        short_sig = bool(row["entry_short"]) and (bool(row["confirm_short"]) if use_confirmation else True)

        entry = float(row["close"])

        # ---- LONG ----
        if long_sig:
            sl = min(float(row["low"]), float(row["asia_low"])) - SL_BUFFER
            if sl >= entry:
                continue
            risk = entry - sl
            tp   = entry + rr_target * risk

            forward = day_df.iloc[i+1:].copy()
            # manage until session+2 hours or daily cap (same style as baseline)
            if session_key == "london":
                t_hour_cap = min(LONDON[1] + 2, TIMEOUT_CAP_HOUR)
            else:
                # if day time falls inside London hours, cap off London; otherwise NY
                t_hour_cap = min((LONDON[1] + 2) if (LONDON[0] <= hour < LONDON[1]) else (NEWYORK[1] + 2),
                                  TIMEOUT_CAP_HOUR)
            forward = forward[forward["time"].dt.hour < t_hour_cap]

            hitSL = hitTP = None
            for _, r2 in forward.iterrows():
                if r2["low"] <= sl:
                    hitSL = r2["time"]; break
                if r2["high"] >= tp:
                    hitTP = r2["time"]; break

            if hitTP:
                rows.append({"date": day_df["date"].iloc[0], "time": ts, "side":"long",
                            "rr": rr_target, "confirm": use_confirmation,
                            "session": session_key, "total_R": rr_target})
            elif hitSL:
                rows.append({"date": day_df["date"].iloc[0], "time": ts, "side":"long",
                            "rr": rr_target, "confirm": use_confirmation,
                            "session": session_key, "total_R": -1.0})
            took_trade = bool(hitTP or hitSL)

        # ---- SHORT ----
        elif short_sig:
            sl = max(float(row["high"]), float(row["asia_high"])) + SL_BUFFER
            if sl <= entry:
                continue
            risk = sl - entry
            tp   = entry - rr_target * risk

            forward = day_df.iloc[i+1:].copy()
            if session_key == "london":
                t_hour_cap = min(LONDON[1] + 2, TIMEOUT_CAP_HOUR)
            else:
                t_hour_cap = min((LONDON[1] + 2) if (LONDON[0] <= hour < LONDON[1]) else (NEWYORK[1] + 2),
                                  TIMEOUT_CAP_HOUR)
            forward = forward[forward["time"].dt.hour < t_hour_cap]

            hitSL = hitTP = None
            for _, r2 in forward.iterrows():
                if r2["high"] >= sl:
                    hitSL = r2["time"]; break
                if r2["low"]  <= tp:
                    hitTP = r2["time"]; break

            if hitTP:
                rows.append({"date": day_df["date"].iloc[0], "time": ts, "side":"short",
                            "rr": rr_target, "confirm": use_confirmation,
                            "session": session_key, "total_R": rr_target})
            elif hitSL:
                rows.append({"date": day_df["date"].iloc[0], "time": ts, "side":"short",
                            "rr": rr_target, "confirm": use_confirmation,
                            "session": session_key, "total_R": -1.0})
            took_trade = bool(hitTP or hitSL)

    return rows

def run_variant(df: pd.DataFrame, rr_target: float, use_confirmation: bool, session_key: str, year: int):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.date
    df = df[df["time"].dt.year == year]

    all_rows = []
    for d, day in df.groupby("date"):
        all_rows.extend(simulate_day(day, rr_target, use_confirmation, session_key))

    res = pd.DataFrame(all_rows)
    if res.empty:
        return pd.DataFrame([{
            "year": year, "session": session_key, "rr": rr_target, "confirm": use_confirmation,
            "trades": 0, "win_rate": 0.0, "avg_R": 0.0, "total_R": 0.0,
            "max_dd_R": 0.0, "max_win_streak": 0, "max_loss_streak": 0
        }])

    res = res.sort_values("time").reset_index(drop=True)
    res["cum_R"] = res["total_R"].cumsum()
    wins = (res["total_R"] > 0)
    losses = ~wins

    # Streak helpers
    def longest_streak(xs, flag=True):
        best = cur = 0
        for v in xs:
            if bool(v) == flag:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        return best

    out = pd.DataFrame([{
        "year": year,
        "session": session_key,
        "rr": rr_target,
        "confirm": use_confirmation,
        "trades": int(len(res)),
        "win_rate": float(wins.mean()),
        "avg_R": float(res["total_R"].mean()),
        "total_R": float(res["total_R"].sum()),
        "max_dd_R": float((res["cum_R"].cummax() - res["cum_R"]).max()),
        "max_win_streak": int(longest_streak(wins.tolist(), True)),
        "max_loss_streak": int(longest_streak(losses.tolist(), True)),
    }])

    # also dump trades per variant (optional but handy)
    trades_path = REP / f"trades_{session_key}_rr{rr_target}_{'confirm' if use_confirmation else 'noc'}_{year}.csv"
    res.drop(columns=["cum_R"]).to_csv(trades_path, index=False)
    return out

def main():
    df = pd.read_parquet(LBL)
    # Guard: required columns per your labeler & baseline
    need = {
        "open","high","low","close","asia_high","asia_low",
        "entry_long","entry_short","confirm_long","confirm_short","time"
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in labels: {miss}")

    grid_rr = [1.5, 2.0]          # as discussed next-step
    grid_confirm = [False, True]  # with/without confirmation filter
    grid_session = ["london", "newyork"]

    parts = []
    for rr in grid_rr:
        for c in grid_confirm:
            for s in grid_session:
                parts.append(run_variant(df, rr, c, s, YEAR))

    summary = pd.concat(parts, ignore_index=True)
    out_path = REP / f"variants_{YEAR}.csv"
    summary.to_csv(out_path, index=False)

    print("[Auralis] Variants written ->", out_path)
    print(summary.sort_values(["session","rr","confirm"]).to_string(index=False))

if __name__ == "__main__":
    main()
