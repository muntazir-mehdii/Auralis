# pipelines/rr2_full_tp.py
# Auralis: Strict RR=2.0 take-profit of 100% at target, with SL moved to BE at 1R.
# Session: London (07:00–11:00 UTC) entries; manage until session_end+2h, capped at 20:00 UTC.
# Inputs: data/labels/auralis_labels.parquet (created by your labeling pipeline)
# Outputs: reports/rr2_fulltp_2025.csv (trade log) and summary printed to console.

from __future__ import annotations
import pandas as pd
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
LBL  = ROOT / "data" / "labels" / "auralis_labels.parquet"
REP  = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
LONDON_START = 7     # 07:00 UTC
LONDON_END   = 11    # entries allowed before 11:00
TIMEOUT_CAP_HOUR = 20  # hard cap for management
SL_BUFFER = 1.00       # $0.10 beyond wick/asia level
YEAR = 2025            # filter to a single year; adjust as needed

# Strict RR policy
RR_TARGET = 2.0        # take 100% at exactly 2R (no partials)
MOVE_BE_AT = 1.0       # move SL to BE once +1R is tagged (no P/L booked at 1R)


# ---------------- Helpers ----------------

def _session_manage_cap(entry_hour: int) -> int:
    """Management window ends 2h after London session end, capped by TIMEOUT_CAP_HOUR.
    If entry is during London window, use LONDON_END+2 else also use LONDON_END+2 (since only London entries allowed).
    """
    return min(LONDON_END + 2, TIMEOUT_CAP_HOUR)


def _check_columns(df: pd.DataFrame):
    need = {
        "time", "open", "high", "low", "close",
        "asia_high", "asia_low",
        "entry_long", "entry_short",
        "confirm_long", "confirm_short",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in labels parquet: {sorted(miss)}")


# ---------------- Core Simulation ----------------

def simulate_rr2_fulltp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["time"].dt.year == YEAR]
    df["date"] = df["time"].dt.date

    trades = []

    for d, day in df.groupby("date"):
        took_trade = False
        day = day.sort_values("time").reset_index(drop=True)

        for i, row in day.iterrows():
            if took_trade:
                break

            ts = row["time"]
            hour = ts.hour
            if not (LONDON_START <= hour < LONDON_END):
                continue

            # Signals (confirmations ON)
            long_sig  = bool(row["entry_long"])  and bool(row["confirm_long"]) 
            short_sig = bool(row["entry_short"]) and bool(row["confirm_short"]) 
            if not (long_sig or short_sig):
                continue

            # --- Next-bar entry to avoid look-ahead bias ---
            if i + 1 >= len(day):
                continue
            entry_row  = day.iloc[i + 1]
            entry      = float(entry_row["open"])
            entry_ts   = entry_row["time"]
            entry_hour = entry_ts.hour

            # Prepare side-specific levels
            side = None
            if long_sig and not short_sig:
                side = "long"
                sl_raw = min(float(row["low"]), float(row["asia_low"])) - SL_BUFFER
                if sl_raw >= entry:
                    continue
                risk   = entry - sl_raw
                tp_2r  = entry + RR_TARGET * risk
                be_level = entry
            elif short_sig and not long_sig:
                side = "short"
                sl_raw = max(float(row["high"]), float(row["asia_high"])) + SL_BUFFER
                if sl_raw <= entry:
                    continue
                risk   = sl_raw - entry
                tp_2r  = entry - RR_TARGET * risk
                be_level = entry
            else:
                # both signaled on same bar: skip to avoid ambiguity (rare)
                continue

            # Forward window starts AFTER the entry bar
            fwd = day.iloc[i + 2:].copy()
            cap_hour = _session_manage_cap(entry_hour)
            fwd = fwd[fwd["time"].dt.hour < cap_hour]

            state = "live"  # live -> at_be -> done
            result_R = None

            for _, r2 in fwd.iterrows():
                low2, high2 = float(r2["low"]), float(r2["high"]) 

                if side == "long":
                    if state == "live":
                        tag_1r_price = entry + MOVE_BE_AT * risk
                        # Priority: +1R first, then same-bar TP2R/BE, then check SL
                        if high2 >= tag_1r_price:
                            state = "at_be"
                            if high2 >= tp_2r:
                                result_R = RR_TARGET; break
                            if low2  <= be_level:
                                result_R = 0.0; break
                            continue
                        # direct 2R without 1R (rare)
                        if high2 >= tp_2r:
                            result_R = RR_TARGET; break
                        # only now consider original SL
                        if low2 <= sl_raw:
                            result_R = -1.0; break
                    elif state == "at_be":
                        if high2 >= tp_2r:
                            result_R = RR_TARGET; break
                        if low2 <= be_level:
                            result_R = 0.0; break

                else:  # short
                    if state == "live":
                        tag_1r_price = entry - MOVE_BE_AT * risk
                        if low2 <= tag_1r_price:
                            state = "at_be"
                            if low2  <= tp_2r:
                                result_R = RR_TARGET; break
                            if high2 >= be_level:
                                result_R = 0.0; break
                            continue
                        if low2 <= tp_2r:
                            result_R = RR_TARGET; break
                        if high2 >= sl_raw:
                            result_R = -1.0; break
                    elif state == "at_be":
                        if low2 <= tp_2r:
                            result_R = RR_TARGET; break
                        if high2 >= be_level:
                            result_R = 0.0; break

            if result_R is None:
                result_R = 0.0

            trades.append({
                "date": d, "time": entry_ts, "side": side,
                "entry": entry, "sl": sl_raw, "tp": tp_2r,
                "result_R": float(result_R)
            })
            took_trade = True

        # end bars loop

    return pd.DataFrame(trades)


# ---------------- Main ----------------

def main():
    df = pd.read_parquet(LBL)
    _check_columns(df)

    trades = simulate_rr2_fulltp(df)

    out_trades = REP / f"rr2_fulltp_{YEAR}.csv"
    trades.to_csv(out_trades, index=False)

    if trades.empty:
        print("[Auralis] No trades generated under current filters.")
        return

    # Summary
    trades = trades.sort_values("time").reset_index(drop=True)
    trades["cum_R"] = trades["result_R"].cumsum()
    wins = (trades["result_R"] > 0)
    losses = (trades["result_R"] < 0)

    def longest_streak(xs, flag=True):
        best = cur = 0
        for v in xs:
            if bool(v) == flag:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        return best

    summary = {
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()),
        "total_R": float(trades["result_R"].sum()),
        "avg_R_per_trade": float(trades["result_R"].mean()),
        "max_dd_R": float((trades["cum_R"].cummax() - trades["cum_R"]).max()),
        "max_win_streak": int(longest_streak(wins.tolist(), True)),
        "max_loss_streak": int(longest_streak(losses.tolist(), True)),
        "first_trade": str(trades["time"].iloc[0]),
        "last_trade": str(trades["time"].iloc[-1]),
    }

    out_summary = REP / f"rr2_fulltp_{YEAR}_summary.json"
    import json
    out_summary.write_text(json.dumps(summary, indent=2))

    print("[Auralis] RR=2.0 (100% TP) with BE@1R complete.")
    print("Trades ->", out_trades)
    print("Summary ->", out_summary)
    print(summary)


if __name__ == "__main__":
    main()
