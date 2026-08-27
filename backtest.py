"""
backtest.py — Scores past recommendations against actual price outcomes.

Run anytime:
    python backtest.py

For each logged recommendation that is >= 30 days old and not yet scored,
it fetches the current/outcome price and calculates whether the call was right.
Results are saved back to the log files and a summary is printed.
"""

import os
import json
import glob
from datetime import datetime, timedelta
import yfinance as yf


OUTCOME_DAYS = 30       # How many days after recommendation to measure
MIN_RETURN_BUY = 0.0    # Buy is "correct" if return > 0% (stock went up)
MIN_RETURN_SELL = 0.0   # Sell is "correct" if return < 0% (stock went down)


def fetch_price_on_date(ticker: str, target_date: datetime) -> float | None:
    """
    Fetches the closing price of a ticker on or near a target date.
    Looks up to 5 trading days forward if the exact date is a weekend/holiday.
    """
    start = target_date
    end = target_date + timedelta(days=7)  # Buffer for weekends/holidays

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return None
        return round(float(hist["Close"].iloc[0]), 2)
    except Exception:
        return None


def score_recommendation(action: str, return_pct: float) -> bool:
    """
    Determines if a recommendation was correct based on subsequent return.
    Buy = correct if stock went up
    Sell/Avoid = correct if stock went down
    Hold = correct if stock stayed within ±5%
    """
    if action == "buy":
        return return_pct > MIN_RETURN_BUY
    elif action == "sell":
        return return_pct < -MIN_RETURN_SELL
    elif action == "hold":
        return -5.0 <= return_pct <= 5.0
    return False


def run_backtest():
    log_files = sorted(glob.glob("logs/*_recommendations.json"))

    if not log_files:
        print("No recommendation logs found. Run pipeline.py first.")
        return

    print(f"\n{'='*60}")
    print(f"BACKTEST — {datetime.today().strftime('%B %d, %Y')}")
    print(f"Outcome window: {OUTCOME_DAYS} days after recommendation")
    print(f"{'='*60}\n")

    all_records = []
    updated_files = 0

    for log_file in log_files:
        with open(log_file, "r") as f:
            records = json.load(f)

        file_updated = False

        for rec in records:
            # Skip already scored
            if rec.get("correct") is not None:
                all_records.append(rec)
                continue

            # Skip if not yet mature
            rec_date = datetime.strptime(rec["date"], "%Y-%m-%d")
            outcome_date = rec_date + timedelta(days=OUTCOME_DAYS)

            if datetime.today() < outcome_date:
                print(f"  ⏳ {rec['ticker']} ({rec['date']}) — outcome date "
                      f"{outcome_date.strftime('%Y-%m-%d')} not reached yet")
                all_records.append(rec)
                continue

            # Fetch outcome price
            if not rec.get("price_at_recommendation"):
                print(f"  ✗ {rec['ticker']} — no entry price recorded, skipping")
                all_records.append(rec)
                continue

            print(f"  Scoring {rec['ticker']} ({rec['date']}) → "
                  f"checking price on {outcome_date.strftime('%Y-%m-%d')}...")

            outcome_price = fetch_price_on_date(rec["ticker"], outcome_date)

            if not outcome_price:
                print(f"    ✗ Could not fetch outcome price")
                all_records.append(rec)
                continue

            entry_price = rec["price_at_recommendation"]
            return_pct = round(((outcome_price - entry_price) / entry_price) * 100, 2)
            correct = score_recommendation(rec["action"], return_pct)

            rec["outcome_date"] = outcome_date.strftime("%Y-%m-%d")
            rec["price_at_outcome"] = outcome_price
            rec["return_pct"] = return_pct
            rec["correct"] = correct

            result_str = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"    {result_str} | Entry: ${entry_price} → "
                  f"Outcome: ${outcome_price} | Return: {return_pct:+.1f}%")

            file_updated = True
            all_records.append(rec)

        if file_updated:
            with open(log_file, "w") as f:
                json.dump(records, f, indent=2)
            updated_files += 1

    # ── Summary stats ────────────────────────────────────────────────────────
    scored = [r for r in all_records if r.get("correct") is not None]
    pending = [r for r in all_records if r.get("correct") is None]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total recommendations logged: {len(all_records)}")
    print(f"  Scored (outcome date reached): {len(scored)}")
    print(f"  Pending (not yet mature):      {len(pending)}")

    if not scored:
        print(f"\n  No scored recommendations yet — check back in "
              f"{OUTCOME_DAYS} days.")
        return

    # Overall accuracy
    correct = [r for r in scored if r["correct"]]
    accuracy = len(correct) / len(scored) * 100
    avg_return = sum(r["return_pct"] for r in scored) / len(scored)

    print(f"\n  Overall accuracy:  {accuracy:.1f}% ({len(correct)}/{len(scored)})")
    print(f"  Average return:    {avg_return:+.1f}%")

    # By action type
    for action in ["buy", "hold", "sell"]:
        action_recs = [r for r in scored if r["action"] == action]
        if not action_recs:
            continue
        action_correct = [r for r in action_recs if r["correct"]]
        action_accuracy = len(action_correct) / len(action_recs) * 100
        action_avg_return = sum(r["return_pct"] for r in action_recs) / len(action_recs)
        print(f"\n  {action.upper()} calls ({len(action_recs)} total):")
        print(f"    Accuracy:     {action_accuracy:.1f}%")
        print(f"    Avg return:   {action_avg_return:+.1f}%")

    # Best and worst calls
    scored_sorted = sorted(scored, key=lambda x: x["return_pct"])
    print(f"\n  Best call:  {scored_sorted[-1]['ticker']} "
          f"({scored_sorted[-1]['return_pct']:+.1f}%) on {scored_sorted[-1]['date']}")
    print(f"  Worst call: {scored_sorted[0]['ticker']} "
          f"({scored_sorted[0]['return_pct']:+.1f}%) on {scored_sorted[0]['date']}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run_backtest()