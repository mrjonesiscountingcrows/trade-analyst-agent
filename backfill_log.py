"""
Reusable backfill script — logs recommendations from any existing report file.
Usage:
    python backfill_log.py 2026-08-27
"""

import os
import sys
import json
import re
import yfinance as yf

if len(sys.argv) < 2:
    print("Usage: python backfill_log.py YYYY-MM-DD")
    sys.exit(1)

REPORT_DATE = sys.argv[1]
REPORT_PATH = f"reports/{REPORT_DATE}.md"

if not os.path.exists(REPORT_PATH):
    print(f"Report not found: {REPORT_PATH}")
    sys.exit(1)

with open(REPORT_PATH, "r") as f:
    report_text = f.read()

print(f"\nReading report: {REPORT_PATH}\n")

pattern = re.compile(
        r'\d+\.\s+\*{0,2}([A-Z]{1,5})[\s\-\*\(].*?\n\s+[-\*]?\s*Action:\s*([\w\s]+)',
        re.MULTILINE
)

seen = set()
records = []

for match in pattern.finditer(report_text):
    ticker = match.group(1).strip()
    action = match.group(2).strip().lower()

    if ticker in seen:
        continue
    seen.add(ticker)

    if any(x in action for x in ["buy", "strong", "speculative"]):
        normalized = "buy"
    elif "sell" in action or "avoid" in action:
        normalized = "sell"
    else:
        normalized = "hold"

    print(f"  Fetching price for {ticker} on {REPORT_DATE}...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=REPORT_DATE, period="5d")
        price = round(float(hist["Close"].iloc[0]), 2) if not hist.empty else None
    except Exception:
        price = None

    records.append({
        "date": REPORT_DATE,
        "ticker": ticker,
        "action": normalized,
        "raw_action": match.group(2).strip(),
        "price_at_recommendation": price,
        "outcome_date": None,
        "price_at_outcome": None,
        "return_pct": None,
        "correct": None,
    })
    print(f"    {ticker} → {normalized} @ ${price}")

if not records:
    print("No recommendations found — check the report format.")
    sys.exit(1)

os.makedirs("logs", exist_ok=True)
log_path = f"logs/{REPORT_DATE}_recommendations.json"
with open(log_path, "w") as f:
    json.dump(records, f, indent=2)

print(f"\n✓ Logged {len(records)} recommendations to {log_path}")