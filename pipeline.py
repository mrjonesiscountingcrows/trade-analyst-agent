import os
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from tools.universe import get_us_tickers
from tools.price import get_price_data
from tools.news import get_news
from tools.fundamentals import get_fundamentals

import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Configuration ────────────────────────────────────────────────────────────

# ── Screening modes ──────────────────────────────────────────────────────────

SCREENING_MODES = {
    "oversold": {
        "label": "Oversold Bounce Candidates",
        "description": "RSI < 35 — beaten down stocks that may revert",
        "candidates_per_mode": 8,
    },
    "momentum": {
        "label": "Momentum Leaders",
        "description": "RSI > 60 and price above MA50 — strong uptrends",
        "candidates_per_mode": 8,
    },
    "breakout": {
        "label": "Breakout Candidates",
        "description": "Within 3% of 52-week high with RSI 50-70",
        "candidates_per_mode": 8,
    },
    "value": {
        "label": "Value / Mean Reversion",
        "description": "Price below MA200 with above-average volume",
        "candidates_per_mode": 6,
    },
}

BASE_FILTERS = {
    "min_price": 10.0,
    "min_market_cap": 500_000_000,
}

MAX_DEEP_ANALYSIS = 20  # Total tickers across all modes


# ── Stage 1: Multi-mode bulk screener ────────────────────────────────────────

def bulk_screen(tickers: list[str]) -> list[dict]:
    """
    Single pass through all tickers, scoring each against all four
    screening modes simultaneously. Returns a combined shortlist.
    """
    results = {mode: [] for mode in SCREENING_MODES}
    total = len(tickers)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"STAGE 1: Multi-mode screen across {total} tickers")
    print(f"Modes: {', '.join(SCREENING_MODES.keys())}")
    print(f"Base filters: Price > ${BASE_FILTERS['min_price']}, "
          f"Market Cap > ${BASE_FILTERS['min_market_cap']:,}")
    print(f"{'='*60}\n")

    import random
    random.shuffle(tickers)
    print(f"  Ticker order randomized — alphabet bias eliminated\n")

    for i, ticker in enumerate(tickers):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (total - i) / rate
            counts = {m: len(v) for m, v in results.items()}
            print(f"  Progress: {i}/{total} | {counts} | "
                  f"~{remaining/60:.1f} mins remaining")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="12mo", auto_adjust=True)

            if hist.empty or len(hist) < 30:
                continue

            closes = hist["Close"]
            volumes = hist["Volume"]

            current_price = float(closes.iloc[-1])
            if current_price < BASE_FILTERS["min_price"]:
                continue

            # Skip foreign OTC stocks — these end in F or Y and have
            # unreliable data; we want proper US-listed equities
            if ticker.endswith("F") or ticker.endswith("Y"):
                continue

            # Skip tickers longer than 4 chars — usually OTC or special issues
            if len(ticker) > 4:
                continue

            # RSI (14-day)
            delta = closes.diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1])
            if pd.isna(rsi):
                continue

            # Guard against bad RSI data (0.0 or 100.0 are artifacts)
            if rsi < 5 or rsi > 98:
                continue

            # Moving averages
            ma50 = float(closes.rolling(window=50).mean().iloc[-1]) if len(closes) >= 50 else None
            ma200 = float(closes.rolling(window=200).mean().iloc[-1]) if len(closes) >= 200 else None

            # 52-week high (from 3mo data — approximate)
            high_52w = float(closes.max())

            # Volume — is today's volume above 20-day average?
            avg_volume = float(volumes.rolling(window=20).mean().iloc[-1])
            volume_today = float(volumes.iloc[-1])
            volume_spike = volume_today > (avg_volume * 1.5)

            # Market cap filter
            info = stock.info
            market_cap = info.get("marketCap") or 0
            if market_cap < BASE_FILTERS["min_market_cap"]:
                continue

            name = info.get("longName") or info.get("shortName", ticker)
            sector = info.get("sector")

            base = {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "current_price": round(current_price, 2),
                "rsi": round(rsi, 2),
                "market_cap": market_cap,
            }

            # ── Mode 1: Oversold ──
            if rsi < 35:
                results["oversold"].append({**base, "signal": f"RSI {rsi:.1f} — oversold"})

            # ── Mode 2: Momentum ──
            if rsi > 60 and ma50 and current_price > ma50:
                results["momentum"].append({**base, "signal": f"RSI {rsi:.1f}, price above MA50"})

            # ── Mode 3: Breakout ──
            if ma50 and 50 < rsi < 70:
                pct_from_high = (high_52w - current_price) / high_52w
                if pct_from_high <= 0.03:
                    results["breakout"].append({
                        **base,
                        "signal": f"Within {pct_from_high*100:.1f}% of 52w high, RSI {rsi:.1f}"
                    })

            # ── Mode 4: Value / Mean Reversion ──
            if ma200 and current_price < ma200 and volume_spike:
                results["value"].append({
                    **base,
                    "signal": f"Below MA200, volume {volume_today/avg_volume:.1f}x average"
                })

            time.sleep(0.05)  # 50ms pause — keeps us under Yahoo Finance rate limit

        except Exception:
            continue

    elapsed_mins = (time.time() - start_time) / 60
    print(f"\n✓ Screening complete in {elapsed_mins:.1f} mins")
    for mode, candidates in results.items():
        print(f"  {SCREENING_MODES[mode]['label']}: {len(candidates)} total found")

    # Now that the full universe is scanned, rank each mode properly
    # and pick the best across the full alphabet — not first-found
    combined = []
    for mode, candidates in results.items():
        limit = SCREENING_MODES[mode]["candidates_per_mode"]

        if mode == "oversold":
            # Most oversold (lowest RSI) first
            candidates.sort(key=lambda x: x["rsi"])
        elif mode == "momentum":
            # Strongest momentum (highest RSI) first
            candidates.sort(key=lambda x: -x["rsi"])
        elif mode == "breakout":
            # Closest to 52w high first — need to store pct_from_high
            # Fall back to market cap for now
            candidates.sort(key=lambda x: -x["market_cap"])
        elif mode == "value":
            # Largest companies with volume spike first
            candidates.sort(key=lambda x: -x["market_cap"])

        top = candidates[:limit]
        print(f"  → Top {len(top)} for deep analysis: "
              f"{[c['ticker'] for c in top]}")

        for c in top:
            c["mode"] = mode
            if not any(existing["ticker"] == c["ticker"] for existing in combined):
                combined.append(c)

    print(f"\n  Combined shortlist: {len(combined)} unique tickers")
    print(f"  Tickers: {[c['ticker'] for c in combined]}\n")

    return combined[:MAX_DEEP_ANALYSIS]

# ── Stage 2: Deep analysis ───────────────────────────────────────────────────

def deep_analyze(candidates: list[dict]) -> list[dict]:
    """
    Runs full analysis on each candidate with rate limit handling.
    """
    print(f"{'='*60}")
    print(f"STAGE 2: Deep analysis on {len(candidates)} candidates")
    print(f"Pausing 3 minutes to let Yahoo Finance rate limit recover...")
    print(f"{'='*60}\n")

    time.sleep(180)  # Let the rate limit recover after bulk screening

    analyzed = []

    for i, candidate in enumerate(candidates):
        ticker = candidate["ticker"]
        print(f"  [{i+1}/{len(candidates)}] Analyzing {ticker} — {candidate['name']}")

        # Retry up to 3 times with backoff on rate limit errors
        for attempt in range(3):
            try:
                price = get_price_data(ticker)
                news = get_news(ticker)
                fundamentals = get_fundamentals(ticker)

                analyzed.append({
                    "ticker": ticker,
                    "name": candidate["name"],
                    "sector": candidate["sector"],
                    "screen_rsi": candidate["rsi"],
                    "price_data": price,
                    "news": news,
                    "fundamentals": fundamentals,
                })
                print(f"     ✓ Done")
                time.sleep(3)  # Polite pause between tickers
                break

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "too many" in error_str:
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                    print(f"     ⚠ Rate limited, waiting {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                else:
                    print(f"     ✗ Skipped ({e})")
                    break

    return analyzed

# ── Stage 3: Agent synthesis ─────────────────────────────────────────────────

SYNTHESIS_PROMPT = """
You are a professional equity research analyst producing a daily morning brief.

You have been given deep analysis data for stocks that passed an overnight 
multi-mode screen. Each ticker has a "mode" field indicating which screen 
surfaced it:

- oversold: RSI < 35, potential bounce plays
- momentum: RSI > 60 and above MA50, strong uptrends  
- breakout: within 3% of 52-week high, RSI 50-70
- value: price below MA200 with volume spike

Produce a report in this exact format:

═══════════════════════════════════════════════════════════
  TRADE ANALYST DAILY BRIEF — {date}
═══════════════════════════════════════════════════════════

MARKET OVERVIEW
[3-4 sentences — what does the distribution of signals across 
modes tell you about today's market environment? Are we seeing 
broad oversold conditions, momentum continuation, breakouts, etc?]

OVERSOLD BOUNCE PLAYS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Top 2-3 oversold picks with: ticker, action, conviction, 
price, RSI, analyst target, 2-sentence thesis, key risk]

MOMENTUM PLAYS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Top 2-3 momentum picks — same format]

BREAKOUT WATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Top 2-3 breakout candidates — same format]

VALUE / MEAN REVERSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Top 2 value plays — same format]

PASS (do not buy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Any screened tickers not worth acting on — one line each]

RISKS TO WATCH TODAY
[2-3 macro or sector risks relevant to today's picks]

═══════════════════════════════════════════════════════════

Be direct and specific. Reference actual numbers. Never fabricate data.
If a section has no candidates, write 'No signals today.'
"""

def synthesize_report(analyzed: list[dict]) -> str:
    """
    Sends all analyzed data to the agent for final synthesis.
    Returns the formatted report as a string.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 3: Agent synthesis")
    print(f"{'='*60}\n")

    today = datetime.today().strftime("%A, %B %d %Y")
    data_payload = json.dumps(analyzed, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3000,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIS_PROMPT.replace("{date}", today)
            },
            {
                "role": "user",
                "content": f"Here is today's screening and analysis data:\n\n{data_payload}"
            }
        ]
    )

    return response.choices[0].message.content

# ── Save report ──────────────────────────────────────────────────────────────

def save_report(report: str) -> str:
    """Saves the report to the reports/ directory with today's date."""
    os.makedirs("reports", exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    filepath = f"reports/{date_str}.md"

    with open(filepath, "w") as f:
        f.write(report)

    return filepath

# ── Main ─────────────────────────────────────────────────────────────────────

import pickle

def save_candidates(candidates: list[dict]):
    """Saves screener results to a dated cache file."""
    os.makedirs("cache", exist_ok=True)
    date_str = datetime.today().strftime("%Y-%m-%d")
    filepath = f"cache/{date_str}_candidates.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(candidates, f)
    print(f"  Candidates cached to: {filepath}")
    return filepath

def load_candidates() -> list[dict] | None:
    """Loads today's cached candidates if they exist."""
    date_str = datetime.today().strftime("%Y-%m-%d")
    filepath = f"cache/{date_str}_candidates.pkl"
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            candidates = pickle.load(f)
        print(f"  ✓ Loaded {len(candidates)} candidates from today's cache")
        print(f"    Skipping Stage 1 — delete {filepath} to re-run screener\n")
        return candidates
    return None

def log_recommendations(report: str, analyzed: list[dict], date_str: str):
    """
    Parses the agent's report and saves structured recommendation records
    to a JSON log file for later backtesting.
    """
    import re

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{date_str}_recommendations.json"

    # Build a price lookup from analyzed data
    prices = {
        item["ticker"]: item["price_data"].get("current_price")
        for item in analyzed
        if item.get("price_data")
    }

    # Simple regex to extract BUY/HOLD/SELL/AVOID actions from the report
    records = []
    action_pattern = re.compile(
        r'\*{0,2}([A-Z]{1,5})\*{0,2}.*?Action.*?:\s*\*{0,2}(Buy|Hold|Sell|Avoid|Watch|Strong Buy|Speculative Buy)',
        re.IGNORECASE | re.DOTALL
    )

    seen = set()
    for match in action_pattern.finditer(report):
        ticker = match.group(1).strip()
        action = match.group(2).strip().lower()

        if ticker in seen or ticker not in prices:
            continue
        seen.add(ticker)

        # Normalize action to buy/hold/sell
        if any(x in action for x in ["buy", "strong"]):
            normalized = "buy"
        elif "sell" in action or "avoid" in action:
            normalized = "sell"
        else:
            normalized = "hold"

        records.append({
            "date": date_str,
            "ticker": ticker,
            "action": normalized,
            "raw_action": match.group(2).strip(),
            "price_at_recommendation": prices.get(ticker),
            "outcome_date": None,       # filled in by backtest.py
            "price_at_outcome": None,   # filled in by backtest.py
            "return_pct": None,         # filled in by backtest.py
            "correct": None,            # filled in by backtest.py
        })

    with open(log_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  Logged {len(records)} recommendations to: {log_path}")
    return records

if __name__ == "__main__":
    overall_start = time.time()

    print("\n🔍 TRADE ANALYST PIPELINE STARTING")
    print(f"   {datetime.today().strftime('%A, %B %d %Y — %I:%M %p')}\n")

    # Stage 1: Check cache first, only screen if needed
    candidates = load_candidates()
    if not candidates:
        tickers = get_us_tickers()
        candidates = bulk_screen(tickers)
        if not candidates:
            print("No candidates found matching criteria today.")
            exit()
        save_candidates(candidates)

    # Stage 2: Deep analysis
    analyzed = deep_analyze(candidates)

    # Stage 3: Synthesize report
    report = synthesize_report(analyzed)

    # Save and display
    filepath = save_report(report)
    date_str = datetime.today().strftime("%Y-%m-%d")
    log_recommendations(report, analyzed, date_str)

    total_mins = (time.time() - overall_start) / 60
    print(f"\n✓ Pipeline complete in {total_mins:.1f} mins")
    print(f"  Report saved to: {filepath}\n")
    print(report)