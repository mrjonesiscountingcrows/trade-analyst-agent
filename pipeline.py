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

SCREENING_CRITERIA = {
    "max_rsi": 35,          # Oversold threshold
    "min_price": 10.0,      # No penny stocks
    "min_market_cap": 500_000_000,  # $500M minimum market cap
    "max_candidates": 15,   # How many to deep-dive on
}

# ── Stage 1: Fast bulk screener ──────────────────────────────────────────────

def bulk_screen(tickers: list[str]) -> list[dict]:
    """
    Loops through all tickers and applies fast filters.
    Only fetches 1 month of price data per ticker to keep it quick.
    Returns a shortlist of candidates sorted by RSI ascending.
    """
    candidates = []
    total = len(tickers)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"STAGE 1: Screening {total} tickers")
    print(f"Criteria: RSI < {SCREENING_CRITERIA['max_rsi']}, "
          f"Price > ${SCREENING_CRITERIA['min_price']}, "
          f"Market Cap > ${SCREENING_CRITERIA['min_market_cap']:,}")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers):
        # Progress update every 500 tickers
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (total - i) / rate
            print(f"  Progress: {i}/{total} tickers scanned | "
                  f"{len(candidates)} candidates so far | "
                  f"~{remaining/60:.1f} mins remaining")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo", auto_adjust=True)

            if hist.empty or len(hist) < 15:
                continue

            # Price filter
            current_price = float(hist["Close"].iloc[-1])
            if current_price < SCREENING_CRITERIA["min_price"]:
                continue

            # RSI calculation
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = float(rsi_series.iloc[-1])

            if pd.isna(rsi) or rsi > SCREENING_CRITERIA["max_rsi"]:
                continue

            # Market cap filter (quick info fetch)
            info = stock.info
            market_cap = info.get("marketCap") or 0
            if market_cap < SCREENING_CRITERIA["min_market_cap"]:
                continue

            candidates.append({
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector"),
                "current_price": round(current_price, 2),
                "rsi": round(rsi, 2),
                "market_cap": market_cap,
            })

        except Exception:
            continue  # Skip anything that errors

    # Sort by RSI ascending (most oversold first)
    candidates.sort(key=lambda x: x["rsi"])
    top = candidates[:SCREENING_CRITERIA["max_candidates"]]

    elapsed_mins = (time.time() - start_time) / 60
    print(f"\n✓ Screening complete in {elapsed_mins:.1f} mins")
    print(f"  Found {len(candidates)} candidates matching criteria")
    print(f"  Taking top {len(top)} for deep analysis\n")

    return top

# ── Stage 2: Deep analysis ───────────────────────────────────────────────────

def deep_analyze(candidates: list[dict]) -> list[dict]:
    """
    Runs full analysis on each candidate with rate limit handling.
    """
    print(f"{'='*60}")
    print(f"STAGE 2: Deep analysis on {len(candidates)} candidates")
    print(f"Pausing 60 seconds to let Yahoo Finance rate limit recover...")
    print(f"{'='*60}\n")

    time.sleep(60)  # Let the rate limit recover after bulk screening

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

You have been given deep analysis data for a set of stocks that passed an 
overnight screen (RSI < 35, price > $10, market cap > $500M). Your job is 
to synthesize this data into a concise, actionable morning report.

For each ticker you have: current price, RSI, moving averages, recent news 
headlines, P/E ratio, EPS, analyst target price, and analyst recommendation.

Produce a report in this exact format:

═══════════════════════════════════════════════════════════
  TRADE ANALYST DAILY BRIEF — {date}
═══════════════════════════════════════════════════════════

MARKET OVERVIEW
[2-3 sentences on what the screen results suggest about the 
broader market environment today]

TOP PICKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[For each of your top 3-5 picks, include:]
[Rank]. [TICKER] — [BUY/HOLD/AVOID] | [High/Medium/Low] Conviction
Company: [Name] | Sector: [Sector]
Price: $[X] | RSI: [X] | Analyst Target: $[X]
Thesis: [2-3 sentences — why this is interesting right now]
Key Risk: [1 sentence]

TICKERS TO WATCH (but not buy yet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2-3 tickers from the screened list that need more confirmation
before buying — briefly explain why]

PASS (do not buy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Remaining tickers with one-line reason to pass]

RISKS TO WATCH TODAY
[2-3 macro or sector risks relevant to today's picks]

═══════════════════════════════════════════════════════════

Be direct and specific. Reference actual numbers. Never fabricate data.
If analyst target or recommendation is null, note it and work around it.
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

    total_mins = (time.time() - overall_start) / 60
    print(f"\n✓ Pipeline complete in {total_mins:.1f} mins")
    print(f"  Report saved to: {filepath}\n")
    print(report)