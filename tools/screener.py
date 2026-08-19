import yfinance as yf
import pandas as pd

# Curated investment universe — S&P 500 large caps + popular ETFs
# The agent scans this list to find candidates matching a given thesis
STOCK_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO", "AMD", "ORCL",
    "CRM", "ADBE", "QCOM", "TXN", "INTC", "NOW", "SNOW", "PLTR", "UBER", "NET",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "BLK", "WFC",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    # Consumer
    "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW", "TJX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    # Industrial / Other
    "CAT", "BA", "HON", "UPS", "GE", "MMM", "RTX", "LMT", "DE", "EMR",
]

ETF_UNIVERSE = [
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLB", "XLRE",
    # Thematic
    "ARKK", "BOTZ", "AIQ", "ROBO", "SOXX", "SMH",
    # International
    "EFA", "EEM", "VEA", "VWO",
    # Fixed income / defensive
    "TLT", "AGG", "GLD", "SHY",
]

def screen_tickers(
    universe: str = "stocks",
    min_rsi: float = None,
    max_rsi: float = None,
    min_price: float = None,
    max_price: float = None,
    sector: str = None,
    limit: int = 10,
) -> dict:
    """
    Screens the investment universe and returns a shortlist of tickers
    matching the given criteria. Used by the agent in Stage 1 discovery.

    Args:
        universe: "stocks", "etfs", or "both"
        min_rsi: minimum RSI (e.g. 30 to exclude oversold)
        max_rsi: maximum RSI (e.g. 40 to find oversold candidates)
        min_price: minimum share price
        max_price: maximum share price
        sector: filter by sector string (e.g. "Technology")
        limit: max tickers to return
    """

    if universe == "stocks":
        tickers = STOCK_UNIVERSE
    elif universe == "etfs":
        tickers = ETF_UNIVERSE
    else:
        tickers = STOCK_UNIVERSE + ETF_UNIVERSE

    candidates = []

    print(f"Scanning {len(tickers)} tickers... (this takes ~30 seconds)")

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")

            if hist.empty:
                continue

            # Current price
            current_price = round(float(hist["Close"].iloc[-1]), 2)

            # RSI (14-day from 1 month of data)
            delta = hist["Close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = round(float(rsi_series.iloc[-1]), 2) if not pd.isna(rsi_series.iloc[-1]) else None

            # Apply filters
            if min_price and current_price < min_price:
                continue
            if max_price and current_price > max_price:
                continue
            if min_rsi and rsi and rsi < min_rsi:
                continue
            if max_rsi and rsi and rsi > max_rsi:
                continue
            if sector:
                ticker_sector = info.get("sector", "")
                if sector.lower() not in (ticker_sector or "").lower():
                    continue

            candidates.append({
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "current_price": current_price,
                "rsi": rsi,
                "recommendation": info.get("recommendationKey"),
            })

        except Exception:
            continue  # Skip any ticker that errors out

    # Sort by RSI ascending (most oversold first) as default ranking
    candidates.sort(key=lambda x: x["rsi"] if x["rsi"] is not None else 999)

    return {
        "universe": universe,
        "scanned": len(tickers),
        "matched": len(candidates[:limit]),
        "candidates": candidates[:limit],
    }


if __name__ == "__main__":
    import json

    # Test: find oversold tech stocks (RSI under 45)
    print("=== Oversold Tech Stocks ===")
    result = screen_tickers(universe="stocks", max_rsi=45, sector="Technology", limit=5)
    print(json.dumps(result, indent=2))