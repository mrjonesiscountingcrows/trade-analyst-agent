import yfinance as yf
import pandas as pd

def get_price_data(ticker: str) -> dict:
    """
    Fetches 1 year of daily price history and calculates
    key technical indicators: RSI, 50-day MA, 200-day MA.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty:
        return {"error": f"No price data found for {ticker}"}

    # Moving averages
    hist["MA50"] = hist["Close"].rolling(window=50).mean()
    hist["MA200"] = hist["Close"].rolling(window=200).mean()

    # RSI (14-day)
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    hist["RSI"] = 100 - (100 / (1 + rs))

    latest = hist.iloc[-1]

    return {
        "ticker": ticker,
        "current_price": round(float(latest["Close"]), 2),
        "ma50": round(float(latest["MA50"]), 2) if not pd.isna(latest["MA50"]) else None,
        "ma200": round(float(latest["MA200"]), 2) if not pd.isna(latest["MA200"]) else None,
        "rsi": round(float(latest["RSI"]), 2) if not pd.isna(latest["RSI"]) else None,
        "volume": int(latest["Volume"]),
        "52w_high": round(float(hist["Close"].max()), 2),
        "52w_low": round(float(hist["Close"].min()), 2),
    }


if __name__ == "__main__":
    result = get_price_data("AAPL")
    print(result)