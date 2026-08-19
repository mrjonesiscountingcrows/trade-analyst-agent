import yfinance as yf

def get_fundamentals(ticker: str) -> dict:
    """
    Fetches key fundamental data for a stock or ETF via yfinance.
    Handles cases where fields may be missing (common for ETFs).
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or "symbol" not in info:
        return {"error": f"Could not fetch fundamentals for {ticker}"}

    def safe_round(val, digits=2):
        try:
            return round(float(val), digits)
        except (TypeError, ValueError):
            return None

    def safe_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": safe_int(info.get("marketCap")),
        "pe_ratio": safe_round(info.get("trailingPE")),
        "forward_pe": safe_round(info.get("forwardPE")),
        "eps": safe_round(info.get("trailingEps")),
        "revenue": safe_int(info.get("totalRevenue")),
        "profit_margin": safe_round(info.get("profitMargins"), 4),
        "dividend_yield": safe_round(info.get("dividendYield"), 4),
        "beta": safe_round(info.get("beta")),
        "52w_high": safe_round(info.get("fiftyTwoWeekHigh")),
        "52w_low": safe_round(info.get("fiftyTwoWeekLow")),
        "analyst_target_price": safe_round(info.get("targetMeanPrice")),
        "recommendation": info.get("recommendationKey"),
    }


if __name__ == "__main__":
    import json

    # Test with a stock
    print("=== AAPL (Stock) ===")
    result = get_fundamentals("AAPL")
    print(json.dumps(result, indent=2))

    # Test with an ETF
    print("\n=== SPY (ETF) ===")
    result = get_fundamentals("SPY")
    print(json.dumps(result, indent=2))