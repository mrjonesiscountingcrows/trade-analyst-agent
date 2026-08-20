import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_us_tickers(min_price: float = 5.0) -> list[str]:
    """
    Fetches all US-listed tickers from EODHD and filters to
    a clean investable universe. Removes penny stocks, ETNs,
    warrants, rights, and preferred shares.

    Args:
        min_price: minimum share price to include (default $5)

    Returns:
        List of ticker strings e.g. ["AAPL", "MSFT", ...]
    """
    api_key = os.getenv("EODHD_API_KEY")
    url = f"https://eodhd.com/api/exchange-symbol-list/US?api_token={api_key}&fmt=json"

    print("Fetching US ticker universe from EODHD...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching ticker universe: {e}")
        return []

    tickers = []

    for item in data:
        # Only common stocks and ETFs
        asset_type = item.get("Type", "")
        if asset_type not in ("Common Stock", "ETF"):
            continue

        ticker = item.get("Code", "")
        if not ticker:
            continue

        # Skip tickers with special characters (warrants, rights, units)
        if any(c in ticker for c in ["-", ".", "+", "^", "/"]):
            continue

        # Skip very long tickers (usually not standard equities)
        if len(ticker) > 5:
            continue

        tickers.append(ticker)

    print(f"Universe built: {len(tickers)} tickers after filtering")
    return tickers


if __name__ == "__main__":
    tickers = get_us_tickers()
    print(f"\nFirst 20 tickers: {tickers[:20]}")
    print(f"Last 20 tickers: {tickers[-20:]}")