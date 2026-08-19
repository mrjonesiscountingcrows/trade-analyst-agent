import finnhub
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def get_news(ticker: str, days_back: int = 7) -> dict:
    """
    Fetches recent news headlines for a ticker from Finnhub
    and returns them with a simple sentiment summary.
    """
    client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        articles = client.company_news(ticker, _from=start_date, to=end_date)
    except Exception as e:
        return {"error": str(e)}

    if not articles:
        return {
            "ticker": ticker,
            "article_count": 0,
            "headlines": [],
            "sentiment_summary": "No recent news found."
        }

    # Take the 10 most recent articles
    articles = articles[:10]

    headlines = [
        {
            "headline": a["headline"],
            "source": a["source"],
            "datetime": datetime.fromtimestamp(a["datetime"]).strftime("%Y-%m-%d"),
            "sentiment": a.get("sentiment", {})
        }
        for a in articles
    ]

    # Simple sentiment scoring across articles
    scores = [
        a.get("sentiment", {}).get("articleScore", None)
        for a in articles
    ]
    scores = [s for s in scores if s is not None]

    if scores:
        avg_score = round(sum(scores) / len(scores), 3)
        if avg_score > 0.6:
            sentiment_label = "Positive"
        elif avg_score < 0.4:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        sentiment_summary = f"{sentiment_label} (avg score: {avg_score})"
    else:
        sentiment_summary = "Sentiment data unavailable"

    return {
        "ticker": ticker,
        "article_count": len(headlines),
        "sentiment_summary": sentiment_summary,
        "headlines": headlines,
    }


if __name__ == "__main__":
    result = get_news("AAPL")
    print(f"Ticker: {result['ticker']}")
    print(f"Articles found: {result['article_count']}")
    print(f"Sentiment: {result['sentiment_summary']}")
    print("\nHeadlines:")
    for h in result["headlines"]:
        print(f"  [{h['datetime']}] {h['source']}: {h['headline']}")