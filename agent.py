import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from tools.price import get_price_data
from tools.news import get_news
from tools.fundamentals import get_fundamentals
from tools.screener import screen_tickers

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Tool definitions (what we expose to the agent) ──────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "screen_tickers",
            "description": (
                "Scans the investment universe and returns a shortlist of tickers "
                "matching the given criteria. Use this first to discover candidates "
                "before doing deep analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "universe": {
                        "type": "string",
                        "enum": ["stocks", "etfs", "both"],
                        "description": "Which universe to scan."
                    },
                    "min_rsi": {"type": "number", "description": "Minimum RSI value."},
                    "max_rsi": {"type": "number", "description": "Maximum RSI value. Use <= 35 to find oversold tickers."},
                    "min_price": {"type": "number", "description": "Minimum share price filter."},
                    "max_price": {"type": "number", "description": "Maximum share price filter."},
                    "sector": {"type": "string", "description": "Sector to filter by, e.g. 'Technology', 'Healthcare'."},
                    "limit": {"type": "integer", "description": "Max number of candidates to return. Default 10."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_data",
            "description": "Fetches 1 year of price history and technical indicators (RSI, MA50, MA200) for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock or ETF ticker symbol."}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetches recent news headlines for a ticker. Use to assess sentiment and recent developments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock or ETF ticker symbol."},
                    "days_back": {"type": "integer", "description": "How many days of news to fetch. Default 7."},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Fetches fundamental data for a ticker: P/E ratio, EPS, revenue, analyst target price, recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock or ETF ticker symbol."}
                },
                "required": ["ticker"],
            },
        },
    },
]

# ── Tool dispatcher ──────────────────────────────────────────────────────────

def dispatch_tool(name: str, args: dict):
    """Routes a tool call from the agent to the right function."""
    if name == "screen_tickers":
        return screen_tickers(**args)
    elif name == "get_price_data":
        return get_price_data(**args)
    elif name == "get_news":
        return get_news(**args)
    elif name == "get_fundamentals":
        return get_fundamentals(**args)
    else:
        return {"error": f"Unknown tool: {name}"}

# ── Agent loop ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a professional equity research analyst. Your job is to help users find 
good trade opportunities in US stocks and ETFs.

When given a query, you follow a two-stage process:

STAGE 1 — DISCOVERY:
Use the screen_tickers tool to find candidate tickers that match the user's thesis.
Choose appropriate filters based on the user's intent (e.g. oversold = low RSI,
momentum = high RSI, specific sector, etc.).

STAGE 2 — DEEP ANALYSIS:
For each promising candidate (aim for 2-3 tickers), call all three analysis tools:
- get_price_data (technicals)
- get_news (recent headlines and sentiment)
- get_fundamentals (valuation and analyst views)

FINAL OUTPUT:
After gathering all data, produce a structured research report with:
1. Executive Summary (2-3 sentences on the overall opportunity)
2. Top Pick — your single best recommendation with ticker, action (Buy/Hold/Sell),
   conviction level (High/Medium/Low), current price, and a price target if available
3. Runner-up picks (if applicable)
4. Key Risks to watch
5. Reasoning — what drove your conclusion across technicals, news, and fundamentals

Be direct, specific, and data-driven. Reference actual numbers from the tools.
Never make up data. If a field is null, note it and work around it.
"""

def run_agent(user_query: str) -> str:
    """
    Runs the full agent loop for a given user query.
    Returns the final research report as a string.
    """
    print(f"\n{'='*60}")
    print(f"Query: {user_query}")
    print(f"{'='*60}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    # Agent loop — keeps running until the model stops calling tools
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # If no tool calls, we have the final answer
        if not message.tool_calls:
            return message.content

        # Process each tool call
        messages.append(message)  # Add assistant message with tool calls

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"→ Calling tool: {tool_name}({tool_args})")

            result = dispatch_tool(tool_name, tool_args)

            print(f"  ✓ Done\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })