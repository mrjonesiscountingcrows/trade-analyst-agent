from agent import run_agent

if __name__ == "__main__":
    # Try a few different queries to see the agent in action
    queries = [
        "Are there any energy stocks with strong fundamentals worth buying?",
    ]

    for query in queries:
        report = run_agent(query)
        print("\n" + "="*60)
        print("RESEARCH REPORT")
        print("="*60)
        print(report)
        print("\n")