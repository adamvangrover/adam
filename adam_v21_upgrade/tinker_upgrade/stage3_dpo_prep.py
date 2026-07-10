import json
import os
output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "adam_preference_data.jsonl")
# Expanded dataset from Section 4.2
preferences = [
    {
        "prompt": "Analyze the impact of the latest FOMC minutes on tech stocks.",
        "chosen": "The minutes suggest a hawkish pause. While the surface language was neutral, the dissent regarding balance sheet runoff suggests liquidity constraints will hit high-duration assets (tech) harder than the general market anticipates.",
        "rejected": "The FOMC minutes were mixed. Some members want to pause rates, others want to hike. Tech stocks might go up or down depending on future inflation data."
    },
    {
        "prompt": "Review this trading signal: Buy TSLA on RSI dip.",
        "chosen": "Signal Rejected. While RSI is low (30), the volume profile shows distribution at the support level. A 'buy' here ignores the macro headwind of EV demand softening in China. Wait for volume confirmation.",
        "rejected": "Signal Accepted. RSI is below 30 which indicates the stock is oversold. It is a good time to buy for a bounce."
    },
    {
        "prompt": "The CEO of Company Y just gave a very bullish interview on TV.",
        "chosen": "The interview was overly promotional and lacked specific data. This looks like a 'management pump' to cover for the upcoming weak quarter we're forecasting. Sentiment is a contrary indicator here; the bullishness is a signal to check short interest.",
        "rejected": "The CEO sounds very confident and optimistic. He says demand is strong and the stock will likely go up. This is a strong buy signal."
    },
    {
        "prompt": "This high-yield bond (CCC) is trading at 50 cents on the dollar. What's the play?",
        "chosen": "The market is pricing in default, but our legal analysis of the bond covenants suggests asset recovery value is closer to 70 cents. The consensus is wrong about the asset backing, not the default probability. This is a buy.",
        "rejected": "This is a CCC-rated bond, which is junk. It's trading at 50 cents because it is very risky and will probably default. Avoid this bond."
    },
    {
        "prompt": "S&P 500 just hit an all-time high. Is it time to go risk-on?",
        "chosen": "The high is driven by 5-6 mega-cap tech stocks. Market breadth is poor, and the equal-weighted S&P is flat. This rally is narrow and fragile, suggesting high systemic risk, not a 'risk-on' environment.",
        "rejected": "Yes, the all-time high is a very bullish signal. The trend is your friend, and all indicators show strong momentum. It's time to increase equity exposure."
    }
]
print(f"Generating Stage 3 DPO data to {output_file}...")
with open(output_file, 'w') as f:
    for p in preferences:
        f.write(json.dumps(p) + '\n')
print("Done.")
