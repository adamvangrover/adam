import os
from datetime import datetime

# Define the 10 high-conviction companies with some dummy but realistic-looking data
# Excluding AMZN, GOOGL, MSFT, and NVDA to avoid overwriting existing complex showcase templates.
COMPANIES = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": "173.50", "target": "200.00", "conviction": "HIGH"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical", "price": "175.22", "target": "220.00", "conviction": "HIGH"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services", "price": "505.40", "target": "580.00", "conviction": "HIGH"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services", "price": "610.35", "target": "700.00", "conviction": "HIGH"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Technology", "price": "170.50", "target": "200.00", "conviction": "HIGH"},
    {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "Technology", "price": "301.20", "target": "350.00", "conviction": "HIGH"},
    {"ticker": "PLTR", "name": "Palantir Technologies", "sector": "Technology", "price": "24.50", "target": "35.00", "conviction": "HIGH"},
    {"ticker": "AVGO", "name": "Broadcom Inc.", "sector": "Technology", "price": "1350.00", "target": "1600.00", "conviction": "HIGH"},
    {"ticker": "COST", "name": "Costco Wholesale Corp.", "sector": "Consumer Defensive", "price": "750.25", "target": "820.00", "conviction": "HIGH"},
    {"ticker": "LLY", "name": "Eli Lilly and Co.", "sector": "Healthcare", "price": "780.00", "target": "900.00", "conviction": "HIGH"}
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v26.0 :: {name} ({ticker}) Deep Dive</title>

    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>

    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Inter', 'Segoe UI', 'sans-serif'],
                        mono: ['Fira Code', 'ui-monospace', 'monospace'],
                        display: ['Oswald', 'Inter', 'sans-serif'],
                    }},
                    colors: {{
                        term: {{
                            bg: '#030712', surface: '#0f172a', card: '#1e293b', border: '#334155',
                            cyan: '#06b6d4', blue: '#3b82f6', red: '#ef4444', amber: '#f59e0b', green: '#10b981',
                            neon: '#38bdf8'
                        }}
                    }},
                    backgroundImage: {{
                        'glass': 'linear-gradient(145deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.9) 100%)',
                    }}
                }}
            }}
        }}
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;600&family=Inter:wght@300;400;600;800&family=Oswald:wght@500;700&display=swap');
        body {{ font-size: 14px; background-color: #030712; color: #cbd5e1; }}
        .glow-text {{ text-shadow: 0 0 12px rgba(6, 182, 212, 0.8); }}
        .glass-panel {{
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
    </style>
</head>
<body class="font-sans h-screen overflow-y-auto selection:bg-term-cyan selection:text-white p-8">

    <div class="max-w-4xl mx-auto">
        <header class="border-b border-white/10 pb-6 mb-8 flex justify-between items-end">
            <div>
                <h1 class="text-4xl font-display font-bold text-white tracking-wider glow-text mb-2">{name}</h1>
                <div class="flex gap-4 font-mono text-sm">
                    <span class="text-term-cyan">TICKER: {ticker}</span>
                    <span class="text-slate-500">|</span>
                    <span class="text-term-amber">SECTOR: {sector}</span>
                </div>
            </div>
            <div class="text-right font-mono text-xs text-slate-500">
                <div class="mb-1">GENERATED: {date}</div>
                <div>SYSTEM: ADAM V26.0</div>
            </div>
        </header>

        <main class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="glass-panel p-6 rounded-lg border-t-2 border-t-term-cyan">
                <div class="text-[10px] text-slate-500 font-mono mb-2 uppercase">Current Price</div>
                <div class="text-3xl font-bold text-white">${price}</div>
            </div>

            <div class="glass-panel p-6 rounded-lg border-t-2 border-t-term-green">
                <div class="text-[10px] text-slate-500 font-mono mb-2 uppercase">Target Price (12M)</div>
                <div class="text-3xl font-bold text-term-green">${target}</div>
            </div>

            <div class="glass-panel p-6 rounded-lg border-t-2 border-t-term-amber">
                <div class="text-[10px] text-slate-500 font-mono mb-2 uppercase">Conviction Score</div>
                <div class="text-3xl font-bold text-term-amber glow-amber">{conviction}</div>
            </div>
        </main>

        <section class="glass-panel p-8 rounded-lg mb-8 prose prose-invert max-w-none">
            <h2 class="text-xl font-display text-white mb-4 border-b border-white/10 pb-2">Alpha Routing Protocol</h2>
            <p class="text-slate-300 leading-relaxed font-mono text-sm">
                > INITIALIZING ASSET ANALYSIS FOR {ticker}...<br>
                > CORRELATION MATRIX: STABLE<br>
                > LIQUIDITY PROFILE: OPTIMAL<br>
                > VOLATILITY SKEW: NEGATIVE<br><br>
                System recommendation is to accumulate {ticker} on any macro-induced weakness.
                The current fundamental trajectory indicates a strong probabilistic advantage
                over the next 12-24 months. Expected loss (EL) is well below the dynamic gate limit,
                qualifying {ticker} for automated pass-through in most institutional portfolios.
            </p>
        </section>

        <footer class="text-center text-[10px] font-mono text-slate-600 mt-12 border-t border-white/5 pt-6">
            CONFIDENTIAL // INTERNAL USE ONLY // ADAM FINANCIAL OS V26.0
        </footer>
    </div>

</body>
</html>"""

def main():
    output_dir = "showcase"
    os.makedirs(output_dir, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    for company in COMPANIES:
        filename = f"{company['ticker'].lower()}_company_report.html"
        filepath = os.path.join(output_dir, filename)

        html_content = HTML_TEMPLATE.format(
            name=company["name"],
            ticker=company["ticker"],
            sector=company["sector"],
            price=company["price"],
            target=company["target"],
            conviction=company["conviction"],
            date=current_date
        )

        with open(filepath, "w") as f:
            f.write(html_content)

        print(f"Generated {filepath}")

if __name__ == "__main__":
    main()