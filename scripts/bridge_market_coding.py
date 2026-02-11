import pandas as pd
import subprocess
import json
import os
from datetime import datetime
from generate_market_mayhem_archive import get_all_data

OUTPUT_FILE = "showcase/market_coding_correlation.html"

def get_git_activity():
    """Gets daily commit counts from the current git repo."""
    try:
        # git log --date=short --pretty=format:%ad
        result = subprocess.run(
            ["git", "log", "--date=short", "--pretty=format:%ad"],
            capture_output=True,
            text=True,
            check=True
        )
        dates = result.stdout.splitlines()
        df = pd.DataFrame(dates, columns=["date"])
        df["date"] = pd.to_datetime(df["date"])
        daily_counts = df.groupby("date").size().reset_index(name="commit_count")
        return daily_counts
    except Exception as e:
        print(f"Error getting git log: {e}")
        return pd.DataFrame(columns=["date", "commit_count"])

def generate_report():
    print("Fetching Market Intelligence...")
    market_data_list = get_all_data()

    # Convert to DataFrame
    market_df = pd.DataFrame(market_data_list)
    # Ensure date format
    market_df["date"] = pd.to_datetime(market_df["date"], errors='coerce')
    market_df = market_df.dropna(subset=["date"])

    # Group by date and take average sentiment
    sentiment_df = market_df.groupby("date")["sentiment_score"].mean().reset_index()

    print("Fetching Coding Activity...")
    git_df = get_git_activity()

    print("Correlating Data...")
    # Merge
    merged_df = pd.merge(sentiment_df, git_df, on="date", how="inner")

    if merged_df.empty:
        print("No overlapping dates found between Market Data and Git History.")
        correlation = 0
    else:
        correlation = merged_df["sentiment_score"].corr(merged_df["commit_count"])

    print(f"Correlation Coefficient: {correlation:.4f}")

    # Prepare data for Chart.js
    merged_df = merged_df.sort_values("date")
    dates = merged_df["date"].dt.strftime("%Y-%m-%d").tolist()
    sentiments = merged_df["sentiment_score"].tolist()
    commits = merged_df["commit_count"].tolist()

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market-Code Correlation Bridge</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{ --bg: #050b14; --text: #e0e0e0; --accent: #00f3ff; --secondary: #cc0000; }}
        body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ font-family: 'JetBrains Mono', monospace; color: var(--accent); border-bottom: 1px solid #333; padding-bottom: 10px; }}
        .stat-box {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px; border: 1px solid #333; margin-bottom: 20px; text-align: center; }}
        .stat-val {{ font-size: 2rem; font-weight: bold; font-family: 'JetBrains Mono'; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; }}
        .explanation {{ font-size: 0.9rem; color: #aaa; margin-top: 20px; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BRIDGE: MARKET SENTIMENT ⇄ GIT ACTIVITY</h1>

        <div class="stat-box">
            <div style="font-size: 0.8rem; text-transform: uppercase; color: #888;">Correlation Coefficient (Pearson)</div>
            <div class="stat-val" style="color: { '#00ff00' if correlation > 0 else '#ff0000' }">{correlation:.4f}</div>
        </div>

        <div class="chart-container">
            <canvas id="correlationChart"></canvas>
        </div>

        <div class="explanation">
            <p><strong>System Analysis:</strong> This bridge script correlates the <em>Market Sentiment Score</em> (derived from internal newsletters and reports) with <em>Developer Activity</em> (Git Commit Frequency).</p>
            <p>A positive correlation suggests that development velocity increases during bullish market sentiment. A negative correlation implies "Panic Coding" during downturns or "Complacency" during booms.</p>
            <p><strong>Data Points:</strong> {len(merged_df)} overlapping days.</p>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('correlationChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates)},
                datasets: [
                    {{
                        label: 'Market Sentiment (0-100)',
                        data: {json.dumps(sentiments)},
                        borderColor: '#00f3ff',
                        backgroundColor: 'rgba(0, 243, 255, 0.1)',
                        yAxisID: 'y',
                        tension: 0.4
                    }},
                    {{
                        label: 'Git Commits (Count)',
                        data: {json.dumps(commits)},
                        borderColor: '#cc0000',
                        backgroundColor: 'rgba(204, 0, 0, 0.1)',
                        yAxisID: 'y1',
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{
                    legend: {{ labels: {{ color: '#ccc' }} }},
                    title: {{ display: true, text: 'Sentiment vs. Velocity', color: '#888' }}
                }},
                scales: {{
                    x: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#666' }} }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#00f3ff' }},
                        title: {{ display: true, text: 'Sentiment', color: '#00f3ff' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#cc0000' }},
                        title: {{ display: true, text: 'Commits', color: '#cc0000' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """

    with open(OUTPUT_FILE, "w") as f:
        f.write(html_content)

    print(f"Report generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()
