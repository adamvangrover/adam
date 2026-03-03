import json
import os
import re

print("Refreshing market mayhem archive...")

# Define paths
SHOWCASE_DIR = "showcase"
HTML_FILE = os.path.join(SHOWCASE_DIR, "market_mayhem_archive.html")
STRATEGIC_DATA_FILE = os.path.join(SHOWCASE_DIR, "data/strategic_command.json")
MARKET_DATA_FILE = os.path.join(SHOWCASE_DIR, "data/sp500_market_data.json")
INDEX_DATA_FILE = os.path.join(SHOWCASE_DIR, "data/market_mayhem_index.json")

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found.")
        return None

def extract_year(date_str):
    if not date_str: return "HISTORICAL"
    # Format might be YYYY-MM-DD or MM-DD-YYYY or MMDD-YY-YY
    # Check for formats like 0404-20-25 -> 2025
    match_odd = re.match(r'^\d{4}-20-(\d{2})$', date_str)
    if match_odd:
        return "20" + match_odd.group(1)

    match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if match:
        return match.group(1)

    return "HISTORICAL"

def format_date_for_sort(date_str):
    if not date_str: return "0000-00-00"

    # Check for formats like 0404-20-25 -> 2025-04-04
    match_odd = re.match(r'^(\d{2})(\d{2})-20-(\d{2})$', date_str)
    if match_odd:
        m, d, y = match_odd.groups()
        return f"20{y}-{m}-{d}"

    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    year = extract_year(date_str)
    if year != "HISTORICAL":
        return f"{year}-{date_str}"

    return date_str

def generate_strategic_html(data):
    if not data: return ""
    house_view = data.get("strategic_directives", {}).get("house_view", "NEUTRAL")
    narrative = data.get("strategic_directives", {}).get("narrative", "")

    color = "#f59e0b"
    if house_view == "BULLISH": color = "#0aff60"
    if house_view == "BEARISH": color = "#ff3333"

    return f"""
                <div class="sidebar-title">
                    <span>STRATEGIC COMMAND</span>
                    <i class="fas fa-chess-king"></i>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="color: #666; font-size: 0.7rem;">HOUSE VIEW</span>
                    <div style="font-size: 1.1rem; color: {color}; font-weight: bold;">{house_view}</div>
                </div>
                <div style="font-size: 0.75rem; line-height: 1.4; color: #ccc;">
                    {narrative}
                </div>
"""

def generate_outlook_html(market_data):
    # Calculate some metrics from market_data
    avg_change = sum(item.get("change_pct", 0) for item in market_data) / len(market_data) if market_data else 0
    sp_target = 6000 + int(avg_change * 100)  # mock calculation
    yield_val = 4.2 - (avg_change * 0.1) # mock
    vix_val = max(10, 15 - avg_change)

    return f"""
                <div class="sidebar-title">
                    <span>FORWARD OUTLOOK</span>
                    <i class="fas fa-binoculars"></i>
                </div>
                <div class="metric-row">
                    <span class="metric-label">S&P 500 Target</span>
                    <span class="metric-val" style="color: #f59e0b;">{sp_target:,.0f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">10Y Yield</span>
                    <span class="metric-val {"val-red" if yield_val > 4 else "val-green"}">{yield_val:.2f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">VIX</span>
                    <span class="metric-val {"val-red" if vix_val > 20 else "val-green"}">{vix_val:.1f}</span>
                </div>
"""

def generate_conviction_html(market_data):
    if not market_data: return ""

    # Sort market data by highest price
    items = sorted(market_data, key=lambda x: x.get("current_price", 0), reverse=True)[:5]
    html = """
                <div class="sidebar-title">
                    <span>TOP CONVICTION</span>
                    <i class="fas fa-star"></i>
                </div>
"""
    for item in items:
        ticker = item.get("ticker", "")
        sector = item.get("sector", "")[:5]
        price = item.get("current_price", 0)
        html += f"""
                <div class="metric-row">
                    <span class="metric-label">{ticker} <span style="font-size:0.6rem; color:#555;">{sector}</span></span>
                    <span class="metric-val val-green">{price}</span>
                </div>
"""
    return html

def generate_watchlist_html(market_data):
    if not market_data: return ""

    # Sort market data by change_pct ascending (worst performers)
    items = sorted(market_data, key=lambda x: x.get("change_pct", 0))[:5]
    html = """
                <div class="sidebar-title">
                    <span>WATCH LIST</span>
                    <i class="fas fa-eye"></i>
                </div>
"""
    for item in items:
        ticker = item.get("ticker", "")
        change = item.get("change_pct", 0)
        color_class = "val-green" if change >= 0 else "val-red"
        html += f"""
                <div class="metric-row">
                    <span class="metric-label">{ticker}</span>
                    <span class="metric-val {color_class}">{change}%</span>
                </div>
"""
    return html

def generate_sector_risk_html(market_data):
    if not market_data: return ""

    # Calculate simple risk by sector based on volatility/change
    sectors = {}
    for item in market_data:
        sec = item.get("sector", "Unknown")
        change = item.get("change_pct", 0)
        # Mock risk calculation: lower change = higher risk
        if sec not in sectors: sectors[sec] = []
        sectors[sec].append(change)

    risk_scores = []
    for sec, changes in sectors.items():
        avg_change = sum(changes) / len(changes)
        risk = max(10, min(99, int(50 - (avg_change * 10))))
        risk_scores.append({"name": sec[:4], "score": risk})

    risk_scores.sort(key=lambda x: x["score"], reverse=True)

    html = """
                <div class="sidebar-title">
                    <span>SECTOR RISK</span>
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="risk-grid">
"""
    for item in risk_scores[:6]:
        score = item["score"]
        if score >= 80:
            css_class = "risk-high"
        elif score >= 50:
            css_class = "risk-med"
        else:
            css_class = "risk-low"

        html += f"""
                    <div class="risk-cell {css_class}">
                        <div>{item['name']}</div>
                        <div style="font-size: 1rem; font-weight: bold;">{score}</div>
                    </div>
"""
    html += "                </div>\n"
    return html

def generate_archive_grid_html(index_data):
    if not index_data: return ""

    # Sort by formatted date descending
    index_data.sort(key=lambda x: format_date_for_sort(x.get("date", "")), reverse=True)

    html = "\n"
    for item in index_data:
        title = item.get("title", "")
        date = item.get("date", "")
        type_ = item.get("type", "HISTORICAL")
        summary = item.get("summary", "")
        sent = item.get("sentiment_score", 50)
        conv = item.get("conviction", 50)
        sem = item.get("semantic_score", 50)
        filename = item.get("filename", "#")
        year = extract_year(date)

        safe_title = title.lower().replace('"', '&quot;')
        safe_summary = summary.lower().replace('"', '&quot;')

        html += f"""
            <div class="archive-item type-{type_}" data-year="{year}" data-type="{type_}" data-title="{safe_title} {safe_summary}">
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">{date}</span>
                        <span class="type-badge">{type_}</span>
                        <span class="mono" style="font-size:0.7rem; color:#444;">SENT: {sent} | CONV: {conv} | SEM: {sem}</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem;">{title}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#aaa;">{summary}</p>
                </div>
                <a href="{filename}" class="cyber-btn" style="align-self:center;">ACCESS</a>
            </div>
"""
    return html

def replace_section(html, start_marker, end_marker, new_content):
    pattern = re.compile(f"({re.escape(start_marker)}).*?({re.escape(end_marker)})", re.DOTALL)
    if not pattern.search(html):
        print(f"Warning: Could not find section between {start_marker} and {end_marker}")
        return html
    return pattern.sub(f"\\1{new_content}\\2", html)

def main():
    strategic_data = load_json(STRATEGIC_DATA_FILE)
    market_data = load_json(MARKET_DATA_FILE)
    index_data = load_json(INDEX_DATA_FILE)

    with open(HTML_FILE, 'r') as f:
        html = f.read()

    # 1. Update Strategic Command
    strategic_html = generate_strategic_html(strategic_data)
    html = replace_section(html, '<!-- Strategic Command -->\n            <div class="sidebar-panel">', '            </div>\n\n            <!-- Forward Outlook -->', f"\n{strategic_html}\n")

    # 2. Update Forward Outlook
    outlook_html = generate_outlook_html(market_data)
    html = replace_section(html, '<!-- Forward Outlook -->\n            <div class="sidebar-panel">', '            </div>\n\n            <!-- Top Conviction -->', f"\n{outlook_html}\n")

    # 3. Update Top Conviction
    conviction_html = generate_conviction_html(market_data)
    html = replace_section(html, '<!-- Top Conviction -->\n            <div class="sidebar-panel">', '            </div>\n\n            <!-- Watch List -->', f"\n{conviction_html}\n")

    # 4. Update Watch List
    watchlist_html = generate_watchlist_html(market_data)
    html = replace_section(html, '<!-- Watch List -->\n            <div class="sidebar-panel">', '            </div>\n\n            <!-- Risk Matrix -->', f"\n{watchlist_html}\n")

    # 5. Update Risk Matrix
    risk_html = generate_sector_risk_html(market_data)
    html = replace_section(html, '<!-- Risk Matrix -->\n            <div class="sidebar-panel">', '            </div>\n\n        </aside>', f"\n{risk_html}\n")

    # 6. Update Archive Grid
    archive_html = generate_archive_grid_html(index_data)
    html = replace_section(html, '<!-- Archive Items Grid (Injected) -->\n            <div id="archiveGrid" style="padding-bottom: 50px;">', '            </div>\n\n        </main>', f"\n{archive_html}\n")

    # 7. Update chart data logic in <script>
    if index_data:
        # Sort ascending for chart using formatted date
        # Also remove duplicate dates to clean up X axis
        seen_dates = set()
        unique_index_data = []
        for item in index_data:
            date = item.get("date", "")
            if date and date not in seen_dates:
                seen_dates.add(date)
                unique_index_data.append(item)

        unique_index_data.sort(key=lambda x: format_date_for_sort(x.get("date", "")))

        labels = [item.get("date", "") for item in unique_index_data]
        sentiment_scores = [item.get("sentiment_score", 50) for item in unique_index_data]

        js_data_inject = f"""
        const chartData = {{
            labels: {json.dumps(labels)},
            datasets: [
                {{
                    label: 'Sentiment Score',
                    data: {json.dumps(sentiment_scores)},
                    borderColor: '#00f3ff',
                    backgroundColor: 'rgba(0, 243, 255, 0.1)',
                    tension: 0.4,
                    fill: true
                }}
            ]
        }}
"""
        html = replace_section(html, '// Data Handling', '// Add Simulated Projections to Data', f"\n{js_data_inject}\n        ")

    with open(HTML_FILE, 'w') as f:
        f.write(html)

    print("Refresh complete.")

if __name__ == "__main__":
    main()
