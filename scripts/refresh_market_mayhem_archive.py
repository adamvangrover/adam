import json
import os
import re
from datetime import datetime

# Paths
STRATEGIC_PATH = 'showcase/data/strategic_command.json'
MARKET_DATA_PATH = 'showcase/data/sp500_market_data.json'
INDEX_PATH = 'showcase/data/market_mayhem_index.json'
TARGET_HTML_PATH = 'showcase/market_mayhem_archive.html'

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_sidebar(strategic, market, index_data):
    # 1. Strategic Command
    house_view = strategic['strategic_directives']['house_view']
    narrative = strategic['strategic_directives']['narrative']

    view_color = '#f59e0b'
    if house_view == 'BULLISH': view_color = '#0aff60'
    if house_view == 'BEARISH': view_color = '#ff3333'

    strategic_html = f"""
            <!-- Strategic Command -->
            <div class="sidebar-panel">
                <div class="sidebar-title">
                    <span>STRATEGIC COMMAND</span>
                    <i class="fas fa-chess-king"></i>
                </div>
                <div style="margin-bottom: 10px;">
                    <span style="color: #666; font-size: 0.7rem;">HOUSE VIEW</span>
                    <div style="font-size: 1.1rem; color: {view_color}; font-weight: bold;">{house_view}</div>
                </div>
                <div style="font-size: 0.75rem; line-height: 1.4; color: #ccc;">
                    {narrative}
                </div>
            </div>"""

    # 2. Forward Outlook
    # Use latest report from index to find metrics if possible, or use fallback mock data if regex fails
    sorted_index = sorted(index_data, key=lambda x: x.get('date', '1900-01-01'), reverse=True)
    latest_report = sorted_index[0] if sorted_index else {}
    body = latest_report.get('full_body', '')

    sp500_match = re.search(r'S&P 500:.*?([\d,]+)', body)
    vix_match = re.search(r'VIX:.*?([\d\.]+)', body)
    yield_match = re.search(r'10Y Treasury:.*?([\d\.]+)%', body)

    sp500_val = sp500_match.group(1) if sp500_match else "6,200"
    vix_val = vix_match.group(1) if vix_match else "14.2"
    yield_val = (yield_match.group(1) + "%") if yield_match else "4.45%"

    outlook_html = f"""
            <!-- Forward Outlook -->
            <div class="sidebar-panel">
                <div class="sidebar-title">
                    <span>FORWARD OUTLOOK</span>
                    <i class="fas fa-binoculars"></i>
                </div>
                <div class="metric-row">
                    <span class="metric-label">S&P 500 Target</span>
                    <span class="metric-val" style="color: #f59e0b;">{sp500_val}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">10Y Yield</span>
                    <span class="metric-val val-red">{yield_val}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">VIX</span>
                    <span class="metric-val val-green">{vix_val}</span>
                </div>
            </div>"""

    # 3. Top Conviction
    high_conviction = [m for m in market if m.get('outlook', {}).get('conviction') == 'High']
    high_conviction.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
    top_5_conviction = high_conviction[:5]

    conviction_rows = ""
    for item in top_5_conviction:
        ticker = item.get('ticker')
        sector = item.get('sector', 'Unknown')[:8]
        val = item.get('risk_score', 0)
        conviction_rows += f"""
                <div class="metric-row">
                    <span class="metric-label">{ticker} <span style="font-size:0.6rem; color:#555;">{sector}</span></span>
                    <span class="metric-val val-green">{val}</span>
                </div>"""

    conviction_html = f"""
            <!-- Top Conviction -->
            <div class="sidebar-panel">
                <div class="sidebar-title">
                    <span>TOP CONVICTION</span>
                    <i class="fas fa-star"></i>
                </div>
                {conviction_rows}
            </div>"""

    # 4. Watch List
    medium_conviction = [m for m in market if m.get('outlook', {}).get('conviction') == 'Medium']
    medium_conviction.sort(key=lambda x: x.get('change_pct', 0), reverse=True)
    watch_list = medium_conviction[:5]

    watch_rows = ""
    for item in watch_list:
        ticker = item.get('ticker')
        pct = item.get('change_pct', 0)
        color_class = "val-green" if pct >= 0 else "val-red"
        watch_rows += f"""
                <div class="metric-row">
                    <span class="metric-label">{ticker}</span>
                    <span class="metric-val {color_class}">{pct:+.2f}%</span>
                </div>"""

    watch_html = f"""
            <!-- Watch List -->
            <div class="sidebar-panel">
                <div class="sidebar-title">
                    <span>WATCH LIST</span>
                    <i class="fas fa-eye"></i>
                </div>
                {watch_rows}
            </div>"""

    # 5. Sector Risk
    sector_risks = {}
    for item in market:
        sec = item.get('sector', 'Other')
        score = item.get('risk_score', 50)
        if sec not in sector_risks:
            sector_risks[sec] = []
        sector_risks[sec].append(score)

    avg_sector_risks = []
    for sec, scores in sector_risks.items():
        avg = sum(scores) / len(scores)
        avg_sector_risks.append((sec, int(avg)))

    avg_sector_risks.sort(key=lambda x: x[1], reverse=True)

    risk_cells = ""
    for sec, score in avg_sector_risks[:6]:
        risk_class = "risk-low"
        if score > 50: risk_class = "risk-med"
        if score > 75: risk_class = "risk-high"

        short_sec = sec.split()[0][:6]
        risk_cells += f"""
                    <div class="risk-cell {risk_class}">
                        <div>{short_sec}</div>
                        <div style="font-size: 1rem; font-weight: bold;">{score}</div>
                    </div>"""

    risk_html = f"""
            <!-- Risk Matrix -->
            <div class="sidebar-panel">
                <div class="sidebar-title">
                    <span>SECTOR RISK</span>
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="risk-grid">
                    {risk_cells}
                </div>
            </div>"""

    return f"""
        <aside class="sidebar mono">
            {strategic_html}
            {outlook_html}
            {conviction_html}
            {watch_html}
            {risk_html}
        </aside>"""

def generate_chart_json(index_data):
    sorted_data = sorted(index_data, key=lambda x: x.get('date', '1900-01-01'))

    labels = [item['date'] for item in sorted_data if 'sentiment_score' in item]
    sentiment_data = [item['sentiment_score'] for item in sorted_data if 'sentiment_score' in item]

    return json.dumps({
        "labels": labels,
        "datasets": [{
            "label": "Sentiment Score",
            "data": sentiment_data,
            "borderColor": "#00f3ff",
            "backgroundColor": "rgba(0, 243, 255, 0.1)",
            "tension": 0.4,
            "fill": True
        }]
    }, indent=4)

def generate_archive_grid(index_data):
    sorted_data = sorted(index_data, key=lambda x: x.get('date', '1900-01-01'), reverse=True)

    html = '<div id="archiveGrid" style="padding-bottom: 50px;">\n'

    for item in sorted_data:
        date = item.get('date', 'Unknown')
        year = date[:4] if len(date) >= 4 else 'HISTORICAL'
        type_ = item.get('type', 'NEWSLETTER')
        title = item.get('title', 'Untitled')
        summary = item.get('summary', 'No summary available.')
        filename = item.get('filename', '#')
        sent = item.get('sentiment_score', 0)
        conv = item.get('conviction', 0)
        sem = item.get('semantic_score', 0)

        html += f"""
            <div class="archive-item type-{type_}" data-year="{year}" data-type="{type_}" data-title="{title.lower()} {summary.lower()}">
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
            </div>\n"""

    html += '            </div>'
    return html

def update_html():
    strategic = load_json(STRATEGIC_PATH)
    market = load_json(MARKET_DATA_PATH)
    index_data = load_json(INDEX_PATH)

    new_sidebar = generate_sidebar(strategic, market, index_data)
    new_chart_json = generate_chart_json(index_data)
    new_archive_grid = generate_archive_grid(index_data)

    with open(TARGET_HTML_PATH, 'r') as f:
        content = f.read()

    # Replace Sidebar
    content = re.sub(r'<aside class="sidebar mono">.*?</aside>', new_sidebar, content, flags=re.DOTALL)

    # Replace Chart Data
    start_marker = "const chartData = {"
    end_marker = "// Add Simulated Projections"

    s_idx = content.find(start_marker)
    if s_idx != -1:
        e_idx = content.find(end_marker, s_idx)
        if e_idx != -1:
            js_replacement = f"const chartData = {new_chart_json};\n\n        "
            content = content[:s_idx] + js_replacement + content[e_idx:]
        else:
            print("Warning: Could not find end marker for chartData.")
    else:
        print("Warning: Could not find start marker for chartData.")

    # Replace Archive Grid
    grid_start_marker = '<div id="archiveGrid" style="padding-bottom: 50px;">'
    gs_idx = content.find(grid_start_marker)
    if gs_idx != -1:
        main_end_idx = content.find('</main>', gs_idx)
        if main_end_idx != -1:
            content = content[:gs_idx] + new_archive_grid + "\n\n        " + content[main_end_idx:]

    with open(TARGET_HTML_PATH, 'w') as f:
        f.write(content)

    print(f"Successfully updated {TARGET_HTML_PATH}")

if __name__ == "__main__":
    update_html()
