with open('index.html', 'r') as f:
    content = f.read()

replacement = """
            <div class="portal-card" style="border-color: #ff3333;">
                <div>
                    <div class="portal-title" style="color: #ff3333;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">📉</span> Market Mayhem
                    </div>
                    <p class="portal-desc">
                        Live Strategic Market Dashboard and Archives.
                    </p>
                </div>
                <a href="showcase/market_mayhem_dashboard.html" class="cyber-btn" style="text-align: center; border-color: #ff3333; color: #ff3333;">OPEN DASHBOARD &rarr;</a>
            </div>
"""

content = content.replace('<!-- Main Portals -->\n        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-bottom: 60px;">', '<!-- Main Portals -->\n        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 30px; margin-bottom: 60px;">\n' + replacement)

with open('index.html', 'w') as f:
    f.write(content)
