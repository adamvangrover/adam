import re

with open('scripts/generate_market_mayhem_archive.py', 'r') as f:
    content = f.read()

# Replace <a href="dashboard.html">DASHBOARD</a> with <a href="market_mayhem_dashboard.html">DASHBOARD</a>
if '<a href="dashboard.html">DASHBOARD</a>' in content:
    content = content.replace('<a href="dashboard.html">DASHBOARD</a>', '<a href="market_mayhem_dashboard.html">DASHBOARD</a>')

with open('scripts/generate_market_mayhem_archive.py', 'w') as f:
    f.write(content)
