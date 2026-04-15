import re

with open('index_archive.html', 'r') as f:
    content = f.read()

# Update link in index.html to point to dashboard
content = content.replace('showcase/market_mayhem_dashboard.html', 'market_mayhem_dashboard.html')

with open('index_archive.html', 'w') as f:
    f.write(content)
