with open('index_archive.html', 'r') as f:
    content = f.read()

content = content.replace('ACCESS ARCHIVE', 'OPEN DASHBOARD')

with open('index_archive.html', 'w') as f:
    f.write(content)

with open('index.html', 'r') as f:
    content = f.read()

content = content.replace('market_mayhem_archive.html', 'market_mayhem_dashboard.html')
content = content.replace('ACCESS ARCHIVE &rarr;', 'OPEN DASHBOARD &rarr;')

with open('index.html', 'w') as f:
    f.write(content)
