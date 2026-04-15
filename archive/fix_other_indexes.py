import glob

files = glob.glob('showcase/*.html')

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()

    # If it has a navigation link to the archive, update it to the dashboard
    if 'href="market_mayhem_archive.html"' in content and 'MARKET MAYHEM' in content:
        content = content.replace('href="market_mayhem_archive.html"', 'href="market_mayhem_dashboard.html"')
        with open(filepath, 'w') as f:
            f.write(content)
