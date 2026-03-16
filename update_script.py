import re

with open('showcase/index.html', 'r') as f:
    content = f.read()

# Update link in index.html to point to dashboard
content = content.replace('market_mayhem_archive.html', 'market_mayhem_dashboard.html')

# Add "DASHBOARD" to text
content = content.replace('ACCESS ARCHIVE', 'OPEN DASHBOARD')

with open('showcase/index.html', 'w') as f:
    f.write(content)

with open('showcase/index2.html', 'r') as f:
    content = f.read()

# Make sure Market Mayhem is in index2
if 'market_mayhem_dashboard.html' not in content:
    replacement = """
                    <a href="market_mayhem_dashboard.html" class="p-4 bg-slate-800/50 hover:bg-slate-700 border border-slate-700 hover:border-red-500 rounded transition group text-center">
                        <i class="fas fa-chart-line text-2xl text-red-500 mb-2 group-hover:scale-110 transition-transform"></i>
                        <div class="text-xs font-bold text-slate-300 group-hover:text-white">Market Mayhem</div>
                    </a>
                    """
    content = content.replace('<!-- Module Quick Access -->', '<!-- Module Quick Access -->\n' + replacement)

with open('showcase/index2.html', 'w') as f:
    f.write(content)
