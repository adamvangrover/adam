import re

# In market_mayhem_dashboard.html, we need to make sure the ARCHIVE link points to the market_mayhem_archive.html
with open('showcase/market_mayhem_dashboard.html', 'r') as f:
    dashboard_content = f.read()

# Make the ARCHIVE link clickable in the top nav
if '<span class="active">DASHBOARD</span>' in dashboard_content and '<span>ARCHIVE</span>' in dashboard_content:
    dashboard_content = dashboard_content.replace('<span>ARCHIVE</span>', '<span onclick="window.location.href=\'market_mayhem_archive.html\'">ARCHIVE</span>')

if '<button class="btn-full-width mono" onclick="document.getElementById(\'archiveModal\').style.display=\'flex\'">VIEW FULL HISTORIC ARCHIVE</button>' in dashboard_content:
    replacement = '<button class="btn-full-width mono" onclick="window.location.href=\'market_mayhem_archive.html\'">VIEW FULL HISTORIC ARCHIVE</button>'
    dashboard_content = dashboard_content.replace(
        '<button class="btn-full-width mono" onclick="document.getElementById(\'archiveModal\').style.display=\'flex\'">VIEW FULL HISTORIC ARCHIVE</button>',
        replacement
    )


with open('showcase/market_mayhem_dashboard.html', 'w') as f:
    f.write(dashboard_content)

# In market_mayhem_archive.html, we need to make sure the DASHBOARD link points to the market_mayhem_dashboard.html
with open('showcase/market_mayhem_archive.html', 'r') as f:
    archive_content = f.read()

if 'href="dashboard.html"' in archive_content:
    archive_content = archive_content.replace('href="dashboard.html"', 'href="market_mayhem_dashboard.html"')
elif 'href="#" class="active">ARCHIVE</a>' in archive_content and '<a href="dashboard.html">DASHBOARD</a>' in archive_content:
    archive_content = archive_content.replace('<a href="dashboard.html">DASHBOARD</a>', '<a href="market_mayhem_dashboard.html">DASHBOARD</a>')


with open('showcase/market_mayhem_archive.html', 'w') as f:
    f.write(archive_content)
