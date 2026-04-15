import os
import glob
import re

files_to_update = [
    'showcase/site_map.json',
    'showcase/js/nav.js',
    'showcase/js/mock_data.js',
    'showcase/js/nav_swarm.js',
    'showcase/js/nexus_apps_extras.js',
    'showcase/js/report_navigation.js',
    'showcase/js/command_palette.js',
    'showcase/js/office_system.js',
    'index2.html',
    'index3.html',
    'showcase/index2.html'
]

for filepath in files_to_update:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()

        if 'market_mayhem_archive.html' in content:
            content = content.replace('market_mayhem_archive.html', 'market_mayhem_dashboard.html')
            with open(filepath, 'w') as f:
                f.write(content)
