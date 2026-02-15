import os
import glob
from bs4 import BeautifulSoup

SHOWCASE_DIR = 'showcase'

def get_target_files():
    patterns = [
        os.path.join(SHOWCASE_DIR, '*market_mayhem*.html'),
        os.path.join(SHOWCASE_DIR, '*Market_Mayhem*.html'),
        os.path.join(SHOWCASE_DIR, 'MM*.html') # Catch MM04042025.html etc
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p))

    # Filter out archive files as they might be index pages, not reports
    # But the prompt says "Market Mayhem HTML files". The archive is also a dashboard.
    # The prompt mentions "Refactor and enhance the 'Market Mayhem' HTML files... transform them from static reports".
    # Archives are already dashboards. I should probably include them or at least the newsletters.
    # Let's include everything matching.
    return list(files)

def create_hud_header(soup):
    header = soup.new_tag('header', attrs={'class': 'cyber-hud-header'})

    # Left: Conviction
    left = soup.new_tag('div', attrs={'class': 'hud-section left'})
    gauge_container = soup.new_tag('div', attrs={'class': 'conviction-gauge-container'})

    span_label = soup.new_tag('span')
    span_label.string = "CONVICTION:"

    bar = soup.new_tag('div', attrs={'class': 'conviction-bar'})
    fill = soup.new_tag('div', attrs={'class': 'conviction-fill'})
    bar.append(fill)

    span_val = soup.new_tag('span', attrs={'class': 'conviction-value'})
    span_val.string = "--%"

    gauge_container.append(span_label)
    gauge_container.append(bar)
    gauge_container.append(span_val)
    left.append(gauge_container)

    # Center: Tags
    center = soup.new_tag('div', attrs={'class': 'hud-section center'})
    tags = soup.new_tag('div', attrs={'class': 'metadata-tags'})
    center.append(tags)

    # Right: Search
    right = soup.new_tag('div', attrs={'class': 'hud-section right'})
    inp = soup.new_tag('input', attrs={
        'type': 'text',
        'id': 'hud-search-input',
        'class': 'hud-search',
        'placeholder': 'SEARCH INTEL...'
    })
    right.append(inp)

    header.append(left)
    header.append(center)
    header.append(right)
    return header

def create_terminal_footer(soup):
    footer = soup.new_tag('footer', attrs={'class': 'system2-terminal collapsed'})

    header = soup.new_tag('div', attrs={'class': 'terminal-header'})
    span = soup.new_tag('span')
    span.string = "SYSTEM 2 DIAGNOSTICS // AGENT TRACE"

    icon = soup.new_tag('i', attrs={'class': 'fas fa-chevron-up toggle-icon'})

    header.append(span)
    header.append(icon)

    body = soup.new_tag('div', attrs={'class': 'terminal-body'})

    footer.append(header)
    footer.append(body)
    return footer

def process_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Check if already processed
    if soup.find('div', class_='dashboard-container'):
        print(f"Skipping {filepath} - already processed.")
        return

    # Add CSS link
    if soup.head:
        link = soup.new_tag('link', rel='stylesheet', href='css/cyberpunk-core.css')
        soup.head.append(link)

    # Add JS script
    if soup.body:
        script = soup.new_tag('script', src='js/dashboard-logic.js', defer=None)
        soup.body.append(script)

    # Create Wrapper
    container = soup.new_tag('div', attrs={'class': 'dashboard-container'})

    # Header
    container.append(create_hud_header(soup))

    # Main Content
    main_content = soup.new_tag('main', attrs={'class': 'cyber-main-content'})

    # Move existing body children to main_content
    # We need to be careful not to move the scripts we just added or nav.js if possible,
    # but simplest is to move everything that was there.
    # Actually, let's move everything currently in body into main_content,
    # except scripts that look like system scripts?
    # Better: Move everything. The layout handles it.
    # Exception: The new script 'js/dashboard-logic.js' is currently at the end of body.
    # If we iterate valid children, we can append them to main_content.

    body_children = list(soup.body.contents)
    for child in body_children:
        # Don't move the dashboard-logic script we just appended?
        # Actually we appended it to soup.body, so it IS in body_children.
        # Let's move it too, it's fine inside main or outside.
        # But wait, dashboard-container uses flex column.
        # main-content is flex-1.
        # Scripts are invisible so it doesn't matter much layout-wise,
        # but semantically better outside main if possible.

        # We will append valid elements to main_content.
        # And clear body.
        # Then append container to body.
        child.extract()
        main_content.append(child)

    container.append(main_content)

    # Footer
    container.append(create_terminal_footer(soup))

    soup.body.append(container)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup))

if __name__ == "__main__":
    files = get_target_files()
    print(f"Found {len(files)} files to process.")
    for f in files:
        process_file(f)
