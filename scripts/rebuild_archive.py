import json
import os

INDEX_FILE = "showcase/data/market_mayhem_index.json"
ARCHIVE_FILE = "showcase/market_mayhem_archive.html"

def load_index():
    try:
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {INDEX_FILE} not found.")
        return []

def generate_archive_html(index_data):
    html_items = ""
    # Sort by date descending
    sorted_data = sorted(index_data, key=lambda x: x.get('date', '0000-00-00'), reverse=True)

    for item in sorted_data:
        # Determine Year
        date_str = item.get('date', 'Unknown')
        year = date_str.split('-')[0] if '-' in date_str else "HISTORICAL"
        if year.isdigit() and int(year) < 2024:
            year_filter = "HISTORICAL"
        elif not year.isdigit():
            year_filter = "HISTORICAL"
        else:
            year_filter = year

        # Determine Type
        item_type = item.get('type', 'NEWSLETTER').upper()

        # Determine Title and Summary
        title = item.get('title', 'Untitled').replace('"', '&quot;')
        summary = item.get('summary', 'No summary available.').replace('"', '&quot;')
        filename = item.get('filename', '#')

        # Stats (Mock or Real)
        sent = item.get('sentiment_score', 50)
        conv = item.get('conviction', 50)
        sem = item.get('semantic_score', 50) # Fallback

        html_items += f'''
            <div class="archive-item type-{item_type}" data-year="{year_filter}" data-type="{item_type}" data-title="{title.lower()} {summary.lower()}">
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">{date_str}</span>
                        <span class="type-badge">{item_type}</span>
                        <span class="mono" style="font-size:0.7rem; color:#444;">SENT: {sent} | CONV: {conv} | SEM: {sem}</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem;">{title}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#aaa;">{summary}</p>
                </div>
                <a href="{filename}" class="cyber-btn" style="align-self:center;">ACCESS</a>
            </div>'''

    return html_items

def update_archive_file(html_content):
    with open(ARCHIVE_FILE, "r") as f:
        original_html = f.read()

    start_marker = '<div id="archiveGrid" style="padding-bottom: 50px;">'
    end_marker = '</main>'

    idx_start = original_html.find(start_marker)
    if idx_start == -1:
        print(f"Error: Could not find '{start_marker}' in HTML.")
        return

    # Find </main> strictly after the start marker
    idx_end_main = original_html.find(end_marker, idx_start)

    if idx_end_main == -1:
        print(f"Error: Could not find '{end_marker}' after the start marker.")
        return

    # Construct the new content
    # Everything before the grid div
    pre_content = original_html[:idx_start]

    # The grid div itself + content + closing div
    # We replace everything from start_marker up to </main>
    # Note: This assumes the grid div is the last direct child of <main> before </main>
    # and that there isn't anything else important after the grid div inside <main>.
    # Based on file inspection, this is true.

    new_section = f"{start_marker}\n{html_content}\n            </div>\n\n        "

    # Everything after </main> (including </main> itself? No, we need to keep </main>)
    post_content = original_html[idx_end_main:]

    new_full_html = pre_content + new_section + post_content

    with open(ARCHIVE_FILE, "w") as f:
        f.write(new_full_html)
    print(f"Updated {ARCHIVE_FILE} with {len(html_content)} bytes of content.")

def main():
    data = load_index()
    if not data:
        return

    html = generate_archive_html(data)
    update_archive_file(html)

if __name__ == "__main__":
    main()
