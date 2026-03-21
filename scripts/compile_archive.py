import os
import json
import re
from datetime import datetime

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWSLETTER_DIR = os.path.join(BASE_DIR, 'core', 'libraries_and_archives', 'restored_newsletters')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'market_mayhem_archive.json')

def extract_date_from_filename(filename):
    """Attempt to extract a date from a filename like newsletter_market_mayhem_apr_04_2025.html"""
    try:
        # Match pattern like apr_04_2025 or oct_1929 or jan_2025
        match = re.search(r'_([a-z]{3})_?(\d{2})?_?(\d{4})\.', filename)
        if match:
            month_str, day_str, year_str = match.groups()
            
            # Month mapping
            months = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = months.get(month_str.lower(), 1)
            day = int(day_str) if day_str else 1
            year = int(year_str)
            
            # Format as YYYY-MM-DD
            return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception as e:
        print(f"Error parsing date from {filename}: {e}")
        pass
    
    return "Archive"

def extract_title(content, filename):
    """Extract a title from the HTML content or fallback to filename."""
    match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        # Clean up
        title = title.replace('ADAM v23.5 ::', '').strip()
        if title:
            return title
            
    match = re.search(r'<h[1-3][^>]*>(.*?)</h[1-3]>', content, re.IGNORECASE | re.DOTALL)
    if match:
        title = match.group(1).strip()
        # Clean up HTML tags if any
        title = re.sub(r'<[^>]+>', '', title)
        if title:
            return title
            
    # Fallback to readable filename
    clean_name = filename.replace('newsletter_', '').replace('_', ' ').replace('.html', '').replace('.md', '').title()
    return clean_name

def extract_snippet(content):
    """Extract a short snippet of text from the content."""
    # Remove script and style elements
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', content, flags=re.IGNORECASE | re.DOTALL)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Grab the first meaningful chunk
    words = text.split()
    snippet = ' '.join(words[:40])
    if len(words) > 40:
        snippet += '...'
    
    # Basic cleanup if empty
    if not snippet.strip():
        snippet = "Archived Market Mayhem Briefing."
        
    return snippet

def compile_archive():
    print(f"Compiling newsletters from {NEWSLETTER_DIR}...")
    
    if not os.path.exists(NEWSLETTER_DIR):
        print(f"Directory not found: {NEWSLETTER_DIR}")
        return
        
    archive_items = []
    
    # Add files from the directory
    for filename in os.listdir(NEWSLETTER_DIR):
        if filename.endswith('.html') or filename.endswith('.md'):
            filepath = os.path.join(NEWSLETTER_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                date_str = extract_date_from_filename(filename)
                title = extract_title(content, filename)
                snippet = extract_snippet(content)
                
                # Assign a mock conviction for the intelligence
                conviction = min(100, max(50, 75 + len(content) % 25))
                
                # Relative path for the UI
                relative_path = f"core/libraries_and_archives/restored_newsletters/{filename}"
                
                archive_items.append({
                    "id": filename.replace('.', '_'),
                    "title": title,
                    "date": date_str,
                    "snippet": snippet,
                    "path": relative_path,
                    "type": "report" if 'report' in filename.lower() else "newsletter",
                    "conviction": conviction,
                    "tags": ["Market Mayhem", "Archive"]
                })
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    # Sort by date descending
    archive_items.sort(key=lambda x: x['date'], reverse=True)
    
    print(f"Processed {len(archive_items)} items.")
    
    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(archive_items, f, indent=2)
        
    print(f"Archive successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    compile_archive()
