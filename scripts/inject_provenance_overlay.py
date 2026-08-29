import sys

def inject_overlay(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if 'system2-overlay' in content:
        print(f"Skipping {filepath}, already contains overlay.")
        return

    with open('scripts/overlay_template.html', 'r', encoding='utf-8') as f:
        overlay_html = f.read()

    if '</body>' in content:
        content = content.replace('</body>', overlay_html + '\n</body>')
    else:
        content += '\n' + overlay_html

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Successfully injected overlay into {filepath}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for fp in sys.argv[1:]:
            inject_overlay(fp)
    else:
        print("Usage: uv run python scripts/inject_provenance_overlay.py <file1.html> ...")
