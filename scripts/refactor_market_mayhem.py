import os
import glob

def refactor_files():
    base_dir = "showcase"
    patterns = [
        "newsletter_market_mayhem_*.html",
        "market_mayhem_*.html"
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(base_dir, p)))

    print(f"Found {len(files)} potential files.")

    css_link = '<link rel="stylesheet" href="css/cyberpunk-core.css">\n'
    js_script = '<script src="js/dashboard-logic.js" defer></script>\n'
    init_script = '<script>document.addEventListener("DOMContentLoaded", () => window.CyberDashboard.init());</script>\n'

    processed_count = 0

    for filepath in files:
        # Skip archive and index files as they are already dashboards
        if "archive" in filepath or "index" in filepath or "rebuild" in filepath:
            print(f"Skipping dashboard/archive file: {filepath}")
            continue

        print(f"Processing {filepath}...")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            modified = False

            # Inject CSS
            if "cyberpunk-core.css" not in content:
                if "</head>" in content:
                    content = content.replace("</head>", css_link + "</head>")
                    modified = True

            # Inject JS
            if "dashboard-logic.js" not in content:
                if "</body>" in content:
                    content = content.replace("</body>", js_script + init_script + "</body>")
                    modified = True

            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"Refactoring complete. Modified {processed_count} files.")

if __name__ == "__main__":
    refactor_files()
