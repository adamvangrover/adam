import os
import re

base_dir = "showcase/data/adam_daily"
output_file = "showcase/adam_daily_hub_index.html"

# Read the original file to keep the nice styling
if os.path.exists("showcase/adam_daily_hub.html"):
    with open("showcase/adam_daily_hub.html", "r") as f:
        content = f.read()
else:
    print("Base file missing.")
    exit(1)

# Try to find all date directories
dates = []
for item in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, item)) and re.match(r"^\d{4}-\d{2}-\d{2}$", item):
        dates.append(item)

dates.sort(reverse=True) # newest first

# Generate HTML blocks
html_blocks = []
for date in dates:
    date_dir = os.path.join(base_dir, date)

    # We only care about HTML and MD files
    files = []
    for f in os.listdir(date_dir):
        if f.endswith(".html") or f.endswith(".md"):
            files.append(f)

    files.sort(reverse=True) # sort files alphabetically

    if not files:
        continue

    block = f"""
            <div class="relative mb-12 group">
                <div class="timeline-dot top-2 group-hover:bg-term-cyan transition-colors"></div>

                <h3 class="text-2xl text-term-cyan font-display mb-4 border-b border-white/10 pb-2 inline-block pr-8">{date}</h3>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
"""

    for f in files:
        ext = f.split(".")[-1].upper()
        color = "term-cyan" if ext == "HTML" else "term-amber"

        block += f"""
                    <a href="data/adam_daily/{date}/{f}" class="cyber-panel p-4 rounded hover:border-{color}/50 transition-all flex flex-col h-full group/card">
                        <div class="flex justify-between items-start mb-2">
                            <span class="text-[10px] font-mono text-{color} bg-{color}/10 px-2 py-0.5 rounded border border-{color}/20">{ext}</span>
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-slate-600 group-hover/card:text-{color}"><line x1="7" y1="17" x2="17" y2="7"></line><polyline points="7 7 17 7 17 17"></polyline></svg>
                        </div>
                        <h4 class="text-white text-sm truncate font-mono mb-1">{f}</h4>
                        <div class="text-[10px] text-slate-500 font-mono mt-auto pt-2 truncate border-t border-white/5">PATH: data/adam_daily/{date}/{f}</div>
                    </a>
"""

    block += """
                </div>
            </div>
"""
    html_blocks.append(block)

# Try to replace the content inside the timeline container
import re
# Find the start of the timeline container
match = re.search(r'(<div[^>]*class="[^"]*timeline[^"]*"[^>]*>)', content)

if not match:
    # Alternative: replace the masonry container
    match = re.search(r'(<div[^>]*class="[^"]*masonry-grid[^"]*"[^>]*>)', content)

if match:
    print("Found container, replacing content...")
    start_idx = match.end()

    # Replace masonry with a simple block layout container for timeline
    prefix = content[:start_idx].replace('class="masonry-grid w-full"', 'class="relative pl-6 border-l border-white/10"')

    # Find the end of the main tag
    end_match = re.search(r'</main>', content)
    if end_match:
        suffix = "\n        </div>\n    " + content[end_match.start():]

        # update title
        prefix = prefix.replace(">Adam Daily</span> Hub</h1>", ">Adam Daily</span> Archive Index</h1>")
        prefix = prefix.replace("<p class=\"text-slate-400 font-mono text-sm max-w-2xl\">Select a date or event to review intelligence modules, scenario analyses, and structural frameworks.</p>", "<p class=\"text-slate-400 font-mono text-sm max-w-2xl\">Complete chronological index of all Market Mayhem, briefs, and daily intelligence artifacts.</p>")

        # Remove the filter buttons
        prefix = re.sub(r'<div class="flex flex-wrap gap-2 mb-8" id="filter-buttons">.*?</div>', '', prefix, flags=re.DOTALL)

        new_content = prefix + "".join(html_blocks) + suffix

        with open(output_file, "w") as f:
            f.write(new_content)
        print("Done.")
    else:
        print("Could not find </main>")
else:
    print("Could not find container")
