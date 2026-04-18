import os
import re
import json

ROOT_DIR = "showcase/data/adam_daily"
OUTPUT_FILE = "showcase/adam_daily_hub.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM DAILY // INTELLIGENCE HUB</title>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'Segoe UI', 'sans-serif'],
                        mono: ['Fira Code', 'ui-monospace', 'monospace'],
                        display: ['Oswald', 'Inter', 'sans-serif'],
                    },
                    colors: {
                        term: {
                            bg: '#030712', surface: '#0f172a',
                            cyan: '#06b6d4', red: '#ef4444', amber: '#f59e0b', green: '#10b981'
                        }
                    }
                }
            }
        }
    </script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;600&family=Inter:wght@300;400;600;800&family=Oswald:wght@500;700&display=swap');

        body { font-size: 14px; background: #030712; color: #cbd5e1; }

        .glass-card {
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.75rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            border-color: rgba(6, 182, 212, 0.3);
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5), 0 0 15px -5px rgba(6,182,212,0.2);
        }

        .masonry-grid {
            column-count: 1; column-gap: 1.5rem;
        }
        @media (min-width: 768px) { .masonry-grid { column-count: 2; } }
        @media (min-width: 1024px) { .masonry-grid { column-count: 3; } }

        .masonry-item { break-inside: avoid; margin-bottom: 1.5rem; }

        .scanline {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.1));
            background-size: 100% 4px; z-index: 9999; pointer-events: none; opacity: 0.15;
        }

        .gated-content-blur {
            filter: blur(8px);
            user-select: none;
            pointer-events: none;
            transition: filter 0.5s ease-out;
        }

        .unlocked .gated-content-blur {
            filter: blur(0px);
            user-select: auto;
            pointer-events: auto;
        }

        .unlocked #authModal { display: none; }

        .glow-text { text-shadow: 0 0 10px rgba(6, 182, 212, 0.5); }
    </style>
</head>
<body class="font-sans antialiased overflow-x-hidden selection:bg-term-cyan selection:text-black">
    <div class="scanline"></div>

    <header class="sticky top-0 z-50 bg-term-bg/90 backdrop-blur-md border-b border-white/10 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded bg-gradient-to-br from-term-cyan to-blue-600 flex items-center justify-center font-display font-bold text-white shadow-[0_0_15px_rgba(6,182,212,0.4)]">A</div>
            <div>
                <h1 class="font-display text-xl font-bold text-white tracking-widest uppercase glow-text">Adam Daily Hub</h1>
                <p class="text-[10px] font-mono text-term-cyan tracking-widest uppercase">Intelligence Aggregation Engine</p>
            </div>
        </div>
        <div class="flex items-center gap-4">
            <div class="hidden sm:flex items-center gap-2 bg-term-cyan/10 px-3 py-1 rounded-full border border-term-cyan/20">
                <div class="w-2 h-2 rounded-full bg-term-cyan animate-pulse"></div>
                <span class="text-[10px] font-mono text-term-cyan">LIVE SYNC</span>
            </div>
            <a href="adam_nexus_portal.html" class="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded text-xs font-mono transition-colors">RETURN TO NEXUS</a>
        </div>
    </header>

    <!-- Authentication Modal -->
    <div id="authModal" class="fixed inset-0 z-40 bg-term-bg/80 backdrop-blur-sm flex items-center justify-center">
        <div class="glass-card p-8 w-full max-w-md border-term-cyan/30 shadow-[0_0_30px_rgba(6,182,212,0.15)] relative overflow-hidden">
            <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-term-cyan to-transparent"></div>
            <h2 class="font-display text-2xl font-bold text-white mb-2 text-center tracking-widest uppercase">Insider Access Required</h2>
            <p class="text-sm text-slate-400 text-center mb-6">Enter clearance key to decrypt payload.</p>

            <form id="authForm" class="space-y-4">
                <input type="password" id="authKey" placeholder="[ ENTER DECRYPTION KEY ]" class="w-full bg-black/50 border border-white/20 rounded p-3 text-center font-mono text-term-cyan focus:outline-none focus:border-term-cyan transition-colors" autocomplete="off">
                <p id="authError" class="text-xs font-mono text-term-red text-center hidden">ACCESS DENIED. INVALID KEY.</p>
                <button type="submit" class="w-full bg-term-cyan hover:bg-white text-term-bg font-bold py-3 rounded transition-colors shadow-[0_0_15px_rgba(6,182,212,0.3)]">DECRYPT PAYLOAD</button>
            </form>
        </div>
    </div>

    <main class="p-6 md:p-8 max-w-[1600px] mx-auto gated-content-blur">
        <div class="mb-8 flex flex-col md:flex-row md:justify-between md:items-end gap-4">
            <div>
                <h2 class="text-2xl font-display font-bold text-white mb-2">ARCHIVED TRANSMISSIONS</h2>
                <p class="text-sm text-slate-400">Aggregated sub-modules parsed directly from the Adam daily intelligence feed.</p>
            </div>

            <div class="flex items-center gap-2 bg-black/30 p-1 rounded-lg border border-white/5">
                <button class="filter-btn active px-4 py-1.5 rounded-md text-xs font-mono transition-colors bg-white/10 text-white" data-filter="all">ALL</button>
                <button class="filter-btn px-4 py-1.5 rounded-md text-xs font-mono text-slate-400 hover:text-white hover:bg-white/5 transition-colors" data-filter="SYSTEM STATUS">SYSTEM STATUS</button>
                <button class="filter-btn px-4 py-1.5 rounded-md text-xs font-mono text-slate-400 hover:text-white hover:bg-white/5 transition-colors" data-filter="MARKET MAYHEM">MARKET MAYHEM</button>
                <button class="filter-btn px-4 py-1.5 rounded-md text-xs font-mono text-slate-400 hover:text-white hover:bg-white/5 transition-colors" data-filter="WHALESCANNER">WHALESCANNER</button>
            </div>
        </div>

        <div class="masonry-grid" id="moduleGrid">
            <!-- CARDS INJECTED HERE -->
            {CARDS_HTML}
        </div>
    </main>

    <script>
        // Authentication Logic
        const validKeys = ['admin', 'adam', 'genesis'];

        document.getElementById('authForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const key = document.getElementById('authKey').value.toLowerCase().trim();
            if (validKeys.includes(key)) {
                document.body.classList.add('unlocked');
                localStorage.setItem('adam_hub_auth', 'true');
            } else {
                document.getElementById('authError').classList.remove('hidden');
                document.getElementById('authKey').value = '';
                setTimeout(() => document.getElementById('authError').classList.add('hidden'), 3000);
            }
        });

        // Auto-unlock if previously authenticated
        if (localStorage.getItem('adam_hub_auth') === 'true') {
            document.body.classList.add('unlocked');
        }

        // Filtering Logic
        const filterBtns = document.querySelectorAll('.filter-btn');
        const cards = document.querySelectorAll('.module-card');

        filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Update active state
                filterBtns.forEach(b => {
                    b.classList.remove('bg-white/10', 'text-white');
                    b.classList.add('text-slate-400');
                });
                btn.classList.add('bg-white/10', 'text-white');
                btn.classList.remove('text-slate-400');

                const filter = btn.dataset.filter;

                cards.forEach(card => {
                    if (filter === 'all' || card.dataset.group === filter) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            });
        });

        // Initialize Markdown for all content blocks
        document.querySelectorAll('.markdown-content').forEach(el => {
            // Un-escape backticks if necessary, or just rely on the raw HTML
            // Note: The Python script renders raw HTML directly from the templates,
            // so we don't strictly need Marked.js here unless we want to re-parse.
            // We'll leave it as raw HTML for fidelity.
        });
    </script>
</body>
</html>
"""

def extract_modules():
    # Dashboard Schemas: The extracted JavaScript array structures (const modules = [...])
    # are required to correctly render masonry card grids and gated content modals in the UI.
    all_modules = []

    # Regex to find the const modules = [...] array
    module_regex = re.compile(r"const modules = \[\s*(.*?)\s*\];\n", re.DOTALL)

    print("Parsing daily files...")
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)

                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
                date_str = date_match.group(1) if date_match else "Unknown Date"

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # Find the module array text
                        match = module_regex.search(content)
                        if match:
                            module_text = match.group(1)

                            # Extremely hacky parsing of JS object literals to extract fields
                            # A real JS parser would be better, but regex works for this specific templated format

                            # Split by '}, {' to get rough module blocks
                            blocks = re.split(r'\},\s*\{', module_text)

                            for block in blocks:
                                # Clean up edges
                                block = block.strip().lstrip('{').rstrip('}')

                                # Extract fields
                                group_match = re.search(r'group:\s*["\'](.*?)["\']', block)
                                title_match = re.search(r'title:\s*["\'](.*?)["\']', block)
                                sub_match = re.search(r'subtitle:\s*["\'](.*?)["\']', block)

                                # Extract content (between backticks)
                                content_match = re.search(r'content:\s*`(.*?)`', block, re.DOTALL)

                                if group_match and title_match and content_match:
                                    # Strip script tags/canvas from content to make it displayable in masonry
                                    raw_html = content_match.group(1)
                                    clean_html = re.sub(r'<div class="w-full h-48 chart-container.*?</div>', '', raw_html, flags=re.DOTALL)
                                    clean_html = re.sub(r'<canvas.*?</canvas>', '', clean_html, flags=re.DOTALL)

                                    all_modules.append({
                                        'date': date_str,
                                        'source_file': os.path.relpath(file_path, 'showcase/'),
                                        'group': group_match.group(1),
                                        'title': title_match.group(1),
                                        'subtitle': sub_match.group(1) if sub_match else "",
                                        'html': clean_html
                                    })
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    # Sort by date descending
    all_modules.sort(key=lambda x: x['date'], reverse=True)
    return all_modules

def generate_cards_html(modules):
    cards_html = ""

    for mod in modules:
        group_color = "term-cyan"
        if "SYSTEM" in mod['group'].upper(): group_color = "term-green"
        elif "WHALESCANNER" in mod['group'].upper(): group_color = "term-purple"
        elif "MAYHEM" in mod['group'].upper(): group_color = "term-amber"

        card = f"""
        <div class="masonry-item module-card" data-group="{mod['group']}">
            <div class="glass-card p-5 overflow-hidden relative group">
                <div class="absolute top-0 left-0 w-full h-1 bg-{group_color} opacity-50"></div>

                <div class="flex justify-between items-start mb-3">
                    <div>
                        <span class="text-[10px] font-mono text-{group_color} border border-{group_color}/30 bg-{group_color}/10 px-2 py-0.5 rounded uppercase tracking-widest">{mod['group']}</span>
                    </div>
                    <span class="text-xs font-mono text-slate-500">{mod['date']}</span>
                </div>

                <h3 class="font-display font-bold text-white text-lg mb-1 leading-tight">{mod['title']}</h3>
                <p class="text-xs font-mono text-slate-400 mb-4 border-b border-white/5 pb-3">{mod['subtitle']}</p>

                <div class="prose prose-sm prose-invert prose-p:text-slate-300 prose-headings:text-white max-w-none text-sm markdown-content">
                    {mod['html']}
                </div>

                <div class="mt-4 pt-3 border-t border-white/5 text-right">
                    <a href="{mod['source_file']}" class="text-[10px] font-mono text-slate-400 hover:text-{group_color} transition-colors uppercase tracking-widest">Open Source File &rarr;</a>
                </div>
            </div>
        </div>
        """
        cards_html += card

    return cards_html

def main():
    modules = extract_modules()
    print(f"Extracted {len(modules)} modules.")

    cards_html = generate_cards_html(modules)

    final_html = HTML_TEMPLATE.replace("{CARDS_HTML}", cards_html)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Hub generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
