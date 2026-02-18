// Command Palette Logic
class CommandPalette {
    constructor() {
        this.manifestPath = 'data/report_manifest.json';
        this.commands = [
            { title: "Go to Mission Control", url: "mission_control_v3.html", icon: "🚀", tag: "NAV" },
            { title: "Go to Crisis Simulator", url: "crisis_simulator.html", icon: "📉", tag: "NAV" },
            { title: "Go to Agent Registry", url: "agents.html", icon: "👥", tag: "NAV" },
            { title: "Go to Market Archive", url: "market_mayhem_archive.html", icon: "📚", tag: "NAV" },
            { title: "Go to Neural Dashboard", url: "neural_dashboard.html", icon: "🧠", tag: "NAV" },
            { title: "Toggle Dark/Light Theme", action: "toggleTheme", icon: "🌓", tag: "ACTION" },
            { title: "Toggle API Mode (Live/Mock)", action: "toggleApi", icon: "🔌", tag: "ACTION" }
        ];
        this.simOptions = [
            { title: "Run: 2008 Lehman Collapse", action: "runSim", param: "2008_LEHMAN", icon: "📉", tag: "SIM" },
            { title: "Run: 1987 Black Monday", action: "runSim", param: "1987_BLACK_MONDAY", icon: "📉", tag: "SIM" },
            { title: "Run: 2000 Dotcom Bubble", action: "runSim", param: "2000_DOTCOM_BUBBLE", icon: "💾", tag: "SIM" },
            { title: "Run: 2020 COVID Crash", action: "runSim", param: "2020_COVID", icon: "🦠", tag: "SIM" },
            { title: "Run: 2022 Inflation Shock", action: "runSim", param: "2022_INFLATION_SHOCK", icon: "💸", tag: "SIM" }
        ];
        this.searchData = [...this.commands];
        this.isOpen = false;
        this.selectedIndex = 0;
    }

    init() {
        this.injectStyles();
        this.createOverlay();
        this.bindEvents();
        this.fetchManifest();
    }

    injectStyles() {
        if (document.getElementById('cmd-palette-style')) return;
        const style = document.createElement('style');
        style.id = 'cmd-palette-style';
        style.innerHTML = `
            #cmd-palette-overlay {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.6); backdrop-filter: blur(8px);
                z-index: 10000; display: none; align-items: flex-start; justify-content: center;
                padding-top: 15vh; opacity: 0; transition: opacity 0.2s ease;
            }
            #cmd-palette-overlay.visible {
                opacity: 1;
            }
            #cmd-palette {
                width: 650px; max-width: 90%;
                background: #0f172a;
                border: 1px solid #334155;
                box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0, 243, 255, 0.1);
                border-radius: 12px; overflow: hidden;
                font-family: 'Inter', sans-serif;
                transform: scale(0.95); transition: transform 0.2s ease;
            }
            #cmd-palette-overlay.visible #cmd-palette {
                transform: scale(1);
            }
            #cmd-header {
                position: relative;
                border-bottom: 1px solid #1e293b;
            }
            #cmd-input {
                width: 100%; padding: 16px 20px 16px 50px;
                background: transparent; border: none;
                color: #e2e8f0; font-size: 1.1rem; outline: none;
                font-family: 'JetBrains Mono', monospace;
            }
            #cmd-search-icon {
                position: absolute; left: 20px; top: 50%; transform: translateY(-50%);
                color: #64748b; font-size: 1rem;
            }
            #cmd-results {
                max-height: 400px; overflow-y: auto;
                padding: 8px 0;
            }
            .cmd-item {
                padding: 10px 16px; margin: 0 8px; border-radius: 6px;
                cursor: pointer; display: flex; align-items: center; justify-content: space-between;
                color: #94a3b8; transition: all 0.1s;
            }
            .cmd-item:hover, .cmd-item.selected {
                background: #1e293b; color: #f8fafc;
            }
            .cmd-item.selected {
                background: #0ea5e9; color: #fff;
            }
            .cmd-item.selected .cmd-tag {
                background: rgba(255,255,255,0.2); color: #fff;
            }
            .cmd-icon { margin-right: 12px; font-size: 1.1rem; width: 24px; text-align: center; }
            .cmd-text { flex-grow: 1; font-size: 0.95rem; font-weight: 500; }
            .cmd-tag {
                font-size: 0.65rem; background: #1e293b; padding: 2px 6px;
                border-radius: 4px; color: #64748b; font-family: 'JetBrains Mono';
                text-transform: uppercase; letter-spacing: 0.05em;
            }
            #cmd-footer {
                padding: 8px 16px; background: #0b1120; border-top: 1px solid #1e293b;
                display: flex; justify-content: flex-end; gap: 15px;
                font-size: 0.7rem; color: #64748b;
            }
            .shortcut-key {
                background: #1e293b; padding: 1px 5px; border-radius: 3px; border: 1px solid #334155; margin-right: 4px;
            }
            /* Scrollbar */
            #cmd-results::-webkit-scrollbar { width: 6px; }
            #cmd-results::-webkit-scrollbar-track { background: transparent; }
            #cmd-results::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        `;
        document.head.appendChild(style);
    }

    createOverlay() {
        if (document.getElementById('cmd-palette-overlay')) return;
        const overlay = document.createElement('div');
        overlay.id = 'cmd-palette-overlay';
        overlay.innerHTML = `
            <div id="cmd-palette">
                <div id="cmd-header">
                    <i class="fas fa-search" id="cmd-search-icon"></i>
                    <input type="text" id="cmd-input" placeholder="Type a command or search..." autocomplete="off">
                </div>
                <div id="cmd-results"></div>
                <div id="cmd-footer">
                    <span><span class="shortcut-key">↵</span> to select</span>
                    <span><span class="shortcut-key">↑↓</span> to navigate</span>
                    <span><span class="shortcut-key">esc</span> to close</span>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
        this.overlay = overlay;
        this.input = document.getElementById('cmd-input');
        this.resultsContainer = document.getElementById('cmd-results');
    }

    async fetchManifest() {
        try {
            // Determine path relative to root based on current location
            // Simple heuristic: if in showcase, use data/..., if deeper, adjust
            const path = window.location.pathname.includes('/showcase/') ? 'data/report_manifest.json' : 'showcase/data/report_manifest.json';

            const response = await fetch(path);
            if (response.ok) {
                const data = await response.json();
                const reports = data.map(item => ({
                    title: item.title,
                    url: item.path,
                    icon: item.type === 'MARKET_MAYHEM' ? '🌪️' : '📄',
                    tag: item.type || 'REPORT',
                    date: item.date
                }));
                // Add to search data
                this.searchData = [...this.commands, ...reports];
            }
        } catch (e) {
            console.warn("CommandPalette: Failed to load manifest", e);
        }
    }

    bindEvents() {
        // Toggle
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                this.toggle();
            }
            if (this.isOpen) {
                if (e.key === 'Escape') this.close();
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    this.navigate(1);
                }
                if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    this.navigate(-1);
                }
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.execute();
                }
            }
        });

        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) this.close();
        });

        this.input.addEventListener('input', (e) => {
            this.renderResults(e.target.value);
            this.selectedIndex = 0;
        });
    }

    toggle() {
        if (this.isOpen) this.close();
        else this.open();
    }

    open() {
        this.isOpen = true;
        this.overlay.style.display = 'flex';
        // Trigger reflow for transition
        void this.overlay.offsetWidth;
        this.overlay.classList.add('visible');
        this.input.value = '';
        this.input.focus();
        this.renderResults('');
        this.selectedIndex = 0;
    }

    close() {
        this.isOpen = false;
        this.overlay.classList.remove('visible');
        setTimeout(() => {
            if (!this.isOpen) this.overlay.style.display = 'none';
        }, 200);
    }

    navigate(direction) {
        const items = this.resultsContainer.querySelectorAll('.cmd-item');
        if (items.length === 0) return;

        items[this.selectedIndex].classList.remove('selected');

        this.selectedIndex += direction;
        if (this.selectedIndex >= items.length) this.selectedIndex = 0;
        if (this.selectedIndex < 0) this.selectedIndex = items.length - 1;

        const selected = items[this.selectedIndex];
        selected.classList.add('selected');
        selected.scrollIntoView({ block: 'nearest' });
    }

    execute() {
        const items = this.resultsContainer.querySelectorAll('.cmd-item');
        if (items.length === 0) return;

        // Ensure index is within bounds (can drift if filter changes rapidly)
        if (this.selectedIndex >= items.length) this.selectedIndex = 0;

        const selected = items[this.selectedIndex];
        const item = selected._itemData;

        if (item.action) {
            this.handleAction(item.action, item.param);
        } else if (item.url) {
            window.location.href = item.url;
        }
        this.close();
    }

    handleAction(action, param) {
        if (action === 'toggleTheme') {
            if (window.toggleTheme) window.toggleTheme();
        }
        if (action === 'toggleApi') {
            if (window.dataManager) window.dataManager.toggleApiMode();
        }
        if (action === 'runSim') {
            // Navigate to the latest Market Mayhem report with the simulation param
            // We use a fixed recent report for now as the 'host'
            const hostReport = "newsletter_market_mayhem_jan_2026.html";
            window.location.href = `${hostReport}?sim=${param}`;
        }
    }

    renderResults(query) {
        this.resultsContainer.innerHTML = '';
        const q = query.toLowerCase();

        let filtered = [];

        // Special Mode: Simulation
        if (q.startsWith('sim')) {
            // If they just typed 'sim', show all options
            // If 'sim: 2008', filter options
            const simQuery = q.replace('sim', '').replace(':', '').trim();
            if (simQuery) {
                filtered = this.simOptions.filter(item =>
                    item.title.toLowerCase().includes(simQuery) ||
                    item.param.toLowerCase().includes(simQuery)
                );
            } else {
                filtered = this.simOptions;
            }
        } else {
            // Standard Search
            filtered = this.searchData.filter(item =>
                item.title.toLowerCase().includes(q) ||
                item.tag.toLowerCase().includes(q)
            );
        }

        // Limit results
        const displayResults = filtered.slice(0, 50);

        displayResults.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = 'cmd-item';
            if (index === 0) div.classList.add('selected');
            div._itemData = item; // Attach data

            div.innerHTML = `
                <span class="cmd-icon">${item.icon}</span>
                <span class="cmd-text">${item.title}</span>
                <span class="cmd-tag">${item.tag}</span>
            `;

            div.onmouseenter = () => {
                this.resultsContainer.querySelectorAll('.cmd-item.selected').forEach(el => el.classList.remove('selected'));
                div.classList.add('selected');
                this.selectedIndex = index;
            };

            div.onclick = () => this.execute();

            this.resultsContainer.appendChild(div);
        });

        if (displayResults.length === 0) {
             this.resultsContainer.innerHTML = '<div style="padding:20px; color:#64748b; text-align:center;">No matching commands.</div>';
        }
    }
}

// Auto-Initialize
if (!window.commandPalette) {
    window.commandPalette = new CommandPalette();
    // Defer init slightly
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => window.commandPalette.init());
    } else {
        window.commandPalette.init();
    }
}
