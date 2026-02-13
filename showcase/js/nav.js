/**
 * ADAM v23.5 APEX NAVIGATOR SYSTEM
 * -----------------------------------------------------------------------------
 * Architect: System Core
 * Status:    Active
 * Context:   Universal Navigation & Dependency Injector
 * * "The map is not the territory, but a broken map is a broken territory."
 * -----------------------------------------------------------------------------
 */

class AdamNavigator {
    constructor() {
        this.version = "23.5.Apex";
        this.config = {
            dependencies: {
                tailwind: "https://cdn.tailwindcss.com",
                fontAwesome: "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
            }
        };
        
        // State
        this.rootPath = '.';
        this.showcasePath = './showcase';
        this.isGitHub = window.location.hostname.includes("github.io");
        this.currentPath = window.location.pathname;
    }

    /**
     * MASTER BOOT SEQUENCE
     */
    init() {
        try {
            console.log(`[AdamNavigator] Initializing v${this.version}...`);

            // 1. Prevent Double Injection
            if (document.getElementById('side-nav')) {
                console.warn("[AdamNavigator] Navigation already active.");
                return;
            }

            // 2. Resolve Environment Paths
            this._resolvePaths();

            // 3. Inject External Dependencies (CSS/Fonts)
            this._injectDependencies();

            // 4. Boot Core Application Logic (Mock Data/App.js)
            this._bootCoreSystem();

            // 5. Render Interface
            this._renderNavigation();
            this._injectCommandPalette();
            this._injectGlobalStyles();

            // 6. Bind Human Interaction Events
            this._bindEvents();

            console.log("[AdamNavigator] System Online.");

        } catch (e) {
            console.error("[AdamNavigator] Critical Failure:", e);
            this._renderSafeMode(e);
        }
    }

    /**
     * Determines relative paths based on script tag location.
     * Critical for sub-directory navigation.
     */
    _resolvePaths() {
        const scriptTag = document.querySelector('script[src*="nav.js"]');
        const dataRoot = scriptTag ? scriptTag.getAttribute('data-root') : null;
        
        this.rootPath = dataRoot || '.';
        const cleanRoot = this.rootPath.replace(/\/$/, '');
        this.showcasePath = `${cleanRoot}/showcase`;
        
        console.log(`[AdamNavigator] Environment: ${this.isGitHub ? 'GITHUB' : 'LOCAL'} | Root: ${this.rootPath}`);
    }

    /**
     * Sanitizes URL paths to prevent double slashes (e.g., .//showcase)
     */
    _sanitizePath(path) {
        return path.replace(/([^:]\/)\/+/g, "$1");
    }

    /**
     * Loads Tailwind and FontAwesome if missing.
     */
    _injectDependencies() {
        setTimeout(() => {
            if (!document.querySelector('script[src*="tailwindcss"]')) {
                const script = document.createElement('script');
                script.src = this.config.dependencies.tailwind;
                document.head.appendChild(script);
            }
            if (!document.querySelector('link[href*="font-awesome"]')) {
                const link = document.createElement('link');
                link.rel = "stylesheet";
                link.href = this.config.dependencies.fontAwesome;
                document.head.appendChild(link);
            }
        }, 0);
    }

    /**
     * Loads Mock Data and App Logic sequentially.
     */
    _bootCoreSystem() {
        if (!window.dataManager) {
            const mockScript = document.createElement('script');
            mockScript.src = this._sanitizePath(`${this.showcasePath}/js/mock_data.js`);
            document.body.appendChild(mockScript);

            mockScript.onload = () => {
                const appScript = document.createElement('script');
                appScript.src = this._sanitizePath(`${this.showcasePath}/js/app.js`);
                document.body.appendChild(appScript);
            };
            
            mockScript.onerror = () => console.warn("[AdamNavigator] Mock Data module missing or failed.");
        }
    }

    /**
     * Returns the Menu Configuration Object
     */
    _getMenuConfig() {
        return [
            { name: 'Mission Control', icon: 'fa-tachometer-alt', link: 'index.html' },
            { name: 'Chat Portal', icon: 'fa-comments', link: 'chat.html' },
            { name: 'Terminal', icon: 'fa-terminal', link: 'terminal.html' },
            { name: 'Workflow Orchestrator', icon: 'fa-sitemap', link: 'workflow_visualizer.html' },
            { type: 'divider' },
            { name: 'Credit Automation', icon: 'fa-file-invoice-dollar', link: 'credit_memo_automation.html' },
            { name: 'Credit Analyst', icon: 'fa-user-tie', link: 'credit_memo_v2.html' },
            { name: 'Sovereign Dashboard', icon: 'fa-globe-americas', link: 'sovereign_dashboard.html' },
            { type: 'divider' },
            { name: 'Trading Platform', icon: 'fa-chart-line', link: 'trading.html' },
            { name: 'Robo Advisor', icon: 'fa-robot', link: 'robo_advisor.html' },
            { name: 'Research Lab', icon: 'fa-flask', link: 'research.html' },
            { type: 'divider' },
            { name: 'Neural Dashboard', icon: 'fa-brain', link: 'neural_dashboard.html' },
            { name: 'Deep Dive Analyst', icon: 'fa-search-dollar', link: 'deep_dive.html' },
            { name: 'Digital Twin', icon: 'fa-project-diagram', link: 'financial_twin.html' },
            { name: 'Agent Registry', icon: 'fa-users-cog', link: 'agents.html' },
            { name: 'Prompt Library', icon: 'fa-scroll', link: 'prompts.html' },
            { type: 'divider' },
            { name: 'Reports Library', icon: 'fa-file-alt', link: 'reports.html' },
            { name: 'Data Vault', icon: 'fa-database', link: 'data.html' },
            { name: 'App Deploy', icon: 'fa-rocket', link: 'deployment.html' },
            { name: 'Quantum Infra', icon: 'fa-network-wired', link: 'quantum_infrastructure.html' },
            { name: 'Quantum Search', icon: 'fa-search-location', link: 'quantum_search.html' },
            { name: 'Evolution Hub', icon: 'fa-dna', link: 'evolution.html' },
            { type: 'divider' },
            { name: 'Root Directory', icon: 'fa-folder-tree', link: 'ROOT' }
        ];
    }

    /**
     * Builds and Injects the Sidebar HTML
     */
    _renderNavigation() {
        const nav = document.createElement('nav');
        nav.id = 'side-nav';
        nav.className = 'fixed top-0 left-0 h-full w-64 bg-[#0f172a] border-r border-slate-700/50 transform -translate-x-full md:translate-x-0 transition-transform duration-300 z-40 flex flex-col glass-panel';

        // --- Header & Search ---
        let html = `
            <div class="p-4 border-b border-slate-700/50">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-lg font-bold tracking-tight text-white mono">ADAM <span class="text-cyan-400">v23.5</span></h2>
                    <button id="nav-close" class="text-slate-400 hover:text-white md:hidden" aria-label="Close navigation">
                        <i class="fas fa-times" aria-hidden="true"></i>
                    </button>
                </div>
                <div class="relative group">
                    <input type="text" id="global-search" placeholder="Search Agents, Docs... (Ctrl+K)"
                        aria-label="Global search"
                        class="w-full bg-slate-900/50 border border-slate-700 rounded px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors font-mono">
                    <i class="fas fa-search absolute right-3 top-2 text-slate-500 text-xs" aria-hidden="true"></i>
                    <div id="search-results" class="absolute left-0 top-full mt-2 w-64 bg-slate-800 border border-slate-700 rounded shadow-xl hidden z-50 max-h-64 overflow-y-auto"></div>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto py-2 custom-scrollbar">
                <ul class="space-y-1 px-3 list-none">
        `;

        // --- Menu Items ---
        const menuItems = this._getMenuConfig();
        const pathFileName = this.currentPath.split('/').pop().split('?')[0] || 'index.html';

        menuItems.forEach(item => {
            if (item.type === 'divider') {
                html += `<li class="my-2 border-t border-slate-800"></li>`;
                return;
            }

            let linkUrl;
            let isActive = false;

            if (item.link === 'ROOT') {
                linkUrl = this._sanitizePath(`${this.rootPath}/index.html`);
            } else {
                linkUrl = this._sanitizePath(`${this.showcasePath}/${item.link}`);
                
                // Active State Logic
                const itemFilename = item.link.split('/').pop();
                isActive = pathFileName === itemFilename;

                // Special Case: Distinguish 'Mission Control' (showcase/index) from 'Root' (./index)
                if (item.name === 'Mission Control' && pathFileName === 'index.html') {
                    isActive = window.location.href.includes('/showcase/');
                }
            }

            const activeClass = isActive
                ? 'bg-cyan-900/20 text-cyan-400 border-l-2 border-cyan-400'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white border-l-2 border-transparent';

            html += `
                <li>
                    <a href="${linkUrl}" class="flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition group ${activeClass}">
                        <i class="fas ${item.icon} w-5 text-center ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-white'}" aria-hidden="true"></i>
                        ${item.name}
                    </a>
                </li>
            `;
        });

        // --- Footer ---
        const isLive = localStorage.getItem('adam_live_mode') === 'true';
        const envText = this.isGitHub ? 'GITHUB' : 'LOCAL';

        html += `
                </ul>
            </div>
            <div class="p-4 border-t border-slate-700/50 bg-slate-900/20">
                <div class="flex items-center justify-between text-xs mb-3">
                    <span class="text-slate-400">Environment</span>
                    <span class="text-cyan-400 font-mono">${envText}</span>
                </div>
                <div class="flex items-center justify-between text-xs mb-3">
                    <span class="text-slate-400">Theme</span>
                    <button id="theme-toggle" class="text-cyan-400 hover:text-white transition"><i class="fas fa-adjust" aria-hidden="true"></i> Switch</button>
                </div>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-slate-400">Data Source</span>
                    <button id="api-toggle" class="font-bold ${isLive ? 'text-emerald-400' : 'text-slate-500'} hover:text-white transition font-mono border border-slate-700 px-2 py-0.5 rounded">${isLive ? 'LIVE' : 'MOCK'}</button>
                </div>
                <div class="mt-4 flex items-center gap-3">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20 flex items-center justify-center text-xs font-bold text-white font-mono">OP</div>
                    <div>
                        <div class="text-sm font-bold text-white">Operator</div>
                        <div class="text-[10px] text-emerald-400 font-mono">Security Clearance: L4</div>
                    </div>
                </div>
            </div>
        `;

        nav.innerHTML = html;
        document.body.prepend(nav);
    }

    /**
     * Injects the Global Command Palette Modal
     */
    _injectCommandPalette() {
        const palette = document.createElement('div');
        palette.id = 'command-palette-overlay';
        palette.className = 'fixed inset-0 bg-black/80 backdrop-blur-sm z-[100] hidden flex items-start justify-center pt-20 transition-opacity duration-200 opacity-0';
        palette.innerHTML = `
            <div class="w-full max-w-2xl bg-[#0f172a] border border-slate-700 rounded-lg shadow-2xl transform scale-95 transition-transform duration-200">
                <div class="border-b border-slate-700 p-4 flex items-center gap-3">
                    <i class="fas fa-search text-cyan-400"></i>
                    <input type="text" id="cp-input" placeholder="Type a command or search..."
                        class="w-full bg-transparent text-white text-lg focus:outline-none font-mono placeholder-slate-500">
                    <span class="text-xs text-slate-500 border border-slate-700 px-2 py-1 rounded">ESC</span>
                </div>
                <div class="max-h-[60vh] overflow-y-auto custom-scrollbar p-2" id="cp-results">
                    <div class="text-center py-10 text-slate-500 font-mono text-sm">
                        Waiting for input...
                    </div>
                </div>
                <div class="border-t border-slate-700 p-2 bg-slate-900/50 rounded-b-lg flex justify-between px-4 text-xs text-slate-500 font-mono">
                    <span><i class="fas fa-level-down-alt"></i> to select</span>
                    <span><i class="fas fa-arrow-up"></i><i class="fas fa-arrow-down"></i> to navigate</span>
                </div>
            </div>
        `;
        document.body.appendChild(palette);
    }

    /**
     * Injects layout styles for the main content area
     */
    _injectGlobalStyles() {
        const style = document.createElement('style');
        style.innerHTML = `
            @media (min-width: 768px) {
                body.has-global-nav { padding-left: 16rem; }
            }
            .glass-panel {
                backdrop-filter: blur(12px);
                background: rgba(15, 23, 42, 0.95);
            }
            #command-palette-overlay.active {
                opacity: 1;
            }
            #command-palette-overlay.active > div {
                transform: scale(100%);
            }
        `;
        document.head.appendChild(style);
        document.body.classList.add('has-global-nav');
    }

    /**
     * Binds Mobile Toggle, Search, and Settings events
     */
    _bindEvents() {
        // 1. Mobile Toggle & Overlay
        const overlay = document.createElement('div');
        overlay.id = 'nav-overlay';
        overlay.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm z-30 opacity-0 pointer-events-none transition-opacity duration-300 md:hidden';
        document.body.appendChild(overlay);

        if (!document.getElementById('nav-toggle')) {
            const toggleBtn = document.createElement('button');
            toggleBtn.id = 'nav-toggle';
            toggleBtn.className = 'fixed top-4 left-4 z-50 p-2 bg-slate-800 text-white rounded md:hidden border border-slate-700 shadow-lg';
            toggleBtn.style.zIndex = '1001'; // Ensure it's above .cyber-header (z-100) and .scan-line (z-999)
            toggleBtn.innerHTML = '<i class="fas fa-bars" aria-hidden="true"></i>';
            toggleBtn.setAttribute('aria-label', 'Toggle navigation');
            toggleBtn.setAttribute('aria-expanded', 'false');
            toggleBtn.setAttribute('aria-controls', 'side-nav');
            document.body.appendChild(toggleBtn);
        }

        const nav = document.getElementById('side-nav');
        const toggleBtn = document.getElementById('nav-toggle');
        const handleToggle = () => {
            const isClosed = nav.classList.contains('-translate-x-full');
            nav.classList.toggle('-translate-x-full', !isClosed);
            overlay.classList.toggle('opacity-0', !isClosed);
            overlay.classList.toggle('pointer-events-none', !isClosed);

            // Update aria-expanded state
            if (toggleBtn) {
                toggleBtn.setAttribute('aria-expanded', isClosed ? 'true' : 'false');
            }
        };

        document.getElementById('nav-toggle')?.addEventListener('click', handleToggle);
        document.getElementById('nav-close')?.addEventListener('click', handleToggle);
        overlay.addEventListener('click', handleToggle);

        // 2. Settings Toggles
        document.getElementById('theme-toggle')?.addEventListener('click', () => {
            if (window.settingsManager) window.settingsManager.toggleTheme();
        });

        document.getElementById('api-toggle')?.addEventListener('click', () => {
            if (window.dataManager) {
                window.dataManager.toggleApiMode();
                location.reload(); // Refresh to update UI state
            }
        });

        // 3. Search Logic (Sidebar)
        const searchInput = document.getElementById('global-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this._handleSearch(e.target.value, 'search-results'));
            
            document.addEventListener('click', (e) => {
                const results = document.getElementById('search-results');
                if (results && !searchInput.contains(e.target) && !results.contains(e.target)) {
                    results.classList.add('hidden');
                }
            });

            // Keyboard Shortcut (Ctrl+K or Cmd+K)
            document.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    searchInput.focus();
                }
            });
        }

        // 4. Command Palette Logic (Ctrl+K)
        document.addEventListener('keydown', (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                this._toggleCommandPalette();
            }
            if (e.key === 'Escape') {
                const palette = document.getElementById('command-palette-overlay');
                if (!palette.classList.contains('hidden')) {
                    this._toggleCommandPalette();
                }
            }
        });

        // Close on overlay click
        document.getElementById('command-palette-overlay')?.addEventListener('click', (e) => {
            if (e.target.id === 'command-palette-overlay') {
                this._toggleCommandPalette();
            }
        });

        // CP Input Handler
        const cpInput = document.getElementById('cp-input');
        if (cpInput) {
            cpInput.addEventListener('input', (e) => this._handleSearch(e.target.value, 'cp-results'));
        }
    }

    _toggleCommandPalette() {
        const palette = document.getElementById('command-palette-overlay');
        const input = document.getElementById('cp-input');

        if (palette.classList.contains('hidden')) {
            palette.classList.remove('hidden');
            // Trigger reflow for transition
            void palette.offsetWidth;
            palette.classList.add('active');
            input.value = '';
            input.focus();
            this._handleSearch('', 'cp-results'); // Show default state
        } else {
            palette.classList.remove('active');
            setTimeout(() => {
                palette.classList.add('hidden');
            }, 200);
        }
    }

    /**
     * Search Handler
     */
    _handleSearch(query, containerId) {
        const resultsContainer = document.getElementById(containerId);
        if (!query && containerId === 'search-results') {
            resultsContainer.classList.add('hidden');
            return;
        }

        let results = [];
        
        // Priority 1: Window Global Search (from app.js)
        if (window.globalSearch) {
            results = window.globalSearch.search(query);
        } 
        // Priority 2: Fallback to REPO_DATA if available
        else if (window.REPO_DATA) {
            const lowerQuery = query.toLowerCase();
            results = (window.REPO_DATA.nodes || [])
                .filter(n => n.label && n.label.toLowerCase().includes(lowerQuery))
                .slice(0, 5)
                .map(n => ({
                    title: n.label,
                    type: n.group,
                    link: this._sanitizePath(`${this.showcasePath}/data.html?file=${n.path}`)
                }));
        }

        this._renderSearchResults(results, resultsContainer);
    }

    _renderSearchResults(results, container) {
        if (!results || results.length === 0) {
            container.classList.add('hidden');
            return;
        }

        container.classList.remove('hidden');
        container.innerHTML = '';

        results.forEach(r => {
            const link = document.createElement('a');
            link.href = r.link;
            link.className = 'block px-4 py-2 hover:bg-slate-700 border-b border-slate-700 last:border-0 transition-colors';

            const headerDiv = document.createElement('div');
            headerDiv.className = 'flex justify-between';

            const typeSpan = document.createElement('span');
            typeSpan.className = 'text-xs text-cyan-400 font-bold font-mono';
            typeSpan.textContent = r.type || 'FILE';

            const icon = document.createElement('i');
            icon.className = 'fas fa-chevron-right text-[10px] text-slate-600';
            icon.setAttribute('aria-hidden', 'true');

            headerDiv.appendChild(typeSpan);
            headerDiv.appendChild(icon);

            const titleDiv = document.createElement('div');
            titleDiv.className = 'text-sm text-white font-medium';
            titleDiv.textContent = r.title;

            link.appendChild(headerDiv);
            link.appendChild(titleDiv);
            container.appendChild(link);
        });
    }

    /**
     * Fallback UI in case of JS crash
     */
    _renderSafeMode(error) {
        const safeNav = document.createElement('div');
        safeNav.style = "position:fixed; top:0; left:0; width:100%; height:40px; background:#ef4444; color:white; z-index:9999; display:flex; align-items:center; justify-content:center; font-family:monospace; font-size:12px; font-weight:bold;";

        const icon = document.createElement('i');
        icon.className = 'fas fa-exclamation-triangle';
        icon.style.marginRight = '10px';

        safeNav.appendChild(icon);
        safeNav.appendChild(document.createTextNode(` NAV SYSTEM FAILURE: ${error.message}`));
        document.body.prepend(safeNav);
    }
}

// === LAUNCH SYSTEM ===
document.addEventListener('DOMContentLoaded', () => {
    window.adamNavigator = new AdamNavigator();
    window.adamNavigator.init();
});
