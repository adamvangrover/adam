document.addEventListener('DOMContentLoaded', () => {
    // 1. Check if nav already exists to prevent duplicate injection
    if (document.getElementById('side-nav')) return;

    // 2. Create the Navigation Element
    const nav = document.createElement('nav');
    nav.id = 'side-nav';
    nav.className = 'fixed top-0 left-0 h-full w-64 bg-[#0f172a] border-r border-slate-700/50 transform -translate-x-full md:translate-x-0 transition-transform duration-300 z-40 flex flex-col glass-panel';

    // 3. Define Menu Items
    // Clean up path to just filename for comparison
    const path = window.location.pathname.split('/').pop();
    const currentPage = path === '' ? 'index.html' : path.split('?')[0]; // Handle empty path as index and ignore query params

    const menuItems = [
        { name: 'Mission Control', icon: 'fa-tachometer-alt', link: 'index.html' },
        { name: 'Chat Portal', icon: 'fa-comments', link: 'chat.html' },
        { name: 'Terminal', icon: 'fa-terminal', link: 'terminal.html' },
        { type: 'divider' },
        { name: 'Trading Platform', icon: 'fa-chart-line', link: 'trading.html' },
        { name: 'Robo Advisor', icon: 'fa-robot', link: 'robo_advisor.html' },
        { name: 'Research Lab', icon: 'fa-flask', link: 'research.html' },
        { type: 'divider' },
        { name: 'Neural Dashboard', icon: 'fa-brain', link: 'neural_dashboard.html' },
        { name: 'Deep Dive Analyst', icon: 'fa-search-dollar', link: 'deep_dive.html' },
        { name: 'Digital Twin', icon: 'fa-project-diagram', link: 'financial_twin.html' },
        { name: 'Agent Registry', icon: 'fa-users-cog', link: 'agents.html' },
        { type: 'divider' },
        { name: 'Reports Library', icon: 'fa-file-alt', link: 'reports.html' },
        { name: 'Data Vault', icon: 'fa-database', link: 'data.html' },
        { name: 'Scenario Engine', icon: 'fa-chess', link: 'scenario_dashboard.html' },
        { name: 'App Deploy', icon: 'fa-rocket', link: 'deployment.html' },
    ];

    // 4. Build HTML
    let menuHtml = `
        <div class="p-4 border-b border-slate-700/50">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-lg font-bold tracking-tight text-white mono">ADAM <span class="text-cyan-400">v23.5</span></h2>
                <button id="nav-close" class="text-slate-400 hover:text-white md:hidden">
                    <i class="fas fa-times"></i>
                </button>
            </div>

            <!-- Global Search -->
            <div class="relative group">
                <input type="text" id="global-search" placeholder="Search Agents, Docs..."
                    class="w-full bg-slate-900/50 border border-slate-700 rounded px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 transition-colors font-mono">
                <i class="fas fa-search absolute right-3 top-2 text-slate-500 text-xs"></i>

                <!-- Search Results Dropdown -->
                <div id="search-results" class="absolute left-0 top-full mt-2 w-64 bg-slate-800 border border-slate-700 rounded shadow-xl hidden z-50 max-h-64 overflow-y-auto">
                    <!-- Results injected here -->
                </div>
            </div>
        </div>

        <div class="flex-1 overflow-y-auto py-2 custom-scrollbar">
            <ul class="space-y-1 px-3">
    `;

    menuItems.forEach(item => {
        if (item.type === 'divider') {
            menuHtml += `<li class="my-2 border-t border-slate-800"></li>`;
        } else {
            // Check active state
            const isActive = currentPage === item.link;
            const activeClass = isActive
                ? 'bg-cyan-900/20 text-cyan-400 border-l-2 border-cyan-400'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white border-l-2 border-transparent';

            menuHtml += `
                <li>
                    <a href="${item.link}" class="flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition group ${activeClass}">
                        <i class="fas ${item.icon} w-5 text-center ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-white'}"></i>
                        ${item.name}
                    </a>
                </li>
            `;
        }
    });

    const isLive = localStorage.getItem('adam_live_mode') === 'true';
    const apiClass = isLive ? 'text-emerald-400' : 'text-slate-500';
    const apiText = isLive ? 'LIVE' : 'MOCK';

    menuHtml += `
            </ul>
        </div>
        <div class="p-4 border-t border-slate-700/50 bg-slate-900/20">
            <div class="flex items-center justify-between text-xs mb-3">
                <span class="text-slate-400">Theme</span>
                <button id="theme-toggle" class="text-cyan-400 hover:text-white transition"><i class="fas fa-adjust"></i> Switch</button>
            </div>
            <div class="flex items-center justify-between text-xs">
                <span class="text-slate-400">Data Source</span>
                <button id="api-toggle" class="font-bold ${apiClass} hover:text-white transition font-mono border border-slate-700 px-2 py-0.5 rounded">${apiText}</button>
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

    nav.innerHTML = menuHtml;
    document.body.prepend(nav); // Prepend to be the first element

    // 5. Inject Layout Adjustments
    // Find the main content area and add margin for desktop
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.classList.add('transition-all', 'duration-300', 'md:ml-64');
    } else {
        // Fallback: Add to body if no main tag (though good practice uses main)
        // document.body.style.paddingLeft = '16rem'; // Only if desktop media query matches
        // Better to wrap existing content if possible, but for now assuming <main> exists based on index.html
    }

    // 6. Interaction Logic

    // Overlay for Mobile
    const overlay = document.createElement('div');
    overlay.id = 'nav-overlay';
    overlay.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm z-30 opacity-0 pointer-events-none transition-opacity duration-300 md:hidden';
    document.body.appendChild(overlay);

    const toggleBtn = document.getElementById('nav-toggle');
    const closeBtn = document.getElementById('nav-close');

    const toggleNav = () => {
        const isClosed = nav.classList.contains('-translate-x-full');
        if (isClosed) {
            nav.classList.remove('-translate-x-full');
            overlay.classList.remove('opacity-0', 'pointer-events-none');
        } else {
            nav.classList.add('-translate-x-full');
            overlay.classList.add('opacity-0', 'pointer-events-none');
        }
    };

    if (toggleBtn) toggleBtn.addEventListener('click', toggleNav);
    if (closeBtn) closeBtn.addEventListener('click', toggleNav);
    overlay.addEventListener('click', toggleNav);

    // Settings Integration
    const themeBtn = document.getElementById('theme-toggle');
    const apiBtn = document.getElementById('api-toggle');

    if (themeBtn && window.settingsManager) {
        themeBtn.addEventListener('click', () => window.settingsManager.toggleTheme());
    }
    if (apiBtn && window.dataManager) {
        apiBtn.addEventListener('click', () => window.dataManager.toggleApiMode());
    }

    // Search Logic
    const searchInput = document.getElementById('global-search');
    const searchResults = document.getElementById('search-results');

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value;
            if (window.globalSearch) {
                const results = window.globalSearch.search(query);
                renderResults(results);
            }
        });

        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.classList.add('hidden');
            }
        });
    }

    function renderResults(results) {
        if (!results || results.length === 0) {
            searchResults.classList.add('hidden');
            return;
        }

        searchResults.classList.remove('hidden');
        searchResults.innerHTML = results.map(r => `
            <a href="${r.link}" class="block px-4 py-2 hover:bg-slate-700 border-b border-slate-700 last:border-0 transition-colors">
                <div class="flex justify-between">
                    <span class="text-xs text-cyan-400 font-bold font-mono">${r.type}</span>
                    <i class="fas fa-chevron-right text-[10px] text-slate-600"></i>
                </div>
                <div class="text-sm text-white font-medium">${r.title}</div>
                <div class="text-[10px] text-slate-400 truncate">${r.subtitle}</div>
            </a>
        `).join('');
    }
});
