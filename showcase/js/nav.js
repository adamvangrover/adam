document.addEventListener('DOMContentLoaded', () => {
    // 0. Inject Dependencies (Tailwind & FontAwesome) if missing
    setTimeout(() => {
        if (!document.querySelector('script[src*="tailwindcss"]')) {
            const script = document.createElement('script');
            script.src = "https://cdn.tailwindcss.com";
            document.head.appendChild(script);
        }
        if (!document.querySelector('link[href*="font-awesome"]')) {
            const link = document.createElement('link');
            link.rel = "stylesheet";
            link.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css";
            document.head.appendChild(link);
        }
    }, 0);

    // 1. Check if nav already exists
    if (document.getElementById('side-nav')) return;

    // 2. Determine Paths
    const scriptTag = document.querySelector('script[src*="nav.js"]');
    const rootPath = scriptTag ? (scriptTag.getAttribute('data-root') || '.') : '.';
    const cleanRoot = rootPath.replace(/\/$/, '');
    const showcasePath = `${cleanRoot}/showcase`;

    // 3. Inject App Logic (Mock Data & App.js) if missing
    if (!window.dataManager) {
        const mockScript = document.createElement('script');
        mockScript.src = `${showcasePath}/js/mock_data.js`;
        document.body.appendChild(mockScript);

        mockScript.onload = () => {
            const appScript = document.createElement('script');
            appScript.src = `${showcasePath}/js/app.js`;
            document.body.appendChild(appScript);
        };
    }

    // 4. Define Menu Items
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
        { name: 'App Deploy', icon: 'fa-rocket', link: 'deployment.html' },
        { type: 'divider' },
        { name: 'Root', icon: 'fa-folder-tree', link: 'ROOT' }
    ];

    // 5. Create Nav Element
    const nav = document.createElement('nav');
    nav.id = 'side-nav';
    nav.className = 'fixed top-0 left-0 h-full w-64 bg-[#0f172a] border-r border-slate-700/50 transform -translate-x-full md:translate-x-0 transition-transform duration-300 z-40 flex flex-col glass-panel';

    // 6. Build HTML
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
                <div id="search-results" class="absolute left-0 top-full mt-2 w-64 bg-slate-800 border border-slate-700 rounded shadow-xl hidden z-50 max-h-64 overflow-y-auto"></div>
            </div>
        </div>

        <div class="flex-1 overflow-y-auto py-2 custom-scrollbar">
            <ul class="space-y-1 px-3 list-none">
    `;

    const path = window.location.pathname.split('/').pop();
    const currentPage = path === '' ? 'index.html' : path.split('?')[0];

    menuItems.forEach(item => {
        if (item.type === 'divider') {
            menuHtml += `<li class="my-2 border-t border-slate-800"></li>`;
        } else {
            let finalLink;
            let isActive = false;

            if (item.link === 'ROOT') {
                finalLink = `${cleanRoot}/index.html`;
            } else {
                finalLink = `${showcasePath}/${item.link}`;
                finalLink = finalLink.replace(/([^:]\/)\/+/g, "$1");
                const itemFilename = item.link.split('/').pop();
                isActive = currentPage === itemFilename;

                if (item.name === 'Mission Control' && currentPage === 'index.html') {
                    if (window.location.href.includes('/showcase/')) isActive = true;
                    else isActive = false;
                }
            }

            const activeClass = isActive
                ? 'bg-cyan-900/20 text-cyan-400 border-l-2 border-cyan-400'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white border-l-2 border-transparent';

            menuHtml += `
                <li>
                    <a href="${finalLink}" class="flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition group ${activeClass}">
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
    document.body.prepend(nav);

    // 7. Inject Layout Styles
    const style = document.createElement('style');
    style.innerHTML = `
        @media (min-width: 768px) {
            body.has-global-nav {
                padding-left: 16rem;
            }
        }
    `;
    document.head.appendChild(style);
    document.body.classList.add('has-global-nav');

    // 8. Interaction Logic
    const overlay = document.createElement('div');
    overlay.id = 'nav-overlay';
    overlay.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm z-30 opacity-0 pointer-events-none transition-opacity duration-300 md:hidden';
    document.body.appendChild(overlay);

    if (!document.getElementById('nav-toggle')) {
        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'nav-toggle';
        toggleBtn.className = 'fixed top-4 left-4 z-50 p-2 bg-slate-800 text-white rounded md:hidden border border-slate-700 shadow-lg';
        toggleBtn.innerHTML = '<i class="fas fa-bars"></i>';
        document.body.appendChild(toggleBtn);
    }

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

    // Settings
    const themeBtn = document.getElementById('theme-toggle');
    const apiBtn = document.getElementById('api-toggle');

    if (themeBtn && window.settingsManager) {
        themeBtn.addEventListener('click', () => window.settingsManager.toggleTheme());
    }
    if (apiBtn && window.dataManager) {
        apiBtn.addEventListener('click', () => window.dataManager.toggleApiMode());
    }

    // Search
    const searchInput = document.getElementById('global-search');
    const searchResults = document.getElementById('search-results');

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value;
            if (window.globalSearch) {
                const results = window.globalSearch.search(query);
                renderResults(results);
            } else if (window.REPO_DATA) {
                 const lowerQuery = query.toLowerCase();
                 const nodes = window.REPO_DATA.nodes || [];
                 const results = nodes.filter(n => n.label && n.label.toLowerCase().includes(lowerQuery)).slice(0, 5).map(n => ({
                     title: n.label,
                     type: n.group,
                     link: `${showcasePath}/data.html?file=${n.path}`
                 }));
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
                    <span class="text-xs text-cyan-400 font-bold font-mono">${r.type || 'FILE'}</span>
                    <i class="fas fa-chevron-right text-[10px] text-slate-600"></i>
                </div>
                <div class="text-sm text-white font-medium">${r.title}</div>
            </a>
        `).join('');
    }
});
