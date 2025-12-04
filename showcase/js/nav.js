document.addEventListener('DOMContentLoaded', () => {
    // Check if nav already exists
    if (document.getElementById('side-nav')) return;

    const nav = document.createElement('nav');
    nav.id = 'side-nav';
    nav.className = 'fixed top-0 left-0 h-full w-64 bg-[#0f172a] border-r border-slate-700/50 transform -translate-x-full transition-transform duration-300 z-40 flex flex-col';

    const currentPage = window.location.pathname.split('/').pop() || 'index.html';

    const menuItems = [
        { name: 'Mission Control', icon: 'fa-tachometer-alt', link: 'index.html' },
        { name: 'Trading Platform', icon: 'fa-chart-line', link: 'trading.html', active: currentPage === 'trading.html' },
        { name: 'Robo Advisor', icon: 'fa-robot', link: 'robo_advisor.html', active: currentPage === 'robo_advisor.html' },
        { name: 'Research Lab', icon: 'fa-flask', link: 'research.html', active: currentPage === 'research.html' },
        { type: 'divider' },
        { name: 'Neural Dashboard', icon: 'fa-brain', link: 'neural_dashboard.html' },
        { name: 'Deep Dive Analyst', icon: 'fa-search-dollar', link: 'deep_dive.html' },
        { name: 'Digital Twin', icon: 'fa-project-diagram', link: 'financial_twin.html' },
        { name: 'Agent Registry', icon: 'fa-users-cog', link: 'agents.html' },
        { type: 'divider' },
        { name: 'Reports Library', icon: 'fa-file-alt', link: 'reports.html' },
        { name: 'Prompt Engineering', icon: 'fa-terminal', link: 'prompts.html' },
        { name: 'Data Explorer', icon: 'fa-database', link: 'data.html' },
        { type: 'divider' },
        { name: 'Settings', icon: 'fa-cog', link: '#' },
    ];

    let menuHtml = `
        <div class="h-16 flex items-center px-6 border-b border-slate-700/50">
            <h2 class="text-lg font-bold tracking-tight text-white mono">ADAM <span class="text-cyan-400">NAV</span></h2>
            <button id="nav-close" class="ml-auto text-slate-400 hover:text-white">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="flex-1 overflow-y-auto py-4">
            <ul class="space-y-1 px-3">
    `;

    menuItems.forEach(item => {
        if (item.type === 'divider') {
            menuHtml += `<li class="my-2 border-t border-slate-800"></li>`;
        } else {
            const activeClass = (currentPage === item.link || item.active) ? 'bg-cyan-900/30 text-cyan-400 border-l-2 border-cyan-400' : 'text-slate-400 hover:bg-slate-800 hover:text-white border-l-2 border-transparent';
            menuHtml += `
                <li>
                    <a href="${item.link}" class="flex items-center gap-3 px-3 py-2 rounded text-sm font-medium transition ${activeClass}">
                        <i class="fas ${item.icon} w-5 text-center"></i>
                        ${item.name}
                    </a>
                </li>
            `;
        }
    });

    menuHtml += `
            </ul>
        </div>
        <div class="p-4 border-t border-slate-700/50">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600"></div>
                <div>
                    <div class="text-sm font-bold text-white">Operator</div>
                    <div class="text-xs text-emerald-400">Online</div>
                </div>
            </div>
        </div>
    `;

    nav.innerHTML = menuHtml;
    document.body.appendChild(nav);

    // Overlay
    const overlay = document.createElement('div');
    overlay.id = 'nav-overlay';
    overlay.className = 'fixed inset-0 bg-black/50 backdrop-blur-sm z-30 opacity-0 pointer-events-none transition-opacity duration-300';
    document.body.appendChild(overlay);

    // Toggle Logic
    const toggleBtn = document.getElementById('nav-toggle');
    const closeBtn = document.getElementById('nav-close');

    const openNav = () => {
        nav.classList.remove('-translate-x-full');
        overlay.classList.remove('opacity-0', 'pointer-events-none');
    };

    const closeNav = () => {
        nav.classList.add('-translate-x-full');
        overlay.classList.add('opacity-0', 'pointer-events-none');
    };

    if (toggleBtn) toggleBtn.addEventListener('click', openNav);
    if (closeBtn) closeBtn.addEventListener('click', closeNav);
    overlay.addEventListener('click', closeNav);
});
