const ADAM_NAV_ITEMS = [
    { label: 'MISSION CONTROL', href: 'index.html', icon: 'M' },
    { label: 'AGENTS', href: 'agents.html', icon: 'A' },
    { label: 'PROMPTS', href: 'prompts.html', icon: 'P' },
    { label: 'REPORTS', href: 'reports.html', icon: 'R' },
    { label: 'KNOWLEDGE GRAPH', href: 'graph.html', icon: 'K' },
    { label: 'DIGITAL TWIN', href: 'financial_twin.html', icon: 'T' },
    { label: 'DATA & LIBS', href: 'data.html', icon: 'D' },
    { label: 'NAVIGATOR', href: 'navigator.html', icon: 'N' },
    { label: 'CHAT', href: 'chat.html', icon: 'C' },
];

function renderNav() {
    const currentPath = window.location.pathname.split('/').pop() || 'index.html';

    const header = document.createElement('header');
    header.className = 'bg-slate-900 border-b border-slate-700 p-4 flex justify-between items-center sticky top-0 z-50 shadow-lg';

    // Logo / Title Area
    const logoDiv = document.createElement('div');
    logoDiv.className = 'flex items-center space-x-4';
    logoDiv.innerHTML = `
        <div class="flex flex-col">
            <h1 class="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300 tracking-wider font-mono">ADAM v23.0</h1>
            <span class="text-[0.6rem] text-blue-500 tracking-[0.2em] uppercase">Adaptive Hive Mind System</span>
        </div>
        <div id="sys-status" class="hidden md:block px-2 py-1 text-[0.6rem] font-bold rounded bg-slate-800 text-blue-400 border border-blue-900 shadow-[0_0_10px_rgba(59,130,246,0.2)]">
            SYSTEM READY
        </div>
    `;

    // Nav Links
    const nav = document.createElement('nav');
    nav.className = 'hidden xl:flex space-x-6 text-xs font-mono tracking-wide';

    ADAM_NAV_ITEMS.forEach(item => {
        const a = document.createElement('a');
        a.href = item.href;
        a.textContent = item.label;

        // Active state
        if (currentPath === item.href) {
            a.className = 'text-cyan-400 font-bold border-b-2 border-cyan-400 pb-1 cursor-default';
        } else {
            a.className = 'text-slate-400 hover:text-cyan-300 transition-colors pb-1 border-b-2 border-transparent hover:border-cyan-900';
        }
        nav.appendChild(a);
    });

    // Mobile Menu Button (Hamburger) - Simplified for now
    const mobileBtn = document.createElement('button');
    mobileBtn.className = 'xl:hidden text-slate-300 hover:text-white';
    mobileBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
    `;
    mobileBtn.onclick = () => {
        const mobileMenu = document.getElementById('mobile-menu');
        if (mobileMenu) mobileMenu.classList.toggle('hidden');
    };

    header.appendChild(logoDiv);
    header.appendChild(nav);
    header.appendChild(mobileBtn);

    // Insert header at top of body
    document.body.prepend(header);

    // Create Mobile Menu Container
    const mobileMenu = document.createElement('div');
    mobileMenu.id = 'mobile-menu';
    mobileMenu.className = 'hidden xl:hidden bg-slate-900 border-b border-slate-700 p-4 absolute w-full z-40 top-[73px] shadow-2xl';

    ADAM_NAV_ITEMS.forEach(item => {
        const a = document.createElement('a');
        a.href = item.href;
        a.className = 'block py-2 text-sm font-mono text-slate-300 hover:text-cyan-400 border-b border-slate-800 last:border-0';
        a.innerHTML = `<span class="inline-block w-6 text-center text-slate-600 font-bold mr-2">${item.icon}</span> ${item.label}`;
        mobileMenu.appendChild(a);
    });

    document.body.appendChild(mobileMenu);
}

// Auto-run
document.addEventListener('DOMContentLoaded', () => {
    // Remove existing header if present (to avoid duplicates if migrating)
    const existingHeader = document.querySelector('header');
    if (existingHeader) existingHeader.remove();

    renderNav();
});
