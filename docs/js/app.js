// Global Application State & Router
const AppState = {
    currentBrand: 'adam-core',
    isMachineMode: false,
    telemetry: {
        latency: 12,
        activeAgents: 142,
        memoryUsage: '4.2GB'
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initBrandSwitcher();
    initTelemetry();
    initKeyboardShortcuts();

    // Inject global header/footer if container exists
    const headerContainer = document.getElementById('global-header');
    if (headerContainer) {
        headerContainer.innerHTML = generateGlobalHeader();
    }
});

function initBrandSwitcher() {
    const switchers = document.querySelectorAll('.brand-switch');
    switchers.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const brand = e.currentTarget.dataset.brand;
            if(brand) {
                // In a real app this would route, here we just update state or redirect
                console.log(`Switching to brand: ${brand}`);
                AppState.currentBrand = brand;
                if(brand === 'home') window.location.href = '/docs/index.html';
                else window.location.href = `/docs/brands/${brand}.html`;
            }
        });
    });
}

function initTelemetry() {
    const latencyEl = document.getElementById('tel-latency');
    const agentsEl = document.getElementById('tel-agents');

    if (latencyEl && agentsEl) {
        setInterval(() => {
            // Simulate jitter
            const jitter = Math.floor(Math.random() * 5) - 2;
            latencyEl.textContent = `${Math.max(5, AppState.telemetry.latency + jitter)}ms`;

            if (Math.random() > 0.8) {
                AppState.telemetry.activeAgents += (Math.random() > 0.5 ? 1 : -1);
                agentsEl.textContent = AppState.telemetry.activeAgents;
            }
        }, 1000);
    }
}

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('global-search');
            if (searchInput) {
                searchInput.focus();
            } else {
                window.location.href = '/docs/archive/index.html';
            }
        }
    });
}

function generateGlobalHeader() {
    return `
        <header class="glass-header w-full px-6 py-4 flex justify-between items-center">
            <div class="flex items-center gap-4">
                <a href="../index.html" class="flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors">
                    <i data-lucide="cpu" class="w-6 h-6"></i>
                    <span class="font-mono font-bold text-lg tracking-wider">ADAM_OS</span>
                </a>
                <div class="h-6 w-px bg-slate-700 mx-2"></div>
                <div class="flex gap-2">
                    <a href="../brands/adam-core.html" class="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition-colors">CORE</a>
                    <a href="../brands/market-mayhem.html" class="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition-colors">MAYHEM</a>
                    <a href="../brands/fortress-hunt.html" class="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition-colors">FORTRESS</a>
                    <a href="../archive/index.html" class="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition-colors">ARCHIVE</a>
                    <a href="../demo/agent-sandbox.html" class="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition-colors">SANDBOX</a>
                </div>
            </div>
            <div class="flex items-center gap-4">
                <div class="hidden md:flex items-center gap-2 glass-panel px-3 py-1.5 text-xs font-mono text-slate-400">
                    <i data-lucide="search" class="w-3 h-3"></i>
                    <span>Search</span>
                    <kbd class="ml-2 px-1.5 py-0.5 bg-slate-800 rounded text-[10px]">⌘K</kbd>
                </div>
                <a href="https://github.com/adamvangrover/adam" target="_blank" class="text-slate-400 hover:text-white transition-colors">
                    <i data-lucide="github" class="w-5 h-5"></i>
                </a>
            </div>
        </header>
    `;
}
