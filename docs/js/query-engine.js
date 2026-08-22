// In-browser Search, Filter, and Neuro-Symbolic Prompt Sandbox

class QueryEngine {
    constructor() {
        this.artifacts = [];
        this.filteredArtifacts = [];
        this.activeFilters = {
            brand: 'All',
            tag: 'All'
        };
    }

    initArchiveSearch() {
        // Mock Pre-Hydrated Artifacts
        this.artifacts = [
            { id: 1, title: 'ADAM Architecture Whitepaper', brand: 'adam-core', tag: 'Agentic Systems', date: '2026-01-15', format: 'PDF/Markdown' },
            { id: 2, title: 'Pre-IPO Private Credit Facility', brand: 'fortress-hunt', tag: 'Private Credit', date: '2026-02-10', format: 'JSON/Memo' },
            { id: 3, title: 'SpaceX S-1 Valuation & Multi-Agent Simulation', brand: 'market-mayhem', tag: 'Macro', date: '2026-03-05', format: 'Interactive' },
            { id: 4, title: 'Shared National Credit Guidelines', brand: 'fortress-hunt', tag: 'Credit Risk', date: '2025-11-20', format: 'Guidelines' },
            { id: 5, title: 'Procedural Lore Generation Engine', brand: 'exiled-spark', tag: 'Game Dev', date: '2026-04-01', format: 'Schema' }
        ];

        this.filteredArtifacts = [...this.artifacts];
        this.renderArchiveList();
        this.bindArchiveEvents();
    }

    bindArchiveEvents() {
        const searchInput = document.getElementById('archive-search');
        if(searchInput) {
            searchInput.addEventListener('input', (e) => this.handleSearch(e.target.value));
        }

        const filterBtns = document.querySelectorAll('.filter-btn');
        filterBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = e.target.dataset.type; // brand or tag
                const val = e.target.dataset.value;
                this.activeFilters[type] = val;

                // update active state UI
                document.querySelectorAll(`.filter-btn[data-type="${type}"]`).forEach(b => b.classList.remove('bg-cyan-900/50', 'text-cyan-400'));
                e.target.classList.add('bg-cyan-900/50', 'text-cyan-400');

                this.applyFilters();
            });
        });
    }

    handleSearch(query) {
        query = query.toLowerCase();
        this.filteredArtifacts = this.artifacts.filter(a =>
            a.title.toLowerCase().includes(query) ||
            a.tag.toLowerCase().includes(query)
        );
        this.applyFilters();
    }

    applyFilters() {
        let results = [...this.artifacts];

        const searchInput = document.getElementById('archive-search');
        if(searchInput && searchInput.value) {
            const q = searchInput.value.toLowerCase();
            results = results.filter(a => a.title.toLowerCase().includes(q) || a.tag.toLowerCase().includes(q));
        }

        if(this.activeFilters.brand !== 'All') {
            results = results.filter(a => a.brand === this.activeFilters.brand);
        }

        if(this.activeFilters.tag !== 'All') {
            results = results.filter(a => a.tag === this.activeFilters.tag);
        }

        this.filteredArtifacts = results;
        this.renderArchiveList();
    }

    renderArchiveList() {
        const container = document.getElementById('archive-results');
        if(!container) return;

        if(this.filteredArtifacts.length === 0) {
            container.innerHTML = '<div class="p-8 text-center text-slate-500 font-mono">No artifacts found matching criteria.</div>';
            return;
        }

        container.innerHTML = this.filteredArtifacts.map(a => `
            <div class="glass-panel p-4 mb-4 flex justify-between items-center interactive-card border-slate-700/50" onclick="engine.loadDocument(${a.id})">
                <div>
                    <h4 class="font-bold text-slate-200">${a.title}</h4>
                    <div class="flex gap-3 mt-2 text-xs font-mono text-slate-400">
                        <span class="px-2 py-0.5 bg-slate-800 rounded">${a.brand}</span>
                        <span class="px-2 py-0.5 bg-slate-800 rounded text-cyan-400">${a.tag}</span>
                        <span>${a.date}</span>
                    </div>
                </div>
                <div class="text-slate-500">
                    <i data-lucide="chevron-right" class="w-5 h-5"></i>
                </div>
            </div>
        `).join('');

        if (window.lucide) lucide.createIcons();
    }

    loadDocument(id) {
        const doc = this.artifacts.find(a => a.id === id);
        const reader = document.getElementById('document-reader');
        if(reader && doc) {
            reader.innerHTML = `
                <div class="p-6 border-b border-slate-800">
                    <h2 class="text-2xl font-bold mb-2">${doc.title}</h2>
                    <div class="flex gap-4 text-sm font-mono text-slate-400 mb-4">
                        <span>${doc.date}</span>
                        <span>|</span>
                        <span>${doc.format}</span>
                    </div>
                    <button class="px-4 py-2 bg-emerald-900/40 text-emerald-400 text-sm font-mono rounded hover:bg-emerald-800/60 transition-colors">
                        <i data-lucide="download" class="w-4 h-4 inline-block mr-1 mb-1"></i> Export Payload
                    </button>
                </div>
                <div class="p-6 font-mono text-sm leading-relaxed text-slate-300">
                    <p class="mb-4">Loading encrypted contents for ${doc.title}...</p>
                    <div class="p-4 bg-slate-900/50 rounded border border-slate-800 mb-4 font-mono text-xs text-slate-400">
                        // Simulated Content Extraction<br/>
                        const artifact_metadata = {<br/>
                        &nbsp;&nbsp;brand_origin: "${doc.brand}",<br/>
                        &nbsp;&nbsp;classification: "${doc.tag}",<br/>
                        &nbsp;&nbsp;compliance_status: "VERIFIED"<br/>
                        };
                    </div>
                    <p>W3C PROV-O compliance trace established. Deterministic signatures valid.</p>
                </div>
            `;
            if (window.lucide) lucide.createIcons();
        }
    }

    // --- Sandbox Simulator Logic ---
    initSandbox() {
        const runBtn = document.getElementById('run-sim-btn');
        if(runBtn) {
            runBtn.addEventListener('click', () => this.runSimulation());
        }
    }

    runSimulation() {
        const terminal = document.getElementById('sim-terminal-body');
        const mode = document.querySelector('input[name="sim-mode"]:checked').value; // 'human' or 'machine'

        if(!terminal) return;

        terminal.innerHTML = '';

        const steps = [
            `[SYS] Initializing ${mode.toUpperCase()} query context...`,
            `[GOV] Validating input against schema registry (FIBO taxonomy)...`,
            `[GOV] Schema strictness check passed.`,
            `[SYS] Dispatching to Neuro-Symbolic router...`,
            `[SYS] Symbolic Rule Check: SUCCESS (Deterministic constraints met)`,
            `[AGENT] Synthesizing final state transition...`,
            `[OUT] Execution complete. W3C PROV-O trace generated.`
        ];

        let i = 0;
        function printStep() {
            if (i < steps.length) {
                // Ensure unique keys by relying on DOM append, avoiding React anti-patterns
                const line = document.createElement('div');
                line.className = 'mb-1 ' + (steps[i].includes('SUCCESS') || steps[i].includes('passed') ? 'text-emerald-400' : 'text-slate-300');
                line.textContent = `> ${steps[i]}`;
                terminal.appendChild(line);
                terminal.scrollTop = terminal.scrollHeight;
                i++;
                setTimeout(printStep, 400 + Math.random() * 400); // jittered typing effect
            } else {
                const final = document.createElement('div');
                final.className = 'mt-4 text-cyan-400 font-bold';
                final.textContent = '>> SYSTEM IDLE';
                terminal.appendChild(final);
                terminal.scrollTop = terminal.scrollHeight;
            }
        }

        printStep();
    }
}

const engine = new QueryEngine();

document.addEventListener('DOMContentLoaded', () => {
    if(document.getElementById('archive-search')) {
        engine.initArchiveSearch();
    }
    if(document.getElementById('run-sim-btn')) {
        engine.initSandbox();
    }
});
