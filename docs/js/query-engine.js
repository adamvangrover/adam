// Search, Filter, and Neuro-Symbolic Prompt Sandbox Logic

const mockIndex = [
    { id: 1, title: 'ADAM Architecture Whitepaper', brand: 'adam-core', tag: 'Agentic Systems', format: 'pdf', date: '2026-08-15', snippet: 'Defining the baseline for neuro-symbolic deterministic orchestration...' },
    { id: 2, title: 'SpaceX S-1 Valuation', brand: 'market-mayhem', tag: 'Macro', format: 'json', date: '2026-10-01', snippet: 'Multi-Agent Simulation Report on Starlink revenue projections...' },
    { id: 3, title: 'Pre-IPO Private Credit Facility', brand: 'fortress-hunt', tag: 'Private Credit', format: 'md', date: '2026-11-12', snippet: 'Hardware Collateral Memo detailing LTV ratios and downside protection...' },
    { id: 4, title: 'Shared National Credit Guidelines', brand: 'fortress-hunt', tag: 'Macro', format: 'pdf', date: '2026-09-20', snippet: 'Corporate Debt Covenant Guidelines and regulatory impacts...' }
];

function initArchiveSearch() {
    const searchInput = document.getElementById('archive-search');
    const resultsContainer = document.getElementById('archive-results');
    const filterSelects = document.querySelectorAll('.archive-filter');

    if (!searchInput || !resultsContainer) return;

    const renderResults = (results) => {
        if (results.length === 0) {
            resultsContainer.innerHTML = '<div class="text-slate-400 p-8 text-center font-mono">No intelligence artifacts found matching criteria.</div>';
            return;
        }

        resultsContainer.innerHTML = results.map(item => `
            <div class="glass-panel p-4 hover:border-cyan-500/50 transition-colors cursor-pointer flex justify-between items-start">
                <div>
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">${item.brand}</span>
                        <span class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">${item.tag}</span>
                    </div>
                    <h3 class="text-lg font-semibold text-white mb-1">${item.title}</h3>
                    <p class="text-sm text-slate-400">${item.snippet}</p>
                </div>
                <div class="flex flex-col items-end gap-2">
                    <span class="text-xs font-mono text-slate-500">${item.date}</span>
                    <span class="text-xs font-mono px-2 py-1 rounded bg-emerald-900/30 text-emerald-400 border border-emerald-800/50 uppercase">${item.format}</span>
                </div>
            </div>
        `).join('');
    };

    const executeSearch = () => {
        const query = searchInput.value.toLowerCase();
        let filters = {};
        filterSelects.forEach(select => {
            if (select.value !== 'all') {
                filters[select.dataset.filterType] = select.value;
            }
        });

        const filtered = mockIndex.filter(item => {
            const matchesQuery = item.title.toLowerCase().includes(query) || item.snippet.toLowerCase().includes(query);
            const matchesFilters = Object.keys(filters).every(key => item[key] === filters[key]);
            return matchesQuery && matchesFilters;
        });

        renderResults(filtered);
    };

    searchInput.addEventListener('input', executeSearch);
    filterSelects.forEach(select => select.addEventListener('change', executeSearch));

    // Initial render
    renderResults(mockIndex);
}

// Sandbox Simulator Logic
function initSandbox() {
    const runBtn = document.getElementById('run-sim-btn');
    const terminal = document.getElementById('sandbox-terminal');
    const promptInput = document.getElementById('sandbox-prompt');

    if (!runBtn || !terminal) return;

    const appendLog = (msg, type = 'info') => {
        const colors = {
            info: 'text-slate-300',
            success: 'text-emerald-400',
            warn: 'text-amber-400',
            error: 'text-rose-400',
            system: 'text-cyan-400'
        };
        const el = document.createElement('div');
        el.className = `font-mono text-xs mb-1 ${colors[type]}`;
        const timeSpan = document.createElement('span');
        timeSpan.className = 'opacity-50';
        timeSpan.textContent = `[${new Date().toISOString().split('T')[1].slice(0,-1)}] `;
        el.appendChild(timeSpan);
        el.appendChild(document.createTextNode(msg));
        terminal.appendChild(el);
        terminal.scrollTop = terminal.scrollHeight;
    };

    runBtn.addEventListener('click', async () => {
        terminal.innerHTML = '';
        const prompt = promptInput ? promptInput.value : 'Executing default payload...';

        appendLog(`INITIALIZING NEURO-SYMBOLIC ORCHESTRATION`, 'system');
        appendLog(`Payload: "${prompt}"`);

        await new Promise(r => setTimeout(r, 600));
        appendLog(`[Gov Gatekeeper] Validating against deterministic schemas...`, 'warn');

        await new Promise(r => setTimeout(r, 800));
        appendLog(`[Gov Gatekeeper] Schema validation PASS (FIBO Topology Matched)`, 'success');

        await new Promise(r => setTimeout(r, 1200));
        appendLog(`[LLM Core] Synthesizing structural parameters...`);

        await new Promise(r => setTimeout(r, 1500));
        appendLog(`[System 2] Resolving logical constraints via Rust Engine...`, 'system');

        await new Promise(r => setTimeout(r, 900));
        appendLog(`EXECUTION COMPLETE. Generated W3C PROV-O Trace.`, 'success');
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initArchiveSearch();
    initSandbox();
});
