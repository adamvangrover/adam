document.addEventListener('DOMContentLoaded', () => {
    // State
    const state = {
        data: [],
        filteredData: [],
        viewMode: 'HUMAN', // 'HUMAN' or 'MACHINE'
        filters: {
            search: '',
            type: 'ALL',
            sentiment: 'ALL'
        }
    };

    // DOM Elements
    const gridContainer = document.getElementById('gridContainer');
    const machineContainer = document.getElementById('machineContainer');
    const searchInput = document.getElementById('searchInput');
    const typeFilter = document.getElementById('typeFilter');
    const viewToggle = document.getElementById('viewToggle');

    // Metrics Elements
    const elTotal = document.getElementById('metricTotal');
    const elSentiment = document.getElementById('metricSentiment');
    const elHighConviction = document.getElementById('metricHighConviction');
    const elSystemLoad = document.getElementById('metricSystemLoad');

    // --- Init ---
    async function init() {
        try {
            const response = await fetch('data/market_mayhem_index.json');
            state.data = await response.json();
            state.filteredData = [...state.data];

            updateMetrics();
            render();
            bindEvents();

            console.log('[REPO_V2] System Online. Data Loaded:', state.data.length);
        } catch (e) {
            console.error('[REPO_V2] Data Load Error:', e);
            gridContainer.innerHTML = `<div style="color:red; font-family:monospace; padding:20px;">CRITICAL FAILURE: ${e.message}</div>`;
        }
    }

    // --- Logic ---

    function filterData() {
        const { search, type, sentiment } = state.filters;
        const lowerSearch = search.toLowerCase();

        state.filteredData = state.data.filter(item => {
            // Search Text
            const matchSearch = (item.title || '').toLowerCase().includes(lowerSearch) ||
                                (item.summary || '').toLowerCase().includes(lowerSearch) ||
                                (item.filename || '').toLowerCase().includes(lowerSearch);

            // Type
            const matchType = type === 'ALL' || item.type === type;

            // Sentiment
            let matchSentiment = true;
            if (sentiment === 'POSITIVE') matchSentiment = item.sentiment_score >= 60;
            if (sentiment === 'NEUTRAL') matchSentiment = item.sentiment_score >= 40 && item.sentiment_score < 60;
            if (sentiment === 'NEGATIVE') matchSentiment = item.sentiment_score < 40;

            return matchSearch && matchType && matchSentiment;
        });

        updateMetrics();
        render();
    }

    function updateMetrics() {
        // Total
        elTotal.textContent = state.filteredData.length;

        // Avg Sentiment
        const totalSent = state.filteredData.reduce((acc, item) => acc + (item.sentiment_score || 0), 0);
        const avgSent = state.filteredData.length ? Math.round(totalSent / state.filteredData.length) : 0;
        elSentiment.textContent = avgSent;
        elSentiment.style.color = avgSent >= 60 ? '#10b981' : (avgSent < 40 ? '#ef4444' : '#f59e0b');

        // High Conviction
        const highConv = state.filteredData.filter(i => i.conviction >= 80).length;
        elHighConviction.textContent = highConv;

        // System Load (Simulated)
        elSystemLoad.textContent = Math.floor(Math.random() * 20 + 30) + '%';
    }

    // --- Rendering ---

    function render() {
        if (state.viewMode === 'HUMAN') {
            gridContainer.style.display = 'grid';
            machineContainer.style.display = 'none';
            renderGrid();
        } else {
            gridContainer.style.display = 'none';
            machineContainer.style.display = 'block';
            renderMachine();
        }
    }

    function renderGrid() {
        gridContainer.innerHTML = '';
        state.filteredData.forEach(item => {
            const el = document.createElement('div');
            el.className = `grid-item type-${item.type}`;

            // Date formatting
            const dateStr = item.date || 'UNKNOWN';

            // Tags
            let tagsHtml = '';
            if (item.entities && item.entities.keywords) {
                tagsHtml = item.entities.keywords.slice(0,3).map(k =>
                    `<span class="badge-outline">${k}</span>`
                ).join(' ');
            }

            el.innerHTML = `
                <div class="item-header">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.65rem; color:#64748b;">${dateStr}</span>
                        <span class="cyber-badge" style="font-size:0.6rem;">${item.type}</span>
                    </div>
                    <h3 style="margin:0; font-size:1rem; color:var(--text-primary); line-height:1.2;">${item.title}</h3>
                </div>
                <div class="item-body">
                    ${(item.summary || '').substring(0, 120)}...
                </div>
                <div class="item-footer">
                    <div style="display:flex; gap:5px;">
                        ${tagsHtml}
                    </div>
                    <a href="${item.filename}" class="cyber-btn" style="padding:4px 8px; font-size:0.6rem;">ACCESS</a>
                </div>
            `;
            gridContainer.appendChild(el);
        });
    }

    function renderMachine() {
        // Pretty Print JSON with syntax highlighting
        // We'll construct HTML string manually for performance on large lists
        let html = '';
        state.filteredData.forEach(item => {
            const jsonStr = JSON.stringify(item, null, 2);
            // Simple syntax highlight via regex replacement
            const coloredJson = jsonStr.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    if (/:$/.test(match)) {
                        cls = 'json-key';
                    } else {
                        cls = 'json-string';
                    }
                } else if (/true|false/.test(match)) {
                    cls = 'json-boolean';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return '<span class="' + cls + '">' + match + '</span>';
            });

            html += `<div class="json-node">${coloredJson}</div>`;
        });
        machineContainer.innerHTML = html;
    }

    // --- Events ---

    function bindEvents() {
        // Search
        searchInput.addEventListener('input', (e) => {
            state.filters.search = e.target.value;
            filterData();
        });

        // Type
        typeFilter.addEventListener('change', (e) => {
            state.filters.type = e.target.value;
            filterData();
        });

        // Toggle View
        viewToggle.addEventListener('change', (e) => {
            state.viewMode = e.target.checked ? 'MACHINE' : 'HUMAN';

            // Visual glitch effect on toggle
            document.body.classList.add('glitch-active');
            setTimeout(() => document.body.classList.remove('glitch-active'), 200);

            render();
        });

        // Mock Upload
        document.getElementById('btnUpload').addEventListener('click', () => {
            alert('SYSTEM MESSAGE: Secure Upload Channel Established. Drop encrypted packets now.');
        });

        // Mock Export
        document.getElementById('btnExport').addEventListener('click', () => {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.filteredData));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "adam_intelligence_export.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        });
    }

    init();
});
