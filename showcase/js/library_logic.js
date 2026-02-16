/**
 * INTELLIGENCE LIBRARY LOGIC
 * Handles data fetching, rendering, and interaction for the Intelligence Library.
 */

const LibraryManager = {
    data: [],
    viewMode: 'human', // 'human' or 'machine'
    filters: {
        search: '',
        type: 'ALL',
        sentiment: 'ALL'
    },

    async init() {
        console.log("[LibraryManager] Initializing...");
        await this.fetchData();
        this.bindEvents();
        this.render();
    },

    async fetchData() {
        try {
            const response = await fetch('data/market_mayhem_index.json');
            if (!response.ok) throw new Error("Failed to load data");
            this.data = await response.json();
            console.log(`[LibraryManager] Loaded ${this.data.length} artifacts.`);
        } catch (e) {
            console.error("[LibraryManager] Data Fetch Error:", e);
            // Fallback mock data if fetch fails (for standalone testing)
            this.data = [
                { title: "System Error: Data Source Offline", date: new Date().toISOString(), type: "ERROR", summary: "Could not retrieve intelligence index.", sentiment_score: 0 }
            ];
        }
    },

    bindEvents() {
        // Search Input
        document.getElementById('searchInput')?.addEventListener('input', (e) => {
            this.filters.search = e.target.value.toLowerCase();
            this.render();
        });

        // Type Filter
        document.getElementById('typeFilter')?.addEventListener('change', (e) => {
            this.filters.type = e.target.value;
            this.render();
        });

        // Sentiment Filter
        document.getElementById('sentimentFilter')?.addEventListener('change', (e) => {
            this.filters.sentiment = e.target.value;
            this.render();
        });

        // View Toggle
        document.getElementById('viewToggle')?.addEventListener('change', (e) => {
            this.viewMode = e.target.checked ? 'machine' : 'human';
            this.toggleView();
        });

        // Upload Mock
        document.getElementById('btnUpload')?.addEventListener('click', () => {
            alert("[SYSTEM] UPLOAD INTERFACE ACTIVATED.\n\nSimulating secure file transfer to data lake...");
            setTimeout(() => alert("[SYSTEM] Transfer Complete. Indexing..."), 1000);
        });

        // Export Mock
        document.getElementById('btnExport')?.addEventListener('click', () => {
            const dataStr = JSON.stringify(this.getFilteredData(), null, 2);
            const blob = new Blob([dataStr], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `intelligence_export_${new Date().toISOString().slice(0,10)}.json`;
            a.click();
        });
    },

    getFilteredData() {
        return this.data.filter(item => {
            const matchSearch = (item.title || '').toLowerCase().includes(this.filters.search) ||
                                (item.summary || '').toLowerCase().includes(this.filters.search);
            const matchType = this.filters.type === 'ALL' || item.type === this.filters.type;

            let matchSentiment = true;
            if (this.filters.sentiment === 'POSITIVE') matchSentiment = item.sentiment_score > 60;
            if (this.filters.sentiment === 'NEGATIVE') matchSentiment = item.sentiment_score < 40;
            if (this.filters.sentiment === 'NEUTRAL') matchSentiment = item.sentiment_score >= 40 && item.sentiment_score <= 60;

            return matchSearch && matchType && matchSentiment;
        });
    },

    toggleView() {
        const humanView = document.getElementById('humanView');
        const machineView = document.getElementById('machineView');

        if (this.viewMode === 'human') {
            humanView.style.display = 'grid';
            machineView.style.display = 'none';
        } else {
            humanView.style.display = 'none';
            machineView.style.display = 'block';
        }
        this.render(); // Re-render to ensure filtered data is shown
    },

    render() {
        const filtered = this.getFilteredData();
        this.renderHuman(filtered);
        this.renderMachine(filtered);
        this.updateStats(filtered);
    },

    updateStats(data) {
        document.getElementById('countDisplay').innerText = `${data.length} ARTIFACTS`;
    },

    renderHuman(data) {
        const container = document.getElementById('humanView');
        if (!container) return;
        container.innerHTML = '';

        data.forEach(item => {
            const card = document.createElement('div');
            card.className = `intel-card type-${item.type}`;

            // Determine sentiment color
            let sentColor = '#94a3b8'; // gray
            if (item.sentiment_score > 60) sentColor = '#10b981'; // green
            if (item.sentiment_score < 40) sentColor = '#ef4444'; // red

            card.innerHTML = `
                <div class="card-meta">
                    <span class="mono">${item.date || 'UNKNOWN_DATE'}</span>
                    <span class="mono">${item.type || 'UNKNOWN'}</span>
                </div>
                <div class="card-title">${item.title || 'Untitled Artifact'}</div>
                <div class="card-summary">${item.summary || 'No summary available.'}</div>
                <div class="card-footer">
                    <span class="sentiment-badge mono" style="color: ${sentColor}; border: 1px solid ${sentColor}">
                        SENT: ${item.sentiment_score || 0}
                    </span>
                    <a href="${item.filename || '#'}" class="btn-cyber" style="font-size: 0.6rem; padding: 4px 8px;">
                        ACCESS <i class="fas fa-arrow-right"></i>
                    </a>
                </div>
            `;
            container.appendChild(card);
        });
    },

    renderMachine(data) {
        const container = document.getElementById('machineView');
        if (!container) return;

        let html = '<div class="terminal-line">[SYSTEM] RENDERING DATA STREAM...</div>';

        data.forEach((item, index) => {
            const json = JSON.stringify(item, null, 2)
                .replace(/"([^"]+)":/g, '<span class="json-key">"$1":</span>')
                .replace(/: "([^"]+)"/g, ': <span class="json-string">"$1"</span>')
                .replace(/: (\d+)/g, ': <span class="json-number">$1</span>');

            html += `
                <div class="terminal-line" style="margin-top: 10px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                    <span style="color: #666">Index [${index}]:</span>
                    <pre style="margin:0; white-space: pre-wrap;">${json}</pre>
                </div>
            `;
        });

        container.innerHTML = html;
    }
};

// Auto-boot
document.addEventListener('DOMContentLoaded', () => {
    LibraryManager.init();
});
