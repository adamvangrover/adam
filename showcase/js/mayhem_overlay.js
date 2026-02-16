class MayhemOverlay {
    constructor() {
        this.data = null;
        this.isOpen = false;
    }

    init() {
        if (document.getElementById('mayhem-overlay')) return;

        const overlay = document.createElement('div');
        overlay.id = 'mayhem-overlay';
        overlay.innerHTML = `
            <div class="mayhem-header">
                <div class="mayhem-title">
                    <span><i class="fas fa-brain"></i> NEURAL ARCHIVE LINK</span>
                    <button class="mayhem-close-btn" onclick="window.mayhemOverlay.toggle()">CLOSE [X]</button>
                </div>
                <div class="prompt-container">
                    <input type="text" id="mayhem-prompt" placeholder="Query the archive (e.g. 'Show me energy crisis reports from 2026')..." autocomplete="off">
                    <i class="fas fa-bolt prompt-icon"></i>
                </div>
                <div id="mayhem-analysis"></div>
            </div>
            <div class="mayhem-grid" id="mayhem-grid">
                <!-- Cards injected here -->
            </div>
        `;
        document.body.appendChild(overlay);

        // Bind Events
        document.getElementById('mayhem-prompt').addEventListener('keyup', (e) => {
            this.handlePrompt(e.target.value);
        });

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.toggle();
        });

        this.loadData();
    }

    async loadData() {
        try {
            const response = await fetch('data/market_mayhem_index.json');
            this.data = await response.json();
            this.renderArchives(this.data);
        } catch (e) {
            console.error("Mayhem Overlay: Failed to load data", e);
        }
    }

    toggle() {
        const overlay = document.getElementById('mayhem-overlay');
        this.isOpen = !this.isOpen;
        if (this.isOpen) {
            overlay.classList.add('active');
            document.getElementById('mayhem-prompt').focus();
        } else {
            overlay.classList.remove('active');
        }
    }

    renderArchives(items) {
        const grid = document.getElementById('mayhem-grid');
        grid.innerHTML = items.map(item => {
            const sentClass = (item.sentiment_score > 60) ? 'sent-high' : (item.sentiment_score < 40 ? 'sent-low' : 'sent-mid');
            return `
            <div class="mayhem-card" onclick="window.location.href='${item.filename}'">
                <div class="card-meta">
                    <span>${item.date}</span>
                    <span class="card-type">${item.type}</span>
                </div>
                <div class="card-title">${item.title}</div>
                <div class="card-summary">${item.summary || 'No summary available.'}</div>
                <div class="card-footer">
                    <div class="sentiment-dot ${sentClass}" title="Sentiment: ${item.sentiment_score}"></div>
                    <span style="font-size:0.7rem; color:#666;">CONVICTION: ${item.conviction || 50}%</span>
                </div>
            </div>
            `;
        }).join('');
    }

    handlePrompt(query) {
        if (!this.data) return;
        const lowerQ = query.toLowerCase();

        // 1. Filter Data
        const filtered = this.data.filter(item =>
            (item.title && item.title.toLowerCase().includes(lowerQ)) ||
            (item.summary && item.summary.toLowerCase().includes(lowerQ)) ||
            (item.keywords && item.keywords.some(k => k.toLowerCase().includes(lowerQ)))
        );

        this.renderArchives(filtered);

        // 2. Simulate Analysis if query is long enough
        const analysisBox = document.getElementById('mayhem-analysis');
        if (query.length > 5) {
            analysisBox.style.display = 'block';
            analysisBox.innerText = `> ANALYZING QUERY VECTOR: "${query}"\n> FOUND ${filtered.length} MATCHES.\n> SENTIMENT VARIANCE DETECTED.`;
        } else {
            analysisBox.style.display = 'none';
        }
    }
}

// Global Instance
window.mayhemOverlay = new MayhemOverlay();

// Auto-init if DOM is ready (for dynamic injection)
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    window.mayhemOverlay.init();
} else {
    document.addEventListener('DOMContentLoaded', () => {
        window.mayhemOverlay.init();
    });
}
