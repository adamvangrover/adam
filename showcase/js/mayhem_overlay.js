class MayhemOverlay {
    constructor() {
        this.data = null;
        this.isOpen = false;
        this.sortMode = 'relevance'; // relevance, date, importance
        this.portfolio = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "TechCo", "SafeHarbor"];
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

                <div class="sort-controls">
                    <span class="sort-label">PRIORITY:</span>
                    <button class="mayhem-sort-btn active" data-sort="relevance" onclick="window.mayhemOverlay.setSortMode('relevance')">RELEVANCE</button>
                    <button class="mayhem-sort-btn" data-sort="date" onclick="window.mayhemOverlay.setSortMode('date')">TIMELINE</button>
                    <button class="mayhem-sort-btn" data-sort="importance" onclick="window.mayhemOverlay.setSortMode('importance')">CONVICTION</button>
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
            // 1. Fetch Market Mayhem Reports
            let mayhemData = [];
            try {
                const response = await fetch('data/market_mayhem_index.json');
                if (response.ok) mayhemData = await response.json();
            } catch (e) { console.warn("Failed to load market_mayhem_index.json", e); }

            // 2. Fetch Credit Memos
            let creditData = [];
            try {
                const response = await fetch('data/credit_memo_library.json');
                if (response.ok) creditData = await response.json();
            } catch (e) { console.warn("Failed to load credit_memo_library.json", e); }

            // 3. Define Static Dashboards/Tools
            const staticTools = [
                {
                    title: "Sovereign Dashboard",
                    date: new Date().toISOString().split('T')[0],
                    type: "DASHBOARD",
                    summary: "Real-time sovereign risk monitoring and geopolitical conflict simulation.",
                    filename: "sovereign_dashboard.html",
                    sentiment_score: 50,
                    conviction: 90,
                    entities: { keywords: ["Sovereign", "Risk", "Geopolitics"] }
                },
                {
                    title: "Credit Memo Automation",
                    date: new Date().toISOString().split('T')[0],
                    type: "TOOL",
                    summary: "Enterprise-grade credit analysis automation suite.",
                    filename: "credit_memo_automation.html",
                    sentiment_score: 50,
                    conviction: 85,
                    entities: { keywords: ["Credit", "Automation", "Enterprise"] }
                },
                {
                    title: "Credit Memo V2",
                    date: new Date().toISOString().split('T')[0],
                    type: "TOOL",
                    summary: "Next-generation credit memo interface with advanced analytics.",
                    filename: "credit_memo_v2.html",
                    sentiment_score: 50,
                    conviction: 85,
                    entities: { keywords: ["Credit", "Analytics", "V2"] }
                }
            ];

            // 4. Normalize and Combine
            this.data = [
                ...mayhemData.map(d => this.normalizeData(d, 'MARKET_MAYHEM')),
                ...creditData.map(d => this.normalizeData(d, 'CREDIT_MEMO')),
                ...staticTools.map(d => this.normalizeData(d, 'STATIC'))
            ];

            this.renderArchives(this.data);
        } catch (e) {
            console.error("Mayhem Overlay: Failed to load data", e);
        }
    }

    normalizeData(item, source) {
        // Standardize structure: title, date, type, summary, url, score, entities
        let normalized = {
            title: item.title || item.borrower_name || "Untitled",
            date: item.date || item.report_date?.split('T')[0] || new Date().toISOString().split('T')[0],
            type: item.type || (source === 'CREDIT_MEMO' ? 'CREDIT_MEMO' : 'UNKNOWN'),
            summary: item.summary || "No summary available.",
            url: item.filename || item.file || "#",
            score: item.sentiment_score || item.risk_score || 50,
            conviction: item.conviction || 50,
            entities: item.entities || { keywords: [] },
            source: source
        };

        // Ensure entities is an object with keywords array
        if (!normalized.entities.keywords) {
             // If entities is just a list or keywords are missing, try to infer
             normalized.entities.keywords = [];
             if (item.ticker) normalized.entities.keywords.push(item.ticker);
             if (item.sector) normalized.entities.keywords.push(item.sector);
        }

        // For credit memos, add ticker to keywords if not present
        if (source === 'CREDIT_MEMO' && item.ticker) {
            if (!normalized.entities.keywords.includes(item.ticker)) {
                normalized.entities.keywords.push(item.ticker);
            }
        }

        return normalized;
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
            const scoreClass = (item.score > 60) ? 'sent-high' : (item.score < 40 ? 'sent-low' : 'sent-mid');
            const portfolioBadge = item._isPortfolio ? `<span class="portfolio-badge"><i class="fas fa-briefcase"></i> PORTFOLIO</span>` : '';
            const typeBadgeClass = item.type === 'DASHBOARD' ? 'type-dashboard' : (item.type === 'TOOL' ? 'type-tool' : 'type-report');

            return `
            <div class="mayhem-card ${item._isPortfolio ? 'card-portfolio' : ''}" onclick="window.location.href='${item.url}'">
                <div class="card-meta">
                    <span>${item.date}</span>
                    <span class="card-type ${typeBadgeClass}">${item.type}</span>
                </div>
                <div class="card-title">
                    ${item.title}
                    ${portfolioBadge}
                </div>
                <div class="card-summary">${item.summary || 'No summary available.'}</div>
                <div class="card-footer">
                    <div class="score-indicator">
                        <div class="sentiment-dot ${scoreClass}"></div>
                        <span class="score-text">SCORE: ${item.score}</span>
                    </div>
                    <span class="conviction-text">CONVICTION: ${item.conviction}%</span>
                </div>
            </div>
            `;
        }).join('');
    }

    handlePrompt(query) {
        if (!this.data) return;
        const lowerQ = query.toLowerCase();

        // 1. Filter Data
        let filtered = this.data.filter(item => {
            const inTitle = item.title && item.title.toLowerCase().includes(lowerQ);
            const inSummary = item.summary && item.summary.toLowerCase().includes(lowerQ);
            const inKeywords = item.entities.keywords && item.entities.keywords.some(k => k.toLowerCase().includes(lowerQ));
            return inTitle || inSummary || inKeywords;
        });

        // 2. Calculate Relevance Scores
        filtered.forEach(item => {
            item._relevance = this.calculateRelevance(item, lowerQ);
        });

        // 3. Sort
        filtered = this.sortData(filtered);

        this.renderArchives(filtered);

        // 4. Simulate Analysis if query is long enough
        const analysisBox = document.getElementById('mayhem-analysis');
        if (query.length > 5) {
            analysisBox.style.display = 'block';
            analysisBox.innerText = `> ANALYZING QUERY VECTOR: "${query}"\n> FOUND ${filtered.length} MATCHES.\n> SORTED BY: ${this.sortMode.toUpperCase()}`;
        } else {
            analysisBox.style.display = 'none';
        }
    }

    calculateRelevance(item, query) {
        let score = 0;

        // Query Match
        if (query) {
            if (item.title.toLowerCase().includes(query)) score += 10;
            if (item.entities.keywords.some(k => k.toLowerCase().includes(query))) score += 5;
        }

        // Portfolio Match
        const isPortfolio = item.entities.keywords.some(k => this.portfolio.includes(k));
        if (isPortfolio) {
            score += 20;
            item._isPortfolio = true;
        } else {
            item._isPortfolio = false;
        }

        // Recency (Simple decay)
        const date = new Date(item.date);
        const now = new Date();
        const diffDays = (now - date) / (1000 * 60 * 60 * 24);
        if (diffDays < 7) score += 15;
        else if (diffDays < 30) score += 10;
        else if (diffDays < 365) score += 5;

        // Importance/Conviction
        if (item.conviction > 80) score += 5;

        // Type Boost
        if (item.type === 'DASHBOARD') score += 50; // Dashboards always high relevance
        if (item.type === 'TOOL') score += 40;

        return score;
    }

    sortData(items) {
        return items.sort((a, b) => {
            if (this.sortMode === 'relevance') {
                return b._relevance - a._relevance;
            } else if (this.sortMode === 'date') {
                return new Date(b.date) - new Date(a.date);
            } else if (this.sortMode === 'importance') {
                return b.conviction - a.conviction;
            }
            return 0;
        });
    }

    setSortMode(mode) {
        this.sortMode = mode;
        const query = document.getElementById('mayhem-prompt').value;
        this.handlePrompt(query);

        // Update UI buttons (will be implemented in next step)
        document.querySelectorAll('.mayhem-sort-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.sort === mode);
        });
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
