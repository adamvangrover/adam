
/**
 * MARKET MAYHEM ARCHIVE V2 CONTROLLER (ENHANCED)
 * -----------------------------------------------------------------------------
 * Integrates Strategic Command, Market Data, and Archive Index.
 * Features: Time Filtering, Semantic Clouds, Alpha Calculation, System 2 Critique.
 * -----------------------------------------------------------------------------
 */

class MarketMayhemController {
    constructor() {
        this.data = {
            strategic: null,
            market: null,
            archive: null
        };
        this.dom = {
            strategicPanel: document.getElementById('strategicPanel'),
            convictionPanel: document.getElementById('convictionPanel'),
            watchListPanel: document.getElementById('watchListPanel'),
            sectorRiskPanel: document.getElementById('sectorRiskPanel'),
            archiveGrid: document.getElementById('archiveGrid'),
            featuredGrid: document.getElementById('featuredGrid'),
            semanticCloud: document.getElementById('semanticCloud'),
            modal: document.getElementById('reportModal')
        };
        this.state = {
            timePeriod: 'ALL',
            alphaChartInstance: null,
            mainChartInstance: null
        };
    }

    async init() {
        console.log("[MarketMayhem] Initializing V2 Controller...");

        try {
            await this.loadData();
            this.renderSidebar();
            this.renderChart();
            this.renderFeatured();
            this.renderArchive();
            this.renderSemanticCloud();
            this.setupEventHandlers();
            console.log("[MarketMayhem] Initialization Complete.");
        } catch (e) {
            console.error("[MarketMayhem] Initialization Failed:", e);
        }
    }

    async loadData() {
        const [strategic, market, archive] = await Promise.all([
            fetch('data/strategic_command.json').then(r => r.json()),
            fetch('data/sp500_market_data.json').then(r => r.json()),
            fetch('data/market_mayhem_index.json').then(r => r.json())
        ]);

        this.data.strategic = strategic;
        this.data.market = market;
        this.data.archive = archive;
    }

    // --- Sidebar Rendering ---

    renderSidebar() {
        // 1. Strategic Command
        if (this.data.strategic) {
            const { house_view, score, narrative } = this.data.strategic.strategic_directives;
            let color = '#f59e0b'; // Neutral
            if (house_view === 'BULLISH') color = '#0aff60';
            if (house_view === 'BEARISH') color = '#ff3333';

            this.dom.strategicPanel.innerHTML = `
                <h3>
                    <span>STRATEGIC COMMAND</span>
                    <i class="fas fa-chess-king"></i>
                </h3>
                <div style="margin-bottom: 10px;">
                    <span style="color: #666; font-size: 0.7rem;">HOUSE VIEW</span>
                    <div style="font-size: 1.1rem; color: ${color}; font-weight: bold;">${house_view}</div>
                </div>
                <div style="font-size: 0.75rem; line-height: 1.4; color: #ccc;">
                    ${narrative}
                </div>
            `;
        }

        // 2. Top Conviction
        if (this.data.market) {
            const convictions = this.data.market
                .filter(item => item.outlook && item.outlook.conviction === 'High')
                .sort((a, b) => b.risk_score - a.risk_score)
                .slice(0, 5);

            this.dom.convictionPanel.innerHTML = `
                <h3>
                    <span>TOP CONVICTION</span>
                    <i class="fas fa-star"></i>
                </h3>
                ${convictions.map(item => `
                    <div class="metric-row">
                        <span class="metric-label">${item.ticker} <span style="font-size:0.6rem; color:#555;">${item.sector ? item.sector.substring(0,8) : ''}</span></span>
                        <span class="metric-val ${item.change_pct >= 0 ? 'val-green' : 'val-red'}">${item.current_price}</span>
                    </div>
                `).join('')}
            `;
        }

        // 3. Watch List
        if (this.data.market) {
            const watchlist = this.data.market
                .filter(item => item.outlook && item.outlook.conviction === 'Medium')
                .slice(0, 5);

            this.dom.watchListPanel.innerHTML = `
                <h3>
                    <span>WATCH LIST</span>
                    <i class="fas fa-eye"></i>
                </h3>
                ${watchlist.map(item => `
                    <div class="metric-row">
                        <span class="metric-label">${item.ticker}</span>
                        <span class="metric-val ${item.change_pct >= 0 ? 'val-green' : 'val-red'}">${item.change_pct}%</span>
                    </div>
                `).join('')}
            `;
        }

        // 4. Sector Risk
        if (this.data.market) {
            const sectorRisk = {};
            const sectorCount = {};

            this.data.market.forEach(item => {
                if (!sectorRisk[item.sector]) {
                    sectorRisk[item.sector] = 0;
                    sectorCount[item.sector] = 0;
                }
                sectorRisk[item.sector] += item.risk_score || 50;
                sectorCount[item.sector]++;
            });

            const sortedSectors = Object.keys(sectorRisk)
                .map(sector => ({
                    sector,
                    avgRisk: Math.round(sectorRisk[sector] / sectorCount[sector])
                }))
                .sort((a, b) => b.avgRisk - a.avgRisk)
                .slice(0, 6);

            this.dom.sectorRiskPanel.innerHTML = `
                <h3>
                    <span>SECTOR RISK</span>
                    <i class="fas fa-exclamation-triangle"></i>
                </h3>
                <div class="risk-grid" style="display:grid; grid-template-columns: 1fr 1fr; gap:5px;">
                    ${sortedSectors.map(item => `
                        <div style="background:rgba(255,255,255,0.05); padding:5px; border-radius:3px; text-align:center;">
                            <div style="color: ${item.avgRisk > 80 ? '#ff3333' : '#f59e0b'}; font-size:0.7rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.sector ? item.sector.split(' ')[0] : 'Unknown'}</div>
                            <div style="font-size: 0.9rem; font-weight: bold;">${item.avgRisk}</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }
    }

    // --- Charting ---

    renderChart() {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const houseView = this.data.strategic ? this.data.strategic.strategic_directives.house_view : 'NEUTRAL';

        let dataPoints = 30;
        if (this.state.timePeriod === '1M') dataPoints = 30;
        if (this.state.timePeriod === '3M') dataPoints = 90;
        if (this.state.timePeriod === '1Y') dataPoints = 100; // Condensed

        // Synthetic History based on filtered Archive Sentiment
        const history = this.generateSentimentHistory(dataPoints);
        const lastVal = history[history.length - 1];

        // Projections
        const projectionSteps = Math.round(dataPoints / 3);
        const projections = [];
        let drift = 0;
        if (houseView === 'BULLISH') drift = 1.0;
        if (houseView === 'BEARISH') drift = -1.0;

        let current = lastVal;
        for(let i=0; i<projectionSteps; i++) {
            current += drift + (Math.random() * 4 - 2);
            // Bounds check
            if (current > 100) current = 100;
            if (current < 0) current = 0;
            projections.push(current);
        }

        // Config
        if (this.state.mainChartInstance) this.state.mainChartInstance.destroy();

        this.state.mainChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [...Array(history.length).fill('').map((_,i)=>i), ...Array(projections.length).fill('').map((_,i)=>'Proj')],
                datasets: [
                    {
                        label: 'Historical Sentiment',
                        data: [...history, ...Array(projections.length).fill(null)],
                        borderColor: '#00f3ff',
                        backgroundColor: 'rgba(0, 243, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0
                    },
                    {
                        label: 'Projected Outlook',
                        data: [...Array(history.length).fill(null), lastVal, ...projections],
                        borderColor: '#d946ef',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: {
                    legend: { labels: { color: '#ccc', font: { family: 'JetBrains Mono' } } }
                },
                scales: {
                    y: { grid: { color: '#222' }, ticks: { color: '#666' }, min: 0, max: 100 },
                    x: { grid: { color: '#222' }, ticks: { display: false } }
                }
            }
        });
    }

    generateSentimentHistory(points) {
        // Try to build from real data if available
        if (!this.data.archive) return Array.from({length: points}, () => 50);

        // Sort archive by date ascending
        const sorted = [...this.data.archive]
            .filter(item => item.date && item.sentiment_score)
            .sort((a, b) => new Date(a.date) - new Date(b.date));

        if (sorted.length === 0) return Array.from({length: points}, () => 50);

        // Take the last N points or interpolate
        const result = [];
        const step = Math.max(1, Math.floor(sorted.length / points));

        for (let i = 0; i < sorted.length; i += step) {
            result.push(sorted[i].sentiment_score);
        }

        // Pad or Trim
        return result.slice(-points);
    }

    // --- Featured Section ---

    renderFeatured() {
        if (!this.data.archive) return;

        // Rank by (Quality + Conviction + Sentiment)
        const featured = this.data.archive
            .map(item => ({
                ...item,
                score: (item.quality || 0) + (item.conviction || 0) + (Math.abs(item.sentiment_score - 50))
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);

        this.dom.featuredGrid.innerHTML = featured.map(item => `
            <div class="featured-card" onclick="window.controller.openModal('${item.filename}')">
                <span class="featured-badge">TOP INTEL</span>
                <div style="font-size:0.9rem; font-weight:bold; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    ${item.title}
                </div>
                <div style="font-size:0.7rem; color:#888; display:flex; justify-content:space-between;">
                    <span>${item.date}</span>
                    <span style="color:var(--accent-green);">CONV: ${item.conviction}%</span>
                </div>
            </div>
        `).join('');
    }

    // --- Semantic Cloud ---

    renderSemanticCloud() {
        if (!this.data.archive) return;

        const counts = {};
        this.data.archive.forEach(item => {
            if (item.entities && item.entities.keywords) {
                item.entities.keywords.forEach(kw => {
                    counts[kw] = (counts[kw] || 0) + 1;
                });
            }
        });

        const sortedTags = Object.keys(counts)
            .map(key => ({ text: key, count: counts[key] }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 15);

        this.dom.semanticCloud.innerHTML = sortedTags.map(tag => `
            <span class="tag" onclick="document.getElementById('searchInput').value='${tag.text}'; window.applyFilters();">
                ${tag.text} <span style="opacity:0.5;">${tag.count}</span>
            </span>
        `).join('');
    }

    // --- Archive List ---

    renderArchive() {
        if (!this.data.archive) return;
        this.dom.archiveGrid.innerHTML = '';

        this.data.archive.forEach(item => {
            const el = document.createElement('div');
            el.className = `archive-item type-${item.type || 'NEWSLETTER'}`;

            el.dataset.year = item.date ? item.date.substring(0, 4) : 'HISTORICAL';
            el.dataset.type = item.type || 'NEWSLETTER';
            el.dataset.title = item.title;
            el.dataset.conviction = item.conviction || 0;
            el.dataset.sentiment = item.sentiment_score || 50;

            const sentimentColor = item.sentiment_score > 60 ? 'var(--accent-green)' : (item.sentiment_score < 40 ? 'var(--accent-red)' : '#888');

            el.innerHTML = `
                <div style="flex-grow: 1;" onclick="window.controller.openModal('${item.filename}')">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">${item.date || 'Unknown'}</span>
                        <span class="type-badge">${item.type || 'ARCHIVE'}</span>
                        <span class="mono" style="font-size:0.7rem; color:${sentimentColor};">SENT: ${item.sentiment_score || '-'}</span>
                        <span class="mono" style="font-size:0.7rem; color:#aaa;">CONV: ${item.conviction || '-'}</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem; color:#fff;">${item.title}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#888;">${item.summary || 'Click to view full intelligence report...'}</p>
                </div>
                <button onclick="window.controller.openModal('${item.filename}')" class="cyber-btn" style="align-self:center;">
                    <i class="fas fa-search"></i> VIEW
                </button>
            `;

            this.dom.archiveGrid.appendChild(el);
        });
    }

    // --- Modal & Alpha Logic ---

    openModal(filename) {
        const report = this.data.archive.find(r => r.filename === filename);
        if (!report) return;

        // Populate Main Content
        const modalMain = document.getElementById('modalMain');
        modalMain.innerHTML = `
            <h2 style="margin-top:0; color:var(--accent-blue);">${report.title}</h2>
            <div style="margin-bottom:20px; border-bottom:1px solid #333; padding-bottom:10px; font-family:var(--font-mono); font-size:0.8rem; color:#666;">
                DATE: ${report.date} | AGENT: ${report.type} | HASH: ${report.provenance_hash ? report.provenance_hash.substring(0,12) : 'N/A'}...
            </div>
            <div style="line-height:1.6; color:#ddd;">
                ${report.full_body || report.summary}
            </div>
        `;

        // Populate Sidebar Critique
        const sentimentColor = (report.sentiment_score || 50) > 60 ? '#0aff60' : ((report.sentiment_score || 50) < 40 ? '#ff3333' : '#f59e0b');
        const convictionColor = (report.conviction || 0) > 75 ? '#0aff60' : '#f59e0b';

        const gaugesHTML = `
            <div style="margin-top: 15px;">
                <div style="margin-bottom: 5px; font-size: 0.7rem; color: #888;">SENTIMENT: ${report.sentiment_score || 50}/100</div>
                <div style="width: 100%; background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px; overflow: hidden;">
                    <div style="width: ${report.sentiment_score || 50}%; background: ${sentimentColor}; height: 100%;"></div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="margin-bottom: 5px; font-size: 0.7rem; color: #888;">CONVICTION: ${report.conviction || 0}%</div>
                <div style="width: 100%; background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px; overflow: hidden;">
                    <div style="width: ${report.conviction || 0}%; background: ${convictionColor}; height: 100%;"></div>
                </div>
            </div>
        `;

        document.getElementById('modalCritique').innerHTML = `
            <div style="margin-bottom:10px;"><strong>VERDICT:</strong> ${(report.conviction || 0) > 75 ? 'HIGH CONVICTION' : 'REVIEW REQUIRED'}</div>
            <div style="font-size:0.75rem; margin-bottom: 10px;">${report.critique || 'System 2 analysis pending...'}</div>
            ${gaugesHTML}
        `;

        // Alpha Calculation
        this.renderAlphaChart(report);

        // Metadata
        const metaHTML = `
            <div style="margin-top:20px;">
                <h4 style="color:#888; font-size:0.8rem;">ENTITIES</h4>
                <div style="display:flex; flex-wrap:wrap; gap:5px;">
                    ${report.entities && report.entities.keywords ? report.entities.keywords.map(k => `<span class="tag">${k}</span>`).join('') : ''}
                </div>
            </div>
        `;
        document.getElementById('modalMeta').innerHTML = metaHTML;

        this.dom.modal.style.display = 'flex';
    }

    renderAlphaChart(report) {
        const ctx = document.getElementById('alphaChart').getContext('2d');
        if (this.state.alphaChartInstance) this.state.alphaChartInstance.destroy();

        // 1. Identify Tickers mentioned
        const tickers = [];
        // Heuristic: check keywords for known tickers in market data
        if (report.entities && report.entities.tickers && report.entities.tickers.length > 0) {
            tickers.push(...report.entities.tickers);
        } else if (this.data.market) {
            // scan summary/title
            const text = (report.title + ' ' + report.summary).toUpperCase();
            this.data.market.forEach(m => {
                if (text.includes(m.ticker)) tickers.push(m.ticker);
            });
        }

        // If no tickers found, show a generic "System Alpha" or blank
        if (tickers.length === 0) {
            document.getElementById('alphaStats').innerText = "No specific tickers linked.";
            // Draw a flat line or placeholder
            this.state.alphaChartInstance = new Chart(ctx, {
                type: 'line',
                data: { labels: ['Start', 'End'], datasets: [] },
                options: { plugins: { legend: { display: false } } }
            });
            return;
        }

        const primaryTicker = tickers[0]; // Just take first for simplicity
        const marketItem = this.data.market.find(m => m.ticker === primaryTicker);

        if (!marketItem || !marketItem.price_history) {
             document.getElementById('alphaStats').innerText = `Data unavailable for ${primaryTicker}`;
             return;
        }

        // Simulate "Since Report Date"
        // Since we don't have exact dates in price_history array, we take the last 30 points as proxy for "recent performance"
        // In a real system, we'd map report.date to an index.
        const history = marketItem.price_history.slice(-30);
        const labels = history.map((_, i) => i);

        // Calculate return
        const startPrice = history[0];
        const endPrice = history[history.length - 1];
        const returnPct = ((endPrice - startPrice) / startPrice * 100).toFixed(2);
        const color = returnPct >= 0 ? '#0aff60' : '#ff3333';

        document.getElementById('alphaStats').innerHTML = `
            ${primaryTicker}: <span style="color:${color}">${returnPct > 0 ? '+' : ''}${returnPct}%</span> (30d Realized)
        `;

        this.state.alphaChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: primaryTicker,
                    data: history,
                    borderColor: color,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false } // Sparkline style
                }
            }
        });
    }

    // --- Filters ---

    setupEventHandlers() {
        // Time Period Buttons
        const buttons = document.getElementById('timePeriodControls').getElementsByTagName('button');
        Array.from(buttons).forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Toggle active class
                Array.from(buttons).forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');

                // Update State & Chart
                this.state.timePeriod = e.target.dataset.period;
                this.renderChart();
            });
        });

        // Global Filter Function
        window.applyFilters = () => {
            const search = document.getElementById('searchInput').value.toLowerCase();
            
            const yearSelect = document.getElementById('yearFilter');
            const typeSelect = document.getElementById('typeFilter');
            const sortSelect = document.getElementById('sortFilter');

            const yearValue = yearSelect ? yearSelect.value : 'all';
            const typeValue = typeSelect ? typeSelect.value : 'all';
            const sortValue = sortSelect ? sortSelect.value : 'newest';

            const grid = document.getElementById('archiveGrid');
            const items = Array.from(grid.getElementsByClassName('archive-item'));

            // Sort
            items.sort((a, b) => {
                if (sortValue === 'newest') return b.dataset.year.localeCompare(a.dataset.year);
                if (sortValue === 'oldest') return a.dataset.year.localeCompare(b.dataset.year);
                // For conviction/sentiment, handle descending
                if (sortValue === 'conviction') return (b.dataset.conviction || 0) - (a.dataset.conviction || 0);
                if (sortValue === 'sentiment') return (b.dataset.sentiment || 0) - (a.dataset.sentiment || 0);
                return 0;
            });

            // Re-append to update DOM order
            items.forEach(item => grid.appendChild(item));

            // Filter
            items.forEach(item => {
                const itemType = item.dataset.type;
                const itemYear = item.dataset.year;
                const itemTitle = item.dataset.title.toLowerCase();

                let matchSearch = itemTitle.includes(search);
                let matchType = typeValue === 'all' || itemType === typeValue;
                let matchYear = yearValue === 'all' || itemYear === yearValue;

                item.style.display = (matchSearch && matchType && matchYear) ? 'flex' : 'none';
            });
        };

        // Attach listeners to selects
        ['yearFilter', 'typeFilter', 'sortFilter', 'searchInput'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener(id === 'searchInput' ? 'input' : 'change', window.applyFilters);
            }
        });
    }
}

// Launch
document.addEventListener('DOMContentLoaded', () => {
    window.controller = new MarketMayhemController();
    window.controller.init();
});
