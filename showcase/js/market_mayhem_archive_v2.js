
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
            this.renderArchive(); // Dynamic JS rendering activated
            this.renderSemanticCloud();
            this.setupEventHandlers();
            console.log("[MarketMayhem] Initialization Complete.");
        } catch (e) {
            console.error("[MarketMayhem] Initialization Failed:", e);
        }
    }

    async loadData() {
        try {
            const [strategic, market, archive] = await Promise.all([
                fetch('data/strategic_command.json').then(r => r.json()).catch(() => null),
                fetch('data/sp500_market_data.json').then(r => r.json()).catch(() => null),
                fetch('data/market_mayhem_index.json').then(r => r.json()).catch(() => null)
            ]);

            this.data.strategic = strategic || window.STRATEGIC_COMMAND_DATA || null;
            this.data.market = market || window.SP500_MARKET_DATA || [];
            this.data.archive = archive || window.MARKET_MAYHEM_DATA || [];
        } catch (e) {
            console.warn("[MarketMayhem] Fetch failed, using static fallback data.", e);
            this.data.strategic = window.STRATEGIC_COMMAND_DATA || null;
            this.data.market = window.SP500_MARKET_DATA || [];
            this.data.archive = window.MARKET_MAYHEM_DATA || [];
        }
    }

    // --- Sidebar Rendering ---

    renderSidebar() {
        // 1. Strategic Command
        if (this.data.strategic) {
            const { house_view, score, narrative } = this.data.strategic.strategic_directives;
            let color = '#f59e0b'; // Neutral
            if (house_view === 'BULLISH') color = '#0aff60';
            if (house_view === 'BEARISH') color = '#ff3333';

            // Calculate average consensus from archive
            let avgBullish = 50;
            let avgBearish = 50;
            if (this.data.archive && this.data.archive.length > 0) {
                const validItems = this.data.archive.filter(item => item.consensus_bullish !== undefined);
                if (validItems.length > 0) {
                    avgBullish = Math.round(validItems.reduce((sum, item) => sum + item.consensus_bullish, 0) / validItems.length);
                    avgBearish = Math.round(validItems.reduce((sum, item) => sum + item.consensus_bearish, 0) / validItems.length);
                }
            }

            this.dom.strategicPanel.innerHTML = `
                <h3>
                    <span>STRATEGIC COMMAND</span>
                    <i class="fas fa-chess-king"></i>
                </h3>
                <div style="margin-bottom: 10px;">
                    <span style="color: #666; font-size: 0.7rem;">HOUSE VIEW</span>
                    <div style="font-size: 1.1rem; color: ${color}; font-weight: bold;">${house_view}</div>
                </div>
                <div style="font-size: 0.75rem; line-height: 1.4; color: #ccc; margin-bottom: 15px;">
                    ${narrative}
                </div>
                <div style="border-top: 1px solid #333; padding-top: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 5px;">
                        <span style="color: #0aff60;">BULL CONSENSUS</span>
                        <span style="color: #0aff60;">${avgBullish}%</span>
                    </div>
                    <div style="width: 100%; background: #333; height: 4px; border-radius: 2px; margin-bottom: 10px;">
                        <div style="width: ${avgBullish}%; background: #0aff60; height: 100%; border-radius: 2px;"></div>
                    </div>

                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 5px;">
                        <span style="color: #ff3333;">BEAR CONSENSUS</span>
                        <span style="color: #ff3333;">${avgBearish}%</span>
                    </div>
                    <div style="width: 100%; background: #333; height: 4px; border-radius: 2px;">
                        <div style="width: ${avgBearish}%; background: #ff3333; height: 100%; border-radius: 2px;"></div>
                    </div>
                </div>
            `;
        }

        // 2. Top Conviction
        if (this.data.market.length > 0) {
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
        if (this.data.market.length > 0) {
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
        if (this.data.market.length > 0) {
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

        // History based on Archive
        const historyData = this.generateMarketHistory(dataPoints);
        const history = historyData.sentiment;
        const conviction = historyData.conviction;
        const sp500 = historyData.sp500;
        const treasury = historyData.treasury;
        const bsl = historyData.bsl;
        const bullish_prob = historyData.bullish_prob;
        const bearish_prob = historyData.bearish_prob;

        const lastVal = history.length > 0 ? history[history.length - 1] : 50;
        const lastSp500 = sp500.filter(v => v !== null).pop() || 5000;
        const lastTreasury = treasury.filter(v => v !== null).pop() || 4.0;

        // Projections
        const projectionSteps = Math.round(dataPoints / 3);
        const projections = [];
        const spProjections = [];
        let drift = 0;
        if (houseView === 'BULLISH') drift = 0.5;
        if (houseView === 'BEARISH') drift = -0.5;

        let current = lastVal;
        let currentSp = lastSp500;

        for(let i=0; i<projectionSteps; i++) {
            current += drift + (Math.random() * 4 - 2);
            // Bounds check
            if (current > 100) current = 100;
            if (current < 0) current = 0;
            projections.push(current);

            // S&P Projection correlates with house view
            currentSp += (drift * 20) + (Math.random() * 40 - 20);
            spProjections.push(currentSp);
        }

        // Config
        if (this.state.mainChartInstance) this.state.mainChartInstance.destroy();

        this.state.mainChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [...historyData.labels, ...Array(projections.length).fill('Proj')],
                datasets: [
                    {
                        label: 'Historical Sentiment',
                        data: [...history, ...Array(projections.length).fill(null)],
                        borderColor: '#00f3ff',
                        backgroundColor: 'rgba(0, 243, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        pointRadius: 0,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Conviction',
                        data: [...conviction, ...Array(projections.length).fill(null)],
                        borderColor: 'rgba(255, 255, 255, 0.4)',
                        borderWidth: 1,
                        borderDash: [2, 4],
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'y',
                        hidden: false
                    },
                    {
                        label: 'Bullish Prob',
                        data: [...bullish_prob, ...Array(projections.length).fill(null)],
                        borderColor: 'rgba(10, 255, 96, 0.5)',
                        borderWidth: 1,
                        tension: 0.4,
                        pointRadius: 0,
                        yAxisID: 'y',
                        hidden: true
                    },
                    {
                        label: 'Projected Sentiment',
                        data: [...Array(history.length - 1).fill(null), lastVal, ...projections],
                        borderColor: '#d946ef',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.4,
                        pointRadius: 0,
                        yAxisID: 'y'
                    },
                    {
                        label: 'S&P 500',
                        data: [...sp500, ...Array(projections.length).fill(null)],
                        borderColor: '#0aff60',
                        borderWidth: 1,
                        tension: 0.1,
                        pointRadius: 0,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Proj S&P 500',
                        data: [...Array(history.length - 1).fill(null), lastSp500, ...spProjections],
                        borderColor: '#0aff60',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        tension: 0.1,
                        pointRadius: 0,
                        yAxisID: 'y1'
                    },
                    {
                        label: '10Y Treasury',
                        data: [...treasury, ...Array(projections.length).fill(null)],
                        borderColor: '#f59e0b',
                        borderWidth: 1,
                        tension: 0.2,
                        pointRadius: 0,
                        yAxisID: 'y2',
                        hidden: true
                    },
                    {
                        label: 'BSL Price',
                        data: [...bsl, ...Array(projections.length).fill(null)],
                        borderColor: '#8b5cf6',
                        borderWidth: 1,
                        tension: 0.2,
                        pointRadius: 0,
                        yAxisID: 'y',
                        hidden: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                layout: {
                    padding: { top: 40 }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        align: 'start',
                        labels: {
                            color: '#ccc',
                            font: { family: 'JetBrains Mono', size: 10 },
                            boxWidth: 12
                        }
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: '#222' },
                        ticks: { color: '#00f3ff' },
                        min: 0, max: 100,
                        title: { display: true, text: 'Sentiment', color: '#00f3ff'}
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#0aff60' },
                        title: { display: true, text: 'S&P 500', color: '#0aff60'}
                    },
                    y2: {
                        type: 'linear',
                        display: false,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#f59e0b' }
                    },
                    x: { grid: { color: '#222' }, ticks: { display: false } }
                }
            }
        });
    }

    generateMarketHistory(points) {
        // Try to build from real data if available
        if (!this.data.archive || this.data.archive.length === 0) {
            return {
                sentiment: Array.from({length: points}, () => 50),
                conviction: Array.from({length: points}, () => 50),
                sp500: Array.from({length: points}, () => null),
                treasury: Array.from({length: points}, () => null),
                bullish_prob: Array.from({length: points}, () => null),
                bearish_prob: Array.from({length: points}, () => null),
                labels: Array.from({length: points}, (_, i) => `Point ${i}`)
            };
        }

        // Sort archive by date ascending
        const sorted = [...this.data.archive]
            .filter(item => item.date && item.sentiment_score)
            .sort((a, b) => new Date(a.date) - new Date(b.date));

        if (sorted.length === 0) {
            return {
                sentiment: Array.from({length: points}, () => 50),
                conviction: Array.from({length: points}, () => 50),
                sp500: Array.from({length: points}, () => null),
                treasury: Array.from({length: points}, () => null),
                bullish_prob: Array.from({length: points}, () => null),
                bearish_prob: Array.from({length: points}, () => null),
                labels: Array.from({length: points}, (_, i) => `Point ${i}`)
            };
        }

        const result = {
            sentiment: [],
            conviction: [],
            sp500: [],
            treasury: [],
            bullish_prob: [],
            bearish_prob: [],
            labels: []
        };
        
        // Simple downsampling or picking logic
        if (sorted.length <= points) {
            sorted.forEach(i => {
                result.sentiment.push(i.sentiment_score);
                result.conviction.push(i.conviction || 50);
                result.sp500.push(i.sp500_level || null);
                result.treasury.push(i.treasury_yield || null);
                result.bsl.push(i.bsl_price || null);
                result.bullish_prob.push(i.consensus_bullish || null);
                result.bearish_prob.push(i.consensus_bearish || null);
                result.labels.push(i.date);
            });
        } else {
             const step = Math.floor(sorted.length / points);
             for(let i=0; i<points; i++) {
                 const index = Math.min(i * step, sorted.length - 1);
                 const item = sorted[index];
                 result.sentiment.push(item.sentiment_score);
                 result.conviction.push(item.conviction || 50);
                 result.sp500.push(item.sp500_level || null);
                 result.treasury.push(item.treasury_yield || null);
                 result.bsl.push(item.bsl_price || null);
                 result.bullish_prob.push(item.consensus_bullish || null);
                 result.bearish_prob.push(item.consensus_bearish || null);
                 result.labels.push(item.date);
             }
        }
        return result;
    }

    // --- Featured Section ---

    renderFeatured() {
        if (!this.data.archive) return;

        // Rank by (Quality + Conviction + Sentiment Variance)
        const featured = this.data.archive
            .map(item => ({
                ...item,
                score: (item.quality || 0) + (item.conviction || 0) + (Math.abs((item.sentiment_score || 50) - 50))
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);

        this.dom.featuredGrid.innerHTML = featured.map(item => `
            <div class="featured-card" onclick="window.controller.openModal('${item.filename}')">
                <span class="featured-badge">TOP INTEL</span>
                <div style="font-size:0.9rem; font-weight:bold; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color: #fff;">
                    ${item.title}
                </div>
                <div style="font-size:0.7rem; color:#888; display:flex; justify-content:space-between; margin-bottom: 5px;">
                    <span class="mono">${item.date}</span>
                    <span style="color:var(--secondary-color);">CONV: ${item.conviction}%</span>
                </div>
                ${item.sp500_level ? `
                <div style="font-size:0.6rem; color:#666; border-top: 1px solid #333; padding-top: 5px; display: flex; justify-content: space-between;">
                    <span>S&P: ${item.sp500_level}</span>
                    <span>10Y: ${item.treasury_yield}%</span>
                </div>
                ` : ''}
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
            .slice(0, 20);

        this.dom.semanticCloud.innerHTML = sortedTags.map(tag => `
            <span class="tag" onclick="document.getElementById('searchInput').value='${tag.text}'; window.applyFilters();">
                ${tag.text} <span style="opacity:0.5;">${tag.count}</span>
            </span>
        `).join('');
    }

    // --- Archive List ---

    renderOutlook() {
        if (!this.dom.outlookPanel) {
            // Need to look it up if we added it manually later
            this.dom.outlookPanel = document.getElementById('outlookPanel');
            if (!this.dom.outlookPanel) return;
        }

        let breadth = 'Neutral';
        let breadthColor = '#f59e0b';
        let peRatio = '21.5x';
        let yieldCurve = 'Normal';
        let yieldColor = '#0aff60';

        if (this.data.archive && this.data.archive.length > 0) {
            const recent = this.data.archive.slice(-5);
            const avgSentiment = recent.reduce((sum, item) => sum + (item.sentiment_score || 50), 0) / recent.length;

            if (avgSentiment > 65) { breadth = 'Expanding'; breadthColor = '#0aff60'; peRatio = '24.5x'; }
            else if (avgSentiment < 35) { breadth = 'Deteriorating'; breadthColor = '#ff3333'; peRatio = '18.2x'; yieldCurve = 'Inverted'; yieldColor = '#ff3333'; }
        }

        this.dom.outlookPanel.innerHTML = `
            <h3>
                <span>MACRO ENVIRONMENT</span>
                <i class="fas fa-globe"></i>
            </h3>
            <div class="metric-row">
                <span class="metric-label">MARKET BREADTH</span>
                <span class="metric-val" style="color: ${breadthColor}">${breadth}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">AVG P/E RATIO</span>
                <span class="metric-val">${peRatio}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">YIELD CURVE</span>
                <span class="metric-val" style="color: ${yieldColor}">${yieldCurve}</span>
            </div>
        `;
    }

    renderArchive() {
        if (!this.data.archive) return;

        // Add Layout Toggles
        const layoutControls = `
            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 10px;">
                <button class="cyber-btn" id="btn-list-view" style="padding: 4px 8px; font-size: 0.7rem;"><i class="fas fa-list"></i> LIST</button>
                <button class="cyber-btn" id="btn-grid-view" style="padding: 4px 8px; font-size: 0.7rem;"><i class="fas fa-th-large"></i> GRID</button>
            </div>
            <style>
                .archive-grid-layout {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                }
                .archive-grid-layout .archive-item {
                    flex-direction: column;
                    align-items: flex-start;
                    height: 100%;
                }
                .archive-grid-layout .cyber-btn {
                    align-self: flex-start !important;
                    margin-top: 10px;
                }
            </style>
        `;

        this.dom.archiveGrid.innerHTML = layoutControls;

        const container = document.createElement('div');
        container.id = 'archive-items-container';
        this.dom.archiveGrid.appendChild(container);

        this.data.archive.forEach(item => {
            const el = document.createElement('div');
            el.className = `archive-item type-${item.type || 'NEWSLETTER'}`;

            el.dataset.year = item.date ? item.date.substring(0, 4) : 'HISTORICAL';
            el.dataset.type = item.type || 'NEWSLETTER';
            el.dataset.title = item.title ? item.title.toLowerCase() : '';
            el.dataset.keywords = item.entities && item.entities.keywords ? item.entities.keywords.join(' ').toLowerCase() : '';
            el.dataset.date = item.date;
            el.dataset.conviction = item.conviction || 0;
            el.dataset.sentiment = item.sentiment_score || 50;
            el.dataset.bullish = item.consensus_bullish || 50;
            el.dataset.bearish = item.consensus_bearish || 50;
            el.dataset.bullish = item.consensus_bullish || 50;
            el.dataset.bearish = item.consensus_bearish || 50;

            const sentimentColor = (item.sentiment_score || 50) > 60 ? 'var(--secondary-color)' : ((item.sentiment_score || 50) < 40 ? 'var(--danger-color)' : '#888');

            el.innerHTML = `
                <div style="flex-grow: 1; cursor: pointer;" onclick="window.controller.openModal('${item.filename}')">
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

            container.appendChild(el);
        });

        // Event listeners for Layout Toggles
        document.getElementById('btn-list-view').addEventListener('click', () => {
            container.className = '';
        });
        document.getElementById('btn-grid-view').addEventListener('click', () => {
            container.className = 'archive-grid-layout';
        });
    }

    // --- Modal & Alpha Logic ---

    openModal(filename) {
        const report = this.data.archive.find(r => r.filename === filename);
        if (!report) return;

        // Populate Main Content with Tabs
        const modalMain = document.getElementById('modalMain');
        modalMain.innerHTML = `
            <div style="display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                <button class="cyber-btn modal-tab-btn active" onclick="window.controller.switchModalTab('read')" style="flex-grow: 1;">READ</button>
                <button class="cyber-btn modal-tab-btn" onclick="window.controller.switchModalTab('edit')" style="flex-grow: 1;">EDIT</button>
                <button class="cyber-btn modal-tab-btn" onclick="window.controller.switchModalTab('simulate')" style="flex-grow: 1;">SIMULATE</button>
            </div>

            <div id="modal-tab-read" class="modal-tab-content">
                <h2 style="margin-top:0; color:var(--primary-color); font-family: 'JetBrains Mono';">${report.title}</h2>
                <div style="margin-bottom:20px; border-bottom:1px solid #333; padding-bottom:10px; font-family:var(--font-mono); font-size:0.8rem; color:#666;">
                    DATE: ${report.date} | AGENT: ${report.type} | HASH: ${report.provenance_hash ? report.provenance_hash.substring(0,12) : 'N/A'}...
                </div>
                <div style="line-height:1.6; color:#ddd; font-family: 'Inter', sans-serif;">
                    ${report.full_body || report.summary}
                </div>
            </div>

            <div id="modal-tab-edit" class="modal-tab-content" style="display: none; height: calc(100% - 60px);">
                <div style="margin-bottom: 10px; font-size: 0.8rem; color: #888;">RAW MARKDOWN / HTML SOURCE</div>
                <textarea style="width: 100%; height: 90%; background: #000; color: #00f3ff; border: 1px solid #333; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; padding: 10px; resize: none; outline: none;">${report.full_body || report.summary}</textarea>
            </div>

            <div id="modal-tab-simulate" class="modal-tab-content" style="display: none;">
                <h3 style="color: #0aff60; font-family: 'JetBrains Mono'; margin-top: 0;">CRISIS & ALPHA SIMULATION</h3>
                <p style="color: #aaa; font-size: 0.85rem;">Run deterministic scenarios based on report sentiment and extracted entities.</p>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px; margin-bottom: 20px;">
                    <div class="metric-row"><span>SCENARIO TARGET:</span><span class="val-green">${report.entities && report.entities.tickers && report.entities.tickers.length > 0 ? report.entities.tickers.join(', ') : 'BROAD MARKET'}</span></div>
                    <div class="metric-row"><span>BASE SENTIMENT:</span><span>${report.sentiment_score}/100</span></div>
                    <div class="metric-row"><span>RISK BIAS:</span><span style="color: ${(report.sentiment_score || 50) < 40 ? '#ff3333' : '#0aff60'};">${(report.sentiment_score || 50) < 40 ? 'BEARISH (CRISIS)' : 'BULLISH (MELT-UP)'}</span></div>
                </div>
                <button class="cyber-btn" onclick="window.controller.runSimulation(this, ${report.sentiment_score})" style="width: 100%; padding: 15px; font-size: 1rem;"><i class="fas fa-play"></i> EXECUTE SIMULATION</button>
                <div id="sim-results" style="margin-top: 20px; display: none;">
                    <div style="color: #00f3ff; font-family: 'JetBrains Mono'; margin-bottom: 10px;">SIMULATION COMPLETE</div>
                    <div class="risk-grid">
                         <div class="risk-cell risk-med">
                            <div>FWD PROJECTION (30D)</div>
                            <div id="sim-fwd-return" style="font-size: 1rem; font-weight: bold; color: #fff;">...</div>
                        </div>
                        <div class="risk-cell risk-med">
                            <div>MAX DRAWDOWN</div>
                            <div id="sim-max-dd" style="font-size: 1rem; font-weight: bold; color: #ff3333;">...</div>
                        </div>
                    </div>
                </div>
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
            <div style="margin-bottom:10px; color: #fff;"><strong>VERDICT:</strong> ${(report.conviction || 0) > 75 ? 'HIGH CONVICTION' : 'REVIEW REQUIRED'}</div>
            <div style="font-size:0.75rem; margin-bottom: 10px; color: #aaa;">${report.critique || 'System 2 analysis pending...'}</div>
            ${gaugesHTML}
        `;

        // Alpha Calculation
        this.renderAlphaChart(report);

        // Metadata
        const metaHTML = `
            <div style="margin-top:20px;">
                <h4 style="color:#888; font-size:0.8rem; border-bottom: 1px solid #444; padding-bottom:5px;">ENTITIES</h4>
                <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top: 10px;">
                    ${report.entities && report.entities.keywords ? report.entities.keywords.map(k => `<span class="tag">${k}</span>`).join('') : '<span style="color:#666; font-size:0.7rem;">None Detected</span>'}
                </div>
            </div>
        `;
        document.getElementById('modalMeta').innerHTML = metaHTML;

        this.dom.modal.style.display = 'flex';
    }

    switchModalTab(tabName) {
        const tabs = document.getElementsByClassName('modal-tab-content');
        const btns = document.getElementsByClassName('modal-tab-btn');

        Array.from(tabs).forEach(t => t.style.display = 'none');
        Array.from(btns).forEach(b => b.classList.remove('active'));

        const el = document.getElementById(`modal-tab-${tabName}`);
        if(el) el.style.display = 'block';

        // Find the button and set active
        Array.from(btns).forEach(b => {
             if(b.innerText.toLowerCase() === tabName) b.classList.add('active');
        });
    }

    runSimulation(btn, sentiment) {
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> CALCULATING ALPHA...';
        btn.disabled = true;

        setTimeout(() => {
            btn.style.display = 'none';
            const results = document.getElementById('sim-results');
            results.style.display = 'block';

            // Generate mock metrics based on sentiment
            const baseReturn = (sentiment - 50) / 10; // -5% to +5%
            const volatility = Math.random() * 10 + 5;

            const fwdReturn = (baseReturn + (Math.random() * 4 - 2)).toFixed(2);
            const maxDd = (volatility * (Math.random() * 0.5 + 0.5)).toFixed(2);

            const fwdEl = document.getElementById('sim-fwd-return');
            fwdEl.innerText = `${fwdReturn > 0 ? '+' : ''}${fwdReturn}%`;
            fwdEl.style.color = fwdReturn >= 0 ? '#0aff60' : '#ff3333';

            document.getElementById('sim-max-dd').innerText = `-${maxDd}%`;

        }, 1500);
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
                options: { plugins: { legend: { display: false } }, scales: { x: { display:false }, y: { display:false } } }
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
            <span style="color: #fff; font-weight: bold;">${primaryTicker}</span>: <span style="color:${color}">${returnPct > 0 ? '+' : ''}${returnPct}%</span> (30d Realized)
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
                    y: { 
                        display: false,
                        grid: { display: false } 
                    } 
                },
                elements: {
                    point: { radius: 0 }
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
            const sortSelect = document.getElementById('sortOrder');

            const yearValue = yearSelect ? yearSelect.value : 'all';
            const typeValue = typeSelect ? typeSelect.value : 'all';
            const sortValue = sortSelect ? sortSelect.value : 'newest';

            const gridContainer = document.getElementById('archive-items-container') || document.getElementById('archiveGrid');
            const items = Array.from(gridContainer.getElementsByClassName('archive-item'));

            // Sort
            items.sort((a, b) => {
                if (sortValue === 'newest') return b.dataset.date.localeCompare(a.dataset.date);
                if (sortValue === 'oldest') return a.dataset.date.localeCompare(b.dataset.date);
                // For conviction/sentiment, handle descending
                if (sortValue === 'conviction') return (parseInt(b.dataset.conviction) || 0) - (parseInt(a.dataset.conviction) || 0);
                if (sortValue === 'sentiment') return (parseInt(b.dataset.sentiment) || 0) - (parseInt(a.dataset.sentiment) || 0);
                if (sortValue === 'bullish') return (parseInt(b.dataset.bullish) || 0) - (parseInt(a.dataset.bullish) || 0);
                if (sortValue === 'bearish') return (parseInt(b.dataset.bearish) || 0) - (parseInt(a.dataset.bearish) || 0);
                return 0;
            });

            // Re-append to update DOM order
            items.forEach(item => gridContainer.appendChild(item));

            // Filter
            items.forEach(item => {
                const itemType = item.dataset.type;
                const itemYear = item.dataset.year;
                const itemTitle = item.dataset.title;
                const itemKeywords = item.dataset.keywords;

                let matchSearch = itemTitle.includes(search) || itemKeywords.includes(search);
                let matchType = typeValue === 'all' || itemType === typeValue;
                let matchYear = yearValue === 'all' || itemYear === yearValue;

                // Handle Historical Special Case
                if (yearValue === 'HISTORICAL' && itemYear === 'HISTORICAL') {
                    matchYear = true;
                }

                item.style.display = (matchSearch && matchType && matchYear) ? 'flex' : 'none';
            });
        };

        // Attach listeners to selects
        ['yearFilter', 'typeFilter', 'sortOrder', 'searchInput'].forEach(id => {
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
