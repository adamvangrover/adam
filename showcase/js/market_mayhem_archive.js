/**
 * Market Mayhem Archive Logic
 * -----------------------------------------------------------------------------
 * Handles data fetching, chart rendering, and archive grid population.
 */

class MarketMayhemArchive {
    constructor() {
        this.data = [];
        this.chartInstance = null;
    }

    async init() {
        console.log("[MarketMayhemArchive] Initializing...");
        await this.fetchData();
        this.renderChart();
        this.renderArchiveGrid();
        this.renderSidebar();
        this.bindEvents();
    }

    async fetchData() {
        try {
            const response = await fetch('data/market_mayhem_index.json');
            if (!response.ok) throw new Error("Failed to load data");
            const jsonData = await response.json();
            this.data = jsonData.archives || [];

            // Sort by date descending
            this.data.sort((a, b) => new Date(b.date) - new Date(a.date));

            console.log(`[MarketMayhemArchive] Loaded ${this.data.length} records.`);
        } catch (e) {
            console.error("[MarketMayhemArchive] Data Load Error:", e);
            document.getElementById('archive-list').innerHTML = `<div style="text-align:center; color:#ff3333; padding:20px;">ERROR LOADING DATA: ${e.message}</div>`;
            // Fallback mock data if fetch fails
            this.data = [];
        }
    }

    renderChart() {
        const ctx = document.getElementById('sentimentChart');
        if (!ctx) return;

        // Process Data for Chart (Chronological)
        const sortedData = [...this.data].sort((a, b) => new Date(a.date) - new Date(b.date));

        // Limit to last 50 points for readability
        const recentData = sortedData.slice(-50);

        const labels = recentData.map(d => d.date);
        const sentimentData = recentData.map(d => d.sentiment || 50);

        // Forward Outlook Mock Logic
        const lastVal = sentimentData.length > 0 ? sentimentData[sentimentData.length - 1] : 50;
        const futureLabels = ['+1W', '+2W', '+1M', '+3M', '+6M'];
        const allLabels = [...labels, ...futureLabels];

        // Extend Datasets
        const paddedSentiment = [...sentimentData, ...Array(futureLabels.length).fill(null)];

        // Bull Case
        const bullPath = [...Array(labels.length).fill(null)];
        bullPath[labels.length - 1] = lastVal;
        bullPath.push(lastVal + 5, lastVal + 10, lastVal + 15, lastVal + 25, lastVal + 35);

        // Bear Case
        const bearPath = [...Array(labels.length).fill(null)];
        bearPath[labels.length - 1] = lastVal;
        bearPath.push(lastVal - 5, lastVal - 15, lastVal - 25, lastVal - 40, lastVal - 50);

        // Consensus
        const consensusPath = [...Array(labels.length).fill(null)];
        consensusPath[labels.length - 1] = lastVal;
        consensusPath.push(lastVal + 2, lastVal + 4, lastVal + 5, lastVal + 8, lastVal + 10);

        this.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allLabels,
                datasets: [
                    {
                        label: 'Historical Sentiment',
                        data: paddedSentiment,
                        borderColor: '#00f3ff',
                        backgroundColor: 'rgba(0, 243, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 3
                    },
                    {
                        label: 'Bull Case',
                        data: bullPath,
                        borderColor: '#0aff60',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'Bear Case',
                        data: bearPath,
                        borderColor: '#ff3333',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'Consensus Outlook',
                        data: consensusPath,
                        borderColor: '#d946ef',
                        borderWidth: 2,
                        tension: 0.4,
                        pointRadius: 4,
                        pointBackgroundColor: '#d946ef'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        labels: { color: '#ccc', font: { family: 'JetBrains Mono', size: 10 }, boxWidth: 10 },
                        position: 'top',
                        align: 'end'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(5, 11, 20, 0.95)',
                        titleFont: { family: 'JetBrains Mono' },
                        bodyFont: { family: 'JetBrains Mono' },
                        borderColor: '#333',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return ` ${context.dataset.label}: ${context.parsed.y}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#666', font: { family: 'JetBrains Mono', size: 9 } },
                        suggestedMin: 0,
                        suggestedMax: 100
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#666', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 9 } }
                    }
                }
            }
        });
    }

    renderArchiveGrid() {
        const grid = document.getElementById('archive-list');
        if (!grid) return;

        grid.innerHTML = '';

        if (this.data.length === 0) {
            grid.innerHTML = '<div style="text-align: center; padding: 40px; color: #666;" class="mono">NO ARCHIVES FOUND</div>';
            return;
        }

        this.data.forEach(item => {
            const year = item.year || item.date.split('-')[0];
            const type = item.type || 'NEWSLETTER';
            const titleLower = (item.title || '').toLowerCase();
            const descLower = (item.description || '').toLowerCase();
            const searchTerms = `${titleLower} ${descLower}`;

            const el = document.createElement('div');
            el.className = `archive-item type-${type}`;
            el.dataset.year = year;
            el.dataset.type = type;
            el.dataset.search = searchTerms;

            // Sentiment Color
            const sentScore = item.sentiment || 50;
            const sentColor = sentScore > 60 ? '#0aff60' : (sentScore < 40 ? '#ff3333' : '#f59e0b');

            el.innerHTML = `
                <div style="flex-grow: 1;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:5px;">
                        <span class="mono" style="font-size:0.7rem; color:#888;">${item.date}</span>
                        <span class="type-badge type-${type}">${type.replace('_', ' ')}</span>
                        <span class="mono" style="font-size:0.7rem; color:${sentColor};">SENT: ${sentScore}</span>
                        <span class="mono" style="font-size:0.7rem; color:#444;">CONV: ${item.conviction || 50}</span>
                    </div>
                    <h3 style="margin:0 0 5px 0; font-size:1.1rem; color: #e0e0e0;">${item.title}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#aaa;">${item.description || 'No summary available.'}</p>
                </div>
                <a href="${item.link || '#'}" class="cyber-btn" style="align-self:center;">ACCESS</a>
            `;
            grid.appendChild(el);
        });
    }

    renderSidebar() {
        // Render Top Conviction
        const convictionList = document.getElementById('conviction-list');
        if (convictionList) {
            const convictions = [
                { ticker: 'NVDA', sector: 'Technology', val: 92.5, change: '+4.2%' },
                { ticker: 'MSFT', sector: 'Technology', val: 88.1, change: '+1.1%' },
                { ticker: 'PLTR', sector: 'Defense', val: 85.4, change: '+3.5%' },
                { ticker: 'ETH', sector: 'Crypto', val: 82.0, change: '+2.8%' },
                { ticker: 'XOM', sector: 'Energy', val: 79.5, change: '-0.5%' }
            ];

            convictionList.innerHTML = convictions.map(c => `
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem;">
                    <span style="font-weight:bold; color:white;">${c.ticker} <span style="font-size:0.6rem; color:#555;">${c.sector}</span></span>
                    <span style="color: ${c.change.includes('+') ? '#0aff60' : '#ff3333'}">${c.val}</span>
                </div>
            `).join('');
        }

        // Render Watch List
        const watchList = document.getElementById('watch-list');
        if (watchList) {
             const watchlist = [
                { ticker: 'TSLA', change: '-2.1%' },
                { ticker: 'AMD', change: '+0.5%' },
                { ticker: 'GOOGL', change: '+1.2%' },
                { ticker: 'AMZN', change: '0.0%' },
                { ticker: 'COIN', change: '+5.4%' }
            ];

            watchList.innerHTML = watchlist.map(w => `
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem;">
                    <span style="color:#aaa;">${w.ticker}</span>
                    <span style="color: ${w.change.includes('+') ? '#0aff60' : (w.change === '0.0%' ? '#f59e0b' : '#ff3333')}">${w.change}</span>
                </div>
            `).join('');
        }
    }

    bindEvents() {
        document.getElementById('searchInput')?.addEventListener('keyup', () => this.applyFilters());
        document.getElementById('yearFilter')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('typeFilter')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('sortOrder')?.addEventListener('change', () => this.applySorting());
    }

    applyFilters() {
        const search = document.getElementById('searchInput').value.toLowerCase();
        const year = document.getElementById('yearFilter').value;
        const type = document.getElementById('typeFilter').value;

        const grid = document.getElementById('archive-list');
        const items = Array.from(grid.getElementsByClassName('archive-item'));

        items.forEach(item => {
            const itemYear = item.dataset.year;
            const itemType = item.dataset.type;
            const itemSearch = item.dataset.search;

            let matchSearch = itemSearch.includes(search);
            let matchYear = year === 'all' || itemYear === year;
            let matchType = type === 'all' || itemType === type;

            item.style.display = (matchSearch && matchYear && matchType) ? 'flex' : 'none';
        });
    }

    applySorting() {
        const sort = document.getElementById('sortOrder').value;
        const grid = document.getElementById('archive-list');
        const items = Array.from(grid.getElementsByClassName('archive-item'));

        items.sort((a, b) => {
            // Extract date from the first .mono span
            const dateA = new Date(a.querySelector('.mono').innerText.trim());
            const dateB = new Date(b.querySelector('.mono').innerText.trim());

            if (sort === 'newest') {
                return dateB - dateA;
            } else {
                return dateA - dateB;
            }
        });

        // Clear and Re-append
        grid.innerHTML = '';
        items.forEach(item => grid.appendChild(item));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.marketMayhem = new MarketMayhemArchive();
    window.marketMayhem.init();
});
