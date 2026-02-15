// Market Mayhem Analytics Hub Logic
// Reads data/market_mayhem_index.json and visualizes trends.

class AnalyticsHub {
    constructor() {
        this.data = null;
        this.charts = {};
        this.currentPeriod = 'ALL';
    }

    async init() {
        console.log("Analytics Hub: Initializing...");
        await this.loadData();
        this.renderStats();
        this.renderCharts();
        this.renderInsights();
        this.bindEvents();
    }

    async loadData() {
        try {
            const response = await fetch('data/market_mayhem_index.json');
            this.data = await response.json();
            // Filter out unsourced if needed, or keep all
            this.data = this.data.filter(d => d.sentiment_score !== undefined);

            // Sort by date ascending
            this.data.sort((a, b) => new Date(a.date) - new Date(b.date));
            console.log(`Loaded ${this.data.length} records.`);
        } catch (e) {
            console.error("Failed to load analytics data:", e);
            document.getElementById('stat-total-reports').innerText = "ERR";
        }
    }

    renderStats() {
        if (!this.data) return;

        // Total Reports
        document.getElementById('stat-total-reports').innerText = this.data.length;

        // Avg Sentiment
        const avgSent = this.data.reduce((acc, curr) => acc + (curr.sentiment_score || 50), 0) / this.data.length;
        document.getElementById('stat-avg-sentiment').innerText = Math.round(avgSent);

        // Calculate Trend (Last 10 vs First 10)
        const first10 = this.data.slice(0, 10).reduce((a, c) => a + (c.sentiment_score||50), 0) / 10;
        const last10 = this.data.slice(-10).reduce((a, c) => a + (c.sentiment_score||50), 0) / 10;
        const delta = last10 - first10;

        const deltaEl = document.getElementById('stat-sentiment-delta');
        deltaEl.innerText = (delta > 0 ? "+" : "") + delta.toFixed(1);
        deltaEl.className = `stat-delta ${delta > 0 ? 'delta-pos' : 'delta-neg'}`;

        // Top Regime
        // Simple logic: if avg > 60 Bull, < 40 Bear, else Neutral
        const regimeEl = document.getElementById('stat-current-regime');
        if (last10 > 60) { regimeEl.innerText = "AGENTIC BULL"; regimeEl.style.color = "#00ff00"; }
        else if (last10 < 40) { regimeEl.innerText = "CRISIS BEAR"; regimeEl.style.color = "#ff0000"; }
        else { regimeEl.innerText = "TRANSITION"; regimeEl.style.color = "#ffff00"; }
    }

    renderCharts() {
        if (!this.data) return;

        // 1. Sentiment vs Conviction Timeline
        const ctxTime = document.getElementById('timelineChart').getContext('2d');
        const labels = this.data.map(d => d.date);
        const sentiments = this.data.map(d => d.sentiment_score);
        const convictions = this.data.map(d => d.conviction || 50);

        this.charts.timeline = new Chart(ctxTime, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Sentiment',
                        data: sentiments,
                        borderColor: '#00f3ff',
                        backgroundColor: 'rgba(0, 243, 255, 0.1)',
                        borderWidth: 1,
                        pointRadius: 1,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Agent Conviction',
                        data: convictions,
                        borderColor: '#ff00ff',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono' } } },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: { display: false },
                    y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } }
                }
            }
        });

        // 2. Entity Gravity (Top 10 Entities)
        const entityCounts = {};
        this.data.forEach(d => {
            if (d.entities && d.entities.tickers) {
                d.entities.tickers.forEach(t => entityCounts[t] = (entityCounts[t] || 0) + 1);
            }
            if (d.entities && d.entities.sovereigns) {
                d.entities.sovereigns.forEach(s => entityCounts[s] = (entityCounts[s] || 0) + 1);
            }
        });

        const sortedEntities = Object.entries(entityCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        const ctxGravity = document.getElementById('gravityChart').getContext('2d');
        this.charts.gravity = new Chart(ctxGravity, {
            type: 'bar', // Using Bar for clarity, "Gravity" implies bubble but bar is safer for chartjs simple
            data: {
                labels: sortedEntities.map(e => e[0]),
                datasets: [{
                    label: 'Mentions',
                    data: sortedEntities.map(e => e[1]),
                    backgroundColor: sortedEntities.map((_, i) => `hsla(${i * 36}, 70%, 50%, 0.6)`),
                    borderColor: '#fff',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
                    y: { ticks: { color: '#e2e8f0', font: { family: 'JetBrains Mono' } } }
                }
            }
        });
    }

    renderInsights() {
        const list = document.getElementById('top-insights');
        if (!list || !this.data) return;

        // Find "Anomalies" (High Divergence or Extreme Sentiment)
        // Or find the newly generated Audit reports
        const audits = this.data.filter(d => d.type === 'SYSTEM_AUDIT');
        const counters = this.data.filter(d => d.type === 'COUNTERFACTUAL');

        let html = '';

        // Add Audits
        audits.forEach(a => {
            html += `
                <li class="insight-item">
                    <span class="insight-score" style="color:#ff00ff;">AUDIT</span>
                    <a href="${a.filename}" style="color:white; text-decoration:none; flex-grow:1;">${a.title}</a>
                    <span style="color:#888; font-size:0.7rem;">${a.date}</span>
                </li>
            `;
        });

        // Add Counterfactuals
        counters.forEach(c => {
            html += `
                <li class="insight-item">
                    <span class="insight-score" style="color:#ffaa00;">SIM</span>
                    <a href="${c.filename}" style="color:white; text-decoration:none; flex-grow:1;">${c.title}</a>
                    <span style="color:#888; font-size:0.7rem;">${c.date}</span>
                </li>
            `;
        });

        // Fallback if empty
        if (html === '') {
            html = '<li class="insight-item" style="color:#666;">No special reports found.</li>';
        }

        list.innerHTML = html;
    }

    bindEvents() {
        // Future interactions (filter by date range etc)
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.analyticsHub = new AnalyticsHub();
    window.analyticsHub.init();
});
