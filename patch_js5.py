import re

with open("showcase/js/comprehensive_credit_dashboard.js", "r") as f:
    js = f.read()

# Charts
js = js.replace("""
function renderChart(memo) {
    const ctx = document.getElementById('financialsChart');
    if (chartInstance) {
        chartInstance.destroy();
    }

    if (!memo.historical_financials || memo.historical_financials.length === 0) {
        ctx.style.display = 'none';
        return;
    }
""", """
function renderChart(memo) {
    const ctx = document.getElementById('financialsChart');
    if (chartInstance) {
        chartInstance.destroy();
    }

    if (memo.financials && memo.financials.historicals) {
        ctx.style.display = 'block';
        const h = memo.financials.historicals;
        const labels = ['2023', '2024'];
        const revs = [h.revenue_2023 || 0, h.revenue_2024 || 0];

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Revenue',
                        data: revs,
                        backgroundColor: 'rgba(34, 211, 238, 0.2)',
                        borderColor: 'rgba(34, 211, 238, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono' } } }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(51, 65, 85, 0.5)' },
                        ticks: { color: '#94a3b8', callback: function(value) { return '$' + value; } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
        return;
    }

    if (!memo.historical_financials || memo.historical_financials.length === 0) {
        ctx.style.display = 'none';
        return;
    }
""")

js = js.replace("""
function renderMonteCarloChart(memo) {
    const ctx = document.getElementById('monteCarloChart');
    if (monteCarloChartInstance) {
        monteCarloChartInstance.destroy();
    }

    if (!memo.dcf_analysis || !memo.dcf_analysis.monte_carlo_forecasts || memo.dcf_analysis.monte_carlo_forecasts.length === 0) {
        ctx.style.display = 'none';
        return;
    }
""", """
function renderMonteCarloChart(memo) {
    const ctx = document.getElementById('monteCarloChart');
    if (monteCarloChartInstance) {
        monteCarloChartInstance.destroy();
    }

    if (memo.financials && memo.financials.monte_carlo_forecasts && memo.financials.monte_carlo_forecasts.metrics) {
        ctx.style.display = 'block';
        const metrics = memo.financials.monte_carlo_forecasts.metrics;
        const rev = metrics.revenue_2025;
        const fcf = metrics.fcf_2025;

        monteCarloChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Revenue 2025', 'FCF 2025'],
                datasets: [
                    {
                        label: 'P10',
                        data: [rev.p10, fcf.p10],
                        backgroundColor: 'rgba(255, 99, 132, 0.5)'
                    },
                    {
                        label: 'P50',
                        data: [rev.p50, fcf.p50],
                        backgroundColor: 'rgba(54, 162, 235, 0.5)'
                    },
                    {
                        label: 'P90',
                        data: [rev.p90, fcf.p90],
                        backgroundColor: 'rgba(75, 192, 192, 0.5)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(51, 65, 85, 0.5)' },
                        ticks: { color: '#94a3b8', callback: function(value) { return '$' + value; } }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
        return;
    }

    if (!memo.dcf_analysis || !memo.dcf_analysis.monte_carlo_forecasts || memo.dcf_analysis.monte_carlo_forecasts.length === 0) {
        ctx.style.display = 'none';
        return;
    }
""")

with open("showcase/js/comprehensive_credit_dashboard.js", "w") as f:
    f.write(js)
