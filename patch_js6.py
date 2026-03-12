import re

with open("showcase/js/comprehensive_credit_dashboard.js", "r") as f:
    js = f.read()

# Make sure we render any dynamic canvas in sections
dynamic_chart_renderer = """
    // Financials Chart
    renderChart(memo);
    renderMonteCarloChart(memo);

    // Initialize custom section charts if they exist
    sectionChartIds.forEach(id => {
        const ctx = document.getElementById(id);
        if (ctx) {
            // Draw a simple chart to fill it based on dcfSensitivityMatrix
            if (id.includes('dcf-chart') && memo.valuation && memo.valuation.dcfSensitivityMatrix) {
                const dataPoints = memo.valuation.dcfSensitivityMatrix.map(m => m.implied_price);
                const labels = memo.valuation.dcfSensitivityMatrix.map(m => `W:${(m.wacc*100).toFixed(1)}% TGR:${(m.tgr*100).toFixed(1)}%`);
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Implied Price',
                            data: dataPoints,
                            borderColor: 'rgba(59, 130, 246, 1)',
                            backgroundColor: 'rgba(59, 130, 246, 0.2)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { grid: { color: 'rgba(51, 65, 85, 0.5)' }, ticks: { color: '#94a3b8' } },
                            x: { display: false }
                        }
                    }
                });
            }
        }
    });
"""

js = js.replace("""
    // Financials Chart
    renderChart(memo);
    renderMonteCarloChart(memo);""", dynamic_chart_renderer.strip('\n'))


with open("showcase/js/comprehensive_credit_dashboard.js", "w") as f:
    f.write(js)
