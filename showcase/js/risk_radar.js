// Risk Radar Logic for Market Mayhem
const riskData = {
    "CAPEX_BUILD": {
        "title": "Capex Build Risks",
        "sentiment": "BEARISH",
        "score": 85,
        "details": "The 'Efficiency Shock' is threatening the trillion-dollar Capex projections. Hyperscalers are re-evaluating GPU cluster ROI as inference costs drop faster than demand scales.",
        "chartData": {
            "labels": ["Q1'25", "Q2'25", "Q3'25", "Q4'25", "Q1'26 (Est)"],
            "datasets": [{
                "label": "Hyperscaler Capex Growth (YoY %)",
                "data": [45, 42, 38, 25, 12],
                "borderColor": "#cc0000",
                "backgroundColor": "rgba(204, 0, 0, 0.2)",
                "fill": true
            }]
        }
    },
    "AI_FRONTIER": {
        "title": "AI Frontier Development",
        "sentiment": "NEUTRAL",
        "score": 50,
        "details": "Model convergence is occurring. The gap between proprietary frontier models and open-source/distilled models is narrowing, eroding the 'Intelligence Moat'.",
        "chartData": {
            "labels": ["GPT-4", "Claude 3", "Gemini 1.5", "Llama 4", "DeepSeek V3"],
            "datasets": [{
                "label": "Benchmark Parity Score",
                "data": [95, 96, 96, 94, 95],
                "borderColor": "#ffff00",
                "backgroundColor": "rgba(255, 255, 0, 0.2)",
                "fill": true
            }]
        }
    },
    "ENTERPRISE_ADOPTION": {
        "title": "Enterprise Adoption",
        "sentiment": "LAGGING",
        "score": 40,
        "details": "The 'Pilot Purgatory' continues. While 90% of enterprises are testing AI, only 12% have deployed production agents at scale due to hallucination risks and data governance.",
        "chartData": {
            "labels": ["Testing", "Pilot", "Production (Internal)", "Production (Customer)"],
            "datasets": [{
                "label": "Adoption Funnel (%)",
                "data": [90, 65, 25, 12],
                "backgroundColor": ["#666", "#888", "#aaa", "#00f3ff"],
                "borderColor": "transparent"
            }]
        }
    },
    "ENERGY_GENERATION": {
        "title": "Energy Generation Constraints",
        "sentiment": "CRITICAL",
        "score": 92,
        "details": "Data center power demand is outpacing grid capacity. Nuclear SMR delays and transmission bottlenecks are the hard cap on AI scaling for the next 36 months.",
        "chartData": {
            "labels": ["2023", "2024", "2025", "2026 (Proj)"],
            "datasets": [{
                "label": "Data Center TWh Demand",
                "data": [120, 150, 190, 240],
                "borderColor": "#ff00ff",
                "backgroundColor": "rgba(255, 0, 255, 0.2)",
                "type": 'line'
            }, {
                "label": "Grid Capacity Additions",
                "data": [10, 15, 18, 20],
                "borderColor": "#00ff00",
                "type": 'line'
            }]
        }
    },
    "SPACE_EXPLORATION": {
        "title": "Space Economy & Defense",
        "sentiment": "BULLISH",
        "score": 75,
        "details": "LEO satellite constellations are becoming critical infrastructure. Defense spending on space assets is decoupling from general fiscal austerity.",
        "chartData": {
            "labels": ["Comm", "Defense", "Earth Obs", "Launch"],
            "datasets": [{
                "label": "Sector Growth YoY",
                "data": [15, 28, 12, 10],
                "backgroundColor": "#00f3ff"
            }]
        }
    },
    "CONSUMPTION_TRENDS": {
        "title": "Data & Energy Consumption",
        "sentiment": "WARNING",
        "score": 88,
        "details": "Training runs for next-gen models (100k+ H100s) require gigawatt-scale power. The environmental impact is inviting regulatory scrutiny and carbon taxes.",
        "chartData": {
            "labels": ["Training", "Inference"],
            "datasets": [{
                "label": "Energy Split 2026",
                "data": [30, 70],
                "backgroundColor": ["#ff9900", "#00cc00"]
            }]
        }
    }
};

let riskChartInstance = null;

function initRiskRadar() {
    const container = document.getElementById('risk-radar-container');
    if (!container) return;

    // Create Grid
    const gridHTML = Object.keys(riskData).map(key => {
        const item = riskData[key];
        let color = '#888';
        if (item.score > 80) color = '#cc0000'; // Critical
        else if (item.score > 60) color = '#ff9900'; // Warning
        else if (item.score < 40) color = '#00cc00'; // Safe/Lagging

        return `
            <div class="risk-card" onclick="renderRiskDetails('${key}')" style="border-left: 3px solid ${color}">
                <div style="font-size:0.7rem; color:${color}; font-weight:bold;">${item.sentiment}</div>
                <div style="font-weight:bold; margin: 5px 0;">${item.title}</div>
                <div class="risk-meter">
                    <div class="risk-fill" style="width: ${item.score}%; background: ${color};"></div>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <h3 class="section-header">/// LIVE RISK TOPOGRAPHY</h3>
        <div class="risk-grid">${gridHTML}</div>
        <div id="risk-detail-view" style="display:none; margin-top:20px; border:1px solid #333; padding:20px; background:rgba(0,0,0,0.3);">
            <div style="display:flex; justify-content:space-between;">
                <h4 id="risk-detail-title" style="margin:0; color:#fff;"></h4>
                <button onclick="closeRiskDetail()" style="background:none; border:none; color:#666; cursor:pointer;">[X]</button>
            </div>
            <p id="risk-detail-text" style="color:#ccc; font-size:0.9rem; margin: 15px 0;"></p>
            <div style="height:250px;">
                <canvas id="riskDetailChart"></canvas>
            </div>
        </div>
    `;
}

function renderRiskDetails(key) {
    const data = riskData[key];
    const view = document.getElementById('risk-detail-view');
    view.style.display = 'block';

    document.getElementById('risk-detail-title').innerText = data.title;
    document.getElementById('risk-detail-text').innerText = data.details;

    const ctx = document.getElementById('riskDetailChart').getContext('2d');
    if (riskChartInstance) riskChartInstance.destroy();

    const config = {
        type: data.chartData.datasets[0].type || 'bar',
        data: data.chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#ccc', font: { family: 'JetBrains Mono' } } }
            },
            scales: {
                y: { grid: { color: '#333' }, ticks: { color: '#888' } },
                x: { grid: { color: '#333' }, ticks: { color: '#888' } }
            }
        }
    };

    riskChartInstance = new Chart(ctx, config);

    // Scroll to detail view
    view.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeRiskDetail() {
    document.getElementById('risk-detail-view').style.display = 'none';
}

document.addEventListener('DOMContentLoaded', initRiskRadar);
