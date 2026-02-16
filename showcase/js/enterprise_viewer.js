let simData = null;
let currentStep = 0;
let isPlaying = false;
let playInterval = null;
let charts = {};

// Load Data
fetch('data/enterprise_simulation_data.json')
    .then(response => response.json())
    .then(data => {
        simData = data;
        console.log("Enterprise Simulation Data Loaded:", simData);
        initCharts();
        updateUI(0);

        // Log initialization
        logSystemEvent("DATASET LOADED: " + simData.metadata.simulation_id);
        logSystemEvent("SCENARIOS: " + simData.metadata.scenarios_included.join(", "));
    })
    .catch(error => console.error("Error loading simulation data:", error));

function initCharts() {
    // Market Charts
    charts.market = new Chart(document.getElementById('marketChart'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'S&P 500', borderColor: '#00f3ff', data: [] },
            { label: 'NASDAQ', borderColor: '#ff00ff', data: [] }
        ]},
        options: { responsive: true, maintainAspectRatio: false, elements: { point: { radius: 0 } }, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });

    charts.vix = new Chart(document.getElementById('vixChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'VIX', borderColor: '#ff4444', data: [] }] },
        options: { responsive: true, maintainAspectRatio: false, elements: { point: { radius: 0 } }, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });

    // Banking Charts
    charts.liquidity = new Chart(document.getElementById('liquidityChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'LCR ($B)', borderColor: '#00ff00', data: [], fill: true, backgroundColor: 'rgba(0,255,0,0.1)' }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });

    charts.capital = new Chart(document.getElementById('capitalChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'CET1 Ratio (%)', borderColor: '#ffff00', data: [] }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });

    charts.sysRisk = new Chart(document.getElementById('systemicRiskChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Systemic Risk', borderColor: '#ff00ff', data: [] }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' }, min: 0, max: 100 } } }
    });

    // Risk Charts
    charts.var = new Chart(document.getElementById('varChart'), {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'VaR 95%', backgroundColor: '#ff4444', data: [] }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });

    charts.stress = new Chart(document.getElementById('stressChart'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Stress Loss ($B)', borderColor: '#ff8800', data: [] }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });

    // Macro Charts
    charts.macro = new Chart(document.getElementById('macroChart'), {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'CPI YoY', borderColor: '#ffff00', data: [] },
            { label: 'Fed Funds Rate', borderColor: '#00f3ff', data: [] }
        ]},
        options: { responsive: true, maintainAspectRatio: false, scales: { x: { display: false }, y: { grid: { color: '#333' } } } }
    });
}

function updateUI(step) {
    if (!simData) return;

    // Ensure step is within bounds
    if (step >= simData.market.length) step = simData.market.length - 1;

    document.getElementById('sim-step-display').innerText = step;
    document.getElementById('timelineSlider').max = simData.market.length - 1;
    document.getElementById('timelineSlider').value = step;

    // Update Market
    const mData = simData.market.slice(0, step + 1);
    updateChart(charts.market, mData.map(d => d.step), [mData.map(d => d.spx), mData.map(d => d.ndx)]);
    updateChart(charts.vix, mData.map(d => d.step), [mData.map(d => d.vix)]);
    document.getElementById('regime-display').innerText = mData[step].regime;
    document.getElementById('regime-display').style.color = mData[step].regime === 'BULL' ? '#00ff00' : (mData[step].regime === 'BEAR' ? '#ff0000' : '#ffff00');

    // Update Banking
    const bData = simData.banking.slice(0, step + 1);
    updateChart(charts.liquidity, bData.map(d => d.step), [bData.map(d => d.liquidity_coverage)]);
    updateChart(charts.capital, bData.map(d => d.step), [bData.map(d => d.cet1_ratio)]);
    updateChart(charts.sysRisk, bData.map(d => d.step), [bData.map(d => d.systemic_risk_score)]);

    // Update Risk
    const rData = simData.risk.slice(0, step + 1);
    updateChart(charts.var, rData.map(d => d.step), [rData.map(d => d.var_95)]);
    updateChart(charts.stress, rData.map(d => d.step), [rData.map(d => d.stress_test_loss_est)]);

    // Update Macro
    const macData = simData.macro.slice(0, step + 1);
    updateChart(charts.macro, macData.map(d => d.step), [macData.map(d => d.cpi_yoy), macData.map(d => d.fed_funds_rate)]);

    // Update Agents Logs
    const aData = simData.agents[step];
    const logContainer = document.getElementById('agent-log-container');
    // For replay, we might want to append only if playing forward, or clear and show recent if scrubbing?
    // Let's simple clear and show "Logs for Step X" + accumulation if feasible, but accumulating DOM elements is heavy.
    // Let's show logs for *current step* only or last 5 steps.

    // Only update logs if step changed
    if (currentStep !== step || logContainer.innerHTML === '') {
        // Just show current step logs for performance in this viewer
        if (aData && aData.logs) {
            aData.logs.forEach(log => {
                const div = document.createElement('div');
                div.className = 'log-entry';
                div.innerHTML = `<span class="log-agent">[${log.agent}]</span> <span class="log-action">${log.action}</span>: ${log.message}`;
                logContainer.prepend(div);
                // Keep only last 50 logs
                if (logContainer.children.length > 50) logContainer.lastChild.remove();
            });
        }
    }

    currentStep = step;
}

function updateChart(chart, labels, dataArrays) {
    chart.data.labels = labels;
    dataArrays.forEach((data, index) => {
        if (chart.data.datasets[index]) {
            chart.data.datasets[index].data = data;
        }
    });
    chart.update('none'); // No animation for performance
}

function switchView(viewName) {
    document.querySelectorAll('.scenario-view').forEach(el => el.classList.remove('active'));
    document.getElementById('view-' + viewName).classList.add('active');

    document.querySelectorAll('.ent-nav-item').forEach(el => el.classList.remove('active'));
    // Ideally find the nav item that was clicked, but we can just use event logic or ID
    event.currentTarget.classList.add('active');

    document.getElementById('page-title').innerText = viewName.toUpperCase().replace('_', ' ') + " SIMULATION";
}

function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('playBtn');
    if (isPlaying) {
        btn.innerHTML = '<i class="fas fa-pause"></i> PAUSE';
        btn.classList.add('active');
        playInterval = setInterval(() => {
            if (currentStep < simData.market.length - 1) {
                updateUI(currentStep + 1);
            } else {
                togglePlay(); // Stop at end
            }
        }, 200);
    } else {
        btn.innerHTML = '<i class="fas fa-play"></i> PLAY';
        btn.classList.remove('active');
        clearInterval(playInterval);
    }
}

function resetSim() {
    if (isPlaying) togglePlay();
    updateUI(0);
    document.getElementById('agent-log-container').innerHTML = '';
}

function scrubTimeline(value) {
    if (isPlaying) togglePlay(); // Pause on scrub
    updateUI(parseInt(value));
}

function logSystemEvent(msg) {
    const stream = document.getElementById('system-log');
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.innerText = "> " + msg;
    stream.prepend(div);
}
