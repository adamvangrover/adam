// Simulation Viewer Logic
console.log("Simulation Viewer Loaded.");

let simData = null;
let currentIndex = 0;
let isPlaying = false;
let intervalId = null;
let charts = {};

// Load Data
try {
    const dataElement = document.getElementById('sim-data');
    if (dataElement) {
        simData = JSON.parse(dataElement.textContent);
        console.log("Simulation Data Loaded:", simData);
        initDashboard();
    } else {
        console.error("Simulation Data Element NOT found.");
    }
} catch (e) {
    console.error("Error parsing simulation data:", e);
}

function initDashboard() {
    document.getElementById('sim-scenario-name').innerText = simData.metadata.scenario;
    initCharts();
    updateUI(0); // Show initial state
}

function initCharts() {
    const ctxPerf = document.getElementById('performanceChart').getContext('2d');
    const ctxSent = document.getElementById('sentimentChart').getContext('2d');

    charts.perf = new Chart(ctxPerf, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Portfolio Value ($)',
                    data: [],
                    borderColor: '#00f3ff',
                    backgroundColor: 'rgba(0, 243, 255, 0.1)',
                    yAxisID: 'y'
                },
                {
                    label: 'S&P 500',
                    data: [],
                    borderColor: '#ff00ff',
                    borderDash: [5, 5],
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { type: 'linear', display: true, position: 'left', grid: { color: '#333' } },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
            }
        }
    });

    charts.sent = new Chart(ctxSent, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Market Sentiment',
                data: [],
                backgroundColor: (context) => {
                    const value = context.raw;
                    return value > 60 ? '#00ff00' : value < 40 ? '#ff0000' : '#ffff00';
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 100, grid: { color: '#333' } }
            }
        }
    });
}

function updateUI(index) {
    if (!simData || index >= simData.timeline.length) {
        pauseReplay();
        return;
    }

    const day = simData.timeline[index];

    // Update Stats
    document.getElementById('stat-date').innerText = day.date;
    document.getElementById('stat-portfolio').innerText = "$" + day.portfolio_value.toLocaleString();
    document.getElementById('stat-sentiment').innerText = day.sentiment;
    document.getElementById('stat-spx').innerText = day.spx.toLocaleString();

    // Update Logs
    const logContainer = document.getElementById('simulation-log');
    // Clear logs if resetting or just starting? No, append or show up to index.
    // For replay, we might want to append.

    // Only append new logs for this specific day
    if (day.actions.length > 0) {
        day.actions.forEach(action => {
            const div = document.createElement('div');
            div.className = 'sim-log-item';
            div.innerHTML = `<span class="sim-log-date">[${day.date}]</span> <span class="sim-log-action">${action}</span>`;
            logContainer.prepend(div); // Add to top
        });
    }

    // Update Charts (Incremental)
    // We need to slice the data up to current index
    const history = simData.timeline.slice(0, index + 1);

    charts.perf.data.labels = history.map(d => d.date);
    charts.perf.data.datasets[0].data = history.map(d => d.portfolio_value);
    charts.perf.data.datasets[1].data = history.map(d => d.spx);
    charts.perf.update('none'); // Update without animation for smooth replay

    charts.sent.data.labels = history.map(d => d.date);
    charts.sent.data.datasets[0].data = history.map(d => d.sentiment);
    charts.sent.update('none');

    currentIndex = index;
}

// Controls
function startReplay() {
    if (isPlaying) return;
    isPlaying = true;

    // If we are at the end, reset
    if (currentIndex >= simData.timeline.length - 1) {
        resetReplay();
        isPlaying = true; // Re-set because resetReplay sets it to false
    }

    intervalId = setInterval(() => {
        currentIndex++;
        if (currentIndex < simData.timeline.length) {
            updateUI(currentIndex);
        } else {
            pauseReplay();
        }
    }, 500); // 500ms per day
}

function pauseReplay() {
    isPlaying = false;
    if (intervalId) clearInterval(intervalId);
}

function resetReplay() {
    pauseReplay();
    currentIndex = 0;
    // Clear logs
    document.getElementById('simulation-log').innerHTML = '';
    // Clear charts
    charts.perf.data.labels = [];
    charts.perf.data.datasets.forEach(ds => ds.data = []);
    charts.perf.update();

    charts.sent.data.labels = [];
    charts.sent.data.datasets.forEach(ds => ds.data = []);
    charts.sent.update();

    updateUI(0);
}
