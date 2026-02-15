class MissionControl {
    constructor() {
        this.agents = [
            "Market_Maker_01", "Risk_Engine_Alpha", "Sentiment_Crawler_V2", "Nexus_Link_Node",
            "Credit_Analyst_Core", "Global_Macro_Eye", "Crypto_Arb_Bot", "News_Synthesizer"
        ];
        this.tasks = [
            "Scanning 10-K filings...", "Monitoring spread duration...", "Rebalancing portfolio...",
            "Verifying source hash...", "Indexing graph nodes...", "Calculating VaR..."
        ];
        this.defconLevel = 5;
    }

    init() {
        console.log("Mission Control v3 Initialized.");
        this.startSimulation();
    }

    startSimulation() {
        // Update Metrics
        setInterval(() => this.updateMetrics(), 2000);
        // Swarm Updates
        setInterval(() => this.updateSwarm(), 1500);
        // System 2 Verification Logs
        setInterval(() => this.injectLog(), 5000);
    }

    updateMetrics() {
        const volatility = (15 + Math.random() * 10).toFixed(2);
        const load = Math.floor(20 + Math.random() * 60);

        document.getElementById('metric-vix').innerText = volatility;
        document.getElementById('metric-load').innerText = load + '%';

        // Update DEFCON based on VIX
        if (volatility > 24) this.setDefcon(3);
        else if (volatility > 20) this.setDefcon(4);
        else this.setDefcon(5);
    }

    setDefcon(level) {
        if (this.defconLevel === level) return;
        this.defconLevel = level;
        const badge = document.getElementById('defcon-badge');
        badge.innerText = `DEFCON ${level}`;

        if (level <= 3) {
            badge.style.color = 'red';
            badge.style.borderColor = 'red';
        } else {
            badge.style.color = '#00f3ff';
            badge.style.borderColor = '#00f3ff';
        }
    }

    updateSwarm() {
        const container = document.getElementById('swarm-list');
        // Randomly update one agent status
        const agentName = this.agents[Math.floor(Math.random() * this.agents.length)];
        const task = this.tasks[Math.floor(Math.random() * this.tasks.length)];

        // Simple rebuild for prototype (in production, update specific DOM elements)
        // For visual flair, we'll just prepend a new "Action" or update existing row logic
        // Here we just keep it static-looking but "alive"

        const existingRow = Array.from(container.children).find(row => row.innerText.includes(agentName));
        if (existingRow) {
            existingRow.querySelector('.agent-task').innerText = task;
            existingRow.querySelector('.agent-status-dot').className = 'agent-status-dot active';
            setTimeout(() => {
                existingRow.querySelector('.agent-status-dot').className = 'agent-status-dot';
            }, 500);
        }
    }

    injectLog() {
        const logContainer = document.getElementById('sys2-log');
        const messages = [
            "VERIFIED: Market_Mayhem_Jan_2026.html [HASH_MATCH]",
            "ALERT: Divergence detected in Bond Yields",
            "SYSTEM: Optimization Cycle Complete",
            "NETWORK: New Node Discovered (Graph ID: 882a)",
            "SECURITY: Handshake Validated"
        ];
        const msg = messages[Math.floor(Math.random() * messages.length)];
        const time = new Date().toLocaleTimeString('en-US', {hour12: false});

        const line = document.createElement('div');
        line.innerText = `[${time}] > ${msg}`;
        logContainer.prepend(line);

        if (logContainer.children.length > 20) {
            logContainer.lastChild.remove();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.missionControl = new MissionControl();
    window.missionControl.init();
});
