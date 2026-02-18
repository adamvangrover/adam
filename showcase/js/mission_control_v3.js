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
        this.manifestPath = 'data/report_manifest.json';
    }

    init() {
        console.log("Mission Control v3 Initialized.");
        this.startSimulation();
        this.fetchManifest();
    }

    startSimulation() {
        // Update Metrics
        setInterval(() => this.updateMetrics(), 2000);
        // Swarm Updates
        setInterval(() => this.updateSwarm(), 1500);
        // System 2 Verification Logs
        setInterval(() => this.injectLog(), 5000);
        // Crisis Sim Status
        setInterval(() => this.updateSimStatus(), 4000);
    }

    updateMetrics() {
        const volatility = (15 + Math.random() * 10).toFixed(2);
        const load = Math.floor(20 + Math.random() * 60);

        const vixEl = document.getElementById('metric-vix');
        const loadEl = document.getElementById('metric-load');

        if(vixEl) vixEl.innerText = volatility;
        if(loadEl) loadEl.innerText = load + '%';

        // Update DEFCON based on VIX
        if (volatility > 24) this.setDefcon(3);
        else if (volatility > 20) this.setDefcon(4);
        else this.setDefcon(5);
    }

    updateSimStatus() {
        const statuses = ["IDLE", "RUNNING", "COMPUTING", "VERIFYING"];
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        const el = document.getElementById('metric-sim-status');
        if (el) {
            el.innerText = status;
            el.style.color = status === "IDLE" ? "#666" : "#00f3ff";
            if (status === "RUNNING") el.style.color = "#cc0000";
        }
    }

    setDefcon(level) {
        if (this.defconLevel === level) return;
        this.defconLevel = level;
        const badge = document.getElementById('defcon-badge');
        if (badge) {
            badge.innerText = `DEFCON ${level}`;

            if (level <= 3) {
                badge.style.color = 'red';
                badge.style.borderColor = 'red';
            } else {
                badge.style.color = '#00f3ff';
                badge.style.borderColor = '#00f3ff';
            }
        }
    }

    updateSwarm() {
        const container = document.getElementById('swarm-list');
        if (!container) return;

        // Randomly update one agent status
        const agentName = this.agents[Math.floor(Math.random() * this.agents.length)];
        const task = this.tasks[Math.floor(Math.random() * this.tasks.length)];

        // Find existing row
        const rows = Array.from(container.children);
        const existingRow = rows.find(row => row.innerText.includes(agentName));

        if (existingRow) {
            const taskEl = existingRow.querySelector('.agent-task');
            const dotEl = existingRow.querySelector('.agent-status-dot');
            if (taskEl) taskEl.innerText = task;
            if (dotEl) {
                dotEl.className = 'agent-status-dot active';
                setTimeout(() => {
                    dotEl.className = 'agent-status-dot';
                }, 500);
            }
        }
    }

    injectLog() {
        const logContainer = document.getElementById('sys2-log');
        if (!logContainer) return;

        const messages = [
            "VERIFIED: Market_Mayhem_Jan_2026.html [HASH_MATCH]",
            "ALERT: Divergence detected in Bond Yields",
            "SYSTEM: Optimization Cycle Complete",
            "NETWORK: New Node Discovered (Graph ID: 882a)",
            "SECURITY: Handshake Validated",
            "RAG: Ingesting 10-K Data...",
            "CRISIS: Scenario '2022_INFLATION_SHOCK' Loaded"
        ];
        const msg = messages[Math.floor(Math.random() * messages.length)];
        const time = new Date().toLocaleTimeString('en-US', {hour12: false});

        const line = document.createElement('div');
        line.innerText = `[${time}] > ${msg}`;
        line.style.opacity = '0';
        line.style.transition = 'opacity 0.5s';

        logContainer.prepend(line);

        // Trigger reflow
        void line.offsetWidth;
        line.style.opacity = '1';

        if (logContainer.children.length > 20) {
            logContainer.lastChild.remove();
        }
    }

    async fetchManifest() {
        try {
            const response = await fetch(this.manifestPath);
            if (!response.ok) throw new Error("Manifest fetch failed");
            const data = await response.json();

            // Filter relevant reports
            const relevant = data.filter(item =>
                item.type === 'MARKET_MAYHEM' ||
                item.type === 'DAILY_BRIEFING' ||
                item.type === 'HOUSE_VIEW'
            );

            // Sort by date (descending)
            // Helper to parse date
            const parseDate = (d) => {
                if (!d || d === "Unknown") return new Date(0);
                return new Date(d);
            };

            relevant.sort((a, b) => parseDate(b.date) - parseDate(a.date));

            // Take top 10
            const top = relevant.slice(0, 10);
            this.renderIntel(top);

        } catch (e) {
            console.error("Failed to load intel:", e);
            const el = document.getElementById('latest-intel');
            if(el) el.innerHTML = '<div style="color:red;">CONNECTION FAILED: MANIFEST_OFFLINE</div>';
        }
    }

    renderIntel(items) {
        const container = document.getElementById('latest-intel');
        if (!container) return;

        container.innerHTML = '';

        items.forEach(item => {
            const div = document.createElement('div');
            div.style.padding = '8px 5px';
            div.style.borderBottom = '1px solid #333';
            div.style.cursor = 'pointer';
            div.className = 'intel-item'; // For hover effects if css added

            const title = document.createElement('div');
            title.style.color = '#ccc';
            title.innerText = item.title;

            const meta = document.createElement('div');
            meta.style.display = 'flex';
            meta.style.justifyContent = 'space-between';
            meta.style.fontSize = '0.65rem';
            meta.style.color = '#666';

            const dateSpan = document.createElement('span');
            dateSpan.innerText = item.date;

            const typeSpan = document.createElement('span');
            typeSpan.innerText = item.type;
            if (item.type === 'MARKET_MAYHEM') typeSpan.style.color = '#cc0000';
            if (item.type === 'DAILY_BRIEFING') typeSpan.style.color = '#00f3ff';

            meta.appendChild(dateSpan);
            meta.appendChild(typeSpan);

            div.appendChild(title);
            div.appendChild(meta);

            div.onclick = () => window.location.href = item.path;

            // Hover effect logic inline for simplicity
            div.onmouseenter = () => div.style.backgroundColor = 'rgba(255,255,255,0.05)';
            div.onmouseleave = () => div.style.backgroundColor = 'transparent';

            container.appendChild(div);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.missionControl = new MissionControl();
    window.missionControl.init();
});
