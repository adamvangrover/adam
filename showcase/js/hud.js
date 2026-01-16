// Adam System HUD
// Injects a fixed status bar at the bottom of the screen

(function() {
    const css = `
        #adam-hud {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            height: 24px;
            background: rgba(2, 6, 23, 0.95);
            border-top: 1px solid #1e293b;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: #64748b;
            z-index: 100;
            backdrop-filter: blur(4px);
        }
        #adam-hud .hud-section { display: flex; align-items: center; gap: 15px; }
        #adam-hud .hud-item { display: flex; align-items: center; gap: 6px; }
        #adam-hud .hud-val { color: #e2e8f0; font-weight: bold; }
        #adam-hud .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 5px #22c55e; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    `;

    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);

    const hud = document.createElement('div');
    hud.id = 'adam-hud';
    hud.innerHTML = `
        <div class="hud-section">
            <div class="hud-item">
                <div class="status-dot"></div>
                <span style="color: #22c55e; letter-spacing: 1px;">SYSTEM ONLINE</span>
            </div>
            <div class="hud-item">
                <span>VERSION:</span> <span class="hud-val">v24.1.0-ALPHA</span>
            </div>
             <div class="hud-item">
                <span>ENV:</span> <span class="hud-val">SIMULATION_MAIN</span>
            </div>
        </div>

        <div class="hud-section">
            <div class="hud-item">
                <i class="fas fa-microchip"></i>
                <span>AGENTS:</span> <span class="hud-val" id="hud-agents">Loading...</span>
            </div>
            <div class="hud-item">
                <i class="fas fa-thermometer-half"></i>
                <span>MARKET TEMP:</span> <span class="hud-val" style="color: #f59e0b">VOLATILE</span>
            </div>
             <div class="hud-item">
                <i class="fas fa-clock"></i>
                <span id="hud-time">00:00:00 UTC</span>
            </div>
        </div>
    `;

    document.body.appendChild(hud);

    // Dynamic Data
    setInterval(() => {
        const now = new Date();
        document.getElementById('hud-time').innerText = now.toISOString().split('T')[1].split('.')[0] + " UTC";
    }, 1000);

    // Mock fetch agent count
    setTimeout(() => {
        // Try to read from window.ADAM_STATE or random
        const count = (window.ADAM_STATE && window.ADAM_STATE.agents) ? window.ADAM_STATE.agents.length : 42;
        document.getElementById('hud-agents').innerText = count;
    }, 1000);

})();
