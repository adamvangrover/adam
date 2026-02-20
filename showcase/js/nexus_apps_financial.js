// Nexus Apps: Financial Suite
// Adds Financial Modeler, Report Viewer, and Briefing Center

(function() {
    const APP_CONFIG = [
        {
            name: 'FinancialModeler',
            icon: 'https://img.icons8.com/color/48/000000/graph-clique.png',
            title: 'Financial Modeler',
            width: 900,
            height: 650
        },
        {
            name: 'ReportViewer',
            icon: 'https://img.icons8.com/color/48/000000/document.png',
            title: 'Report Viewer',
            width: 800,
            height: 800
        },
        {
            name: 'BriefingCenter',
            icon: 'https://img.icons8.com/color/48/000000/news.png',
            title: 'Briefing Center',
            width: 1000,
            height: 700
        }
    ];

    function initFinancialApps() {
        if (!window.officeOS) {
            setTimeout(initFinancialApps, 500);
            return;
        }

        patchAppRegistry();
        injectIcons();
        console.log("Nexus Financial Suite: Loaded.");
    }

    function patchAppRegistry() {
        const originalLaunch = window.officeOS.appRegistry.launch.bind(window.officeOS.appRegistry);

        window.officeOS.appRegistry.launch = function(appName, args) {
            if (appName === 'FinancialModeler') {
                launchFinancialModeler(args);
            } else if (appName === 'ReportViewer') {
                launchReportViewer(args);
            } else if (appName === 'BriefingCenter') {
                launchBriefingCenter(args);
            } else {
                originalLaunch(appName, args);
            }
        };
    }

    function injectIcons() {
        // Desktop
        const desktop = document.getElementById('desktop');
        if (desktop) {
            APP_CONFIG.forEach(app => {
                const el = document.createElement('div');
                el.className = 'desktop-icon';
                el.innerHTML = `<img src="${app.icon}"><span>${app.title}</span>`;
                el.addEventListener('dblclick', () => window.officeOS.appRegistry.launch(app.name));
                desktop.appendChild(el);
            });
        }

        // Start Menu
        const startMenuGrid = document.querySelector('.start-menu-grid');
        if (startMenuGrid) {
            APP_CONFIG.forEach(app => {
                const el = document.createElement('div');
                el.className = 'start-menu-item';
                el.innerHTML = `<img src="${app.icon}"><span>${app.title}</span>`;
                el.addEventListener('click', () => {
                    window.officeOS.appRegistry.launch(app.name);
                    window.officeOS.toggleStartMenu();
                });
                startMenuGrid.appendChild(el);
            });
        }
    }

    // --- Financial Modeler ---
    function launchFinancialModeler(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Financial Modeler',
            icon: APP_CONFIG[0].icon,
            width: APP_CONFIG[0].width,
            height: APP_CONFIG[0].height,
            app: 'FinancialModeler'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.height = '100%';

        // Sidebar
        const sidebar = document.createElement('div');
        sidebar.style.width = '200px';
        sidebar.style.borderRight = '1px solid #ddd';
        sidebar.style.backgroundColor = '#f5f5f5';
        sidebar.style.padding = '10px';
        sidebar.innerHTML = `
            <div style="font-weight:bold; margin-bottom:10px;">Load Model</div>
            <select id="model-select-${winId}" style="width:100%; padding:5px; margin-bottom:10px;">
                <option value="">Select Ticker...</option>
                <!-- Options populated dynamically -->
            </select>
            <button class="cyber-btn" id="load-btn-${winId}" style="width:100%;">Load</button>
        `;

        // Main
        const main = document.createElement('div');
        main.id = `model-main-${winId}`;
        main.style.flex = '1';
        main.style.padding = '20px';
        main.style.overflow = 'auto';
        main.innerHTML = '<div style="text-align:center; margin-top:100px; color:#888;">Load a DCF or EV model to begin.</div>';

        container.appendChild(sidebar);
        container.appendChild(main);
        window.officeOS.windowManager.setWindowContent(winId, container);

        // Populate Tickers (Mock list or fetch from fs)
        const tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "PG"];
        const select = sidebar.querySelector(`#model-select-${winId}`);
        tickers.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.innerText = t;
            select.appendChild(opt);
        });

        // Load Logic
        sidebar.querySelector(`#load-btn-${winId}`).addEventListener('click', async () => {
            const ticker = select.value;
            if (!ticker) return;

            try {
                // Try to fetch Enterprise Model v2 first, then fallback to DCF
                let res = await fetch(`data/models/${ticker}_Financial_Model_v2.json`);
                if (res.ok) {
                    const data = await res.json();
                    renderEnterpriseModel(winId, data);
                } else {
                    res = await fetch(`data/models/${ticker}_DCF.json`);
                    if (res.ok) {
                        const data = await res.json();
                        renderDCFModel(winId, data);
                    } else {
                        main.innerHTML = `Error: No model found for ${ticker}.`;
                    }
                }
            } catch (e) {
                main.innerHTML = `Error loading model: ${e}`;
            }
        });
    }

    function renderEnterpriseModel(winId, data) {
        const main = document.getElementById(`model-main-${winId}`);
        const fins = data.financials[2025];
        const audit = data.audit_trail || {};

        main.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <div>
                    <h2 style="margin:0;">${data.ticker} Enterprise Model</h2>
                    <div style="font-size:12px; color:#28a745;">
                        ✓ Verified by ${audit.auditor_id || 'System'} | Cutoff: ${audit.data_cutoff || 'N/A'}
                    </div>
                </div>
                <button class="cyber-btn" onclick="alert('Model Saved!')" style="background:#28a745; color:white;">Save v2.0</button>
            </div>

            <div style="display:grid; grid-template-columns: 2fr 1fr; gap:20px;">
                <div class="panel" style="background:#fff; padding:15px; border:1px solid #ddd; border-radius:5px;">
                    <h3>2025 Estimates (Base Case)</h3>
                    <table style="width:100%; font-size:14px;">
                        <tr><td>Revenue:</td><td style="font-weight:bold;">$${fins['Income Statement']['Revenue']}B</td></tr>
                        <tr><td>EBITDA:</td><td style="font-weight:bold;">$${fins['Income Statement']['EBITDA']}B</td></tr>
                        <tr><td>Net Income:</td><td style="font-weight:bold;">$${fins['Income Statement']['Net Income']}B</td></tr>
                        <tr><td>Free Cash Flow:</td><td style="font-weight:bold;">$${fins['Cash Flow']['Free Cash Flow']}B</td></tr>
                    </table>
                </div>
                <div class="panel" style="background:#f8f9fa; padding:15px; border:1px solid #ccc; font-size:12px;">
                    <strong>Audit Metadata</strong><br><br>
                    Model: ${audit.model_version}<br>
                    Confidence: ${(audit.confidence_score * 100).toFixed(1)}%<br>
                    Hash: <span style="font-family:monospace;">${audit.data_hash || 'N/A'}</span>
                </div>
            </div>

            <div style="margin-top:20px; padding:15px; background:#fff; border:1px solid #ddd;">
                <h3>3-Year Trajectory</h3>
                <table style="width:100%; text-align:center;">
                    <tr style="background:#eee; font-weight:bold;"><td>Metric</td><td>2023</td><td>2024</td><td>2025</td></tr>
                    <tr>
                        <td style="text-align:left;">Revenue</td>
                        <td>${data.financials[2023]['Income Statement']['Revenue']}</td>
                        <td>${data.financials[2024]['Income Statement']['Revenue']}</td>
                        <td>${data.financials[2025]['Income Statement']['Revenue']}</td>
                    </tr>
                    <tr>
                        <td style="text-align:left;">EBITDA</td>
                        <td>${data.financials[2023]['Income Statement']['EBITDA']}</td>
                        <td>${data.financials[2024]['Income Statement']['EBITDA']}</td>
                        <td>${data.financials[2025]['Income Statement']['EBITDA']}</td>
                    </tr>
                </table>
            </div>
        `;
    }

    function renderDCFModel(winId, data) {
        const main = document.getElementById(`model-main-${winId}`);
        main.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h2>${data.ticker} DCF Analysis</h2>
                <button class="cyber-btn" onclick="alert('Model Saved!')" style="background:#28a745; color:white;">Save Model</button>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                <!-- Inputs -->
                <div class="panel" style="background:#fff; padding:15px; border:1px solid #ddd; border-radius:5px;">
                    <h3>Assumptions</h3>
                    <div style="margin-bottom:15px;">
                        <label>WACC (%)</label>
                        <input type="range" min="5" max="15" step="0.1" value="${(data.wacc*100).toFixed(1)}"
                               oninput="document.getElementById('wacc-val-${winId}').innerText = this.value + '%'; window.recalcDCF('${winId}')"
                               id="wacc-input-${winId}" style="width:100%;">
                        <span id="wacc-val-${winId}" style="float:right; font-weight:bold;">${(data.wacc*100).toFixed(1)}%</span>
                    </div>
                    <div style="margin-bottom:15px;">
                        <label>Growth Rate (%)</label>
                        <input type="range" min="0" max="20" step="0.1" value="${(data.growth_rate*100).toFixed(1)}"
                               oninput="document.getElementById('growth-val-${winId}').innerText = this.value + '%'; window.recalcDCF('${winId}')"
                               id="growth-input-${winId}" style="width:100%;">
                        <span id="growth-val-${winId}" style="float:right; font-weight:bold;">${(data.growth_rate*100).toFixed(1)}%</span>
                    </div>
                    <div style="margin-bottom:15px;">
                        <label>Terminal Growth (%)</label>
                        <input type="number" step="0.1" value="${(data.terminal_growth*100).toFixed(1)}"
                               id="tv-input-${winId}" style="width:100%; padding:5px;">
                    </div>
                </div>

                <!-- Outputs -->
                <div class="panel" style="background:#fff; padding:15px; border:1px solid #ddd; border-radius:5px;">
                    <h3>Valuation Output</h3>
                    <div style="font-size:36px; font-weight:bold; color:#0078d7; text-align:center; margin:20px 0;" id="val-output-${winId}">
                        $${data.implied_value}
                    </div>
                    <div style="text-align:center;">
                        Current Price: $${data.current_price}<br>
                        <span id="upside-${winId}" style="font-weight:bold; color:${data.implied_value > data.current_price ? 'green' : 'red'};">
                            ${((data.implied_value/data.current_price - 1)*100).toFixed(1)}% Upside
                        </span>
                    </div>
                </div>
            </div>

            <div style="margin-top:20px;">
                <h3>Cash Flow Projections</h3>
                <table style="width:100%; border-collapse:collapse; text-align:center;">
                    <tr style="background:#f0f0f0;"><th>Year 1</th><th>Year 2</th><th>Year 3</th><th>Year 4</th><th>Year 5</th></tr>
                    <tr>
                        ${data.fcf_projections.map(f => `<td style="padding:10px; border:1px solid #eee;">$${f}</td>`).join('')}
                    </tr>
                </table>
            </div>

            <!-- Store raw data for recalc -->
            <input type="hidden" id="raw-fcf-${winId}" value="${data.fcf_projections[0] / (1+data.growth_rate)}">
        `;

        // Attach global recalc function
        window.recalcDCF = (wId) => {
            const wacc = parseFloat(document.getElementById(`wacc-input-${wId}`).value) / 100;
            const growth = parseFloat(document.getElementById(`growth-input-${wId}`).value) / 100;
            const tvGrowth = parseFloat(document.getElementById(`tv-input-${wId}`).value) / 100;
            const baseFCF = parseFloat(document.getElementById(`raw-fcf-${wId}`).value);

            let val = 0;
            let currentFCF = baseFCF;
            const projections = [];

            for(let i=1; i<=5; i++) {
                currentFCF *= (1 + growth);
                val += currentFCF / Math.pow(1 + wacc, i);
                projections.push(currentFCF);
            }

            const terminalValue = (currentFCF * (1 + tvGrowth)) / (wacc - tvGrowth);
            val += terminalValue / Math.pow(1 + wacc, 5);

            document.getElementById(`val-output-${wId}`).innerText = '$' + val.toFixed(2);
        };
    }

    // --- Report Viewer ---
    function launchReportViewer(args) {
        const path = args.path || args.url;
        const winId = window.officeOS.windowManager.createWindow({
            title: args.name || 'Report Viewer',
            icon: APP_CONFIG[1].icon,
            width: APP_CONFIG[1].width,
            height: APP_CONFIG[1].height,
            app: 'ReportViewer'
        });

        const iframe = document.createElement('iframe');
        iframe.src = path;
        iframe.style.width = '100%';
        iframe.style.height = '100%';
        iframe.style.border = 'none';
        iframe.style.background = '#fff';

        window.officeOS.windowManager.setWindowContent(winId, iframe);
    }

    // --- Briefing Center ---
    function launchBriefingCenter(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Briefing Center',
            icon: APP_CONFIG[2].icon,
            width: APP_CONFIG[2].width,
            height: APP_CONFIG[2].height,
            app: 'BriefingCenter'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.height = '100%';

        // Sidebar
        const sidebar = document.createElement('div');
        sidebar.style.width = '250px';
        sidebar.style.borderRight = '1px solid #333';
        sidebar.style.backgroundColor = '#222';
        sidebar.style.color = '#fff';
        sidebar.innerHTML = `
            <div style="padding:15px; font-weight:bold; font-size:18px;">Headlines</div>
            <div class="briefing-list" id="briefing-list-${winId}">
                <!-- Items -->
            </div>
        `;

        // Main Reading Pane
        const main = document.createElement('div');
        main.id = `briefing-main-${winId}`;
        main.style.flex = '1';
        main.style.background = '#fcfcfc';

        container.appendChild(sidebar);
        container.appendChild(main);
        window.officeOS.windowManager.setWindowContent(winId, container);

        // Populate Items
        const items = [
            { title: "Morning Call", date: "Feb 19", path: "data/briefings/Morning_Call_Feb_19_2026.html" },
            { title: "Market Mayhem", date: "Feb 2026", path: "data/newsletters/Market_Mayhem_Feb_2026.html" },
            { title: "AI Bubble Analysis", date: "Deep Dive", path: "data/deep_dives/AI_Bubble_Analysis.html" }
        ];

        const list = sidebar.querySelector(`#briefing-list-${winId}`);
        items.forEach(item => {
            const el = document.createElement('div');
            el.style.padding = '10px 15px';
            el.style.borderBottom = '1px solid #333';
            el.style.cursor = 'pointer';
            el.innerHTML = `<div style="font-weight:bold;">${item.title}</div><div style="font-size:11px; color:#888;">${item.date}</div>`;
            el.addEventListener('click', () => {
                const iframe = document.createElement('iframe');
                iframe.src = item.path;
                iframe.style.width = '100%';
                iframe.style.height = '100%';
                iframe.style.border = 'none';
                main.innerHTML = '';
                main.appendChild(iframe);
            });
            list.appendChild(el);
        });

        // Load first item
        if (items.length > 0) list.firstChild.click();
    }

    initFinancialApps();
})();
