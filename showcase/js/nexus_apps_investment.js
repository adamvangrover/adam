// Nexus Apps: Investment Suite
// Adds Portfolio Master and Scenario Lab

(function() {
    const APP_CONFIG = [
        {
            name: 'PortfolioMaster',
            icon: 'https://img.icons8.com/color/48/000000/portfolio.png',
            title: 'Portfolio Master',
            width: 1000,
            height: 700
        },
        {
            name: 'ScenarioLab',
            icon: 'https://img.icons8.com/color/48/000000/test-tube.png',
            title: 'Scenario Lab',
            width: 900,
            height: 600
        }
    ];

    function initInvestmentApps() {
        if (!window.officeOS) {
            setTimeout(initInvestmentApps, 500);
            return;
        }

        patchAppRegistry();
        injectIcons();
        console.log("Nexus Investment Suite: Loaded.");
    }

    function patchAppRegistry() {
        const originalLaunch = window.officeOS.appRegistry.launch.bind(window.officeOS.appRegistry);

        window.officeOS.appRegistry.launch = function(appName, args) {
            if (appName === 'PortfolioMaster') {
                launchPortfolioMaster(args);
            } else if (appName === 'ScenarioLab') {
                launchScenarioLab(args);
            } else {
                originalLaunch(appName, args);
            }
        };
    }

    function injectIcons() {
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
        // Start Menu... (Optional, skipping for brevity as desktop is enough for demo)
    }

    // --- Portfolio Master ---
    function launchPortfolioMaster(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Portfolio Master',
            icon: APP_CONFIG[0].icon,
            width: APP_CONFIG[0].width,
            height: APP_CONFIG[0].height,
            app: 'PortfolioMaster'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.height = '100%';
        container.innerHTML = `
            <div style="background:#eee; padding:10px; border-bottom:1px solid #ccc; display:flex; gap:10px;">
                <select id="fund-select-${winId}" style="padding:5px;">
                    <option value="Global_Macro_Opportunities_Fund">Global Macro Opportunities</option>
                    <option value="AI_Disruption_Alpha_Fund">AI Disruption Alpha</option>
                    <option value="Secure_Income_Yield_Fund">Secure Income Yield</option>
                </select>
                <button class="cyber-btn" id="load-fund-${winId}">Load Fund</button>
            </div>
            <div id="fund-view-${winId}" style="flex:1; padding:20px; overflow:auto;">
                <div style="text-align:center; margin-top:100px; color:#888;">Select a fund to view holdings.</div>
            </div>
        `;
        window.officeOS.windowManager.setWindowContent(winId, container);

        container.querySelector(`#load-fund-${winId}`).addEventListener('click', async () => {
            const fund = container.querySelector(`#fund-select-${winId}`).value;
            try {
                const res = await fetch(`data/portfolios/${fund}.json`);
                if(res.ok) {
                    const data = await res.json();
                    renderPortfolio(winId, data);
                }
            } catch(e) {
                console.error(e);
            }
        });
    }

    function renderPortfolio(winId, data) {
        const view = document.getElementById(`fund-view-${winId}`);
        const audit = data.audit_trail;

        let html = `
            <div style="display:flex; justify-content:space-between; margin-bottom:20px;">
                <div>
                    <h2 style="margin:0;">${data.fund_name}</h2>
                    <div style="color:#666;">AUM: $${(data.aum/1000000000).toFixed(2)}B | Strategy: ${data.strategy}</div>
                </div>
                <div style="text-align:right; font-size:12px; background:#f8f9fa; padding:10px; border:1px solid #ddd;">
                    <strong>COMPLIANCE CHECK</strong><br>
                    <span style="color:green;">✔ ${audit.risk_check}</span> | ID: ${audit.compliance_officer}<br>
                    Last Rebalance: ${audit.last_rebalance}
                </div>
            </div>

            <table style="width:100%; border-collapse:collapse; font-size:14px;">
                <thead>
                    <tr style="background:#333; color:white;">
                        <th style="padding:10px; text-align:left;">Ticker</th>
                        <th style="padding:10px; text-align:left;">Name</th>
                        <th style="padding:10px; text-align:right;">Weight</th>
                        <th style="padding:10px; text-align:right;">Qty</th>
                        <th style="padding:10px; text-align:right;">Cost Basis</th>
                        <th style="padding:10px; text-align:right;">Mkt Price</th>
                        <th style="padding:10px; text-align:right;">Mkt Value</th>
                        <th style="padding:10px; text-align:right;">Unrealized PnL</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.holdings.forEach(h => {
            const pnlColor = h.pnl >= 0 ? 'green' : 'red';
            html += `
                <tr style="border-bottom:1px solid #eee;">
                    <td style="padding:8px;"><strong>${h.ticker}</strong></td>
                    <td style="padding:8px;">${h.name}</td>
                    <td style="padding:8px; text-align:right;">${(h.weight*100).toFixed(2)}%</td>
                    <td style="padding:8px; text-align:right;">${h.quantity.toLocaleString()}</td>
                    <td style="padding:8px; text-align:right;">$${h.avg_cost.toFixed(2)}</td>
                    <td style="padding:8px; text-align:right;">$${h.current_price.toFixed(2)}</td>
                    <td style="padding:8px; text-align:right;">$${(h.market_value/1000000).toFixed(1)}M</td>
                    <td style="padding:8px; text-align:right; color:${pnlColor}; font-weight:bold;">
                        $${(h.pnl/1000000).toFixed(1)}M (${(h.pnl_pct*100).toFixed(1)}%)
                    </td>
                </tr>
            `;
        });

        html += `</tbody></table>`;
        view.innerHTML = html;
    }

    // --- Scenario Lab ---
    function launchScenarioLab(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Scenario Lab',
            icon: APP_CONFIG[1].icon,
            width: APP_CONFIG[1].width,
            height: APP_CONFIG[1].height,
            app: 'ScenarioLab'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.innerHTML = `
            <h3>Stress Test Simulator</h3>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
                <div>
                    <label>Select Portfolio</label>
                    <select id="sim-fund-${winId}" style="width:100%; padding:8px;">
                         <option value="Global_Macro_Opportunities_Fund">Global Macro Opportunities</option>
                         <option value="AI_Disruption_Alpha_Fund">AI Disruption Alpha</option>
                    </select>
                </div>
                <div>
                    <label>Select Scenario</label>
                    <select id="sim-scen-${winId}" style="width:100%; padding:8px;">
                        <option value="Hard_Landing_2026">Hard Landing (Yield Curve Inversion)</option>
                        <option value="AI_Supercycle_2026">AI Supercycle (Productivity Boom)</option>
                        <option value="Stagflation_Shock">Stagflation Shock (Oil Crisis)</option>
                    </select>
                </div>
            </div>
            <button class="cyber-btn" id="run-sim-${winId}" style="width:100%; margin-bottom:20px;">Run Simulation</button>
            <div id="sim-results-${winId}" style="border:1px solid #ddd; padding:20px; min-height:200px;">
                Results will appear here.
            </div>
        `;
        window.officeOS.windowManager.setWindowContent(winId, container);

        container.querySelector(`#run-sim-${winId}`).addEventListener('click', async () => {
            const fundName = container.querySelector(`#sim-fund-${winId}`).value;
            const scenName = container.querySelector(`#sim-scen-${winId}`).value;
            const resDiv = container.querySelector(`#sim-results-${winId}`);

            resDiv.innerHTML = 'Simulating market shock...';

            try {
                const [fundRes, scenRes] = await Promise.all([
                    fetch(`data/portfolios/${fundName}.json`).then(r=>r.json()),
                    fetch(`data/market_scenarios/${scenName}.json`).then(r=>r.json())
                ]);

                simulateScenario(fundRes, scenRes, resDiv);
            } catch(e) {
                resDiv.innerHTML = 'Simulation Error: ' + e;
            }
        });
    }

    function simulateScenario(portfolio, scenario, container) {
        let initialVal = 0;
        let shockedVal = 0;
        const impacts = [];

        portfolio.holdings.forEach(h => {
            initialVal += h.market_value;
            // Determine shock
            let shock = scenario.impact_factors[h.sector] || 0.0;
            // Default shock if not specified
            if (shock === 0) shock = -0.05; // Generic correlation

            const newVal = h.market_value * (1 + shock);
            shockedVal += newVal;

            impacts.push({
                ticker: h.ticker,
                shock: shock,
                pnl: newVal - h.market_value
            });
        });

        // Cash assumed safe? Or FX impact? Assume safe for now.
        const totalInitial = initialVal + portfolio.cash;
        const totalShocked = shockedVal + portfolio.cash;
        const change = totalShocked - totalInitial;
        const changePct = change / totalInitial;

        const color = change >= 0 ? 'green' : 'red';

        container.innerHTML = `
            <div style="text-align:center; padding:20px; border-bottom:1px solid #eee;">
                <h4>Impact Analysis: ${scenario.title}</h4>
                <div style="font-size:36px; font-weight:bold; color:${color};">
                    ${(changePct*100).toFixed(2)}%
                </div>
                <div style="color:#666;">
                    PnL Impact: $${(change/1000000).toFixed(2)}M
                </div>
            </div>

            <div style="margin-top:20px;">
                <strong>Top Contributors</strong>
                <ul style="list-style:none; padding:0;">
                    ${impacts.sort((a,b) => Math.abs(b.pnl) - Math.abs(a.pnl)).slice(0,5).map(i => `
                        <li style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f9f9f9;">
                            <span>${i.ticker} (${(i.shock*100).toFixed(1)}%)</span>
                            <span style="color:${i.pnl >= 0 ? 'green' : 'red'};">$${(i.pnl/1000000).toFixed(2)}M</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }

    initInvestmentApps();
})();
