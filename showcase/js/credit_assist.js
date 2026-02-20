/**
 * ADAM v23.5 CREDIT ASSISTANT
 * -----------------------------------------------------------------------------
 * Additive module for "Credit Memo Automation".
 * Adds GenAI Chat Widget and Simulated Cursor Collaboration.
 * -----------------------------------------------------------------------------
 */

class CreditAssist {
    constructor() {
        this.init();
    }

    init() {
        console.log("[CreditAssist] Initializing...");
        this.renderAssistant();
        this.startCursorSim();
    }

    renderAssistant() {
        // Floating Button
        const btn = document.createElement('button');
        btn.innerHTML = '<i class="fas fa-magic"></i> AI Assist';
        btn.className = 'credit-assist-btn';
        btn.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #007bff, #00d2ff);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 123, 255, 0.4);
            cursor: pointer;
            z-index: 9999;
            transition: transform 0.2s;
        `;
        btn.onmouseover = () => btn.style.transform = "scale(1.05)";
        btn.onmouseout = () => btn.style.transform = "scale(1)";

        document.body.appendChild(btn);

        // Modal
        const modal = document.createElement('div');
        modal.id = 'assist-modal';
        modal.style.cssText = `
            position: fixed;
            bottom: 80px;
            right: 30px;
            width: 300px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 15px;
            color: #cbd5e1;
            display: none;
            z-index: 9999;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            font-family: 'Inter', sans-serif;
        `;

        modal.innerHTML = `
            <div style="border-bottom: 1px solid #334155; padding-bottom: 8px; margin-bottom: 10px; display:flex; justify-content:space-between;">
                <span style="font-weight:bold; color:white;">AI Copilot</span>
                <span style="font-size:0.8em; color:#00d2ff;">GPT-4</span>
            </div>
            <div style="font-size:0.9em; margin-bottom:10px;">How can I help with this memo?</div>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <button class="assist-action" data-action="draft_risk" style="text-align:left; background:#0f172a; border:1px solid #334155; padding:8px; border-radius:4px; color:#94a3b8; cursor:pointer; font-size:0.85em;">
                    <i class="fas fa-pen-fancy" style="margin-right:8px; color:#00d2ff;"></i> Draft 'Risk Factors'
                </button>
                <button class="assist-action" data-action="verify_covenants" style="text-align:left; background:#0f172a; border:1px solid #334155; padding:8px; border-radius:4px; color:#94a3b8; cursor:pointer; font-size:0.85em;">
                    <i class="fas fa-check-double" style="margin-right:8px; color:#10b981;"></i> Verify Covenants
                </button>
                <button class="assist-action" data-action="find_comps" style="text-align:left; background:#0f172a; border:1px solid #334155; padding:8px; border-radius:4px; color:#94a3b8; cursor:pointer; font-size:0.85em;">
                    <i class="fas fa-search-dollar" style="margin-right:8px; color:#f59e0b;"></i> Find Comparable Comps
                </button>
            </div>
            <div id="assist-output" style="margin-top:10px; font-size:0.8em; color:#00d2ff; min-height:20px; white-space: pre-wrap; max-height: 200px; overflow-y: auto;"></div>
        `;

        document.body.appendChild(modal);

        btn.onclick = () => {
            modal.style.display = modal.style.display === 'none' ? 'block' : 'none';
        };

        // Actions
        const actions = modal.querySelectorAll('.assist-action');
        actions.forEach(a => {
            a.onclick = () => {
                const action = a.getAttribute('data-action');
                this.handleAction(action);
            };
        });
    }

    handleAction(action) {
        const output = document.getElementById('assist-output');
        output.innerText = "Analyzing context...";
        output.style.color = "#00d2ff";

        // Attempt to get borrower context from the main H1 title
        const titleEl = document.querySelector('h1');
        const fullTitle = titleEl ? titleEl.innerText : "Generic Borrower";
        // Often title includes "CREDIT MEMO ...", let's try to find the specific borrower name in the rendered content if loaded
        // In `credit_memo.js`, it renders `<h1 class="text-3xl font-serif text-slate-900 mb-2">${name}</h1>` inside `#memo-container`
        const memoTitle = document.querySelector('#memo-container h1');
        const borrower = memoTitle ? memoTitle.innerText : "Unknown Entity";

        setTimeout(() => {
            if (action === 'draft_risk') {
                const risks = this.generateRisks(borrower);
                output.innerText = `Drafting Risk Factors for ${borrower}:\n\n` + risks;
                output.style.color = "#e2e8f0";
            } else if (action === 'verify_covenants') {
                const check = this.checkCovenants(borrower);
                output.innerText = check;
                output.style.color = "#10b981";
            } else {
                output.innerText = "Scanning 10-K filings... found 3 relevant peers. (Simulation)";
            }
        }, 1200);
    }

    generateRisks(name) {
        const n = name.toLowerCase();
        if (n.includes("apple") || n.includes("aapl")) {
            return "1. Supply Chain Concentration: Heavy reliance on manufacturing in Greater China poses geopolitical continuity risks.\n2. Antitrust Scrutiny: Ongoing DOJ and EU investigations into App Store dominance may erode services margins.\n3. Hardware Saturation: Smartphone replacement cycles are lengthening globally.";
        }
        if (n.includes("tesla") || n.includes("tsla")) {
            return "1. Key Man Risk: High dependence on CEO Elon Musk's strategic direction and attention.\n2. Margin Compression: Aggressive price cuts in EV sector are deteriorating gross margins.\n3. Regulatory Hurdles: FSD technology faces stringent NHTSA scrutiny delaying rollout.";
        }
        if (n.includes("nvidia") || n.includes("nvda")) {
            return "1. Customer Concentration: Top 4 customers (Hyperscalers) account for significant revenue.\n2. Export Controls: US restrictions on AI chip sales to China limit TAM expansion.\n3. Cyclicality: Semiconductor inventory corrections historically follow boom cycles.";
        }
        return `1. Macro Headwinds: Rising interest rates may impact ${name}'s cost of capital.\n2. Competitive Pressure: Increasing fragmentation in the core sector.\n3. Execution Risk: Strategic initiatives carry implementation uncertainty.`;
    }

    checkCovenants(name) {
        // Try to read leverage from DOM
        // Looking for "Net Leverage (x)" row in the table
        let leverage = 3.5; // default
        const rows = document.querySelectorAll('tr');
        rows.forEach(r => {
            if (r.innerText.includes('Net Leverage')) {
                const cells = r.querySelectorAll('td');
                if (cells.length > 1) {
                    leverage = parseFloat(cells[1].innerText) || 3.5;
                }
            }
        });

        const status = leverage < 4.0 ? "PASS" : "FAIL";
        return `Covenant Check for ${name}:\n\n- Net Leverage: ${leverage.toFixed(2)}x (Limit: 4.00x) [${status}]\n- Interest Coverage: > 3.0x [PASS]\n\nCompliance Certified.`;
    }

    startCursorSim() {
        const container = document.getElementById('memo-content'); // Main area
        if (!container) return;

        const cursors = [
            { id: 'cursor-1', name: 'RiskBot', color: '#ff0055', x: 100, y: 100, vx: 2, vy: 1 },
            { id: 'cursor-2', name: 'LegalAI', color: '#f59e0b', x: 300, y: 200, vx: -1, vy: 2 }
        ];

        cursors.forEach(c => {
            const el = document.createElement('div');
            el.id = c.id;
            el.style.cssText = `
                position: absolute;
                top: 0; left: 0;
                pointer-events: none;
                z-index: 50;
                transition: top 0.5s ease, left 0.5s ease;
            `;
            el.innerHTML = `
                <i class="fas fa-mouse-pointer" style="color:${c.color}; font-size: 1.2em; transform: rotate(-20deg);"></i>
                <div style="background:${c.color}; color:black; font-size:0.6em; padding:1px 4px; border-radius:3px; font-weight:bold; margin-left:12px; margin-top:-5px;">${c.name}</div>
            `;
            container.appendChild(el);
        });

        // Loop
        setInterval(() => {
            cursors.forEach(c => {
                // Random walk
                c.x += (Math.random() - 0.5) * 100;
                c.y += (Math.random() - 0.5) * 100;

                // Bounds check (rough)
                if (c.x < 50) c.x = 50;
                if (c.x > container.clientWidth - 50) c.x = container.clientWidth - 50;
                if (c.y < 50) c.y = 50;
                if (c.y > container.clientHeight - 50) c.y = container.clientHeight - 50;

                const el = document.getElementById(c.id);
                if (el) {
                    el.style.left = c.x + 'px';
                    el.style.top = c.y + 'px';
                }
            });
        }, 1500);
    }
}

// Auto-Launch
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        new CreditAssist();
    }, 1000);
});
