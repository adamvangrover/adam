document.addEventListener('DOMContentLoaded', async () => {
    // --- 1. State Management ---
    let mockData = {};
    let libraryIndex = [];
    let currentMemo = null;

    // centralized DOM Element map with safety checks
    const elements = {
        generateBtn: document.getElementById('generate-btn'),
        memoPanel: document.getElementById('memo-panel'),
        memoContent: document.getElementById('memo-content'),
        evidencePanel: document.getElementById('evidence-panel'),
        pdfViewer: document.getElementById('pdf-viewer'),
        closeEvidenceBtn: document.getElementById('close-evidence'),
        progressContainer: document.getElementById('progress-container'),
        progressFill: document.getElementById('progress-fill'),
        progressText: document.getElementById('progress-text'),
        agentStatus: document.getElementById('agent-status'),
        auditLogPanel: document.getElementById('audit-log-panel'),
        auditLogContent: document.getElementById('audit-log-content'),
        borrowerSelect: document.getElementById('borrower-select'),
        libraryList: document.getElementById('library-list')
    };

    // --- 2. Initialization ---
    try {
        // await Promise.all([loadMockData(), loadLibraryIndex()]); // Legacy
        await loadLibraryIndex();
        renderLibraryList();
        logAudit("System", "Ready", "Credit Memo Orchestrator initialized.");
    } catch (e) {
        console.error("Initialization failed:", e);
        logAudit("System", "Error", "Initialization failed");
    }

    // --- 3. Event Listeners ---
    if(elements.generateBtn) elements.generateBtn.addEventListener('click', startGeneration);
    if(elements.closeEvidenceBtn) elements.closeEvidenceBtn.addEventListener('click', hideEvidence);
    
    // Global Delegate for Tab Switching
    window.switchTab = (tabName) => {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        
        // Reset all tab buttons
        document.querySelectorAll('.tab-btn').forEach(el => {
            el.style.borderBottom = '2px solid transparent';
            el.style.color = 'var(--text-secondary, #888)';
        });
        
        // Show target content
        const target = document.getElementById(`tab-${tabName}`);
        if(target) target.style.display = 'block';
        
        // Highlight target button
        const btn = document.getElementById(`btn-tab-${tabName}`);
        if(btn) {
            btn.style.borderBottom = '2px solid var(--accent-color, #007bff)';
            btn.style.color = 'var(--accent-color, #007bff)';
        }
    };

    // --- 4. Data Loading ---
    // loadMockData removed - handled by UniversalLoader

    async function loadLibraryIndex() {
        try {
            libraryIndex = await window.universalLoader.loadLibrary();
        } catch(e) { console.warn("Library index fetch failed", e); }
    }

    async function loadCreditMemo(identifier) {
        try {
            const memo = await window.universalLoader.loadCreditMemo(identifier);
            if (!memo) throw new Error("No data found");

            // Check if data is already in the final structure (has financials)
            if (memo.historical_financials && Array.isArray(memo.historical_financials) && memo.historical_financials.length > 0) {
                 currentMemo = memo;
                 renderFullMemoUI(memo);
            } else {
                 // Fallback to adapter for partial data
                 renderMemoFromMock(memo);
            }
        } catch(e) {
            console.error("Could not load memo file:", e);
            logAudit("System", "Error", `Failed to load ${identifier}`);
        }
    }

    // --- 5. Core Logic: Simulation ---
    function startGeneration() {
        const borrower = elements.borrowerSelect ? elements.borrowerSelect.value : "Apple Inc.";
        
        // UI Reset
        if(elements.memoContent) elements.memoContent.innerHTML = ''; 
        if(elements.progressContainer) elements.progressContainer.style.display = 'block';
        hideEvidence();

        // Simulate Enterprise Agent Workflow
        simulateAgent("Archivist", `Retrieving ${borrower} EDGAR filings...`, 0, 800)
            .then(() => simulateAgent("Quant", "Normalizing EBITDA & spreading comps...", 33, 1200))
            .then(() => simulateAgent("Risk Officer", "Analyzing covenant compliance...", 66, 1000))
            .then(() => simulateAgent("Writer", "Synthesizing executive summary...", 90, 1500))
            .then(() => {
                if(elements.progressContainer) elements.progressContainer.style.display = 'none';
                
                // Load data using Universal Loader
                loadCreditMemo(borrower);
                
                logAudit("Orchestrator", "Complete", `Memo generated for ${borrower}`);
            });
    }

    function simulateAgent(agentName, statusText, progressStart, duration) {
        return new Promise(resolve => {
            if(elements.progressText) elements.progressText.innerText = `${agentName} Active`;
            if(elements.agentStatus) elements.agentStatus.innerText = statusText;
            if(elements.progressFill) elements.progressFill.style.width = `${progressStart}%`;

            logAudit(agentName, "Start", statusText);

            setTimeout(() => {
                logAudit(agentName, "Complete", "Task finished");
                resolve();
            }, duration);
        });
    }

    // --- 6. Core Logic: Rendering ---

    // A. Sidebar Library Renderer
    function renderLibraryList() {
        if(!elements.libraryList) return;
        elements.libraryList.innerHTML = '';
        
        libraryIndex.forEach(item => {
            const itemDiv = document.createElement('div');
            // Hybrid Styling: Using inline styles for structure but variables for theme
            itemDiv.style.padding = '12px';
            itemDiv.style.borderBottom = '1px solid var(--glass-border, rgba(255,255,255,0.1))';
            itemDiv.style.cursor = 'pointer';
            itemDiv.className = 'library-item group'; // Allows hover effects if CSS exists

            const scoreColor = item.risk_score < 60 ? '#ff4444' : (item.risk_score < 80 ? '#ffbb33' : '#00C851');
            
            itemDiv.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-weight:bold; color: var(--text-primary, #fff); font-size:0.9em;">${item.borrower_name}</span>
                    <span style="font-size:0.75em; color: var(--text-secondary, #888);">${new Date(item.report_date).toLocaleDateString()}</span>
                </div>
                <div style="font-size:0.75em; color: var(--text-secondary, #888); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                    ${item.summary || 'No summary available.'}
                </div>
                <div style="margin-top:8px; display:flex; justify-content:space-between; align-items:center;">
                     <span style="font-family:monospace; font-size:0.8em; font-weight:bold; color:${scoreColor}">Risk: ${item.risk_score}/100</span>
                </div>
            `;
            
            itemDiv.onclick = () => {
                startGeneration(); 
            };
            
            elements.libraryList.appendChild(itemDiv);
        });
    }

    // B. Main Memo Container Renderer
    function renderFullMemoUI(memo) {
        if(!elements.memoContent) return;

        // 1. Tab Navigation
        const tabsHtml = `
            <div style="display: flex; gap: 20px; border-bottom: 1px solid var(--glass-border, rgba(255,255,255,0.1)); margin-bottom: 20px; overflow-x: auto;">
                <div id="btn-tab-memo" class="tab-btn" onclick="switchTab('memo')" style="padding: 10px; cursor: pointer; color: var(--accent-color, #007bff); border-bottom: 2px solid var(--accent-color, #007bff); white-space: nowrap;">Memo</div>
                <div id="btn-tab-financials" class="tab-btn" onclick="switchTab('financials')" style="padding: 10px; cursor: pointer; color: var(--text-secondary, #888); white-space: nowrap;">Financials</div>
                <div id="btn-tab-dcf" class="tab-btn" onclick="switchTab('dcf')" style="padding: 10px; cursor: pointer; color: var(--text-secondary, #888); white-space: nowrap;">Valuation (DCF)</div>
                <div id="btn-tab-cap" class="tab-btn" onclick="switchTab('cap')" style="padding: 10px; cursor: pointer; color: var(--text-secondary, #888); white-space: nowrap;">Cap Structure</div>
                <div id="btn-tab-risk" class="tab-btn" onclick="switchTab('risk')" style="padding: 10px; cursor: pointer; color: var(--text-secondary, #888); white-space: nowrap;">Risk Quant</div>
                <div id="btn-tab-sys2" class="tab-btn" onclick="switchTab('sys2')" style="padding: 10px; cursor: pointer; color: var(--text-secondary, #888); white-space: nowrap;">System 2</div>
            </div>
        `;

        // 2. Tab Contents (Delegating to specific generators)
        const contentHtml = `
            <div id="tab-memo" class="tab-content" style="display:block;">${generateMemoHtml(memo)}</div>
            <div id="tab-financials" class="tab-content" style="display:none;">${generateFinancialsHtml(memo)}</div>
            <div id="tab-dcf" class="tab-content" style="display:none;">${generateDcfHtml(memo)}</div>
            <div id="tab-cap" class="tab-content" style="display:none;">${generateCapStructureHtml(memo)}</div>
            <div id="tab-risk" class="tab-content" style="display:none;">${generateRiskQuantHtml(memo)}</div>
            <div id="tab-sys2" class="tab-content" style="display:none;">${generateSystem2Html(memo)}</div>
        `;

        elements.memoContent.innerHTML = tabsHtml + contentHtml;
        elements.memoContent.style.display = 'block';

        // Initialize Charts
        initRiskCharts(memo);
    }

    // --- 7. Generators (HTML Builders) ---

    // Generator 1: Main Narrative (Executive Summary + Risks)
    function generateMemoHtml(memo) {
        // Safe accessors
        const name = memo.borrower_name || memo.borrower_details?.name;
        const rating = memo.rating || memo.borrower_details?.rating || "N/A";
        const sector = memo.sector || memo.borrower_details?.sector || "General";
        const scoreColor = memo.risk_score < 60 ? 'text-red-500' : (memo.risk_score < 80 ? 'text-yellow-500' : 'text-emerald-500');

        // Header
        let html = `
            <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:20px; padding-bottom:20px; border-bottom:1px solid var(--glass-border, #333);">
                <div>
                    <h1 style="margin:0; font-size:2em; font-family:serif;">${name}</h1>
                    <div style="font-size: 0.9em; color: var(--text-secondary, #888); margin-top:5px;">
                        Rating: <span style="color: var(--accent-color, #fff); font-weight:bold;">${rating}</span> | Sector: ${sector}
                    </div>
                </div>
                <div style="text-align:right;">
                     <div style="font-size:0.8em; text-transform:uppercase; letter-spacing:1px; color:var(--text-secondary, #888);">Risk Score</div>
                     <div class="${scoreColor}" style="font-size:1.8em; font-weight:bold; font-family:monospace;">${memo.risk_score}/100</div>
                </div>
            </div>
        `;

        // Sections
        const sections = memo.sections || [];
        if (sections.length > 0) {
            sections.forEach(sec => {
                // Citation regex replacement
                let content = sec.content.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
                     return `<span class="citation-tag" onclick="viewEvidence('${docId}')" style="cursor:pointer; color:var(--accent-color, #007bff); font-size:0.8em; margin-left:5px;"><i class="fas fa-search"></i> [${docId}]</span>`;
                });
                
                html += `
                    <div style="margin-bottom:25px;">
                        <h2 style="color: var(--accent-color, #fff); font-size:1.2em; border-bottom: 1px solid var(--glass-border, #333); padding-bottom: 5px; margin-bottom:10px;">${sec.title}</h2>
                        <p style="line-height: 1.6; color: var(--text-primary, #ddd); font-size:0.95em;">${content}</p>
                    </div>
                `;
            });
        } else if (memo.documents) {
            // Fallback if no specific sections
            const doc = memo.documents[0];
            doc.chunks.forEach(chunk => {
                if(chunk.type === 'narrative') {
                     html += `<p style="margin-bottom:10px;">${chunk.content} <span class="citation-tag" onclick="renderEvidence('${memo.borrower_name}', '${doc.doc_id}', '${chunk.chunk_id}')" style="cursor:pointer; color:var(--accent-color);">[${doc.doc_id}]</span></p>`;
                }
            });
        }
        return html;
    }

    // Generator 2: Financials (Table)
    function generateFinancialsHtml(memo) {
        if (!memo.historical_financials) return '<p style="color:var(--text-secondary);">No structured financial data.</p>';
        
        let html = `<h3 style="color:var(--text-primary);">Historical Performance</h3>`;
        html += `<table style="width:100%; border-collapse: collapse; margin-top:15px; font-family: monospace; font-size:0.9em;">`;
        
        // Headers
        const periods = memo.historical_financials.map(d => d.period);
        html += `<thead style="color: var(--text-secondary); border-bottom: 1px solid var(--glass-border);"><tr><th style="text-align:left; padding:10px;">Metric</th>${periods.map(p => `<th style="text-align:right; padding:10px;">${p}</th>`).join('')}</tr></thead>`;
        
        // Rows
        const metrics = [
            { key: "revenue", label: "Revenue" },
            { key: "ebitda", label: "EBITDA" },
            { key: "net_income", label: "Net Income" },
            { key: "leverage_ratio", label: "Leverage (x)", fmt: (v) => v.toFixed(2) + 'x' }
        ];

        html += `<tbody>`;
        metrics.forEach(m => {
            html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">`;
            html += `<td style="padding:10px; color: var(--accent-color);">${m.label}</td>`;
            memo.historical_financials.forEach(period => {
                let val = period[m.key];
                val = m.fmt ? m.fmt(val) : formatCurrency(val);
                html += `<td style="text-align:right; padding:10px; color:var(--text-primary);">${val}</td>`;
            });
            html += `</tr>`;
        });
        html += `</tbody></table>`;
        return html;
    }

    // Generator 3: Valuation (DCF) - Ported from "Left" side logic
    function generateDcfHtml(memo) {
        if (!memo.dcf_analysis) return '<p style="color:var(--text-secondary);">No valuation data available.</p>';
        const dcf = memo.dcf_analysis;

        // Cards style
        const cardStyle = "background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px; border: 1px solid var(--glass-border, #333);";
        const labelStyle = "font-size: 0.7em; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 1px; margin-bottom: 5px;";
        const valueStyle = "font-size: 1.2em; font-weight: bold; font-family: monospace; color: var(--text-primary);";

        let html = `
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px;">
                 <div style="${cardStyle}">
                    <div style="${labelStyle}">WACC</div>
                    <div style="${valueStyle}">${(dcf.wacc * 100).toFixed(1)}%</div>
                 </div>
                 <div style="${cardStyle}">
                    <div style="${labelStyle}">Term Growth</div>
                    <div style="${valueStyle}">${(dcf.growth_rate * 100).toFixed(1)}%</div>
                 </div>
                 <div style="${cardStyle}">
                    <div style="${labelStyle}">Implied Price</div>
                    <div style="${valueStyle}; color: #00C851;">$${dcf.share_price.toFixed(2)}</div>
                 </div>
                 <div style="${cardStyle}">
                    <div style="${labelStyle}">Enterprise Val</div>
                    <div style="${valueStyle}; color: var(--accent-color);">${formatCurrency(dcf.enterprise_value)}</div>
                 </div>
            </div>
        `;

        // Projection Table
        html += `<h3 style="font-size: 0.9em; font-weight: bold; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 15px;">Projected Free Cash Flow</h3>`;
        html += `<table style="width:100%; text-align:left; font-family: monospace; font-size: 0.85em; border-collapse: collapse;">`;
        html += `<thead style="color: var(--text-secondary); border-bottom: 1px solid var(--glass-border);"><tr><th style="padding:8px;">Period</th><th style="padding:8px; text-align:right;">Year 1</th><th style="padding:8px; text-align:right;">Year 2</th><th style="padding:8px; text-align:right;">Year 3</th><th style="padding:8px; text-align:right;">Year 4</th><th style="padding:8px; text-align:right;">Year 5</th></tr></thead>`;
        html += `<tbody><tr style="border-bottom: 1px solid rgba(255,255,255,0.05);"><td style="padding:8px; font-weight:bold; color:var(--text-primary);">Unlevered FCF</td>`;
        
        if (dcf.free_cash_flow) {
            dcf.free_cash_flow.forEach(v => {
                html += `<td style="padding:8px; text-align:right; color:var(--text-primary);">${formatCurrency(v)}</td>`;
            });
        }
        html += `</tr></tbody></table>`;

        // Terminal Value Box
        html += `
            <div style="margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); display: flex; justify-content: space-between; align-items: center;">
                 <div>
                    <div style="font-size: 0.9em; font-weight: bold; color: var(--text-primary);">Terminal Value Calculation</div>
                    <div style="font-size: 0.7em; color: var(--text-secondary); font-family: monospace; margin-top: 4px;">Method: Gordon Growth</div>
                 </div>
                 <div style="text-align: right;">
                    <div style="font-size: 1.4em; font-weight: bold; color: var(--text-primary); font-family: monospace;">${formatCurrency(dcf.terminal_value)}</div>
                    <div style="font-size: 0.7em; color: var(--text-secondary);">Present Value: ${formatCurrency(dcf.terminal_value / Math.pow(1+dcf.wacc, 5))}</div>
                 </div>
            </div>
        `;
        return html;
    }

    // Generator 4: Cap Structure - Ported from "Left" side logic
    function generateCapStructureHtml(memo) {
        let html = '';
        
        // Equity Data
        if (memo.equity_data) {
            const eq = memo.equity_data;
            html += `
                <div style="margin-bottom: 30px;">
                     <h3 style="font-size: 0.8em; font-weight: bold; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 15px;">Equity Market Data</h3>
                     <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                        <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:4px;">
                            <div style="font-size:0.7em; color:var(--text-secondary);">Market Cap</div>
                            <div style="font-weight:bold; color:var(--text-primary); font-family:monospace;">${formatCurrency(eq.market_cap)}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:4px;">
                            <div style="font-size:0.7em; color:var(--text-secondary);">P/E Ratio</div>
                            <div style="font-weight:bold; color:var(--accent-color); font-family:monospace;">${eq.pe_ratio ? eq.pe_ratio.toFixed(1)+'x' : 'N/A'}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:4px;">
                            <div style="font-size:0.7em; color:var(--text-secondary);">Beta</div>
                            <div style="font-weight:bold; color:var(--text-primary); font-family:monospace;">${eq.beta ? eq.beta.toFixed(2) : 'N/A'}</div>
                        </div>
                     </div>
                </div>
            `;
        }

        // Debt Facilities Table
        if (memo.debt_facilities && memo.debt_facilities.length > 0) {
            html += `<div><h3 style="font-size: 0.8em; font-weight: bold; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 15px;">Debt Facilities</h3>`;
            html += `<table style="width:100%; text-align:left; font-family: monospace; font-size: 0.8em; border-collapse: collapse;">`;
            html += `<thead style="color: var(--text-secondary); border-bottom: 1px solid var(--glass-border);">
                        <tr>
                            <th style="padding:8px;">Type</th>
                            <th style="padding:8px; text-align:right;">Committed</th>
                            <th style="padding:8px; text-align:right;">Drawn</th>
                            <th style="padding:8px; text-align:right;">Rate</th>
                            <th style="padding:8px; text-align:center;">Rating</th>
                            <th style="padding:8px; text-align:center;">LTV</th>
                        </tr>
                     </thead><tbody>`;
            
            memo.debt_facilities.forEach(d => {
                // Style logic for Rating
                let ratingColor = '#00C851'; // Green
                if (d.snc_rating === "Special Mention") ratingColor = '#ffbb33';
                if (d.snc_rating === "Substandard" || d.snc_rating === "Doubtful") ratingColor = '#ff4444';

                // LTV Bar
                const ltvPct = (d.ltv || 0) * 100;
                const ltvColor = ltvPct < 60 ? '#00C851' : (ltvPct < 80 ? '#ffbb33' : '#ff4444');

                html += `
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px; color:var(--text-primary); font-weight:bold;">${d.facility_type}</td>
                        <td style="padding:8px; text-align:right; color:var(--text-primary);">${formatCurrency(d.amount_committed)}</td>
                        <td style="padding:8px; text-align:right; color:var(--text-secondary);">${formatCurrency(d.amount_drawn)}</td>
                        <td style="padding:8px; text-align:right; color:var(--accent-color);">${d.interest_rate}</td>
                        <td style="padding:8px; text-align:center;"><span style="color:${ratingColor}; border:1px solid ${ratingColor}; padding:2px 4px; border-radius:3px; font-size:0.8em;">${d.snc_rating}</span></td>
                        <td style="padding:8px; vertical-align:middle;">
                            <div style="width:60px; height:4px; background:#333; margin:auto; position:relative; border-radius:2px;">
                                <div style="width:${ltvPct}%; height:100%; background:${ltvColor}; border-radius:2px;"></div>
                            </div>
                        </td>
                    </tr>
                `;
            });
            html += `</tbody></table></div>`;
        } else {
            html += '<p style="color:var(--text-secondary); font-style:italic;">No debt facility data available.</p>';
        }
        
        return html;
    }

    // Generator 5: Risk Quant
    function generateRiskQuantHtml(memo) {
        return `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px;">
                    <h3 style="color:var(--text-primary); margin-bottom:10px;">Probability of Default (PD)</h3>
                    <div style="height: 200px;">
                        <canvas id="pdChart"></canvas>
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px;">
                    <h3 style="color:var(--text-primary); margin-bottom:10px;">Loss Given Default (LGD)</h3>
                    <div style="display:flex; flex-direction:column; justify-content:center; height:100%; align-items:center;">
                        <div style="font-size:3em; font-weight:bold; color:#ff4444;">${((memo.risk_metrics?.lgd || 0.45) * 100).toFixed(1)}%</div>
                        <div style="color:var(--text-secondary);">Recovery Rate: ${((1 - (memo.risk_metrics?.lgd || 0.45)) * 100).toFixed(1)}%</div>
                    </div>
                </div>
            </div>
            <div style="margin-top:20px; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px;">
                <h3 style="color:var(--text-primary); margin-bottom:10px;">Scenario Analysis</h3>
                <table style="width:100%; text-align:left; color:var(--text-secondary);">
                    <thead><tr><th>Scenario</th><th>PD Shock</th><th>Rev Impact</th><th>Rating Impact</th></tr></thead>
                    <tbody>
                        <tr><td>Recession (Bear)</td><td style="color:#ff4444;">+150 bps</td><td style="color:#ff4444;">-12.5%</td><td>Downgrade (2 notches)</td></tr>
                        <tr><td>Base Case</td><td>0 bps</td><td>+3.2%</td><td>Stable</td></tr>
                        <tr><td>Expansion (Bull)</td><td style="color:#00C851;">-50 bps</td><td style="color:#00C851;">+8.5%</td><td>Upgrade Watch</td></tr>
                    </tbody>
                </table>
            </div>
        `;
    }

    // Generator 6: System 2
    function generateSystem2Html(memo) {
        const critique = memo.system2_critique || {
            strengths: ["Strong market position", "Robust cash flow"],
            weaknesses: ["High leverage", "Sector headwinds"],
            verdict: "The automated analysis is directionally correct but may underestimate the impact of recent regulatory changes.",
            confidence: 0.85
        };

        return `
            <div style="background: rgba(100, 0, 255, 0.1); border: 1px solid #663399; padding: 20px; border-radius: 4px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
                    <h3 style="color:#a020f0; margin:0;"><i class="fas fa-brain"></i> System 2 Critique</h3>
                    <span style="background:#4b0082; color:white; padding:2px 8px; border-radius:4px; font-size:0.8em;">Confidence: ${(critique.confidence * 100).toFixed(0)}%</span>
                </div>
                <p style="color:var(--text-primary); font-style:italic; border-left: 3px solid #a020f0; padding-left: 10px; margin-bottom: 20px;">
                    "${critique.verdict}"
                </p>

                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                    <div>
                        <h4 style="color:#00C851;">Strengths</h4>
                        <ul style="color:var(--text-secondary); padding-left:20px;">
                            ${critique.strengths.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    </div>
                    <div>
                        <h4 style="color:#ff4444;">Weaknesses</h4>
                        <ul style="color:var(--text-secondary); padding-left:20px;">
                            ${critique.weaknesses.map(w => `<li>${w}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    function initRiskCharts(memo) {
        setTimeout(() => {
            const ctx = document.getElementById('pdChart');
            if(ctx && window.Chart) {
                // Destroy existing instance if any (rudimentary check, better handled by tracking instances)
                if (window.myPdChart) window.myPdChart.destroy();

                window.myPdChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
                        datasets: [{
                            label: 'Cumulative PD (%)',
                            data: memo.risk_metrics?.pd_term || [0.5, 1.2, 2.1, 3.5, 5.0],
                            backgroundColor: 'rgba(255, 68, 68, 0.5)',
                            borderColor: '#ff4444',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#888' } },
                            x: { grid: { display: false }, ticks: { color: '#888' } }
                        }
                    }
                });
            }
        }, 100);
    }

    // --- 8. Evidence Viewer Logic ---

    // A. Specific Coordinate Renderer
    window.renderEvidence = (borrowerName, docId, chunkId) => {
        // Fallback search in library or mockData
        let data = currentMemo;
        if (!data || (data.borrower_name !== borrowerName && data.borrower_details?.name !== borrowerName)) {
             // Try to find the right data if currentMemo isn't it
             data = mockData[Object.keys(mockData).find(k => mockData[k].borrower_details.name === borrowerName)] || currentMemo;
        }

        const doc = data?.documents?.find(d => d.doc_id === docId);
        const chunk = doc?.chunks?.find(c => c.chunk_id === chunkId);
        
        if (doc && chunk) {
            setupPdfViewer(docId, chunk.page);
            
            // Draw BBox
            const [x0, y0, x1, y1] = chunk.bbox;
            const highlight = document.createElement('div');
            highlight.className = 'bbox-highlight'; // Ensure CSS class exists or use inline below
            highlight.style.position = 'absolute';
            highlight.style.border = '2px solid var(--accent-color, #007bff)';
            highlight.style.backgroundColor = 'rgba(0, 123, 255, 0.2)';
            highlight.style.left = `${x0}px`;
            highlight.style.top = `${y0}px`;
            highlight.style.width = `${x1 - x0}px`;
            highlight.style.height = `${y1 - y0}px`;
            
            const label = document.createElement('div');
            label.innerText = chunk.type.toUpperCase();
            label.style.background = 'var(--accent-color, #007bff)';
            label.style.color = 'white';
            label.style.fontSize = '10px';
            label.style.fontWeight = 'bold';
            label.style.padding = '2px 4px';
            label.style.position = 'absolute';
            label.style.top = '-18px';
            label.style.left = '-2px';
            
            highlight.appendChild(label);
            
            const container = document.getElementById('pdf-page-container');
            if(container) {
                container.appendChild(highlight);
                highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            logAudit("Frontend", "Evidence", `Displayed precise artifact ${chunkId}`);
        }
    };

    // B. Generic Viewer
    window.viewEvidence = (docId) => {
        setupPdfViewer(docId, 1);
        logAudit("Frontend", "Evidence", `Opened document ${docId}`);
    };

    function setupPdfViewer(docId, pageNum) {
        if(!elements.evidencePanel || !elements.pdfViewer) return;

        elements.evidencePanel.classList.add('active');
        elements.evidencePanel.style.width = '50%'; // Or class based toggle
        
        // Clear previous
        elements.pdfViewer.innerHTML = '';
        
        const pageDiv = document.createElement('div');
        pageDiv.id = 'pdf-page-container';
        // Base styling for the "PDF" page
        pageDiv.style.background = '#fff'; 
        pageDiv.style.color = '#000';
        pageDiv.style.position = 'relative';
        pageDiv.style.minHeight = '1000px'; // Simulated height
        pageDiv.style.width = '100%';
        pageDiv.style.padding = '40px';
        pageDiv.style.boxShadow = '0 0 20px rgba(0,0,0,0.5)';
        
        pageDiv.innerHTML = `
            <div style="border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between;">
                <span style="font-weight: bold; font-family:sans-serif;">${docId}</span>
                <span style="font-family:sans-serif;">Page ${pageNum}</span>
            </div>
            <div style="font-family: Times New Roman, serif; color: #444; line-height: 1.8; font-size:14px;">
                <p style="text-align:center; font-weight:bold; font-size:16px;">UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
                <p style="text-align:center; font-weight:bold;">Form 10-K</p>
                <br>
                <p><strong>ITEM 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</strong></p>
                <p>The following discussion and analysis should be read in conjunction with the Consolidated Financial Statements and related notes included elsewhere in this Annual Report on Form 10-K. This discussion contains forward-looking statements based upon current expectations that involve risks and uncertainties.</p>
                <br>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.</p>
                <br>
                <div style="background:#eee; padding:20px; border:1px solid #ddd; text-align:center; color:#888;">[Simulated PDF Content Visualization]</div>
            </div>
        `;
        
        elements.pdfViewer.appendChild(pageDiv);
    }

    function hideEvidence() {
        if(elements.evidencePanel) {
            elements.evidencePanel.classList.remove('active');
            elements.evidencePanel.style.width = '0';
        }
    }

    // --- 9. Utilities ---
    function logAudit(actor, action, details) {
        if (!elements.auditLogContent) return;
        
        const time = new Date().toLocaleTimeString();
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid var(--glass-border, #333)';
        tr.innerHTML = `
            <td style="padding: 8px; color: var(--text-secondary, #666); font-size: 0.8em; font-family:monospace;">${time}</td>
            <td style="padding: 8px; color: var(--accent-color, #007bff); font-weight: bold;">${actor}</td>
            <td style="padding: 8px; color: var(--text-primary, #ccc);">${action}</td>
            <td style="padding: 8px; color: var(--text-secondary, #888); font-size: 0.9em;">${details}</td>
        `;
        elements.auditLogContent.insertBefore(tr, elements.auditLogContent.firstChild);
    }
    
    function formatCurrency(val) {
        if (val === undefined || val === null) return 'N/A';
        // Handle negative numbers
        const absVal = Math.abs(val);
        let str = '';
        if (absVal >= 1000) str = '$' + (absVal / 1000).toFixed(1) + 'B';
        else str = '$' + absVal.toFixed(1) + 'M';
        
        return val < 0 ? `(${str})` : str;
    }
    
    // Adapter for Mock Data fallback
    function renderMemoFromMock(data) {
        const adapter = {
            borrower_name: data.borrower_details?.name || data.borrower_name,
            borrower_details: data.borrower_details,
            risk_score: data.risk_score || 75,
            rating: data.borrower_details?.rating,
            sector: data.borrower_details?.sector,
            documents: data.documents,
            historical_financials: data.historical_financials, // Assuming mock data has this or it returns undefined
            sections: [
                { title: "Executive Summary", content: data.summary || (data.documents && data.documents[0].chunks.find(c=>c.type==='narrative')?.content) || "Analysis pending." },
                { title: "Risk Factors", content: "Risk factors extracted from filings." }
            ],
            // Mock DCF if not present
            dcf_analysis: data.dcf_analysis || {
                wacc: 0.085,
                growth_rate: 0.025,
                share_price: 150.00,
                enterprise_value: 2000000,
                terminal_value: 2500000,
                free_cash_flow: [12000, 13500, 14200, 15100, 16000]
            }
        };
        currentMemo = adapter;
        renderFullMemoUI(adapter);
    }
});