document.addEventListener('DOMContentLoaded', async () => {
    // --- 1. State Management ---
    let mockData = {};
    let libraryIndex = [];
    let currentMemo = null;

    // centralized DOM Element map
    const elements = {
        generateBtn: document.getElementById('generate-btn'),
        memoPanel: document.getElementById('memo-panel'),
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
    
    window.switchTab = (tabName) => {
        document.querySelectorAll('#tab-memo, #tab-annex-a, #tab-annex-b, #tab-annex-c, #tab-risk-quant, #tab-system-2').forEach(el => {
            el.classList.add('hidden');
        });
        
        document.querySelectorAll('.tab-btn').forEach(el => {
            el.classList.remove('text-blue-400', 'border-blue-500');
            el.classList.add('text-slate-400', 'border-transparent');
        });
        
        const target = document.getElementById(`tab-${tabName}`);
        if(target) target.classList.remove('hidden');
        
        const btn = document.getElementById(`btn-tab-${tabName}`);
        if(btn) {
            btn.classList.remove('text-slate-400', 'border-transparent');
            btn.classList.add('text-blue-400', 'border-blue-500');
        }
    };

    // --- 4. Data Loading ---

    async function loadLibraryIndex() {
        try {
            libraryIndex = await window.universalLoader.loadLibrary();
            console.log("Library Index Loaded:", libraryIndex);
        } catch(e) { console.warn("Library index fetch failed", e); }
    }

    async function loadCreditMemo(identifier) {
        console.log("Loading Credit Memo:", identifier);
        try {
            const memo = await window.universalLoader.loadCreditMemo(identifier);
            if (!memo) throw new Error("No data found");

            currentMemo = memo;

            if (memo.historical_financials && Array.isArray(memo.historical_financials)) {
                 renderFullMemoUI(memo);
            } else {
                 renderMemoFromMock(memo);
            }
        } catch(e) {
            console.error("Could not load memo file:", e);
            logAudit("System", "Error", `Failed to load ${identifier}`);
        }
    }

    // --- 5. Core Logic: Simulation ---
    function startGeneration(identifier) {
        // Handle Event object or string
        let borrower = "credit_memo_Apple_Inc.json";

        if (typeof identifier === 'string') {
            borrower = identifier;
        } else if (elements.borrowerSelect) {
            borrower = elements.borrowerSelect.value;
        }

        console.log("Starting generation for:", borrower);
        
        // UI Reset
        if(elements.progressContainer) elements.progressContainer.style.display = 'block';
        hideEvidence();

        // Simulate Enterprise Agent Workflow
        simulateAgent("Archivist", `Retrieving context...`, 0, 800)
            .then(() => simulateAgent("Quant", "Spreading financials...", 33, 1200))
            .then(() => simulateAgent("Risk Officer", "Analyzing risks...", 66, 1000))
            .then(() => simulateAgent("System 2", "Validating & Critiquing...", 90, 1500))
            .then(() => {
                if(elements.progressContainer) elements.progressContainer.style.display = 'none';
                loadCreditMemo(borrower);
                logAudit("Orchestrator", "Complete", `Memo generated.`);
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

    function renderLibraryList() {
        if(!elements.libraryList) return;
        elements.libraryList.innerHTML = '';
        
        libraryIndex.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'p-3 border-b border-slate-700/50 cursor-pointer hover:bg-slate-800/50 transition';

            const scoreColor = item.risk_score < 60 ? 'text-red-500' : (item.risk_score < 80 ? 'text-yellow-500' : 'text-emerald-500');
            
            itemDiv.innerHTML = `
                <div class="flex justify-between items-center mb-1">
                    <span class="font-bold text-white text-sm">${item.borrower_name}</span>
                    <span class="text-xs text-slate-500">${new Date(item.report_date).toLocaleDateString()}</span>
                </div>
                <div class="text-xs text-slate-500 line-clamp-2 mb-2">
                    ${item.summary || 'No summary available.'}
                </div>
                <div class="flex justify-between items-center">
                     <span class="font-mono text-xs font-bold ${scoreColor}">Risk: ${item.risk_score}/100</span>
                </div>
            `;
            
            itemDiv.onclick = () => {
                console.log("Library Item Clicked:", item.file);
                startGeneration(item.file);
            };
            
            elements.libraryList.appendChild(itemDiv);
        });
    }

    // B. Main Memo Container Renderer (Updated to target specific IDs)
    function renderFullMemoUI(memo) {
        const memoContainer = document.getElementById('memo-container');
        if(memoContainer) memoContainer.innerHTML = generateMemoHtml(memo);

        const finTable = document.getElementById('financials-table');
        if(finTable) finTable.innerHTML = generateFinancialsTableContent(memo);

        const dcfContainer = document.getElementById('dcf-container');
        if(dcfContainer) dcfContainer.innerHTML = generateDcfHtml(memo);

        const capContainer = document.getElementById('cap-structure-container');
        if(capContainer) capContainer.innerHTML = generateCapStructureHtml(memo);

        const riskContainer = document.getElementById('risk-quant-container');
        if(riskContainer) riskContainer.innerHTML = generateRiskQuantHtml(memo);

        const sys2Container = document.getElementById('system-2-container');
        if(sys2Container) sys2Container.innerHTML = generateSystemTwoHtml(memo);

        // Render Repayment Chart
        if (memo.repayment_schedule) {
            renderRepaymentChart(memo);
        }

        // Setup DCF listeners
        setupDCFListeners(memo);

        // Switch to memo tab by default
        window.switchTab('memo');
    }

    // --- 7. Generators (HTML Builders) ---

    // Generator 1: Main Narrative
    function generateMemoHtml(memo) {
        const name = memo.borrower_name || memo.borrower_details?.name;
        const rating = (memo.credit_ratings && memo.credit_ratings.length > 0) ? memo.credit_ratings[0].rating : (memo.rating || "N/A");
        const sector = memo.sector || memo.borrower_details?.sector || "General";
        const scoreColor = memo.risk_score < 60 ? 'text-red-500' : (memo.risk_score < 80 ? 'text-yellow-500' : 'text-emerald-500');

        let html = `
            <div class="flex justify-between items-start mb-8 pb-6 border-b border-slate-700/50">
                <div>
                    <h1 class="text-3xl font-serif text-slate-900 mb-2">${name}</h1>
                    <div class="text-sm text-slate-500">
                        Rating: <span class="font-bold text-slate-700">${rating}</span> | Sector: ${sector}
                    </div>
                </div>
                <div class="text-right">
                     <div class="text-xs uppercase tracking-widest text-slate-400 mb-1">Risk Score</div>
                     <div class="${scoreColor} text-4xl font-mono font-bold">${memo.risk_score}/100</div>
                </div>
            </div>
        `;

        const sections = memo.sections || [];
        sections.forEach(sec => {
            let content = sec.content.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
                 return `<span class="citation-tag bg-blue-50 text-blue-600 px-1 rounded cursor-pointer hover:bg-blue-100 transition" onclick="viewEvidence('${docId}')"><i class="fas fa-search mr-1"></i>[${docId}]</span>`;
            });

            html += `
                <div class="mb-8">
                    <h2 class="text-xl font-bold text-slate-800 border-b border-slate-200 pb-2 mb-4">${sec.title}</h2>
                    <div class="prose prose-slate text-sm max-w-none text-slate-600 leading-relaxed whitespace-pre-line">${content}</div>
                </div>
            `;
        });

        if(memo.key_strengths && memo.key_strengths.length > 0) {
            html += `<div class="grid grid-cols-2 gap-8 mb-8">`;
            html += `<div><h3 class="font-bold text-emerald-600 mb-2 uppercase text-xs tracking-widest">Key Strengths</h3><ul class="list-disc pl-4 space-y-1 text-sm text-slate-600">`;
            memo.key_strengths.forEach(s => html += `<li>${s}</li>`);
            html += `</ul></div>`;

            html += `<div><h3 class="font-bold text-red-600 mb-2 uppercase text-xs tracking-widest">Key Weaknesses</h3><ul class="list-disc pl-4 space-y-1 text-sm text-slate-600">`;
            memo.key_weaknesses.forEach(w => html += `<li>${w}</li>`);
            html += `</ul></div>`;
            html += `</div>`;
        }

        return html;
    }

    // Generator 2: Financials (Table Content Only)
    function generateFinancialsTableContent(memo) {
        if (!memo.historical_financials) return '';
        
        const periods = memo.historical_financials.map(d => d.period);
        let html = `<thead class="text-slate-500 border-b border-slate-700"><tr><th class="p-3">Metric</th>${periods.map(p => `<th class="p-3 text-right">${p}</th>`).join('')}</tr></thead>`;
        
        const metrics = [
            { key: "revenue", label: "Revenue" },
            { key: "ebitda", label: "EBITDA" },
            { key: "net_income", label: "Net Income" },
            { key: "leverage_ratio", label: "Leverage (x)", fmt: (v) => v.toFixed(2) + 'x' },
            { key: "dscr", label: "DSCR (x)", fmt: (v) => v.toFixed(2) + 'x' }
        ];

        html += `<tbody class="divide-y divide-slate-800/50">`;
        metrics.forEach(m => {
            html += `<tr>`;
            html += `<td class="p-3 text-blue-400 font-bold">${m.label}</td>`;
            memo.historical_financials.forEach(period => {
                let val = period[m.key];
                val = m.fmt ? m.fmt(val) : formatCurrency(val);
                html += `<td class="p-3 text-right text-slate-300">${val}</td>`;
            });
            html += `</tr>`;
        });
        html += `</tbody>`;
        return html;
    }

    // Generator 3: Valuation (DCF) with Inputs
    function generateDcfHtml(memo) {
        if (!memo.dcf_analysis) return '<p class="text-slate-500 italic">No valuation data available.</p>';
        const dcf = memo.dcf_analysis;
        const inputs = dcf.inputs || { wacc: dcf.wacc, growth_rate: dcf.growth_rate };

        let html = `
            <div class="grid grid-cols-4 gap-4 mb-8">
                 <div class="bg-slate-800/50 p-4 rounded border border-slate-700/50">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-1">WACC (Input)</div>
                    <input type="number" id="dcf-input-wacc" value="${(inputs.wacc * 100).toFixed(1)}" step="0.1" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono w-full focus:border-blue-500 outline-none">
                 </div>
                 <div class="bg-slate-800/50 p-4 rounded border border-slate-700/50">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-1">Term Growth (Input)</div>
                    <input type="number" id="dcf-input-growth" value="${(inputs.growth_rate * 100).toFixed(1)}" step="0.1" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-mono w-full focus:border-blue-500 outline-none">
                 </div>
                 <div class="bg-slate-800/50 p-4 rounded border border-slate-700/50">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-1">Implied Price</div>
                    <div class="text-xl font-bold text-emerald-400 font-mono" id="dcf-output-price">$${dcf.share_price.toFixed(2)}</div>
                 </div>
                 <div class="bg-slate-800/50 p-4 rounded border border-slate-700/50">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-1">Enterprise Val</div>
                    <div class="text-xl font-bold text-blue-400 font-mono" id="dcf-output-ev">${formatCurrency(dcf.enterprise_value)}</div>
                 </div>
            </div>
        `;

        html += `<h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Projected Free Cash Flow</h3>`;
        html += `<table class="w-full text-left font-mono text-sm border-collapse mb-8">`;
        html += `<thead class="text-slate-500 border-b border-slate-700"><tr><th class="p-2">Period</th><th class="p-2 text-right">Year 1</th><th class="p-2 text-right">Year 2</th><th class="p-2 text-right">Year 3</th><th class="p-2 text-right">Year 4</th><th class="p-2 text-right">Year 5</th></tr></thead>`;
        html += `<tbody><tr class="border-b border-slate-800/50"><td class="p-2 font-bold text-white">Unlevered FCF</td>`;
        
        if (dcf.free_cash_flow) {
            dcf.free_cash_flow.forEach(v => {
                html += `<td class="p-2 text-right text-slate-300">${formatCurrency(v)}</td>`;
            });
        }
        html += `</tr></tbody></table>`;

        html += `
            <div class="p-4 bg-slate-800/30 border border-slate-700/50 rounded flex justify-between items-center">
                 <div>
                    <div class="text-sm font-bold text-white">Terminal Value Calculation</div>
                    <div class="text-xs text-slate-500 font-mono mt-1">Method: Gordon Growth</div>
                 </div>
                 <div class="text-right">
                    <div class="text-2xl font-bold text-white font-mono" id="dcf-output-tv">${formatCurrency(dcf.terminal_value)}</div>
                 </div>
            </div>
        `;
        return html;
    }

    // Generator 4: Cap Structure
    function generateCapStructureHtml(memo) {
        let html = '';
        if (memo.equity_data) {
            const eq = memo.equity_data;
            html += `
                <div class="mb-8">
                     <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Equity Market Data</h3>
                     <div class="grid grid-cols-3 gap-4">
                        <div class="bg-slate-800/50 p-3 rounded border border-slate-700/50">
                            <div class="text-xs text-slate-500">Market Cap</div>
                            <div class="font-bold text-white font-mono">${formatCurrency(eq.market_cap)}</div>
                        </div>
                        <div class="bg-slate-800/50 p-3 rounded border border-slate-700/50">
                            <div class="text-xs text-slate-500">P/E Ratio</div>
                            <div class="font-bold text-blue-400 font-mono">${eq.pe_ratio ? eq.pe_ratio.toFixed(1)+'x' : 'N/A'}</div>
                        </div>
                        <div class="bg-slate-800/50 p-3 rounded border border-slate-700/50">
                            <div class="text-xs text-slate-500">Beta</div>
                            <div class="font-bold text-white font-mono">${eq.beta ? eq.beta.toFixed(2) : 'N/A'}</div>
                        </div>
                     </div>
                </div>
            `;
        }

        if (memo.debt_facilities && memo.debt_facilities.length > 0) {
            html += `<div><h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Debt Facilities</h3>`;
            html += `<table class="w-full text-left font-mono text-xs border-collapse">`;
            html += `<thead class="text-slate-500 border-b border-slate-700">
                        <tr>
                            <th class="p-2">Type</th>
                            <th class="p-2 text-right">Committed</th>
                            <th class="p-2 text-right">Drawn</th>
                            <th class="p-2 text-right">Rate</th>
                            <th class="p-2 text-center">Rating</th>
                            <th class="p-2 text-center">LTV</th>
                        </tr>
                     </thead><tbody>`;
            
            memo.debt_facilities.forEach(d => {
                let ratingColor = 'text-emerald-400 border-emerald-400';
                if (d.snc_rating === "Special Mention") ratingColor = 'text-yellow-400 border-yellow-400';
                if (d.snc_rating === "Substandard" || d.snc_rating === "Doubtful") ratingColor = 'text-red-400 border-red-400';

                const ltvPct = (d.ltv || 0) * 100;
                const ltvColor = ltvPct < 60 ? 'bg-emerald-500' : (ltvPct < 80 ? 'bg-yellow-500' : 'bg-red-500');

                html += `
                    <tr class="border-b border-slate-800/50">
                        <td class="p-2 text-white font-bold">${d.facility_type}</td>
                        <td class="p-2 text-right text-white">${formatCurrency(d.amount_committed)}</td>
                        <td class="p-2 text-right text-slate-400">${formatCurrency(d.amount_drawn)}</td>
                        <td class="p-2 text-right text-blue-400">${d.interest_rate}</td>
                        <td class="p-2 text-center"><span class="${ratingColor} border px-1 rounded text-[10px]">${d.snc_rating}</span></td>
                        <td class="p-2 align-middle">
                            <div class="w-16 h-1 bg-slate-800 rounded mx-auto relative overflow-hidden">
                                <div class="h-full ${ltvColor}" style="width:${ltvPct}%"></div>
                            </div>
                        </td>
                    </tr>
                `;
            });
            html += `</tbody></table></div>`;
        }

        // Add Chart Container
        html += `
            <div class="mt-8">
                <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Debt Repayment Schedule (Forecast)</h3>
                <div class="bg-slate-800/30 p-4 rounded border border-slate-700/50 relative" style="height: 300px;">
                    <canvas id="repayment-chart"></canvas>
                </div>
            </div>
        `;

        return html;
    }

    function renderRepaymentChart(memo) {
        const ctx = document.getElementById('repayment-chart');
        if (!ctx || !memo.repayment_schedule) return;

        // Destroy existing chart if any
        if (window.repaymentChartInstance) {
            window.repaymentChartInstance.destroy();
        }

        const labels = memo.repayment_schedule.map(item => item.year);
        const data = memo.repayment_schedule.map(item => item.amount);
        const backgrounds = memo.repayment_schedule.map(item =>
            item.source.includes('Maturity') ? 'rgba(239, 68, 68, 0.7)' : 'rgba(59, 130, 246, 0.7)'
        );

        window.repaymentChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Repayment Amount ($M)',
                    data: data,
                    backgroundColor: backgrounds,
                    borderColor: backgrounds.map(c => c.replace('0.7', '1.0')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const item = memo.repayment_schedule[context.dataIndex];
                                return `${item.source}: ${formatCurrency(item.amount)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Generator 5: Risk Quant (PD/LGD/Scenarios)
    function generateRiskQuantHtml(memo) {
        if (!memo.pd_model) return '<p class="text-slate-500 italic">No quantitative risk models available.</p>';
        const pd = memo.pd_model;
        const scenarios = memo.scenario_analysis;

        let html = '';

        html += `
            <div class="mb-8">
                <h3 class="text-sm font-bold text-purple-400 uppercase tracking-widest mb-4 border-b border-slate-700 pb-2">Probability of Default (PD) Model</h3>
                <div class="grid grid-cols-2 gap-8">
                    <div class="bg-slate-800/30 p-4 rounded border border-slate-700/50">
                         <div class="flex justify-between items-center mb-4">
                            <span class="text-slate-400 text-sm">Model Score</span>
                            <span class="text-2xl font-mono font-bold text-white">${pd.model_score.toFixed(1)}/100</span>
                         </div>
                         <div class="flex justify-between items-center mb-4">
                            <span class="text-slate-400 text-sm">Implied Rating</span>
                            <span class="text-xl font-mono font-bold text-purple-400 border border-purple-500 px-2 rounded">${pd.implied_rating}</span>
                         </div>
                         <div class="space-y-2">
                             <div class="flex justify-between text-xs">
                                <span class="text-slate-500">1-Year PD</span>
                                <span class="text-white font-mono">${(pd.one_year_pd * 100).toFixed(2)}%</span>
                             </div>
                             <div class="flex justify-between text-xs">
                                <span class="text-slate-500">5-Year PD</span>
                                <span class="text-white font-mono">${(pd.five_year_pd * 100).toFixed(2)}%</span>
                             </div>
                         </div>
                    </div>
                    <div>
                        <h4 class="text-xs text-slate-500 uppercase mb-2">Key Drivers</h4>
                        <ul class="space-y-2 text-sm text-slate-300">
                            ${Object.entries(pd.input_factors).map(([k,v]) => `
                                <li class="flex justify-between border-b border-slate-800 pb-1">
                                    <span>${k}</span>
                                    <span class="font-mono text-blue-400">${typeof v === 'number' ? v.toFixed(2) : v}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;

        if (scenarios) {
            html += `
                <div class="mb-8">
                    <h3 class="text-sm font-bold text-blue-400 uppercase tracking-widest mb-4 border-b border-slate-700 pb-2">Scenario Analysis</h3>
                    <div class="grid grid-cols-3 gap-4">
                        ${scenarios.scenarios.map(s => `
                            <div class="bg-slate-800/30 p-4 rounded border border-slate-700/50 relative overflow-hidden">
                                <div class="absolute top-0 right-0 bg-slate-700 text-[10px] px-2 py-0.5 rounded-bl text-white font-mono">${(s.probability * 100).toFixed(0)}% PROB</div>
                                <h4 class="font-bold text-white mb-2">${s.name}</h4>
                                <div class="text-2xl font-mono font-bold text-emerald-400 mb-2">$${s.implied_share_price.toFixed(2)}</div>
                                <div class="text-xs text-slate-500 space-y-1">
                                    <div>Rev Growth: ${(s.revenue_growth*100).toFixed(1)}%</div>
                                    <div>EBITDA Margin: ${(s.ebitda_margin*100).toFixed(1)}%</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="mt-4 text-center text-sm text-slate-400">
                        Probability Weighted Price: <span class="text-white font-bold font-mono">$${scenarios.weighted_share_price.toFixed(2)}</span>
                    </div>
                </div>
            `;
        }

        return html;
    }

    // Generator 6: System 2 Critique
    function generateSystemTwoHtml(memo) {
        if (!memo.system_two_critique) return '<p class="text-slate-500 italic">No System 2 critique available.</p>';
        const s2 = memo.system_two_critique;
        const statusColor = s2.verification_status === "PASS" ? "text-emerald-400" : "text-yellow-400";
        const icon = s2.verification_status === "PASS" ? "fa-check-circle" : "fa-exclamation-triangle";

        let html = `
            <div class="bg-slate-800/30 border border-slate-700/50 p-6 rounded-lg">
                <div class="flex items-center justify-between mb-6">
                    <div class="flex items-center gap-3">
                         <div class="w-10 h-10 rounded-full bg-pink-900/30 flex items-center justify-center border border-pink-500/30">
                            <i class="fas fa-brain text-pink-400"></i>
                         </div>
                         <div>
                            <h3 class="font-bold text-white text-lg">System 2 Review</h3>
                            <div class="text-xs text-slate-500">Autonomous Critique Agent</div>
                         </div>
                    </div>
                    <div class="text-right">
                        <div class="text-xs text-slate-500 uppercase">Conviction Score</div>
                        <div class="text-2xl font-mono font-bold text-white">${(s2.conviction_score * 100).toFixed(0)}/100</div>
                    </div>
                </div>

                <div class="mb-6">
                    <h4 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Critique Points</h4>
                    <ul class="space-y-3">
                        ${s2.critique_points.map(p => `
                            <li class="flex gap-3 text-slate-300 text-sm bg-black/20 p-3 rounded border border-slate-800/50">
                                <i class="fas fa-comment-dots text-pink-400 mt-1"></i>
                                <span>${p}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>

                <div class="flex items-center justify-between pt-4 border-t border-slate-700/50">
                    <span class="text-xs text-slate-500 font-mono">AGENT: ${s2.author_agent}</span>
                    <div class="flex items-center gap-2 ${statusColor}">
                        <i class="fas ${icon}"></i>
                        <span class="font-bold text-sm tracking-widest">${s2.verification_status}</span>
                    </div>
                </div>
            </div>
        `;
        return html;
    }

    // --- 8. Interactivity ---

    function setupDCFListeners(memo) {
        if(!memo.dcf_analysis) return;

        const waccInput = document.getElementById('dcf-input-wacc');
        const growthInput = document.getElementById('dcf-input-growth');

        const recalculate = () => {
             const newWacc = parseFloat(waccInput.value) / 100;
             const newGrowth = parseFloat(growthInput.value) / 100;

             // Client-side Math (Mirroring backend)
             const currentEbitda = memo.historical_financials[0].ebitda;
             const baseFcf = currentEbitda * 0.65;

             let projectedFcf = [];
             let pvFcf = 0;

             for(let i=1; i<=5; i++) {
                 const g = 0.05 - (0.005 * (i-1));
                 const fcf = baseFcf * Math.pow((1+g), i);
                 projectedFcf.push(fcf);
                 pvFcf += fcf / Math.pow((1+newWacc), i);
             }

             const termVal = (projectedFcf[4] * (1+newGrowth)) / (newWacc - newGrowth);
             const pvTerm = termVal / Math.pow((1+newWacc), 5);
             const ev = pvFcf + pvTerm;

             // Equity
             const debt = memo.historical_financials[0].total_liabilities;
             const eq = ev - debt;
             const shares = eq / memo.dcf_analysis.share_price;
             const price = eq / shares;

             document.getElementById('dcf-output-ev').innerText = formatCurrency(ev);
             document.getElementById('dcf-output-price').innerText = '$' + price.toFixed(2);
             document.getElementById('dcf-output-tv').innerText = formatCurrency(termVal);

             logAudit("User", "Intervention", `DCF Re-calc: WACC=${(newWacc*100).toFixed(1)}%, g=${(newGrowth*100).toFixed(1)}%`);
        };

        if(waccInput) waccInput.addEventListener('change', recalculate);
        if(growthInput) growthInput.addEventListener('change', recalculate);
    }

    // --- 9. Evidence Viewer Logic (Existing) ---
    window.renderEvidence = (borrowerName, docId, chunkId) => {
        let data = currentMemo;
        if (!data || (data.borrower_name !== borrowerName && data.borrower_details?.name !== borrowerName)) {
             data = mockData[Object.keys(mockData).find(k => mockData[k].borrower_details.name === borrowerName)] || currentMemo;
        }

        const doc = data?.documents?.find(d => d.doc_id === docId);
        const chunk = doc?.chunks?.find(c => c.chunk_id === chunkId);
        
        if (doc && chunk) {
            setupPdfViewer(docId, chunk.page);
            const [x0, y0, x1, y1] = chunk.bbox;
            const highlight = document.createElement('div');
            highlight.className = 'bbox-highlight';
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

    window.viewEvidence = (docId) => {
        setupPdfViewer(docId, 1);
        logAudit("Frontend", "Evidence", `Opened document ${docId}`);
    };

    function setupPdfViewer(docId, pageNum) {
        if(!elements.evidencePanel || !elements.pdfViewer) return;
        elements.evidencePanel.classList.add('active');
        elements.evidencePanel.style.width = '50%';
        elements.pdfViewer.innerHTML = '';
        const pageDiv = document.createElement('div');
        pageDiv.id = 'pdf-page-container';
        pageDiv.style.background = '#fff'; 
        pageDiv.style.color = '#000';
        pageDiv.style.position = 'relative';
        pageDiv.style.minHeight = '1000px';
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
                <p>The following discussion and analysis should be read in conjunction with the Consolidated Financial Statements and related notes included elsewhere in this Annual Report on Form 10-K.</p>
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
        const absVal = Math.abs(val);
        let str = '';
        if (absVal >= 1000) str = '$' + (absVal / 1000).toFixed(1) + 'B';
        else str = '$' + absVal.toFixed(1) + 'M';
        return val < 0 ? `(${str})` : str;
    }
    
    function renderMemoFromMock(data) {
        currentMemo = data;
        renderFullMemoUI(data);
    }
});