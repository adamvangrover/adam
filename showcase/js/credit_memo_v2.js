document.addEventListener('DOMContentLoaded', async () => {
    // --- State Management ---
    let library = {};
    let selectedBorrower = null;
    let currentMemoData = null;
    let isEditMode = false;
    let currentChart = null;

    const elements = {
        borrowerSelect: document.getElementById('borrower-select'),
        generateBtn: document.getElementById('generate-btn'),
        editBtn: document.getElementById('edit-btn'),
        exportBtn: document.getElementById('export-btn'),
        memoContainer: document.getElementById('memo-container'), // Main scrollable area
        memoPlaceholder: document.getElementById('memo-placeholder'), // Placeholder
        memoContent: document.getElementById('memo-content'),     // Inner content div
        evidenceViewer: document.getElementById('pdf-mock-canvas'),
        terminal: document.getElementById('agent-terminal'),
        docIdLabel: document.getElementById('doc-id-label') || document.createElement('div'), // Safety fallback
        modeIndicator: document.querySelector('.header-nav span span') 
    };

    // --- Initialization ---
    async function init() {
        log("System", "Initializing CreditOS v2.4...", "info");

        try {
            // Load Library via Universal Loader
            const libArray = await window.universalLoader.loadLibrary();

            // Normalize to object for local lookups if needed, or just use for dropdown
            library = {};
            libArray.forEach(item => {
                // Ensure we have an ID
                const id = item.id || item.borrower_name.replace(/ /g, '_');
                library[id] = item;
                
                const opt = document.createElement('option');
                // Use ID as value if file is missing, but prefer file if available logic elsewhere expects it.
                // However, updated loader prefers ID or Filename.
                // Original used item.file. Let's use item.id for cleaner lookup,
                // but universalLoader.loadCreditMemo handles filename or ID.
                opt.value = item.file || item.id;
                opt.innerText = `${item.borrower_name} (Risk: ${item.risk_score})`;
                opt.dataset.ticker = item.ticker || "UNK";
                opt.dataset.name = item.borrower_name;
                elements.borrowerSelect.appendChild(opt);
            });

            log("Archivist", `Loaded ${libArray.length} entity profiles.`, "success");
            
            // Load Audit Logs (Background)
            loadAuditLogs();

        } catch (e) {
            log("System", "Failed to load artifact library.", "error");
            console.error(e);
        }
    }

    // --- Event Listeners ---
    if(elements.generateBtn) elements.generateBtn.addEventListener('click', runAnalysis);
    if(elements.editBtn) elements.editBtn.addEventListener('click', toggleEditMode);
    if(elements.exportBtn) elements.exportBtn.addEventListener('click', () => window.print());

    // --- Core Workflow ---
    async function runAnalysis() {
        const filename = elements.borrowerSelect.value;
        if (!filename) { alert("Please select a target entity."); return; }
        
        const option = elements.borrowerSelect.options[elements.borrowerSelect.selectedIndex];
        const ticker = option.dataset.ticker;
        const name = option.dataset.name;

        // Reset UI
        elements.memoContent.innerHTML = '';
        clearTerminal();
        
        // 1. Retrieval Phase
        await simulateAgentTask("Archivist", [
            `Querying Vector DB for ${ticker}...`,
            `Found 10-K (${new Date().getFullYear()}) - Indexing...`,
            `Extracting relevant chunks (MD&A, Risk Factors)...`
        ], 600);

        // 2. Load Data (Simulating "Live" Fetch)
        try {
            currentMemoData = await window.universalLoader.loadCreditMemo(filename);
            if (!currentMemoData) throw new Error("File not found");
            selectedBorrower = currentMemoData; // Sync state
        } catch(e) {
            console.error(e);
            log("System", "Error loading memo data", "error");
            return;
        }

        // 3. Quant Phase
        const assets = currentMemoData.historical_financials?.[0]?.total_assets || 'N/A';
        await simulateAgentTask("Quant", [
            `OCR-ing Financial Tables...`,
            `Mapping line items to FIBO ontology...`,
            `Validating Balance Sheet: Assets (${assets}) = Liab + Equity...`,
            `PASS: Checksum valid.`
        ], 800);

        // 4. Synthesis Phase
        await simulateAgentTask("Writer", [
            `Synthesizing Executive Summary...`,
            `Injecting Citations...`,
            `Compiling Final Report...`
        ], 800);

        // Render
        renderMemo(currentMemoData);
        log("Orchestrator", `Analysis Complete for ${name}`, "success");
    }

    // --- Rendering Logic ---
    function renderMemo(memo) {
        // Toggle visibility
        elements.memoPlaceholder.style.display = 'none';
        elements.memoContent.style.display = 'block';

        const contentDiv = elements.memoContent;
        contentDiv.innerHTML = '';
        contentDiv.className = "memo-paper"; // Apply the paper CSS class

        // Header
        const riskColor = memo.risk_score < 60 ? 'red' : (memo.risk_score < 80 ? 'orange' : 'green');

        // Source Badge Logic
        const isRag = (memo.source && memo.source.includes('RAG')) || (memo.executive_summary && memo.executive_summary.includes('RAG Analysis'));
        const sourceBadge = isRag
            ? `<span style="background: #e0f7fa; color: #006064; padding: 2px 6px; border-radius: 4px; border: 1px solid #006064; font-weight: bold;">SOURCE: RAG (10-K)</span>`
            : `<span style="background: #f5f5f5; color: #999; padding: 2px 6px; border-radius: 4px; border: 1px solid #ddd;">SOURCE: MOCK/HISTORICAL</span>`;

        const header = document.createElement('div');
        header.innerHTML = `
            <h1 class="editable-content">${memo.borrower_name}</h1>
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 30px; font-family: var(--font-mono); font-size: 0.8rem; color: #666;">
                <span>DATE: ${new Date(memo.report_date).toLocaleDateString()}</span>
                <span>RISK SCORE: <b style="color: ${riskColor}">${memo.risk_score}/100</b></span>
                ${sourceBadge}
            </div>
        `;
        contentDiv.appendChild(header);

        // Executive Summary
        if (memo.executive_summary) {
            const execSumDiv = document.createElement('div');
            execSumDiv.innerHTML = `
                <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Executive Summary</h3>
                <div class="editable-content" style="text-align: justify; margin-bottom: 30px; font-family: var(--font-mono); font-size: 0.9rem;">
                    ${memo.executive_summary}
                </div>
            `;
            contentDiv.appendChild(execSumDiv);
        }

        // Financial Snapshot & Chart (Merged Feature)
        if (memo.financial_ratios) {
            const fr = memo.financial_ratios;
            const finDiv = document.createElement('div');
            finDiv.style.marginBottom = '30px';
            finDiv.style.background = '#f9f9f9';
            finDiv.style.padding = '20px';
            finDiv.style.border = '1px solid #ddd';

            window._fr_sources = fr.sources || {};
            const bind = (key) => window._fr_sources[key] ? `style="cursor:pointer; color:blue; text-decoration:underline;" onclick='window.viewEvidence(window._fr_sources["${key}"])'` : "";

            finDiv.innerHTML = `
                <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Financial Snapshot</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; font-family: var(--font-mono); font-size: 0.8rem; margin-bottom: 20px;">
                    <div><b>Leverage:</b> ${fr.leverage_ratio?.toFixed(2)}x</div>
                    <div><b>EBITDA:</b> <span ${bind('ebitda')}>${formatCurrency(fr.ebitda)}</span></div>
                    <div><b>Revenue:</b> <span ${bind('revenue')}>${formatCurrency(fr.revenue)}</span></div>
                    <div><b>DSCR:</b> ${fr.dscr?.toFixed(2)}x</div>
                </div>
                <div style="height: 250px; width: 100%;">
                    <canvas id="finChart"></canvas>
                </div>
            `;
            contentDiv.appendChild(finDiv);
            
            // Render Chart after DOM insertion
            setTimeout(() => {
                if(memo.historical_financials) renderFinChart(memo.historical_financials);
            }, 0);
        }

        // --- NEW: Consensus Data ---
        if (memo.consensus_data) {
             renderConsensus(memo, contentDiv);
        }

        // --- NEW: Capital Structure & SNC ---
        if (memo.capital_structure) renderCapStructure(memo, contentDiv);
        if (memo.snc_rating) renderSNC(memo, contentDiv);
        if (memo.projections) renderProjections(memo, contentDiv);

        // --- NEW: System 2 Audit ---
        if (memo.system2_audit) renderSystem2Audit(memo.system2_audit, contentDiv);

        // Sections
        if (memo.sections) {
            memo.sections.forEach(section => {
                const sectionDiv = document.createElement('div');
                let contentHtml = section.content.replace(/\n/g, '<br>');
                // Citations
                contentHtml = contentHtml.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
                     return `<span class="citation-tag" onclick="viewEvidence('${docId}', 'chunk_001')">${docId}</span>`; // Fallback chunk ID
                });

                sectionDiv.innerHTML = `
                    <h2 class="editable-content">${section.title}</h2>
                    <div class="editable-content" style="text-align: justify;">${contentHtml}</div>
                `;
                contentDiv.appendChild(sectionDiv);
            });
        }

        // DCF (Interactive)
        if (memo.dcf_analysis) {
            renderDCF(memo.dcf_analysis, contentDiv);
        }
        
        // --- NEW: Valuation Scenarios ---
        if (memo.price_targets) {
            renderValuationScenarios(memo.price_targets, contentDiv);
        }

        // Risk Quant (Enhanced)
        renderRiskQuant(memo, contentDiv);

        if (isEditMode) applyEditMode();
    }

    function renderFinChart(data) {
        const ctx = document.getElementById('finChart');
        if (!ctx) return;
        
        if (currentChart) currentChart.destroy();

        // Data prep (assuming newest first in array)
        const chartData = [...data].reverse();

        currentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.map(d => d.period),
                datasets: [
                    {
                        label: 'Revenue',
                        data: chartData.map(d => d.revenue),
                        backgroundColor: 'rgba(0, 243, 255, 0.5)',
                        borderColor: 'rgba(0, 243, 255, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'EBITDA',
                        data: chartData.map(d => d.ebitda),
                        backgroundColor: 'rgba(0, 255, 65, 0.5)',
                        borderColor: 'rgba(0, 255, 65, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true } }
            }
        });
    }

    function renderDCF(dcf, container) {
        const dcfDiv = document.createElement('div');
        dcfDiv.innerHTML = `<h2>Annex B: Valuation (Interactive)</h2>`;
        
        dcfDiv.innerHTML += `
            <div style="background: #f0f0f0; padding: 15px; border: 1px dashed #999; margin-bottom: 20px;">
                <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px;">
                    <label>WACC (%): <input type="number" id="dcf-wacc" value="${(dcf.wacc*100).toFixed(1)}" step="0.1" style="width:60px"></label>
                    <label>Growth (%): <input type="number" id="dcf-growth" value="${(dcf.growth_rate*100).toFixed(1)}" step="0.1" style="width:60px"></label>
                    <button onclick="window.recalculateDCF()" style="padding: 2px 8px; font-size: 0.7rem;">UPDATE</button>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 1rem;">
                    <div>Implied Share Price: <span id="dcf-share-price" style="font-weight: bold; color: green;">$${dcf.share_price.toFixed(2)}</span></div>
                    <div>Enterprise Value: <span id="dcf-ev" style="font-weight: bold; color: blue;">${formatCurrency(dcf.enterprise_value)}</span></div>
                </div>
            </div>
        `;
        container.appendChild(dcfDiv);
    }
    
    // --- NEW: Consensus Renderer ---
    function renderConsensus(memo, container) {
        const c = memo.consensus_data;
        const div = document.createElement('div');
        div.style.marginBottom = "30px";

        div.innerHTML = `
            <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Consensus vs System Model</h3>
            <div style="background: #fdfdfd; padding: 15px; border: 1px solid #eee;">
                <table class="fin-table" style="width:100%; font-size:0.8rem;">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th style="text-align:right">Consensus Mean</th>
                            <th style="text-align:right">Street High</th>
                            <th style="text-align:right">Street Low</th>
                            <th style="text-align:right; background: #e0f7fa;">System (RAG)</th>
                            <th style="text-align:right">Delta %</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>Revenue</b></td>
                            <td class="num">${formatCurrency(c.revenue.mean)}</td>
                            <td class="num">${formatCurrency(c.revenue.high)}</td>
                            <td class="num">${formatCurrency(c.revenue.low)}</td>
                            <td class="num" style="background: #e0f7fa;"><b>${formatCurrency(memo.historical_financials[0].revenue)}</b></td>
                            <td class="num" style="color:${c.revenue.system_delta_pct > 0 ? 'green':'red'}">${(c.revenue.system_delta_pct * 100).toFixed(1)}%</td>
                        </tr>
                         <tr>
                            <td><b>EBITDA</b></td>
                            <td class="num">${formatCurrency(c.ebitda.mean)}</td>
                            <td class="num">${formatCurrency(c.ebitda.high)}</td>
                            <td class="num">${formatCurrency(c.ebitda.low)}</td>
                            <td class="num" style="background: #e0f7fa;"><b>${formatCurrency(memo.historical_financials[0].ebitda)}</b></td>
                            <td class="num" style="color:${c.ebitda.system_delta_pct > 0 ? 'green':'red'}">${(c.ebitda.system_delta_pct * 100).toFixed(1)}%</td>
                        </tr>
                    </tbody>
                </table>
                <div style="margin-top:10px; font-size:0.75rem; color:#666;">
                    System Sentiment: <b>${c.sentiment}</b> based on variance from consensus.
                </div>
            </div>
        `;
        container.appendChild(div);
    }

    // --- NEW: Valuation Scenarios ---
    function renderValuationScenarios(targets, container) {
        const div = document.createElement('div');
        div.style.marginBottom = "30px";

        div.innerHTML = `
            <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Valuation Scenarios (12M Price Targets)</h3>
            <div style="display: flex; gap: 10px; justify-content: space-between;">
                <div style="flex:1; background:#ffebee; border:1px solid #ffcdd2; padding:15px; text-align:center;">
                    <div style="font-size:0.7rem; color:#b71c1c; text-transform:uppercase; font-weight:bold;">Bear Case</div>
                    <div style="font-size:1.5rem; font-family:var(--font-mono); color:#b71c1c;">$${targets.bear.toFixed(2)}</div>
                </div>
                <div style="flex:1; background:#f3e5f5; border:1px solid #e1bee7; padding:15px; text-align:center;">
                    <div style="font-size:0.7rem; color:#4a148c; text-transform:uppercase; font-weight:bold;">Base Case</div>
                    <div style="font-size:1.5rem; font-family:var(--font-mono); color:#4a148c;">$${targets.base.toFixed(2)}</div>
                </div>
                <div style="flex:1; background:#e8f5e9; border:1px solid #c8e6c9; padding:15px; text-align:center;">
                    <div style="font-size:0.7rem; color:#1b5e20; text-transform:uppercase; font-weight:bold;">Bull Case</div>
                    <div style="font-size:1.5rem; font-family:var(--font-mono); color:#1b5e20;">$${targets.bull.toFixed(2)}</div>
                </div>
            </div>
        `;
        container.appendChild(div);
    }

    function renderCapStructure(memo, container) {
        const cs = memo.capital_structure;
        if (!cs) return;
        const div = document.createElement('div');
        div.style.marginBottom = "30px";

        window._cs_sources = cs.sources || {};
        const bind = (key) => window._cs_sources[key] ? `style="cursor:pointer; text-decoration:underline; color:blue;" onclick='window.viewEvidence(window._cs_sources["${key}"])'` : "";

        div.innerHTML = `
            <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Capital Structure</h3>
            <div style="background: #fdfdfd; padding: 15px; border: 1px solid #eee;">
                <table class="fin-table" style="width:100%; font-size:0.8rem;">
                    <thead>
                        <tr>
                            <th>Instrument</th>
                            <th style="text-align:right">Amount ($M)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td>Senior Notes</td><td class="num" ${bind('senior_notes')}>${formatCurrency(cs.senior_notes)}</td></tr>
                        <tr><td>Revolver Drawn</td><td class="num" ${bind('revolver')}>${formatCurrency(cs.revolver)}</td></tr>
                        <tr><td>Subordinated Debt</td><td class="num" ${bind('subordinated')}>${formatCurrency(cs.subordinated)}</td></tr>
                        <tr style="border-top:2px solid #000"><td><b>Total Debt</b></td><td class="num"><b>${formatCurrency(cs.senior_notes + cs.revolver + cs.subordinated)}</b></td></tr>
                    </tbody>
                </table>
            </div>
        `;
        container.appendChild(div);
    }

    function renderSNC(memo, container) {
        const snc = memo.snc_rating;
        if (!snc) return;
        const div = document.createElement('div');
        div.style.marginBottom = "30px";
        window._snc_sources = snc.sources || {};

        div.innerHTML = `
            <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">SNC Regulatory Rating</h3>
            <div style="background: #fff3e0; padding: 15px; border-left: 5px solid #ff9800;">
                <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 5px;">
                    <span style="cursor:pointer;" onclick='window.viewEvidence(window._snc_sources.rating)'>${snc.rating}</span>
                </div>
                <div style="font-style: italic; color: #555;">
                    " <span style="cursor:pointer;" onclick='window.viewEvidence(window._snc_sources.justification)'>${snc.justification}</span> "
                </div>
            </div>
        `;
        container.appendChild(div);
    }

    function renderProjections(memo, container) {
        const p = memo.projections;
        if (!p) return;
        const div = document.createElement('div');
        div.style.marginBottom = "30px";
        window._proj_sources = p.sources || {};

        div.innerHTML = `
            <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Management Projections (Guidance)</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: #e8eaf6; padding: 15px; text-align: center;">
                    <div style="font-size:0.8rem; color:#3f51b5;">Revenue Growth</div>
                    <div style="font-size:1.5rem; color:#1a237e; cursor:pointer;" onclick='window.viewEvidence(window._proj_sources.revenue_growth)'>${p.revenue_growth.toFixed(1)}%</div>
                </div>
                <div style="background: #e8eaf6; padding: 15px; text-align: center;">
                    <div style="font-size:0.8rem; color:#3f51b5;">EBITDA Margin Target</div>
                    <div style="font-size:1.5rem; color:#1a237e; cursor:pointer;" onclick='window.viewEvidence(window._proj_sources.ebitda_margin)'>${p.ebitda_margin.toFixed(1)}%</div>
                </div>
            </div>
        `;
        container.appendChild(div);
    }

    function renderSystem2Audit(audit, container) {
        const div = document.createElement('div');
        div.style.marginBottom = "30px";

        const rows = audit.log.map(l => `
            <tr>
                <td>${l.check}</td>
                <td><span style="color: ${l.status==='PASS'?'green':(l.status==='WARN'?'orange':'red')}">${l.status}</span></td>
                <td>${l.msg}</td>
            </tr>
        `).join('');

        div.innerHTML = `
             <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">System 2 Audit Log</h3>
             <div style="background: #333; color: #eee; padding: 15px; font-family: monospace; font-size: 0.8rem;">
                <div style="margin-bottom:10px; border-bottom:1px solid #555;">VERDICT: <span style="color:${audit.verdict==='CLEAN'?'#00e676':'#ff5252'}">${audit.verdict} (Score: ${audit.score})</span></div>
                <table style="width:100%;">
                    <tbody>${rows}</tbody>
                </table>
             </div>
        `;
        container.appendChild(div);
    }

    function renderRiskQuant(memo, container) {
         // Use existing PD model data if available from RAG
         let pd = 0.0, lgd = 0.45, rating = "N/A";
         let debt = 0;

         if (memo.pd_model) {
             pd = memo.pd_model.pd_1yr || 0.02;
             lgd = memo.pd_model.lgd || 0.45;
             rating = memo.pd_model.regulatory_rating || memo.regulatory_rating || "N/A";
         } else {
             // Fallback
             debt = memo.historical_financials?.[0]?.total_liabilities || 0;
             const equity = memo.historical_financials?.[0]?.total_equity || 0;
             const assets = debt + equity;
             const leverage = assets > 0 ? debt / assets : 0;
             pd = (leverage * 0.05).toFixed(4);
         }

         // If debt wasn't set from fallback
         if (!debt && memo.historical_financials && memo.historical_financials[0]) {
             debt = memo.historical_financials[0].total_liabilities || memo.historical_financials[0].gross_debt || 0;
         }

         const el = debt * pd * lgd;
         
         const div = document.createElement('div');
         div.innerHTML = `
            <h2>Risk Model Output & Regulatory Rating</h2>
            <div style="font-family: var(--font-mono); font-size: 0.8rem; border: 1px solid #ccc; padding: 20px; background: #fffbf0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <div style="font-size:0.7rem; color:#666;">Regulatory Rating</div>
                        <div style="font-size:1.2rem; font-weight:bold; color:#000;">${rating}</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:#666;">Probability of Default (1Y)</div>
                        <div style="font-size:1.2rem; font-weight:bold; color:#d32f2f;">${(pd*100).toFixed(2)}%</div>
                    </div>
                     <div>
                        <div style="font-size:0.7rem; color:#666;">Loss Given Default (LGD)</div>
                        <div style="font-size:1.2rem; font-weight:bold; color:#000;">${(lgd*100).toFixed(0)}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.7rem; color:#666;">Expected Loss ($)</div>
                        <div style="font-size:1.2rem; font-weight:bold; color:#000;">${formatCurrency(el)}</div>
                    </div>
                </div>
                <div style="margin-top:15px; font-size:0.75rem; color:#666; font-style:italic;">
                    Rationale: ${memo.pd_model?.rationale || "Automated calculation based on leverage ratios."}
                </div>
            </div>
         `;
         container.appendChild(div);
    }

    // --- Evidence Viewer (Merged) ---
    window.viewEvidence = async function(source) {
        const viewer = elements.evidenceViewer;
        viewer.innerHTML = '<div style="padding:20px; color: #888;">Loading evidence...</div>';

        // Support Legacy Call (docId, chunkId) or New Object
        let docId, start, end;
        // Check arguments for legacy calls
        if (arguments.length > 1 || typeof source === 'string') {
             docId = source;
             start = 0; end = 0;
        } else if (source && typeof source === 'object') {
             docId = source.doc_id;
             start = source.start;
             end = source.end;
        }

        if (!docId) {
            viewer.innerHTML = '<div style="padding:20px; color: red;">No document ID provided.</div>';
            return;
        }
        
        if(elements.docIdLabel) elements.docIdLabel.innerText = docId;

        try {
            // Fetch text file
            const res = await fetch(`data/${docId}`);
            if (!res.ok) throw new Error(`Document ${docId} not found`);
            const text = await res.text();
            
            // Render Text
            const pre = document.createElement('pre');
            pre.style.whiteSpace = "pre-wrap";
            pre.style.fontFamily = "monospace";
            pre.style.fontSize = "11px";
            pre.style.lineHeight = "1.5";
            pre.style.padding = "20px";
            pre.style.color = "#333";
            pre.style.backgroundColor = "#fff";
            pre.style.overflowX = "hidden";
            pre.style.height = "100%";
            pre.style.margin = "0";
            
            if (typeof start === 'number' && typeof end === 'number' && end > start) {
                // Safe Highlight
                const before = text.substring(0, start);
                const target = text.substring(start, end);
                const after = text.substring(end);

                const spanBefore = document.createTextNode(before);

                const spanTarget = document.createElement('span');
                spanTarget.style.backgroundColor = "#ffeb3b"; // Yellow
                spanTarget.style.color = "#000";
                spanTarget.style.border = "1px solid #fbc02d";
                spanTarget.innerText = target;
                spanTarget.id = "highlight-target";

                const spanAfter = document.createTextNode(after);

                pre.appendChild(spanBefore);
                pre.appendChild(spanTarget);
                pre.appendChild(spanAfter);
            } else {
                pre.innerText = text;
            }
            
            viewer.innerHTML = '';
            viewer.appendChild(pre);

            // Scroll to highlight
            setTimeout(() => {
                const el = document.getElementById('highlight-target');
                if(el) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);

            log("Frontend", `User viewed evidence ${docId}`, "audit");

        } catch (e) {
             viewer.innerHTML = `<div style="padding:20px; color: red;">Error: ${e.message}</div>`;
             console.error(e);
        }
    };

    // --- Helpers ---
    function log(agent, message, type="info") {
        const div = document.createElement('div');
        div.className = `log-entry ${type}`; // Uses CSS from previous step
        div.innerHTML = `<span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span><span class="log-agent">${agent}</span><span class="log-message">${message}</span>`;
        elements.terminal.appendChild(div);
        elements.terminal.scrollTop = elements.terminal.scrollHeight;
    }

    function clearTerminal() {
        elements.terminal.innerHTML = '';
    }

    function simulateAgentTask(agent, messages, delay) {
        return new Promise(resolve => {
            let i = 0;
            const interval = setInterval(() => {
                if (i >= messages.length) {
                    clearInterval(interval);
                    resolve();
                } else {
                    log(agent, messages[i]);
                    i++;
                }
            }, delay);
        });
    }

    function formatCurrency(val) {
        if (!val) return '-';
        return val >= 1000 ? `$${(val/1000).toFixed(1)}B` : `$${val.toFixed(0)}M`;
    }

    function toggleEditMode() {
        isEditMode = !isEditMode;
        const btn = elements.editBtn;
        if (isEditMode) {
            btn.classList.add('editing-active');
            btn.innerHTML = 'SAVE CHANGES';
            applyEditMode();
        } else {
            btn.classList.remove('editing-active');
            btn.innerHTML = 'EDIT MODE';
            document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "false");
        }
    }

    function applyEditMode() {
        document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "true");
    }
    
    // Expose DCF recalc to window since it's called by inline onclick
    window.recalculateDCF = function() {
        if (!currentMemoData || !currentMemoData.dcf_analysis) return;
        const wacc = parseFloat(document.getElementById('dcf-wacc').value) / 100;
        const growth = parseFloat(document.getElementById('dcf-growth').value) / 100;
        
        // Mock Calc
        const basePrice = currentMemoData.dcf_analysis.share_price;
        const originalWacc = currentMemoData.dcf_analysis.wacc;
        const factor = originalWacc / wacc; // Simple inverse relationship for demo
        const newPrice = basePrice * factor;
        const newEV = currentMemoData.dcf_analysis.enterprise_value * factor;
        
        document.getElementById('dcf-share-price').textContent = `$${newPrice.toFixed(2)}`;
        document.getElementById('dcf-ev').textContent = formatCurrency(newEV);
    };
    
    function loadAuditLogs() {
        // Mock loading logs
        fetch('data/credit_memo_audit_log.json')
            .then(res => res.json())
            .then(logs => {
                logs.slice(-5).forEach(l => log("System", `[AUDIT] ${l.action}`, "audit"));
            })
            .catch(e => console.log("No audit logs found"));
    }

    // Start
    init();
});