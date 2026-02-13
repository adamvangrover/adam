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
        uploadBtn: document.getElementById('upload-btn'), // New
        downloadBtn: document.getElementById('download-btn'), // New
        memoContainer: document.getElementById('memo-container'),
        memoContent: document.getElementById('memo-content'),
        evidenceViewer: document.getElementById('pdf-mock-canvas'),
        terminal: document.getElementById('agent-terminal'),
        docIdLabel: document.getElementById('doc-id-label') || document.createElement('div'),
        modeIndicator: document.querySelector('.header-nav span span') 
    };

    // --- Initialization ---
    async function init() {
        log("System", "Initializing CreditOS v2.5 (Advanced)...", "info");

        try {
            // Load Library
            const res = await fetch('data/credit_memo_library.json');
            if (res.ok) {
                library = await res.json();
                
                if (Array.isArray(library)) {
                    const libObj = {};
                    library.forEach(item => libObj[item.id] = item);
                    library = libObj;
                }

                // Populate Dropdown
                Object.values(library).forEach(item => {
                    const opt = document.createElement('option');
                    opt.value = item.file || item.id;
                    opt.innerText = `${item.borrower_name} (Risk: ${item.risk_score})`;
                    opt.dataset.ticker = item.ticker || "UNK";
                    opt.dataset.name = item.borrower_name;
                    elements.borrowerSelect.appendChild(opt);
                });
                
                log("Archivist", `Loaded ${Object.keys(library).length} entity profiles.`, "success");
            } else {
                // Non-fatal, might be running without library
                console.warn("Library not found");
            }
            
            loadAuditLogs();

        } catch (e) {
            log("System", "Failed to load artifact library.", "error");
            console.error(e);
        }

        // Setup File Input for Upload
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.json';
        fileInput.style.display = 'none';
        fileInput.id = 'model-upload-input';
        document.body.appendChild(fileInput);

        fileInput.addEventListener('change', handleFileUpload);

        if(elements.uploadBtn) {
            elements.uploadBtn.addEventListener('click', () => fileInput.click());
        }

        if(elements.downloadBtn) {
            elements.downloadBtn.addEventListener('click', downloadModel);
        }
    }

    // --- Event Listeners ---
    if(elements.generateBtn) elements.generateBtn.addEventListener('click', runAnalysis);
    if(elements.editBtn) elements.editBtn.addEventListener('click', toggleEditMode);
    if(elements.exportBtn) elements.exportBtn.addEventListener('click', () => window.print());

    // --- Core Workflow ---
    async function runAnalysis() {
        const val = elements.borrowerSelect.value;
        if (!val) { alert("Please select a target entity."); return; }

        let filename = val;
        if (!val.endsWith('.json')) {
            if (library[val] && library[val].file) {
                filename = library[val].file;
            } else {
                filename = `credit_memo_${val}.json`;
            }
        }
        
        const option = elements.borrowerSelect.options[elements.borrowerSelect.selectedIndex];
        const ticker = option.dataset.ticker || "UNK";
        const name = option.dataset.name || val;

        // Reset UI
        elements.memoContent.innerHTML = '';
        clearTerminal();
        
        // 1. Retrieval Phase
        await simulateAgentTask("Archivist", [
            `Querying Vector DB for ${ticker}...`,
            `Found 10-K (${new Date().getFullYear()}) - Indexing...`,
            `Extracting relevant chunks (MD&A, Risk Factors)...`
        ], 600);

        // 2. Load Data
        try {
            const path = filename.includes('/') ? filename : `data/${filename}`;
            const res = await fetch(path);
            if (!res.ok) throw new Error("File not found");
            currentMemoData = await res.json();

            // Enrich with default enhanced data if missing (for demo purposes)
            enrichData(currentMemoData);

            selectedBorrower = currentMemoData;
        } catch(e) {
            log("System", `Error loading memo data: ${e.message}`, "error");
            // Fallback for demo
             log("System", "Generating synthetic data...", "warning");
             currentMemoData = generateSyntheticData(name, ticker);
             enrichData(currentMemoData);
        }

        // 3. Quant Phase
        const assets = currentMemoData.historical_financials?.[0]?.total_assets || 'N/A';
        await simulateAgentTask("Quant", [
            `OCR-ing Financial Tables...`,
            `Mapping line items to FIBO ontology...`,
            `Validating Balance Sheet: Assets (${assets}) = Liab + Equity...`,
            `PASS: Checksum valid.`,
            `Running DCF Sensitivity Analysis...`
        ], 800);

        // 4. Synthesis Phase
        await simulateAgentTask("Writer", [
            `Synthesizing Executive Summary...`,
            `Injecting Citations...`,
            `Compiling Final Report...`
        ], 800);

        // Render
        try {
            renderMemo(currentMemoData);
            log("Orchestrator", `Analysis Complete for ${name}`, "success");
        } catch(e) {
            log("System", `Render Error: ${e.message}`, "error");
            console.error(e);
        }
    }

    // --- Data Enrichment (Demo) ---
    function enrichData(data) {
        if (!data.outlook) {
            data.outlook = {
                rating: data.risk_score > 80 ? "STRONG BUY" : (data.risk_score > 60 ? "BUY" : "HOLD"),
                conviction: Math.floor(Math.random() * 20) + 80, // 80-99
                price_target_base: (data.dcf_analysis?.share_price || 150) * 1.1,
                price_target_bull: (data.dcf_analysis?.share_price || 150) * 1.3,
                price_target_bear: (data.dcf_analysis?.share_price || 150) * 0.8
            };
        }
        if (!data.forecast_financials && data.historical_financials) {
            const last = data.historical_financials[0]; // Assuming sorted desc
            data.forecast_financials = [
                { period: 'FY2026E', revenue: last.revenue * 1.1, ebitda: last.ebitda * 1.12 },
                { period: 'FY2027E', revenue: last.revenue * 1.2, ebitda: last.ebitda * 1.25 },
                { period: 'FY2028E', revenue: last.revenue * 1.3, ebitda: last.ebitda * 1.40 }
            ];
        }
    }

    function generateSyntheticData(name, ticker) {
         return {
             borrower_name: name,
             report_date: new Date().toISOString(),
             risk_score: 75,
             historical_financials: [
                 { period: 'FY2025', revenue: 100000, ebitda: 30000, total_assets: 200000, total_liabilities: 100000, total_equity: 100000 },
                 { period: 'FY2024', revenue: 90000, ebitda: 25000 },
                 { period: 'FY2023', revenue: 80000, ebitda: 20000 }
             ],
             financial_ratios: { leverage_ratio: 2.5, ebitda: 30000, revenue: 100000, dscr: 4.5 },
             dcf_analysis: { share_price: 200.00, enterprise_value: 2500000, wacc: 0.085, growth_rate: 0.03 },
             sections: [{ title: "Executive Summary", content: "Synthetic generated summary for " + name }]
         };
    }

    // --- Rendering Logic ---
    function renderMemo(memo) {
        const contentDiv = elements.memoContent;

        contentDiv.style.display = 'block';
        const placeholder = document.getElementById('memo-placeholder');
        if(placeholder) placeholder.style.display = 'none';

        contentDiv.innerHTML = '';
        contentDiv.className = "memo-paper";

        // Header
        const riskColor = memo.risk_score < 60 ? 'red' : (memo.risk_score < 80 ? 'orange' : 'green');
        const header = document.createElement('div');

        // Analyst Outlook Section (New)
        const outlookHtml = memo.outlook ? `
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #eee; padding: 10px; margin-top: 10px; border: 1px solid #ccc;">
                <div>
                    <div style="font-size: 0.7rem; color: #666;">RATING</div>
                    <div id="outlook-rating" class="editable-content" style="font-weight: bold; color: ${memo.outlook.rating.includes('BUY') ? 'green' : 'orange'}">${memo.outlook.rating}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #666;">CONVICTION</div>
                    <div id="outlook-conviction" class="editable-content" style="font-weight: bold;">${memo.outlook.conviction}%</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: #666;">TARGET (BASE)</div>
                    <div id="outlook-target" class="editable-content" style="font-weight: bold;">$${memo.outlook.price_target_base.toFixed(2)}</div>
                </div>
                 <div>
                    <div style="font-size: 0.7rem; color: #666;">UPSIDE</div>
                    <div style="font-weight: bold; color: green;">+${((memo.outlook.price_target_base / memo.dcf_analysis?.share_price - 1) * 100).toFixed(1)}%</div>
                </div>
            </div>
        ` : '';

        header.innerHTML = `
            <h1 class="editable-content">${memo.borrower_name}</h1>
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #000; padding-bottom: 5px; font-family: var(--font-mono); font-size: 0.8rem; color: #666;">
                <span>DATE: ${new Date(memo.report_date).toLocaleDateString()}</span>
                <span>RISK SCORE: <b style="color: ${riskColor}">${memo.risk_score}/100</b></span>
                <span>ID: ${memo.borrower_name.substring(0,3).toUpperCase()}-${Math.floor(Math.random()*10000)}</span>
            </div>
            ${outlookHtml}
        `;
        contentDiv.appendChild(header);

        // Financial Snapshot & Chart (Forecast Enabled)
        if (memo.financial_ratios) {
            const finDiv = document.createElement('div');
            finDiv.style.marginBottom = '30px';
            finDiv.style.marginTop = '20px';
            finDiv.style.background = '#f9f9f9';
            finDiv.style.padding = '20px';
            finDiv.style.border = '1px solid #ddd';

            finDiv.innerHTML = `
                <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Financial Snapshot & Forecast</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; font-family: var(--font-mono); font-size: 0.8rem; margin-bottom: 20px;">
                    <div><b>Leverage:</b> ${memo.financial_ratios.leverage_ratio?.toFixed(2)}x</div>
                    <div><b>EBITDA:</b> ${formatCurrency(memo.financial_ratios.ebitda)}</div>
                    <div><b>Revenue:</b> ${formatCurrency(memo.financial_ratios.revenue)}</div>
                    <div><b>DSCR:</b> ${memo.financial_ratios.dscr?.toFixed(2)}x</div>
                </div>
                <div style="height: 300px; width: 100%;">
                    <canvas id="finChart"></canvas>
                </div>
            `;
            contentDiv.appendChild(finDiv);
            
            setTimeout(() => {
                renderFinChart(memo.historical_financials, memo.forecast_financials);
            }, 0);
        }

        // Sections
        if (memo.sections) {
            memo.sections.forEach((section, index) => {
                const sectionDiv = document.createElement('div');
                let contentHtml = section.content.replace(/\n/g, '<br>');
                contentHtml = contentHtml.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
                     return `<span class="citation-tag" onclick="viewEvidence('${docId}', 'chunk_001')">${docId}</span>`;
                });

                sectionDiv.innerHTML = `
                    <h2 class="editable-content" data-section-title-index="${index}">${section.title}</h2>
                    <div class="editable-content" data-section-content-index="${index}" style="text-align: justify;">${contentHtml}</div>
                `;
                contentDiv.appendChild(sectionDiv);
            });
        }

        // DCF (Interactive Matrix)
        if (memo.dcf_analysis) {
            renderDCF(memo.dcf_analysis, contentDiv);
        }
        
        renderRiskQuant(memo, contentDiv);

        if (isEditMode) applyEditMode();
    }

    function renderFinChart(historical, forecast) {
        const ctx = document.getElementById('finChart');
        if (!ctx) return;
        
        if (currentChart) currentChart.destroy();

        // Prepare Data
        // Historical: Newest first usually in raw data? Assuming standard array.
        // Let's sort just in case: Hist (Old -> New), Forecast (New -> Future)

        const histSorted = [...(historical || [])].reverse(); // Assuming input was desc
        const foreSorted = [...(forecast || [])];

        const labels = [...histSorted.map(d => d.period), ...foreSorted.map(d => d.period)];

        // Revenue Data
        const revHist = histSorted.map(d => d.revenue);
        // Pad forecast with nulls for historical period
        const revFore = new Array(histSorted.length).fill(null).concat(foreSorted.map(d => d.revenue));
        // Connect the lines? Add last hist point to forecast?
        if(histSorted.length > 0 && foreSorted.length > 0) {
             // To make line connect, we might need overlap or chart.js features. Keeping simple for now.
        }

        const ebitdaHist = histSorted.map(d => d.ebitda);
        const ebitdaFore = new Array(histSorted.length).fill(null).concat(foreSorted.map(d => d.ebitda));

        currentChart = new Chart(ctx, {
            data: {
                labels: labels,
                datasets: [
                    {
                        type: 'bar',
                        label: 'Historical Rev',
                        data: revHist,
                        backgroundColor: 'rgba(0, 243, 255, 0.5)',
                        borderColor: 'rgba(0, 243, 255, 1)',
                        borderWidth: 1,
                        order: 2
                    },
                    {
                        type: 'line',
                        label: 'Forecast Rev',
                        data: revFore,
                        borderColor: 'rgba(0, 243, 255, 1)',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        fill: false,
                        order: 1
                    },
                    {
                        type: 'bar',
                        label: 'Historical EBITDA',
                        data: ebitdaHist,
                        backgroundColor: 'rgba(0, 255, 65, 0.5)',
                        borderColor: 'rgba(0, 255, 65, 1)',
                        borderWidth: 1,
                        order: 3
                    },
                    {
                        type: 'line',
                        label: 'Forecast EBITDA',
                        data: ebitdaFore,
                        borderColor: 'rgba(0, 255, 65, 1)',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        fill: false,
                        order: 1
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
        dcfDiv.innerHTML = `<h2>Annex B: Valuation & Sensitivity</h2>`;

        const currentPrice = dcf.share_price;
        const currentWacc = dcf.wacc;
        const currentGrowth = dcf.growth_rate;

        // Generate Sensitivity Matrix (3x3 around current)
        // Rows: WACC, Cols: Growth
        const waccSteps = [currentWacc - 0.01, currentWacc, currentWacc + 0.01];
        const growthSteps = [currentGrowth - 0.01, currentGrowth, currentGrowth + 0.01];
        
        let matrixHtml = '<table style="width:100%; font-family: var(--font-mono); font-size: 0.8rem; border-collapse: collapse; text-align: center;">';
        matrixHtml += '<tr><td style="border:none;"></td><th colspan="3">Terminal Growth</th></tr>';
        matrixHtml += `<tr><th>WACC</th>${growthSteps.map(g => `<th>${(g*100).toFixed(1)}%</th>`).join('')}</tr>`;

        waccSteps.forEach(w => {
            matrixHtml += `<tr><th>${(w*100).toFixed(1)}%</th>`;
            growthSteps.forEach(g => {
                // Simple DCF Approx: P = P_base * (W_base / W) * ( (1+g)/(1+g_base) )^Multi
                // Or just recalc EV/Shares.
                // Using simple sensitivity proxy for demo:
                // Price ~ 1/(Wacc - Growth)
                const baseMultiple = 1 / (currentWacc - currentGrowth);
                const newMultiple = 1 / (w - g);
                const price = currentPrice * (newMultiple / baseMultiple);

                // Color coding
                const diff = (price / currentPrice) - 1;
                const color = diff > 0.05 ? 'green' : (diff < -0.05 ? 'red' : 'black');
                const bg = diff > 0.1 ? '#e0ffe0' : (diff < -0.1 ? '#ffe0e0' : '#fff');

                matrixHtml += `<td style="border: 1px solid #ddd; background: ${bg}; color: ${color}; padding: 5px;">$${price.toFixed(0)}</td>`;
            });
            matrixHtml += '</tr>';
        });
        matrixHtml += '</table>';

        dcfDiv.innerHTML += `
            <div style="background: #f0f0f0; padding: 15px; border: 1px dashed #999; margin-bottom: 20px; display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <h3>Assumptions</h3>
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                        <label>WACC (%): <input type="number" id="dcf-wacc" value="${(dcf.wacc*100).toFixed(1)}" step="0.1" style="width:60px"></label>
                        <label>Growth (%): <input type="number" id="dcf-growth" value="${(dcf.growth_rate*100).toFixed(1)}" step="0.1" style="width:60px"></label>
                        <button onclick="window.recalculateDCF()" style="padding: 5px; background: #333; color: white; border: none; cursor: pointer;">UPDATE MODEL</button>
                    </div>
                     <div style="margin-top: 15px; font-family: var(--font-mono);">
                        <div>Implied Price: <span id="dcf-share-price" style="font-weight: bold; font-size: 1.2rem; color: blue;">$${dcf.share_price.toFixed(2)}</span></div>
                    </div>
                </div>
                <div style="flex: 1;">
                    <h3>Sensitivity Analysis</h3>
                    ${matrixHtml}
                </div>
            </div>
        `;
        container.appendChild(dcfDiv);
    }
    
    function renderRiskQuant(memo, container) {
         const debt = memo.historical_financials?.[0]?.total_liabilities || 0;
         const equity = memo.historical_financials?.[0]?.total_equity || 0;
         const assets = debt + equity;
         const leverage = assets > 0 ? debt / assets : 0;
         const pd = (leverage * 0.05).toFixed(4);
         
         const panel = document.getElementById('risk-quant-panel');
         if (panel) {
             panel.innerHTML = '';
             const div = document.createElement('div');
             div.style.padding = '10px';
             div.style.fontFamily = 'var(--font-mono)';
             div.style.fontSize = '0.8rem';
             div.style.color = 'var(--text-primary)';

             div.innerHTML = `
                <div style="margin-bottom: 10px; color: var(--neon-magenta); font-weight: bold;">RISK METRICS</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>PD (1Y):</span>
                    <span style="color: var(--neon-pink);">${(pd*100).toFixed(2)}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>LGD:</span>
                    <span>45.0%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Exp. Loss:</span>
                    <span>${formatCurrency(debt * pd * 0.45)}</span>
                </div>
                 <div style="display: flex; justify-content: space-between; margin-bottom: 5px; margin-top: 10px;">
                    <span>Conviction:</span>
                    <span style="color: cyan;">${memo.outlook ? memo.outlook.conviction : 'N/A'}%</span>
                </div>
                <div style="margin-top: 10px; border-top: 1px solid #333; padding-top: 5px;">
                    <div style="font-size: 0.7rem; color: #666;">MODEL: CREDIT_RF_V2</div>
                </div>
             `;
             panel.appendChild(div);
         } else {
             const div = document.createElement('div');
             div.innerHTML = `
                <h2>Risk Model Output</h2>
                <div style="font-family: var(--font-mono); font-size: 0.8rem; border: 1px solid #ccc; padding: 10px;">
                    <div>Probability of Default (PD): ${(pd*100).toFixed(2)}%</div>
                    <div>Loss Given Default (LGD): 45%</div>
                    <div>Expected Loss: ${formatCurrency(debt * pd * 0.45)}</div>
                </div>
             `;
             container.appendChild(div);
         }
    }

    // --- I/O Functions ---
    async function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const data = JSON.parse(e.target.result);
                currentMemoData = data;
                enrichData(currentMemoData); // Ensure compat
                renderMemo(currentMemoData);
                log("System", `Model loaded from ${file.name}`, "success");
            } catch (err) {
                log("System", "Failed to parse JSON model file.", "error");
            }
        };
        reader.readAsText(file);
    }

    function downloadModel() {
        if (!currentMemoData) { alert("No model loaded."); return; }

        // Capture edits if any
        if (isEditMode) {
            alert("Please save changes (exit Edit Mode) before downloading.");
            return;
        }

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentMemoData, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `${currentMemoData.borrower_name || "credit_memo"}_model.json`);
        document.body.appendChild(downloadAnchorNode); // required for firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
        log("System", "Model downloaded successfully.", "success");
    }

    // --- Evidence Viewer (Merged) ---
    window.viewEvidence = (docId, chunkId) => {
        const viewer = elements.evidenceViewer;
        viewer.innerHTML = '';

        let doc, chunk;
        if(currentMemoData && currentMemoData.documents) {
             doc = currentMemoData.documents.find(d => d.doc_id === docId);
             chunk = doc?.chunks.find(c => c.chunk_id === chunkId);
        }

        const page = document.createElement('div');
        page.style.width = '100%';
        page.style.height = '1000px';
        page.style.background = '#fff';
        page.style.position = 'relative';
        page.style.padding = '40px';
        page.style.boxSizing = 'border-box';
        
        page.innerHTML = `<div style="color: #ddd; font-size: 6px; overflow:hidden; height:100%; word-break: break-all;">
            ${"CONTENT ".repeat(5000)}
        </div>`;

        if (chunk && chunk.bbox) {
            const [x0, y0, x1, y1] = chunk.bbox;
            const bbox = document.createElement('div');
            bbox.className = 'bbox-highlight';
            bbox.style.left = `${x0}px`;
            bbox.style.top = `${y0}px`;
            bbox.style.width = `${x1 - x0}px`;
            bbox.style.height = `${y1 - y0}px`;
            
            const label = document.createElement('div');
            label.className = 'bbox-label';
            label.innerText = chunk.type.toUpperCase();
            bbox.appendChild(label);
            
            const textOverlay = document.createElement('div');
            textOverlay.style.background = "rgba(255,255,255,0.9)";
            textOverlay.style.color = "black";
            textOverlay.style.fontSize = "12px";
            textOverlay.style.height = "100%";
            textOverlay.style.overflow = "hidden";
            textOverlay.innerText = chunk.content;
            bbox.appendChild(textOverlay);

            page.appendChild(bbox);
            setTimeout(() => bbox.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
        } else {
            page.innerHTML += `<div style="position:absolute; top:100px; left:50px; background:yellow; padding:10px; color:black;">Evidence Loaded: ${docId}</div>`;
        }

        viewer.appendChild(page);
        log("Frontend", `User viewed evidence ${docId}`, "audit");
    };

    // --- Helpers ---
    function log(agent, message, type="info") {
        const div = document.createElement('div');
        div.className = `log-entry ${type}`;
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

            // Capture changes from DOM back to Data Model
            if(currentMemoData) {
                // 1. Capture Outlook Fields if present
                if (currentMemoData.outlook) {
                    const outlookDivs = document.querySelectorAll('.editable-content');
                    // This is brittle without IDs, but for the header section:
                    // We know the order: Name, Title, Content...
                    // Let's rely on specific IDs if we can, or just accept that full DOM scraping is complex.
                    // BETTER: Let's add IDs to the outlook fields in renderMemo to make this robust.

                    const ratingEl = document.getElementById('outlook-rating');
                    const convictionEl = document.getElementById('outlook-conviction');
                    const targetEl = document.getElementById('outlook-target');

                    if(ratingEl) currentMemoData.outlook.rating = ratingEl.innerText;
                    if(convictionEl) currentMemoData.outlook.conviction = parseInt(convictionEl.innerText) || currentMemoData.outlook.conviction;
                    if(targetEl) currentMemoData.outlook.price_target_base = parseFloat(targetEl.innerText.replace('$','')) || currentMemoData.outlook.price_target_base;
                }

                // 2. Capture Name
                const nameEl = document.querySelector('h1.editable-content');
                if(nameEl) currentMemoData.borrower_name = nameEl.innerText;

                // 3. Capture Sections (Narrative)
                if (currentMemoData.sections) {
                    currentMemoData.sections.forEach((section, index) => {
                        const titleEl = document.querySelector(`[data-section-title-index="${index}"]`);
                        const contentEl = document.querySelector(`[data-section-content-index="${index}"]`);

                        if (titleEl) section.title = titleEl.innerText;
                        if (contentEl) {
                            // Convert <br> back to \n roughly, or just save innerText which might lose formatting but keeps content.
                            // innerText preserves newlines as \n usually.
                            section.content = contentEl.innerText;
                        }
                    });
                }

                log("System", "Changes saved to local model.", "success");
            }
        }
    }

    function applyEditMode() {
        document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "true");
    }
    
    window.recalculateDCF = function() {
        if (!currentMemoData || !currentMemoData.dcf_analysis) return;
        const wacc = parseFloat(document.getElementById('dcf-wacc').value) / 100;
        const growth = parseFloat(document.getElementById('dcf-growth').value) / 100;
        
        // Update Model
        currentMemoData.dcf_analysis.wacc = wacc;
        currentMemoData.dcf_analysis.growth_rate = growth;

        // Recalc (Simple Proxy)
        const basePrice = 200; // Fixed baseline for stability in demo or use original
        const factor = (0.085 / wacc) * ((1+growth)/(1+0.03));
        const newPrice = basePrice * factor;
        const newEV = 2500000 * factor;
        
        currentMemoData.dcf_analysis.share_price = newPrice;
        currentMemoData.dcf_analysis.enterprise_value = newEV;

        // Re-render only DCF section? Easier to re-render all for matrix update.
        renderMemo(currentMemoData);
    };
    
    function loadAuditLogs() {
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