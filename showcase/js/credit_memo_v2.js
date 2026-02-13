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
            // Load Library
            const res = await fetch('data/credit_memo_library.json');
            if (res.ok) {
                library = await res.json(); // Array or Object? Handling both.
                
                // Normalize if array
                if (Array.isArray(library)) {
                    const libObj = {};
                    library.forEach(item => libObj[item.id] = item);
                    library = libObj;
                }

                // Populate Dropdown
                Object.values(library).forEach(item => {
                    const opt = document.createElement('option');
                    opt.value = item.file; // Using filename as key to load actual data
                    opt.innerText = `${item.borrower_name} (Risk: ${item.risk_score})`;
                    opt.dataset.ticker = item.ticker || "UNK";
                    opt.dataset.name = item.borrower_name;
                    elements.borrowerSelect.appendChild(opt);
                });
                
                log("Archivist", `Loaded ${Object.keys(library).length} entity profiles.`, "success");
            } else {
                throw new Error("Library not found");
            }
            
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
            const path = filename.includes('/') ? filename : `data/${filename}`;
            const res = await fetch(path);
            if (!res.ok) throw new Error("File not found");
            currentMemoData = await res.json();
            selectedBorrower = currentMemoData; // Sync state
        } catch(e) {
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
        const contentDiv = elements.memoContent;
        contentDiv.innerHTML = '';
        contentDiv.className = "memo-paper"; // Apply the paper CSS class

        // Header
        const riskColor = memo.risk_score < 60 ? 'red' : (memo.risk_score < 80 ? 'orange' : 'green');
        const header = document.createElement('div');
        header.innerHTML = `
            <h1 class="editable-content">${memo.borrower_name}</h1>
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 30px; font-family: var(--font-mono); font-size: 0.8rem; color: #666;">
                <span>DATE: ${new Date(memo.report_date).toLocaleDateString()}</span>
                <span>RISK SCORE: <b style="color: ${riskColor}">${memo.risk_score}/100</b></span>
                <span>ID: ${memo.borrower_name.substring(0,3).toUpperCase()}-${Math.floor(Math.random()*10000)}</span>
            </div>
        `;
        contentDiv.appendChild(header);

        // Financial Snapshot & Chart (Merged Feature)
        if (memo.financial_ratios) {
            const finDiv = document.createElement('div');
            finDiv.style.marginBottom = '30px';
            finDiv.style.background = '#f9f9f9';
            finDiv.style.padding = '20px';
            finDiv.style.border = '1px solid #ddd';

            finDiv.innerHTML = `
                <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Financial Snapshot</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; font-family: var(--font-mono); font-size: 0.8rem; margin-bottom: 20px;">
                    <div><b>Leverage:</b> ${memo.financial_ratios.leverage_ratio?.toFixed(2)}x</div>
                    <div><b>EBITDA:</b> ${formatCurrency(memo.financial_ratios.ebitda)}</div>
                    <div><b>Revenue:</b> ${formatCurrency(memo.financial_ratios.revenue)}</div>
                    <div><b>DSCR:</b> ${memo.financial_ratios.dscr?.toFixed(2)}x</div>
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
        
        // Risk Quant
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
    
    function renderRiskQuant(memo, container) {
         // Simple PD logic from Left Side
         const debt = memo.historical_financials?.[0]?.total_liabilities || 0;
         const equity = memo.historical_financials?.[0]?.total_equity || 0;
         const assets = debt + equity;
         const leverage = assets > 0 ? debt / assets : 0;
         const pd = (leverage * 0.05).toFixed(4);
         
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

    // --- Evidence Viewer (Merged) ---
    window.viewEvidence = (docId, chunkId) => {
        const viewer = elements.evidenceViewer;
        viewer.innerHTML = ''; // Clear

        // 1. Find Data
        let doc, chunk;
        if(currentMemoData && currentMemoData.documents) {
             doc = currentMemoData.documents.find(d => d.doc_id === docId);
             chunk = doc?.chunks.find(c => c.chunk_id === chunkId);
        }

        // 2. Create Page Container
        const page = document.createElement('div');
        page.style.width = '100%';
        page.style.height = '1000px';
        page.style.background = '#fff';
        page.style.position = 'relative';
        page.style.padding = '40px';
        page.style.boxSizing = 'border-box';
        
        // 3. Mock Text Background
        page.innerHTML = `<div style="color: #ddd; font-size: 6px; overflow:hidden; height:100%; word-break: break-all;">
            ${"CONTENT ".repeat(5000)}
        </div>`;

        // 4. BBox Highlight
        if (chunk && chunk.bbox) {
            const [x0, y0, x1, y1] = chunk.bbox;
            const bbox = document.createElement('div');
            bbox.className = 'bbox-highlight'; // Uses CSS from previous step
            bbox.style.left = `${x0}px`;
            bbox.style.top = `${y0}px`;
            bbox.style.width = `${x1 - x0}px`;
            bbox.style.height = `${y1 - y0}px`;
            
            const label = document.createElement('div');
            label.className = 'bbox-label';
            label.innerText = chunk.type.toUpperCase();
            bbox.appendChild(label);
            
            // Overlay actual text for readability
            const textOverlay = document.createElement('div');
            textOverlay.style.background = "rgba(255,255,255,0.9)";
            textOverlay.style.color = "black";
            textOverlay.style.fontSize = "12px";
            textOverlay.style.height = "100%";
            textOverlay.style.overflow = "hidden";
            textOverlay.innerText = chunk.content;
            bbox.appendChild(textOverlay);

            page.appendChild(bbox);
            
            // Scroll to it
            setTimeout(() => bbox.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
        } else {
            // Fallback for missing bbox
            page.innerHTML += `<div style="position:absolute; top:100px; left:50px; background:yellow; padding:10px; color:black;">Evidence Loaded: ${docId}</div>`;
        }

        viewer.appendChild(page);
        log("Frontend", `User viewed evidence ${docId}`, "audit");
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