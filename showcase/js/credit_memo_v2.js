document.addEventListener('DOMContentLoaded', async () => {
    // --- State Management ---
    let selectedBorrower = null;
    let currentMemoData = null;
    let isEditMode = false;
    let currentChart = null;

    const elements = {
        borrowerSelect: document.getElementById('borrower-select'),
        generateBtn: document.getElementById('generate-btn'),
        editBtn: document.getElementById('edit-btn'),
        exportBtn: document.getElementById('export-btn'),
        memoContainer: document.getElementById('memo-container'),
        memoPlaceholder: document.getElementById('memo-placeholder'),
        memoContent: document.getElementById('memo-content'),
        evidenceViewer: document.getElementById('pdf-mock-canvas'),
        terminal: document.getElementById('agent-terminal'),
        docIdLabel: document.getElementById('doc-id-label') || document.createElement('div'),
        modeIndicator: document.querySelector('.header-nav span span') 
    };

    // --- Initialization ---
    async function init() {
        log("System", "Initializing CreditOS v2.4 (Universal Loader)...", "info");

        if (window.UniversalLoader) {
            await window.UniversalLoader.init();
            
            const library = window.UniversalLoader.getLibrary();
            // Populate Dropdown
            library.forEach(item => {
                const opt = document.createElement('option');
                opt.value = item.id;
                opt.innerText = `${item.name} (${item.risk_score})`;
                opt.dataset.ticker = item.ticker;
                opt.dataset.name = item.name;
                elements.borrowerSelect.appendChild(opt);
            });

            log("Archivist", `Loaded ${library.length} entity profiles.`, "success");
            loadAuditLogs();
        } else {
            log("System", "UniversalLoader not found!", "error");
        }
    }

    // --- Event Listeners ---
    if(elements.generateBtn) elements.generateBtn.addEventListener('click', runAnalysis);
    if(elements.editBtn) {
        elements.editBtn.addEventListener('click', toggleEditMode);

        // Inject Import/Export controls
        const toolsGroup = elements.editBtn.parentElement;
        if (toolsGroup) {
             const btnContainer = document.createElement('div');
             btnContainer.style.display = 'flex';
             btnContainer.style.gap = '5px';
             btnContainer.style.marginTop = '10px';

             const dlBtn = document.createElement('button');
             dlBtn.className = 'secondary';
             dlBtn.innerHTML = '<i class="fas fa-download"></i> JSON';
             dlBtn.title = "Download State";
             dlBtn.style.flex = "1";
             dlBtn.onclick = downloadState;

             const ulBtn = document.createElement('button');
             ulBtn.className = 'secondary';
             ulBtn.innerHTML = '<i class="fas fa-upload"></i> LOAD';
             ulBtn.title = "Upload State";
             ulBtn.style.flex = "1";
             ulBtn.onclick = () => document.getElementById('json-upload').click();

             const fileInput = document.createElement('input');
             fileInput.type = 'file';
             fileInput.id = 'json-upload';
             fileInput.style.display = 'none';
             fileInput.accept = '.json';
             fileInput.onchange = uploadState;

             btnContainer.appendChild(dlBtn);
             btnContainer.appendChild(ulBtn);
             btnContainer.appendChild(fileInput);
             toolsGroup.appendChild(btnContainer);
        }
    }
    if(elements.exportBtn) elements.exportBtn.addEventListener('click', () => window.print());

    // --- Core Workflow ---
    async function runAnalysis() {
        const id = elements.borrowerSelect.value;
        if (!id) { alert("Please select a target entity."); return; }
        
        const option = elements.borrowerSelect.options[elements.borrowerSelect.selectedIndex];
        const name = option.dataset.name;

        // Reset UI
        elements.memoContent.innerHTML = '';
        clearTerminal();
        
        // 1. Retrieval Phase
        await simulateAgentTask("Archivist", [
            `Querying Vector DB for ${name}...`,
            `Found Artifacts - Indexing...`,
            `Extracting relevant chunks (MD&A, Risk Factors)...`
        ], 600);

        // 2. Load Data (Unified)
        try {
            currentMemoData = await window.UniversalLoader.loadEntity(id);
            selectedBorrower = currentMemoData;
        } catch(e) {
            log("System", "Error loading memo data", "error");
            return;
        }

        // 3. Quant Phase
        const assets = currentMemoData.financials?.history?.[0]?.total_assets || 'N/A';
        await simulateAgentTask("Quant", [
            `OCR-ing Financial Tables...`,
            `Mapping line items to FIBO ontology...`,
            `Validating Balance Sheet...`,
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
        if (elements.memoPlaceholder) elements.memoPlaceholder.style.display = 'none';

        const contentDiv = elements.memoContent;
        contentDiv.innerHTML = '';
        contentDiv.style.display = 'block'; // Ensure visible
        contentDiv.className = "memo-paper";

        // Header
        const riskColor = memo.risk_score < 60 ? 'red' : (memo.risk_score < 80 ? 'orange' : 'green');
        const header = document.createElement('div');
        header.innerHTML = `
            <h1 class="editable-content">${memo.name}</h1>
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 30px; font-family: var(--font-mono); font-size: 0.8rem; color: #666;">
                <span>DATE: ${new Date(memo.report_date).toLocaleDateString()}</span>
                <span>RISK SCORE: <b style="color: ${riskColor}">${memo.risk_score}/100</b></span>
                <span>ID: ${memo.ticker}-${Math.floor(Math.random()*10000)}</span>
            </div>
        `;
        contentDiv.appendChild(header);

        // Financial Snapshot & Chart
        const history = memo.financials.history;
        const ratios = memo.financials.ratios;

        if (history && history.length > 0) {
            const latest = history[0]; // Assuming unified loader puts latest first, or check logic
            // Wait, normalizeSovereign logic just passes spread.history.
            // normalizeLibrary logic returns [{ period: "FY25", ... }]

            // Let's find latest period safely
            const snapshot = history.length > 0 ? history[history.length -1] : {}; // Usually arrays are sorted old to new?
            // Unified Loader logic:
            // Sovereign: spread.history usually sorted.
            // Library: single item array.

            const revenue = latest.revenue || latest.total_revenue || 0;
            const ebitda = latest.ebitda || 0;

            const finDiv = document.createElement('div');
            finDiv.style.marginBottom = '30px';
            finDiv.style.background = '#f9f9f9';
            finDiv.style.padding = '20px';
            finDiv.style.border = '1px solid #ddd';

            finDiv.innerHTML = `
                <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Financial Snapshot</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; font-family: var(--font-mono); font-size: 0.8rem; margin-bottom: 20px;">
                    <div><b>Leverage:</b> ${ratios["Leverage"] || ratios.leverage_ratio || "N/A"}</div>
                    <div><b>EBITDA:</b> ${formatCurrency(ebitda)}</div>
                    <div><b>Revenue:</b> ${formatCurrency(revenue)}</div>
                    <div><b>Rating:</b> ${memo.rating}</div>
                </div>
                <div style="height: 250px; width: 100%;">
                    <canvas id="finChart"></canvas>
                </div>
            `;
            contentDiv.appendChild(finDiv);
            
            setTimeout(() => {
                renderFinChart(history);
            }, 0);
        }

        // Sections
        if (memo.sections) {
            memo.sections.forEach((section, index) => {
                const sectionDiv = document.createElement('div');
                sectionDiv.dataset.sectionIndex = index;
                let contentHtml = section.content.replace(/\n/g, '<br>');
                // Citations
                contentHtml = contentHtml.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
                     return `<span class="citation-tag" onclick="viewEvidence('${docId}', 'chunk_001')">${docId}</span>`;
                });

                sectionDiv.innerHTML = `
                    <h2 class="editable-content">${section.title}</h2>
                    <div class="editable-content" style="text-align: justify;">${contentHtml}</div>
                `;
                contentDiv.appendChild(sectionDiv);
            });
        }

        // DCF
        if (memo.valuation && memo.valuation.dcf) {
            renderDCF(memo.valuation.dcf, contentDiv);
        }
        
        // Risk Quant
        renderRiskQuant(memo, contentDiv);

        if (isEditMode) applyEditMode();
    }

    function renderFinChart(data) {
        const ctx = document.getElementById('finChart');
        if (!ctx) return;
        
        if (currentChart) currentChart.destroy();

        // Ensure sorted chronologically
        const chartData = [...data].sort((a,b) => (a.fiscal_year || a.period) > (b.fiscal_year || b.period) ? 1 : -1);

        currentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.map(d => d.period || d.fiscal_year),
                datasets: [
                    {
                        label: 'Revenue',
                        data: chartData.map(d => d.revenue || d.total_revenue),
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
         // Simple PD logic from Unified Data
         const debt = memo.financials.history?.[0]?.total_debt || 0;
         const equity = memo.financials.history?.[0]?.total_equity || 0;
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
        viewer.innerHTML = '';

        // 1. Find Data
        let doc, chunk;
        if(currentMemoData && currentMemoData.documents) {
             doc = currentMemoData.documents.find(d => d.doc_id === docId) || currentMemoData.documents[0]; // fallback to first doc
             chunk = doc?.chunks?.find(c => c.chunk_id === chunkId);
        }

        // 2. Create Page Container
        const page = document.createElement('div');
        page.style.width = '100%';
        page.style.height = '1000px';
        page.style.background = '#fff';
        page.style.position = 'relative';
        page.style.padding = '40px';
        page.style.boxSizing = 'border-box';
        
        // 3. Mock Text
        page.innerHTML = `<div style="color: #ddd; font-size: 6px; overflow:hidden; height:100%; word-break: break-all;">
            ${"CONTENT ".repeat(5000)}
        </div>`;

        // 4. BBox Highlight
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
             page.innerHTML += `<div style="position:absolute; top:100px; left:50px; background:yellow; padding:10px; color:black;">Evidence Loaded: ${docId || "N/A"}</div>`;
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
            saveDOMToState();
            btn.classList.remove('editing-active');
            btn.innerHTML = 'EDIT MODE';
            document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "false");
        }
    }

    function applyEditMode() {
        document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "true");
    }

    function saveDOMToState() {
        if (!currentMemoData) return;

        // Title
        const titleEl = document.querySelector('#memo-content h1.editable-content');
        if (titleEl) currentMemoData.name = titleEl.innerText;

        // Sections
        const sectionDivs = document.querySelectorAll('#memo-content div[data-section-index]');
        sectionDivs.forEach(div => {
            const idx = div.dataset.sectionIndex;
            const contentEl = div.querySelector('div.editable-content');
            const titleEl = div.querySelector('h2.editable-content');

            if (currentMemoData.sections[idx]) {
                if (titleEl) currentMemoData.sections[idx].title = titleEl.innerText;
                if (contentEl) currentMemoData.sections[idx].content = contentEl.innerText;
            }
        });

        log("System", "Changes saved to local state.", "success");
    }

    function downloadState() {
        if (!currentMemoData) { alert("No data to export."); return; }
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentMemoData, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `credit_memo_${currentMemoData.ticker || "export"}.json`);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
        log("System", "State exported to JSON.", "success");
    }

    function uploadState(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const json = JSON.parse(e.target.result);
                const normalized = window.UniversalLoader.validateAndNormalize(json);
                if (normalized) {
                    currentMemoData = normalized;
                    selectedBorrower = normalized;
                    renderMemo(normalized);
                    log("System", `Loaded state from ${file.name}`, "success");
                } else {
                    alert("Invalid Credit Memo JSON.");
                }
            } catch (err) {
                console.error(err);
                alert("Error parsing JSON.");
            }
        };
        reader.readAsText(file);
    }
    
    window.recalculateDCF = function() {
        if (!currentMemoData || !currentMemoData.valuation.dcf) return;

        const wacc = parseFloat(document.getElementById('dcf-wacc').value) / 100;
        const growth = parseFloat(document.getElementById('dcf-growth').value) / 100;
        
        const dcf = currentMemoData.valuation.dcf;
        const originalWacc = dcf.wacc || 0.08;
        const originalGrowth = dcf.growth_rate || 0.02;
        const baseEV = dcf.enterprise_value;
        const basePrice = dcf.share_price;

        // Unified Sensitivity Logic
        const waccDelta = originalWacc - wacc; // Lower WACC = Higher Value
        const growthDelta = growth - originalGrowth; // Higher Growth = Higher Value

        // 1% WACC change ~ 15% value change (inverse)
        // 1% Growth change ~ 10% value change
        const multiplier = 1 + (waccDelta * 15) + (growthDelta * 10);

        const newEV = baseEV * multiplier;
        const newPrice = basePrice * multiplier;
        
        document.getElementById('dcf-share-price').textContent = `$${newPrice.toFixed(2)}`;
        document.getElementById('dcf-ev').textContent = formatCurrency(newEV);
    };
    
    function loadAuditLogs() {
        // Use Unified Schema logs if available, else fetch mock
        if (selectedBorrower && selectedBorrower.audit_log) {
            selectedBorrower.audit_log.forEach(l => log("System", `[AUDIT] ${l.action}`, "audit"));
        } else {
            fetch('data/credit_memo_audit_log.json')
            .then(res => res.json())
            .then(logs => {
                logs.slice(-5).forEach(l => log("System", `[AUDIT] ${l.action}`, "audit"));
            })
            .catch(e => console.log("No audit logs found"));
        }
    }

    // Start
    init();
});