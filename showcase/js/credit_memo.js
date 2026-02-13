document.addEventListener('DOMContentLoaded', async () => {
    // --- State Management ---
    let mockData = {};
    let libraryIndex = [];
    let currentMemo = null;

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
        auditLogContent: document.getElementById('audit-log-content'), // mapped to tbody in HTML
        borrowerSelect: document.getElementById('borrower-select'),
        libraryList: document.getElementById('library-list') // Assuming sidebar has a list container
    };

    // --- Initialization ---
    try {
        await Promise.all([loadMockData(), loadLibraryIndex()]);
        renderLibraryList();
        logAudit("System", "Ready", "Credit Memo Orchestrator initialized.");
    } catch (e) {
        console.error("Initialization failed:", e);
    }

    // --- Event Listeners ---
    if(elements.generateBtn) elements.generateBtn.addEventListener('click', startGeneration);
    if(elements.closeEvidenceBtn) elements.closeEvidenceBtn.addEventListener('click', hideEvidence);
    
    // Tab Switching (Global Delegate)
    window.switchTab = (tabName) => {
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.tab-btn').forEach(el => {
            el.style.borderBottom = '2px solid transparent';
            el.style.color = 'var(--text-secondary)';
        });
        
        const target = document.getElementById(`tab-${tabName}`);
        if(target) target.style.display = 'block';
        
        const btn = document.getElementById(`btn-tab-${tabName}`);
        if(btn) {
            btn.style.borderBottom = '2px solid var(--accent-color)';
            btn.style.color = 'var(--accent-color)';
        }
    };

    // --- Data Loading ---
    async function loadMockData() {
        // Fallback or local load
        try {
            const res = await fetch('data/credit_mock_data.json');
            mockData = await res.json();
        } catch(e) { console.warn("Mock data fetch failed, using internal fallback"); }
    }

    async function loadLibraryIndex() {
        try {
            const res = await fetch('data/credit_memo_library.json');
            libraryIndex = await res.json();
        } catch(e) { console.warn("Library index fetch failed"); }
    }

    // --- Core Logic: Simulation (Left Side) ---
    function startGeneration() {
        const borrower = elements.borrowerSelect ? elements.borrowerSelect.value : "Apple Inc.";
        
        // UI Reset
        elements.memoContent.innerHTML = ''; 
        elements.progressContainer.style.display = 'block';
        hideEvidence();

        // Simulate Enterprise Agent Workflow
        simulateAgent("Archivist", `Retrieving ${borrower} EDGAR filings...`, 0, 800)
            .then(() => simulateAgent("Quant", "Normalizing EBITDA & spreading comps...", 33, 1200))
            .then(() => simulateAgent("Risk Officer", " analyzing covenant compliance...", 66, 1000))
            .then(() => simulateAgent("Writer", "Synthesizing executive summary...", 90, 1500))
            .then(() => {
                elements.progressContainer.style.display = 'none';
                
                // Check if we have pre-generated JSON for this, else use mock data fallback
                const libraryEntry = libraryIndex.find(i => i.borrower_name === borrower || i.id === borrower);
                if (libraryEntry) {
                    loadCreditMemo(libraryEntry.file);
                } else if (mockData[borrower]) {
                    renderMemoFromMock(mockData[borrower]);
                } else {
                    // Fallback to first mock entry if specific selection missing
                    renderMemoFromMock(Object.values(mockData)[0]);
                }
                
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

    // --- Core Logic: Rendering (Right Side Logic + Cyber Styling) ---
    
    // 1. Sidebar Library
    function renderLibraryList() {
        if(!elements.libraryList) return;
        elements.libraryList.innerHTML = '';
        
        libraryIndex.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.style.padding = '10px';
            itemDiv.style.borderBottom = '1px solid var(--glass-border)';
            itemDiv.style.cursor = 'pointer';
            itemDiv.style.opacity = '0.8';
            
            const scoreColor = item.risk_score < 60 ? '#ff4444' : (item.risk_score < 80 ? '#ffbb33' : '#00C851');
            
            itemDiv.innerHTML = `
                <div style="display:flex; justify-content:space-between; color: white; font-weight:bold;">
                    <span>${item.borrower_name}</span>
                    <span style="color:${scoreColor}">${item.risk_score}</span>
                </div>
                <div style="font-size:0.8em; color: var(--text-secondary); margin-top:4px;">
                    ${new Date(item.report_date).toLocaleDateString()}
                </div>
            `;
            
            itemDiv.onclick = () => {
                startGeneration(); // Re-run simulation for effect, or direct load: loadCreditMemo(item.file);
            };
            
            elements.libraryList.appendChild(itemDiv);
        });
    }

    async function loadCreditMemo(filename) {
        try {
            const path = filename.includes('/') ? filename : `data/${filename}`;
            const res = await fetch(path);
            const memo = await res.json();
            currentMemo = memo;
            renderFullMemoUI(memo);
        } catch(e) {
            console.error("Could not load memo file:", e);
        }
    }

    // 2. Main Memo Render
    function renderFullMemoUI(memo) {
        // Tab Navigation HTML
        const tabsHtml = `
            <div style="display: flex; gap: 20px; border-bottom: 1px solid var(--glass-border); margin-bottom: 20px;">
                <div id="btn-tab-memo" class="tab-btn" onclick="switchTab('memo')" style="padding: 10px; cursor: pointer; color: var(--accent-color); border-bottom: 2px solid var(--accent-color);">Memo</div>
                <div id="btn-tab-financials" class="tab-btn" onclick="switchTab('financials')" style="padding: 10px; cursor: pointer; color: var(--text-secondary);">Financials</div>
                <div id="btn-tab-dcf" class="tab-btn" onclick="switchTab('dcf')" style="padding: 10px; cursor: pointer; color: var(--text-secondary);">Valuation (DCF)</div>
            </div>
        `;

        // Content Containers
        const contentHtml = `
            <div id="tab-memo" class="tab-content" style="display:block;">${generateMemoHtml(memo)}</div>
            <div id="tab-financials" class="tab-content" style="display:none;">${generateFinancialsHtml(memo)}</div>
            <div id="tab-dcf" class="tab-content" style="display:none;">${generateDcfHtml(memo)}</div>
        `;

        elements.memoContent.innerHTML = tabsHtml + contentHtml;
        elements.memoContent.style.display = 'block';
    }

    // 3. HTML Generators (Adapting Right Side Logic to Left Side CSS)
    function generateMemoHtml(memo) {
        // Fallback for simple mock data structure vs complex library structure
        const name = memo.borrower_name || memo.borrower_details?.name;
        const rating = memo.rating || memo.borrower_details?.rating || "N/A";
        const sector = memo.sector || memo.borrower_details?.sector || "General";
        const sections = memo.sections || []; 

        let html = `<h1 style="margin:0;">${name}</h1>`;
        html += `<div style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 20px;">
                    Rating: <span style="color: var(--accent-color)">${rating}</span> | Sector: ${sector}
                 </div>`;

        // Generate Sections
        if (sections.length > 0) {
            sections.forEach(sec => {
                // Citation logic
                let content = sec.content.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
                     return `<span class="citation-tag" onclick="viewEvidence('${docId}')"><i class="fas fa-search"></i> ${docId}</span>`;
                });
                
                html += `<h2 style="color: var(--accent-color); border-bottom: 1px solid var(--glass-border); padding-bottom: 5px; margin-top: 25px;">${sec.title}</h2>`;
                html += `<p style="line-height: 1.6; color: #d0d0d0;">${content}</p>`;
            });
        } else if (memo.documents) {
            // Fallback for mock data structure
            const doc = memo.documents[0];
            doc.chunks.forEach(chunk => {
                if(chunk.type === 'narrative') {
                     html += `<p>${chunk.content} <span class="citation-tag" onclick="renderEvidence('${memo.borrower_details.name}', '${doc.doc_id}', '${chunk.chunk_id}')">[${doc.doc_id}]</span></p>`;
                }
            });
        }
        return html;
    }

    function generateFinancialsHtml(memo) {
        if (!memo.historical_financials) return '<p>No structured financial data.</p>';
        
        let html = `<h3 style="color:white;">Historical Performance</h3>`;
        html += `<table style="width:100%; border-collapse: collapse; margin-top:15px; font-family: monospace;">`;
        
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
                html += `<td style="text-align:right; padding:10px;">${val}</td>`;
            });
            html += `</tr>`;
        });
        html += `</tbody></table>`;
        return html;
    }

    function generateDcfHtml(memo) {
        if (!memo.dcf_analysis) return '<p>No valuation data.</p>';
        const dcf = memo.dcf_analysis;
        
        return `
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px;">
                    <div style="font-size: 0.8em; color: var(--text-secondary);">Enterprise Value</div>
                    <div style="font-size: 1.5em; color: var(--accent-color); font-weight: bold;">${formatCurrency(dcf.enterprise_value)}</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 4px;">
                    <div style="font-size: 0.8em; color: var(--text-secondary);">Implied Share Price</div>
                    <div style="font-size: 1.5em; color: #00C851; font-weight: bold;">$${dcf.share_price.toFixed(2)}</div>
                </div>
            </div>
            <div style="font-family: monospace; font-size: 0.9em; color: var(--text-secondary);">
                WACC: ${(dcf.wacc * 100).toFixed(1)}% | Terminal Growth: ${(dcf.growth_rate * 100).toFixed(1)}%
            </div>
        `;
    }

    // --- Evidence Viewer Logic (Merged) ---

    // 1. Specific Coordinate Renderer (Left Side Style)
    window.renderEvidence = (borrowerName, docId, chunkId) => {
        const data = mockData[borrowerName] || Object.values(mockData)[0]; // Fallback
        const doc = data?.documents?.find(d => d.doc_id === docId);
        const chunk = doc?.chunks?.find(c => c.chunk_id === chunkId);
        
        if (doc && chunk) {
            setupPdfViewer(docId, chunk.page);
            // Draw BBox
            const [x0, y0, x1, y1] = chunk.bbox;
            const highlight = document.createElement('div');
            highlight.className = 'bbox-highlight';
            // Scale coords if necessary, assuming 1000px height base
            highlight.style.left = `${x0}px`;
            highlight.style.top = `${y0}px`;
            highlight.style.width = `${x1 - x0}px`;
            highlight.style.height = `${y1 - y0}px`;
            
            const label = document.createElement('div');
            label.innerText = chunk.type.toUpperCase();
            label.style.background = 'var(--accent-color)';
            label.style.color = 'black';
            label.style.fontSize = '10px';
            label.style.position = 'absolute';
            label.style.top = '-15px';
            
            highlight.appendChild(label);
            document.getElementById('pdf-page-container').appendChild(highlight);
            highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
            logAudit("Frontend", "Evidence", `Displayed precise artifact ${chunkId}`);
        }
    };

    // 2. Generic Viewer (Right Side Style)
    window.viewEvidence = (docId) => {
        setupPdfViewer(docId, 1);
        
        // Simulated highlight for generic docs
        const highlight = document.createElement('div');
        highlight.className = 'bbox-highlight';
        highlight.style.left = '10%';
        highlight.style.top = '20%';
        highlight.style.width = '80%';
        highlight.style.height = '100px';
        document.getElementById('pdf-page-container').appendChild(highlight);
        logAudit("Frontend", "Evidence", `Opened document ${docId}`);
    };

    function setupPdfViewer(docId, pageNum) {
        elements.evidencePanel.classList.add('active');
        elements.evidencePanel.style.width = '50%';
        
        // Clear previous
        elements.pdfViewer.innerHTML = '';
        
        const pageDiv = document.createElement('div');
        pageDiv.id = 'pdf-page-container';
        pageDiv.className = 'pdf-page'; // Uses CSS from previous step
        pageDiv.style.background = '#fff'; 
        pageDiv.style.color = '#000';
        pageDiv.style.position = 'relative';
        pageDiv.style.minHeight = '1000px';
        pageDiv.style.padding = '40px';
        
        pageDiv.innerHTML = `
            <div style="border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between;">
                <span style="font-weight: bold;">${docId}</span>
                <span>Page ${pageNum}</span>
            </div>
            <div style="font-family: serif; color: #444; line-height: 1.8;">
                <p><strong>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</strong></p>
                <p>Form 10-K</p>
                <br>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
                <br>
                <p>[Simulated Document Content for Contextual Verification]</p>
            </div>
        `;
        
        elements.pdfViewer.appendChild(pageDiv);
    }

    function hideEvidence() {
        elements.evidencePanel.classList.remove('active');
        elements.evidencePanel.style.width = '0';
    }

    // --- Utilities ---
    function logAudit(actor, action, details) {
        if (!elements.auditLogContent) return;
        
        const time = new Date().toLocaleTimeString();
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #333';
        tr.innerHTML = `
            <td style="padding: 5px; color: #666; font-size: 0.8em;">${time}</td>
            <td style="padding: 5px; color: var(--accent-color); font-weight: bold;">${actor}</td>
            <td style="padding: 5px; color: #ccc;">${action}</td>
            <td style="padding: 5px; color: #888;">${details}</td>
        `;
        elements.auditLogContent.insertBefore(tr, elements.auditLogContent.firstChild);
    }
    
    function formatCurrency(val) {
        if (val === undefined || val === null) return 'N/A';
        if (Math.abs(val) >= 1000) return '$' + (val / 1000).toFixed(1) + 'B';
        return '$' + val.toFixed(1) + 'M';
    }
    
    // Internal Helper for Mock Data rendering fallback
    function renderMemoFromMock(data) {
        // Quick adapter to map mock data structure to full UI
        const adapter = {
            borrower_name: data.borrower_details.name,
            borrower_details: data.borrower_details,
            documents: data.documents,
            sections: [
                { title: "Executive Summary", content: data.documents[0].chunks.find(c=>c.type==='narrative')?.content || "Analysis pending." },
                { title: "Risk Factors", content: "Risk factors analyzed from 10-K filing." }
            ]
        };
        currentMemo = adapter;
        renderFullMemoUI(adapter);
    }
});