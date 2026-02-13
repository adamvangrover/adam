document.addEventListener('DOMContentLoaded', async () => {
    // --- State ---
    let libraryData = [];
    let currentReport = null;
    let auditInterval = null;
    let isEditing = false;

    // --- Elements ---
    const els = {
        libraryList: document.getElementById('library-list'),
        memoContainer: document.getElementById('memo-container'),
        financialsTable: document.getElementById('financials-table'),
        dcfContainer: document.getElementById('dcf-container'),
        pdfViewer: document.getElementById('pdf-viewer'),
        mockPdfPage: document.getElementById('mock-pdf-page'),
        highlightBox: document.getElementById('highlight-box'),
        docTitle: document.getElementById('doc-title'),
        auditTableBody: document.getElementById('audit-table-body'),
        tabs: {
            memo: document.getElementById('tab-memo'),
            annexA: document.getElementById('tab-annex-a'),
            annexB: document.getElementById('tab-annex-b')
        },
        btns: {
            memo: document.getElementById('btn-tab-memo'),
            annexA: document.getElementById('btn-tab-annex-a'),
            annexB: document.getElementById('btn-tab-annex-b')
        }
    };

    // --- Init ---
    try {
        // Ensure UniversalLoader is available (injected via HTML)
        if (typeof UniversalLoader === 'undefined') {
            console.error("UniversalLoader not found. Check script imports.");
        }
        await loadLibrary();
        startAuditStream();
        setupGlobalControls();
    } catch (e) {
        console.error("Initialization failed:", e);
    }

    // --- Global Controls (Upload/Download/Edit) ---
    function setupGlobalControls() {
        const header = document.querySelector('header .flex.items-center.gap-6');
        if (!header) return;

        // Edit Toggle
        const editBtn = document.createElement('button');
        editBtn.id = 'global-edit-btn';
        editBtn.innerHTML = `<i class="fas fa-edit mr-1"></i> Edit Mode`;
        editBtn.className = "text-xs font-mono text-slate-400 hover:text-white transition cursor-pointer border border-slate-700 rounded px-2 py-1";
        editBtn.onclick = toggleEditMode;
        header.prepend(editBtn);

        // Upload Button
        const uploadBtn = document.createElement('label');
        uploadBtn.innerHTML = `<i class="fas fa-upload mr-1"></i> Load JSON`;
        uploadBtn.className = "text-xs font-mono text-slate-400 hover:text-white transition cursor-pointer border border-slate-700 rounded px-2 py-1 ml-2";
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = '.json';
        fileInput.style.display = 'none';
        fileInput.onchange = handleFileUpload;
        uploadBtn.appendChild(fileInput);
        header.prepend(uploadBtn);

        // Download Button
        const downloadBtn = document.createElement('button');
        downloadBtn.innerHTML = `<i class="fas fa-download mr-1"></i> Save JSON`;
        downloadBtn.className = "text-xs font-mono text-slate-400 hover:text-white transition cursor-pointer border border-slate-700 rounded px-2 py-1 ml-2";
        downloadBtn.onclick = downloadReport;
        header.prepend(downloadBtn);
    }

    // --- Upload/Download Logic ---
    function handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const json = JSON.parse(e.target.result);
                // Validate structure loosely
                if (json.borrower_name || json.id) {
                    currentReport = json;
                    // Add to library if not exists
                    if (!libraryData.find(i => i.id === json.id)) {
                        libraryData.push(json);
                        renderLibraryList();
                    }
                    loadReport(json.id); // Reload UI
                    logAudit("System", "Import", `Loaded ${json.borrower_name} from file.`);
                } else {
                    alert("Invalid JSON format. Expected Credit Memo structure.");
                }
            } catch (err) {
                console.error(err);
                alert("Error parsing JSON file.");
            }
        };
        reader.readAsText(file);
    }

    function downloadReport() {
        if (!currentReport) {
            alert("No report loaded to download.");
            return;
        }
        // Update currentReport with any edits from DOM
        updateReportFromDOM();

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentReport, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `${currentReport.ticker}_credit_memo.json`);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
        logAudit("System", "Export", `Saved ${currentReport.ticker} locally.`);
    }

    function toggleEditMode() {
        isEditing = !isEditing;
        const btn = document.getElementById('global-edit-btn');
        
        if (isEditing) {
            btn.classList.add('bg-blue-600', 'text-white', 'border-blue-500');
            btn.classList.remove('text-slate-400');
            btn.innerHTML = `<i class="fas fa-check mr-1"></i> Done Editing`;
            document.body.classList.add('editing-active');
            
            // Make elements editable
            document.querySelectorAll('[data-editable]').forEach(el => {
                el.contentEditable = "true";
                el.classList.add('outline-dashed', 'outline-blue-500', 'outline-1', 'p-1');
            });
        } else {
            // Save Changes
            updateReportFromDOM();
            
            btn.classList.remove('bg-blue-600', 'text-white', 'border-blue-500');
            btn.classList.add('text-slate-400');
            btn.innerHTML = `<i class="fas fa-edit mr-1"></i> Edit Mode`;
            document.body.classList.remove('editing-active');

            // Disable editing
            document.querySelectorAll('[data-editable]').forEach(el => {
                el.contentEditable = "false";
                el.classList.remove('outline-dashed', 'outline-blue-500', 'outline-1', 'p-1');
            });
            logAudit("User", "Edit", "Manual overrides applied.");
        }
    }

    function updateReportFromDOM() {
        if (!currentReport) return;

        // Example: Update Summary
        const summaryEl = document.getElementById('memo-summary');
        if (summaryEl) currentReport.summary = summaryEl.innerText;

        // Update Narratives
        document.querySelectorAll('[data-chunk-id]').forEach(el => {
            const chunkId = el.getAttribute('data-chunk-id');
            const doc = currentReport.documents[0];
            const chunk = doc.chunks.find(c => c.chunk_id === chunkId);
            if (chunk) chunk.content = el.innerText;
        });
    }


    // --- Library Logic ---
    async function loadLibrary() {
        try {
            // Use UniversalLoader
            libraryData = await window.UniversalLoader.loadCreditLibrary();
            renderLibraryList();
        } catch (e) {
            console.error(e);
            els.libraryList.innerHTML = `<div class="p-4 text-red-500 text-xs">Error loading library.</div>`;
        }
    }

    function renderLibraryList() {
        els.libraryList.innerHTML = '';
        libraryData.forEach(item => {
            const div = document.createElement('div');
            div.className = "p-4 border-b border-slate-800 hover:bg-[#1e293b] cursor-pointer transition group";
            div.onclick = () => loadReport(item.id);

            const riskColor = item.risk_score < 60 ? 'text-red-400' : (item.risk_score < 80 ? 'text-yellow-400' : 'text-emerald-400');

            div.innerHTML = `
                <div class="flex justify-between items-center mb-1">
                    <span class="font-bold text-slate-200 text-sm group-hover:text-blue-400 transition">${item.borrower_name}</span>
                    <span class="text-xs font-mono ${riskColor}">${item.risk_score}/100</span>
                </div>
                <div class="flex justify-between items-center text-[10px] text-slate-500 font-mono">
                    <span>${item.ticker}</span>
                    <span>${new Date(item.report_date).toLocaleDateString()}</span>
                </div>
            `;
            els.libraryList.appendChild(div);
        });
    }

    async function loadReport(id) {
        currentReport = libraryData.find(i => i.id === id);
        if (!currentReport) return;

        // Reset UI
        switchTab('memo');
        logAudit("Orchestrator", "Loading", `Fetching artifacts for ${currentReport.ticker}...`);

        renderMemo();
        renderFinancials();
        renderDCF();

        els.mockPdfPage.classList.add('hidden');
        els.docTitle.textContent = "Waiting for selection...";

        logAudit("Orchestrator", "Complete", `Report loaded successfully.`);
    }

    // --- Rendering Logic ---

    function renderMemo() {
        const { borrower_name, summary, documents } = currentReport;
        const mainDoc = documents[0];

        let html = `
            <h1 class="text-3xl font-bold text-slate-900 mb-2" data-editable>${borrower_name}</h1>
            <div class="text-xs font-mono text-slate-500 mb-8 border-b border-slate-200 pb-4">
                TICKER: <span class="text-blue-600 font-bold">${currentReport.ticker}</span> |
                SECTOR: ${currentReport.sector} |
                RATING: <span class="bg-blue-100 text-blue-800 px-1 rounded">${currentReport.borrower_details.rating}</span>
            </div>

            <h2 class="text-lg font-bold text-slate-800 mb-4 uppercase tracking-widest border-b-2 border-blue-500 inline-block">Executive Summary</h2>
            <p id="memo-summary" class="text-sm leading-relaxed text-slate-700 mb-8 text-justify" data-editable>
                ${summary}
            </p>
        `;

        // Narrative Sections
        const narratives = mainDoc.chunks.filter(c => c.type === 'narrative');
        if (narratives.length > 0) {
            html += `<h2 class="text-lg font-bold text-slate-800 mb-4 uppercase tracking-widest border-b-2 border-blue-500 inline-block">Key Credit Drivers</h2>`;
            narratives.forEach(chunk => {
                html += `
                    <div class="mb-4 text-sm text-slate-700 leading-relaxed pl-4 border-l-2 border-slate-200 hover:border-blue-400 transition">
                        <p class="mb-1" data-editable data-chunk-id="${chunk.chunk_id}">${chunk.content}</p>
                        <button onclick="viewEvidence('${mainDoc.doc_id}', '${chunk.chunk_id}')" class="text-[10px] bg-slate-100 hover:bg-blue-100 text-slate-500 hover:text-blue-600 px-2 py-0.5 rounded transition cursor-pointer font-mono">
                            <i class="fas fa-search mr-1"></i>Ref: ${chunk.chunk_id}
                        </button>
                    </div>
                `;
            });
        }

        // Risk Factors
        const risks = mainDoc.chunks.filter(c => c.type === 'risk_factor');
        if (risks.length > 0) {
            html += `<h2 class="text-lg font-bold text-slate-800 mb-4 mt-8 uppercase tracking-widest border-b-2 border-red-500 inline-block">Risk Factors</h2>`;
            html += `<ul class="space-y-2">`;
            risks.forEach(chunk => {
                html += `
                    <li class="flex items-start gap-2 text-sm text-slate-700">
                        <i class="fas fa-exclamation-triangle text-red-500 mt-1 text-xs"></i>
                        <span data-editable data-chunk-id="${chunk.chunk_id}">${chunk.content}</span>
                        <button onclick="viewEvidence('${mainDoc.doc_id}', '${chunk.chunk_id}')" class="text-[10px] text-slate-400 hover:text-blue-600 ml-auto cursor-pointer">
                            [View Source]
                        </button>
                    </li>
                `;
            });
            html += `</ul>`;
        }

        els.memoContainer.innerHTML = html;
        if(isEditing) toggleEditMode(); // Re-apply edit state if needed
    }

    function renderFinancials() {
        const tableChunk = currentReport.documents[0].chunks.find(c => c.type === 'financial_table');
        if (!tableChunk || !tableChunk.content_json) {
            els.financialsTable.innerHTML = '<tr><td class="p-4 text-slate-500">No structured financial data available.</td></tr>';
            return;
        }

        const data = tableChunk.content_json;
        const years = ['2023A', '2024A', '2025E'];
        
        let headerHtml = `<thead class="bg-slate-700/50 text-slate-400 text-[10px] uppercase"><tr><th class="p-3">Metric ($Bn)</th>`;
        years.forEach(y => headerHtml += `<th class="p-3 text-right">${y}</th>`);
        headerHtml += `</tr></thead>`;

        let bodyHtml = `<tbody class="divide-y divide-slate-700/30 text-slate-300">`;

        Object.entries(data).forEach(([key, val]) => {
            const label = key.replace(/_/g, ' ').toUpperCase();
            bodyHtml += `
                <tr class="hover:bg-slate-700/20 transition">
                    <td class="p-3 font-bold text-blue-400">${label}</td>
                    <td class="p-3 text-right text-slate-500" data-editable>${(val * 0.9).toFixed(1)}</td>
                    <td class="p-3 text-right text-slate-500" data-editable>${(val * 0.95).toFixed(1)}</td>
                    <td class="p-3 text-right text-white font-mono" data-editable>${val.toFixed(1)}</td>
                </tr>
            `;
        });
        bodyHtml += `</tbody>`;

        els.financialsTable.innerHTML = headerHtml + bodyHtml;
    }

    function renderDCF() {
        const tableChunk = currentReport.documents[0].chunks.find(c => c.type === 'financial_table');
        const ebitda = tableChunk?.content_json?.ebitda || 50.0;
        const wacc = 0.085;
        const growth = 0.025;

        els.dcfContainer.innerHTML = `
            <div class="grid grid-cols-2 gap-8 mb-8">
                <div class="bg-slate-800/50 p-6 rounded border border-slate-700">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-4">Assumptions</div>

                    <div class="mb-6">
                        <div class="flex justify-between text-xs text-slate-300 mb-2">
                            <span>WACC</span>
                            <span id="val-wacc" class="font-mono text-blue-400">8.5%</span>
                        </div>
                        <input type="range" min="50" max="150" value="85" class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="updateDCF(this.value, 'wacc')">
                    </div>

                    <div>
                         <div class="flex justify-between text-xs text-slate-300 mb-2">
                            <span>Terminal Growth</span>
                            <span id="val-growth" class="font-mono text-emerald-400">2.5%</span>
                        </div>
                        <input type="range" min="0" max="50" value="25" class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer" oninput="updateDCF(this.value, 'growth')">
                    </div>
                </div>

                <div class="flex flex-col justify-center space-y-4">
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700 flex justify-between items-center">
                        <span class="text-xs text-slate-500">Implied Enterprise Value</span>
                        <span id="res-ev" class="text-xl font-bold text-white font-mono">$${(ebitda * (1+growth) / (wacc - growth)).toFixed(1)}B</span>
                     </div>
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700 flex justify-between items-center">
                        <span class="text-xs text-slate-500">Implied Share Price</span>
                        <span id="res-share" class="text-xl font-bold text-emerald-400 font-mono">$145.20</span>
                     </div>
                </div>
            </div>

            <div class="bg-slate-900/50 p-4 rounded border border-slate-800 text-[10px] font-mono text-slate-500">
                MODEL: GORDON GROWTH (SIMPLIFIED) | BASE EBITDA: $${ebitda}B
            </div>
        `;

        window.updateDCF = (val, type) => {
            const currentWacc = type === 'wacc' ? val / 1000 : parseFloat(document.getElementById('val-wacc').innerText) / 100;
            const currentGrowth = type === 'growth' ? val / 1000 : parseFloat(document.getElementById('val-growth').innerText) / 100;

            if (type === 'wacc') document.getElementById('val-wacc').innerText = (val/10).toFixed(1) + '%';
            if (type === 'growth') document.getElementById('val-growth').innerText = (val/10).toFixed(1) + '%';

            const denom = Math.max(0.001, currentWacc - currentGrowth);
            const ev = ebitda * (1 + currentGrowth) / denom;
            
            document.getElementById('res-ev').innerText = '$' + ev.toFixed(1) + 'B';
            document.getElementById('res-share').innerText = '$' + (ev * 0.35).toFixed(2);
        };
    }

    // --- Evidence Viewer ---
    window.viewEvidence = (docId, chunkId) => {
        const doc = currentReport.documents.find(d => d.doc_id === docId);
        const chunk = doc?.chunks.find(c => c.chunk_id === chunkId);
        
        if (!doc || !chunk) return;

        els.docTitle.textContent = `${docId} | Page ${chunk.page}`;
        els.mockPdfPage.classList.remove('hidden');

        const [x0, y0, x1, y1] = chunk.bbox || [10, 10, 100, 100];
        const scaleX = 100 / 600;
        const scaleY = 100 / 800;

        els.highlightBox.style.left = (x0 * scaleX) + '%';
        els.highlightBox.style.top = (y0 * scaleY) + '%';
        els.highlightBox.style.width = ((x1 - x0) * scaleX) + '%';
        els.highlightBox.style.height = ((y1 - y0) * scaleY) + '%';

        logAudit("User", "Interaction", `Verified evidence ${chunkId}`);
    };

    // --- Audit Log ---
    function startAuditStream() {
        const actions = ["Indexing", "Vector Search", "Compliance Check", "Model Inference", "Cache Update"];
        auditInterval = setInterval(() => {
            if (Math.random() > 0.7) {
                const action = actions[Math.floor(Math.random() * actions.length)];
                logAudit("System", action, "OK");
            }
        }, 3000);
    }

    function logAudit(actor, action, status) {
        const row = document.createElement('tr');
        row.className = "hover:bg-slate-800/30 transition animate-pulse-once";
        row.innerHTML = `
            <td class="p-3 border-b border-slate-800/50 text-slate-500 font-mono">${new Date().toLocaleTimeString()}</td>
            <td class="p-3 border-b border-slate-800/50 text-blue-400 font-bold">${actor}: ${action}</td>
            <td class="p-3 border-b border-slate-800/50 text-emerald-500">${status}</td>
        `;
        els.auditTableBody.prepend(row);
        
        if (els.auditTableBody.children.length > 20) {
            els.auditTableBody.lastElementChild.remove();
        }
    }

    // --- Tab Logic ---
    window.switchTab = (tabName) => {
        Object.values(els.tabs).forEach(el => el.classList.add('hidden'));
        Object.values(els.btns).forEach(btn => {
            btn.classList.remove('text-blue-400', 'border-blue-500');
            btn.classList.add('text-slate-400', 'border-transparent');
        });

        let targetTab, targetBtn;
        if (tabName === 'memo') { targetTab = els.tabs.memo; targetBtn = els.btns.memo; }
        if (tabName === 'annex-a') { targetTab = els.tabs.annexA; targetBtn = els.btns.annexA; }
        if (tabName === 'annex-b') { targetTab = els.tabs.annexB; targetBtn = els.btns.annexB; }

        if (targetTab && targetBtn) {
            targetTab.classList.remove('hidden');
            targetBtn.classList.remove('text-slate-400', 'border-transparent');
            targetBtn.classList.add('text-blue-400', 'border-blue-500');
        }
    };
});
