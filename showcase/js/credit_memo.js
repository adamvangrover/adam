document.addEventListener('DOMContentLoaded', async () => {
    // --- State Management ---
    let currentMemo = null;

    const elements = {
        pdfViewer: document.getElementById('pdf-viewer'),
        mockPdfPage: document.getElementById('mock-pdf-page'),
        docTitle: document.getElementById('doc-title'),

        // Audit
        auditTableBody: document.getElementById('audit-table-body'),

        // Sidebar
        libraryList: document.getElementById('library-list'),

        // Tabs
        tabBtns: {
            memo: document.getElementById('btn-tab-memo'),
            annexA: document.getElementById('btn-tab-annex-a'),
            annexB: document.getElementById('btn-tab-annex-b')
        },
        tabContents: {
            memo: document.getElementById('tab-memo'),
            annexA: document.getElementById('tab-annex-a'),
            annexB: document.getElementById('tab-annex-b')
        },

        // Containers
        memoContainer: document.getElementById('memo-container'),
        financialsTable: document.getElementById('financials-table'),
        dcfContainer: document.getElementById('dcf-container')
    };

    // --- Initialization ---
    try {
        if (window.UniversalLoader) {
            await window.UniversalLoader.init();
            renderLibraryList();
            injectControls();
            logAudit("System", "Ready", "Credit Memo Orchestrator initialized.");
        } else {
            console.error("UniversalLoader not found!");
        }
    } catch (e) {
        console.error("Initialization failed:", e);
    }

    // --- Global Tab Switcher ---
    window.switchTab = (tabName) => {
        // Hide all contents
        Object.values(elements.tabContents).forEach(el => el.classList.add('hidden'));
        
        // Reset all buttons
        Object.values(elements.tabBtns).forEach(el => {
            el.classList.remove('text-blue-400', 'border-blue-500');
            el.classList.add('text-slate-400', 'border-transparent');
        });
        
        // Activate target
        const content = document.getElementById(`tab-${tabName}`);
        const btn = document.getElementById(`btn-tab-${tabName}`);

        if (content) content.classList.remove('hidden');
        if (btn) {
            btn.classList.remove('text-slate-400', 'border-transparent');
            btn.classList.add('text-blue-400', 'border-blue-500');
        }
    };

    // --- Core Logic ---

    function renderLibraryList() {
        if(!elements.libraryList) return;
        elements.libraryList.innerHTML = '';
        
        const library = window.UniversalLoader.getLibrary();

        library.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = "p-3 border-b border-slate-800 hover:bg-[#1e293b] cursor-pointer transition group";
            
            const scoreColor = item.risk_score > 80 ? 'text-emerald-400' : (item.risk_score > 60 ? 'text-yellow-400' : 'text-red-400');
            
            itemDiv.innerHTML = `
                <div class="flex justify-between items-center mb-1">
                    <span class="text-xs font-bold text-white group-hover:text-blue-400 transition">${item.name}</span>
                    <span class="text-[10px] font-mono ${scoreColor}">${item.risk_score}</span>
                </div>
                <div class="flex justify-between items-center text-[10px] text-slate-500">
                    <span>${item.ticker}</span>
                    <span>${item.sector}</span>
                </div>
            `;
            
            itemDiv.onclick = () => loadEntity(item.id);
            elements.libraryList.appendChild(itemDiv);
        });
    }

    async function loadEntity(id) {
        logAudit("Orchestrator", "Loading", `Fetching artifacts for ${id}...`);

        // Simulate loading state if desired
        elements.memoContainer.innerHTML = `<div class="flex flex-col items-center justify-center h-full text-blue-400 animate-pulse"><i class="fas fa-circle-notch fa-spin fa-3x mb-4"></i><p>Running Analysis...</p></div>`;

        try {
            const data = await window.UniversalLoader.loadEntity(id);
            currentMemo = data;

            // Artificial delay for effect
            await new Promise(r => setTimeout(r, 600));

            renderMemo(data);
            renderFinancials(data);
            renderValuation(data);
            renderAuditLog(data);

            logAudit("Orchestrator", "Complete", `Report generated for ${data.name}`);

        } catch (e) {
            console.error("Load failed", e);
            elements.memoContainer.innerHTML = `<p class="text-red-500">Error loading data.</p>`;
        }
    }

    // --- Rendering ---

    function renderMemo(data) {
        let html = `
            <div class="border-b-2 border-slate-900 pb-6 mb-8">
                <div class="flex justify-between items-end mb-2">
                    <h1 class="text-3xl font-bold text-slate-900 font-mono tracking-tighter editable-field">${data.name.toUpperCase()}</h1>
                    <span class="text-sm font-bold bg-slate-900 text-white px-3 py-1 rounded">${data.rating}</span>
                </div>
                <div class="flex gap-4 text-xs text-slate-500 font-mono uppercase tracking-widest">
                    <span>${data.ticker}</span>
                    <span>|</span>
                    <span>${data.sector}</span>
                    <span>|</span>
                    <span>${new Date(data.report_date).toLocaleDateString()}</span>
                </div>
            </div>
        `;

        data.sections.forEach((section, index) => {
            let content = section.content;

            // Replace [Ref] or similar if present in content, though Unified schema likely has clean text or simple refs.
            // Let's support a simple [DocID] pattern for highlighting
            // content = content.replace(/\[(.*?)\]/g, `<span class="text-blue-600 font-bold cursor-pointer hover:underline" onclick="viewEvidence('$1')">[$1]</span>`);

            html += `
                <div class="mb-8" data-section-index="${index}">
                    <h3 class="text-sm font-bold text-blue-900 uppercase tracking-widest border-b border-blue-100 pb-2 mb-4 editable-field">${section.title}</h3>
                    <div class="text-xs leading-relaxed text-slate-700 font-serif text-justify space-y-2 whitespace-pre-line editable-field">
                        ${content}
                    </div>
                </div>
            `;
        });

        elements.memoContainer.innerHTML = html;
    }

    function renderFinancials(data) {
        const table = elements.financialsTable;
        if (!table) return;
        table.innerHTML = '';

        const history = data.financials.history;
        if (!history || history.length === 0) {
            table.innerHTML = '<tr><td class="p-4 text-slate-500">No structured financial data available.</td></tr>';
            return;
        }

        // Headers
        const periods = history.map(h => h.period || h.fiscal_year);
        let thead = `<thead class="bg-[#0f172a] text-slate-400"><tr><th class="p-3">Metric</th>${periods.map(p => `<th class="p-3 text-right">${p}</th>`).join('')}<th class="p-3 text-right">Trend</th></tr></thead>`;
        
        // Rows
        const metrics = [
            { key: "revenue", label: "Total Revenue" },
            { key: "ebitda", label: "EBITDA" },
            { key: "net_income", label: "Net Income" },
            { key: "total_assets", label: "Total Assets" },
            { key: "total_debt", label: "Total Debt" }
        ];

        let tbody = `<tbody class="divide-y divide-slate-700/50 text-slate-300">`;
        metrics.forEach(m => {
            tbody += `<tr><td class="p-3 font-bold text-blue-400">${m.label}</td>`;
            let values = [];
            history.forEach(h => {
                let val = h[m.key];
                if (val === undefined) val = h[m.key.replace("total_", "")] || 0; // fallback
                values.push(val);
                tbody += `<td class="p-3 text-right font-mono">${formatCurrency(val)}</td>`;
            });

            // Simple Sparkline (HTML/CSS Bar)
            const max = Math.max(...values, 1);
            const trendHtml = `<div class="flex items-end justify-end gap-[1px] h-6 w-16 opacity-70">
                ${values.map(v => `<div class="w-2 bg-emerald-500" style="height: ${(v/max)*100}%"></div>`).join('')}
            </div>`;

            tbody += `<td class="p-3 text-right">${trendHtml}</td></tr>`;
        });
        tbody += `</tbody>`;

        table.innerHTML = thead + tbody;
    }

    function renderValuation(data) {
        const container = elements.dcfContainer;
        if (!container) return;

        const dcf = data.valuation.dcf;
        
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-6">
                <div class="bg-[#0a101d] p-6 rounded border border-slate-700">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-2">Enterprise Value</div>
                    <div class="text-2xl font-bold text-white font-mono" id="val-ev">${formatCurrency(dcf.enterprise_value)}</div>
                </div>
                <div class="bg-[#0a101d] p-6 rounded border border-slate-700">
                    <div class="text-xs text-slate-500 uppercase tracking-widest mb-2">Implied Share Price</div>
                    <div class="text-2xl font-bold text-emerald-400 font-mono" id="val-share-price">$${dcf.share_price.toFixed(2)}</div>
                </div>
            </div>

            <div class="bg-[#0a101d] p-6 rounded border border-slate-700 mt-6">
                <h3 class="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Assumptions (Interactive)</h3>
                <div class="grid grid-cols-2 gap-8 text-sm font-mono">
                    <div class="flex flex-col gap-2">
                        <div class="flex justify-between">
                             <span class="text-slate-500">WACC</span>
                             <span class="text-blue-400" id="lbl-wacc">${(dcf.wacc * 100).toFixed(1)}%</span>
                        </div>
                        <input type="range" min="4" max="15" step="0.1" value="${dcf.wacc * 100}" class="w-full accent-blue-500 cursor-pointer" oninput="updateValuation(this.value, 'wacc')">
                    </div>
                    <div class="flex flex-col gap-2">
                        <div class="flex justify-between">
                            <span class="text-slate-500">Terminal Growth</span>
                            <span class="text-blue-400" id="lbl-growth">${(dcf.growth_rate * 100).toFixed(1)}%</span>
                        </div>
                        <input type="range" min="0" max="8" step="0.1" value="${dcf.growth_rate * 100}" class="w-full accent-emerald-500 cursor-pointer" oninput="updateValuation(this.value, 'growth')">
                    </div>
                </div>
            </div>
        `;
        
        // attach updater to window for simplicity
        window.updateValuation = (val, type) => {
            if (!currentMemo) return;
            const dcf = currentMemo.valuation.dcf;
            
            let wacc = dcf.wacc;
            let growth = dcf.growth_rate;
            
            if (type === 'wacc') {
                wacc = parseFloat(val) / 100;
                document.getElementById('lbl-wacc').innerText = val + "%";
            } else {
                growth = parseFloat(val) / 100;
                document.getElementById('lbl-growth').innerText = val + "%";
            }

            // Simplified DCF sensitivity logic
            // Sensitivity Factor: 1% WACC change ~ 15% value change (inverse)
            // 1% Growth change ~ 10% value change

            const originalWacc = dcf.wacc || 0.08;
            const originalGrowth = dcf.growth_rate || 0.02;
            const baseEV = dcf.enterprise_value;
            const basePrice = dcf.share_price;

            const waccDelta = originalWacc - wacc; // Lower WACC = Higher Value
            const growthDelta = growth - originalGrowth; // Higher Growth = Higher Value

            const multiplier = 1 + (waccDelta * 15) + (growthDelta * 10);
            const newEV = baseEV * multiplier;
            const newPrice = basePrice * multiplier;

            document.getElementById('val-ev').innerText = formatCurrency(newEV);
            document.getElementById('val-share-price').innerText = "$" + newPrice.toFixed(2);
        };
    }

    function renderAuditLog(data) {
        const tbody = elements.auditTableBody;
        if (!tbody) return;
        tbody.innerHTML = '';
        
        data.audit_log.forEach(entry => {
            logAudit("System", entry.action, `Status: ${entry.status}`);
        });
    }

    // --- Evidence Viewer ---

    // We can expose this to be called from onclick handlers in the HTML content
    window.viewEvidence = (docId) => {
        const viewer = elements.pdfViewer;
        const mockPage = elements.mockPdfPage;
        const title = elements.docTitle;
        
        if (title) title.innerText = docId || "Document Preview";
        
        // Show the mock page
        if (mockPage) {
            mockPage.classList.remove('hidden');
            // Randomize position of highlight for effect
            const highlight = document.getElementById('highlight-box');
            if (highlight) {
                highlight.style.top = `${Math.random() * 60 + 10}%`;
                highlight.style.height = `${Math.random() * 20 + 5}%`;
            }
        }
    };

    // --- Utilities ---

    function logAudit(actor, action, details) {
        if (!elements.auditTableBody) return;
        
        const time = new Date().toLocaleTimeString();
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-800/50 transition";

        const statusColor = details.toLowerCase().includes('success') ? 'text-emerald-400' : 'text-slate-400';

        tr.innerHTML = `
            <td class="p-3 border-b border-slate-800/50 text-slate-500">${time}</td>
            <td class="p-3 border-b border-slate-800/50 font-bold text-blue-400">${actor} :: ${action}</td>
            <td class="p-3 border-b border-slate-800/50 ${statusColor}">${details}</td>
        `;
        elements.auditTableBody.insertBefore(tr, elements.auditTableBody.firstChild);
    }

    function formatCurrency(val) {
        if (val === undefined || val === null) return '-';
        if (Math.abs(val) >= 1000) return '$' + (val / 1000).toFixed(1) + 'B';
        return '$' + val.toFixed(1) + 'M';
    }

    // --- Edit & Persistence Controls ---
    let isEditMode = false;

    function injectControls() {
        const headerTools = document.querySelector('.h-12 .flex.gap-2');
        if (headerTools) {
             // Load
            const uploadBtn = document.createElement('button');
            uploadBtn.className = "px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded transition mr-2";
            uploadBtn.innerHTML = '<i class="fas fa-upload mr-1"></i> Load';
            uploadBtn.onclick = () => document.getElementById('json-upload-v1').click();
            headerTools.prepend(uploadBtn);

            // Save JSON
            const jsonBtn = document.createElement('button');
            jsonBtn.className = "px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded transition mr-2";
            jsonBtn.innerHTML = '<i class="fas fa-download mr-1"></i> Save JSON';
            jsonBtn.onclick = downloadState;
            headerTools.prepend(jsonBtn);

            // Edit
            const editBtn = document.createElement('button');
            editBtn.className = "px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded transition mr-2";
            editBtn.innerHTML = '<i class="fas fa-pen mr-1"></i> Edit';
            editBtn.onclick = function() { toggleEditMode(this); };
            headerTools.prepend(editBtn);

            // Hidden Input
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.id = 'json-upload-v1';
            fileInput.style.display = 'none';
            fileInput.accept = '.json';
            fileInput.onchange = uploadState;
            headerTools.appendChild(fileInput);
        }
    }

    function toggleEditMode(btn) {
        isEditMode = !isEditMode;
        const els = document.querySelectorAll('.editable-field');
        els.forEach(el => {
            el.contentEditable = isEditMode;
            el.classList.toggle('bg-yellow-50', isEditMode);
            el.classList.toggle('p-2', isEditMode); // add padding
            el.classList.toggle('rounded', isEditMode);
            el.classList.toggle('text-slate-900', isEditMode); // Ensure contrast
            if (!isEditMode) el.classList.remove('bg-yellow-50', 'p-2', 'rounded', 'text-slate-900');
        });

        if (isEditMode) {
             btn.innerHTML = '<i class="fas fa-save mr-1"></i> Save Changes';
             btn.classList.add('bg-emerald-600', 'hover:bg-emerald-500');
             btn.classList.remove('bg-slate-700', 'hover:bg-slate-600');
        } else {
             saveDOMToState();
             btn.innerHTML = '<i class="fas fa-pen mr-1"></i> Edit';
             btn.classList.remove('bg-emerald-600', 'hover:bg-emerald-500');
             btn.classList.add('bg-slate-700', 'hover:bg-slate-600');
        }
    }

    function saveDOMToState() {
        if (!currentMemo) return;

        const titleEl = document.querySelector('h1.editable-field');
        if (titleEl) currentMemo.name = titleEl.innerText;

        const sectionDivs = document.querySelectorAll('div[data-section-index]');
        sectionDivs.forEach(div => {
            const idx = div.dataset.sectionIndex;
            const titleEl = div.querySelector('h3.editable-field');
            const contentEl = div.querySelector('div.editable-field');

            if (currentMemo.sections[idx]) {
                if (titleEl) currentMemo.sections[idx].title = titleEl.innerText;
                if (contentEl) currentMemo.sections[idx].content = contentEl.innerText;
            }
        });
        logAudit("System", "Save", "Changes persisted to local state.");
    }

    function downloadState() {
        if (!currentMemo) { alert("No report loaded."); return; }
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentMemo, null, 2));
        const anchor = document.createElement('a');
        anchor.href = dataStr;
        anchor.download = `credit_memo_${currentMemo.ticker || "report"}.json`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        logAudit("System", "Export", "Report exported to JSON.");
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
                    currentMemo = normalized;
                    renderMemo(normalized);
                    renderFinancials(normalized);
                    renderValuation(normalized);
                    renderAuditLog(normalized);
                    logAudit("System", "Load", `Loaded state from ${file.name}`);
                } else {
                    alert("Invalid JSON format.");
                }
            } catch (err) {
                console.error(err);
                alert("Error parsing JSON.");
            }
        };
        reader.readAsText(file);
    }

});
