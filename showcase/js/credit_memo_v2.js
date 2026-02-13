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
        uploadInput: document.getElementById('upload-json'),
        memoContent: document.getElementById('memo-content'),
        evidenceViewer: document.getElementById('pdf-mock-canvas'),
        terminal: document.getElementById('agent-terminal')
    };

    // --- Initialization ---
    async function init() {
        log("System", "Initializing CreditOS v2.4 (Editable)...", "info");

        try {
            // Load Library via UniversalLoader
            const data = await window.UniversalLoader.loadCreditLibrary();
            library = Array.isArray(data) ? data.reduce((acc, item) => ({...acc, [item.id]: item}), {}) : data;

            Object.values(library).forEach(item => {
                const opt = document.createElement('option');
                opt.value = item.id;
                opt.innerText = `${item.borrower_name} (Risk: ${item.risk_score})`;
                elements.borrowerSelect.appendChild(opt);
            });
            
            log("Archivist", `Loaded ${Object.keys(library).length} entity profiles via UniversalLoader.`, "success");
        } catch (e) {
            console.error(e);
            log("System", "Failed to load library. Upload manually.", "error");
        }
    }

    // --- Event Listeners ---
    if(elements.generateBtn) elements.generateBtn.addEventListener('click', () => runAnalysis(elements.borrowerSelect.value));
    if(elements.editBtn) elements.editBtn.addEventListener('click', toggleEditMode);

    // Export JSON
    if(elements.exportBtn) elements.exportBtn.addEventListener('click', () => {
        if(!currentMemoData) return;
        updateDataFromDOM(); // Sync edits
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentMemoData, null, 2));
        const a = document.createElement('a');
        a.href = dataStr;
        a.download = `${currentMemoData.ticker}_credit_memo.json`;
        a.click();
        log("System", "Exported current memo.", "success");
    });

    // Upload JSON
    if(elements.uploadInput) elements.uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if(!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            try {
                const json = JSON.parse(evt.target.result);
                currentMemoData = json;
                renderMemo(json);
                log("System", `Loaded ${json.borrower_name} from file.`, "success");
            } catch(err) {
                console.error(err);
                alert("Invalid JSON");
            }
        };
        reader.readAsText(file);
    });

    // --- Core Workflow ---
    async function runAnalysis(id) {
        if (!id) { alert("Please select a target entity."); return; }
        const item = library[id];
        
        // Reset UI
        elements.memoContent.innerHTML = '';
        clearTerminal();
        
        await simulateAgentTask("Archivist", [`Fetching ${item.ticker} artifacts...`], 500);

        // Use library item directly as data source (since it's now full JSON)
        currentMemoData = item;

        await simulateAgentTask("Quant", [`Running spreads...`, `Validating...`], 500);
        renderMemo(currentMemoData);
        log("Orchestrator", `Analysis Complete for ${item.borrower_name}`, "success");
    }

    // --- Rendering Logic ---
    function renderMemo(memo) {
        const contentDiv = elements.memoContent;
        contentDiv.innerHTML = '';
        contentDiv.style.display = 'block';
        document.getElementById('memo-placeholder').style.display = 'none';

        // Header
        const riskColor = memo.risk_score < 60 ? 'red' : (memo.risk_score < 80 ? 'orange' : 'green');
        const header = document.createElement('div');
        header.innerHTML = `
            <h1 class="editable-content" data-key="borrower_name">${memo.borrower_name}</h1>
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 30px; font-family: var(--font-mono); font-size: 0.8rem; color: #666;">
                <span>DATE: ${new Date(memo.report_date).toLocaleDateString()}</span>
                <span>RISK SCORE: <b style="color: ${riskColor}">${memo.risk_score}/100</b></span>
                <span>ID: ${memo.ticker}-${Math.floor(Math.random()*10000)}</span>
            </div>
        `;
        contentDiv.appendChild(header);

        // Summary
        const summaryDiv = document.createElement('div');
        summaryDiv.innerHTML = `
            <h3>Executive Summary</h3>
            <p class="editable-content" data-key="summary">${memo.summary}</p>
        `;
        contentDiv.appendChild(summaryDiv);

        // Sections (Narratives)
        if (memo.documents && memo.documents[0]) {
             const narratives = memo.documents[0].chunks.filter(c => c.type === 'narrative');
             if(narratives.length) {
                 const secDiv = document.createElement('div');
                 secDiv.innerHTML = `<h3>Key Drivers</h3>`;
                 narratives.forEach(chunk => {
                     secDiv.innerHTML += `
                        <div style="margin-bottom:15px; padding-left:10px; border-left:2px solid #444;">
                            <div class="editable-content" data-chunk-id="${chunk.chunk_id}">${chunk.content}</div>
                            <span class="citation-tag" onclick="viewEvidence('${memo.documents[0].doc_id}', '${chunk.chunk_id}')">[REF: ${chunk.chunk_id}]</span>
                        </div>
                     `;
                 });
                 contentDiv.appendChild(secDiv);
             }
        }

        if (isEditMode) applyEditMode();
    }

    // --- Edit Logic ---
    function toggleEditMode() {
        isEditMode = !isEditMode;
        const btn = elements.editBtn;
        if (isEditMode) {
            btn.innerHTML = '<i class="fas fa-save"></i> SAVE DOM';
            btn.style.borderColor = 'var(--neon-cyan)';
            btn.style.color = 'var(--neon-cyan)';
            document.body.classList.add('editing-active');
            applyEditMode();
        } else {
            updateDataFromDOM(); // Commit changes to JS object
            btn.innerHTML = '<i class="fas fa-edit"></i> EDIT';
            btn.style.borderColor = '#444';
            btn.style.color = '#888';
            document.body.classList.remove('editing-active');
            document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "false");
            log("System", "Changes committed to memory.", "info");
        }
    }

    function applyEditMode() {
        document.querySelectorAll('.editable-content').forEach(el => el.contentEditable = "true");
    }

    function updateDataFromDOM() {
        if (!currentMemoData) return;
        
        // Update top-level keys
        document.querySelectorAll('[data-key]').forEach(el => {
            const key = el.getAttribute('data-key');
            if(currentMemoData[key]) currentMemoData[key] = el.innerText;
        });

        // Update Chunks
        if(currentMemoData.documents) {
            document.querySelectorAll('[data-chunk-id]').forEach(el => {
                const cid = el.getAttribute('data-chunk-id');
                const doc = currentMemoData.documents[0];
                const chunk = doc.chunks.find(c => c.chunk_id === cid);
                if(chunk) chunk.content = el.innerText;
            });
        }
    }

    // --- Evidence Viewer ---
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
        page.style.height = '600px';
        page.style.background = '#fff';
        page.style.position = 'relative';
        page.style.padding = '20px';
        page.style.boxSizing = 'border-box';
        
        page.innerHTML = `<div style="color: #ddd; font-size: 6px; overflow:hidden; height:100%; word-break: break-all;">${"CONTENT ".repeat(2000)}</div>`;

        if (chunk && chunk.bbox) {
            const [x0, y0, x1, y1] = chunk.bbox;
            // Scaling for smaller viewer
            const scaleX = 100 / 600;
            const scaleY = 100 / 800;
            
            const bbox = document.createElement('div');
            bbox.className = 'bbox-highlight';
            bbox.style.left = (x0 * scaleX) + '%';
            bbox.style.top = (y0 * scaleY) + '%';
            bbox.style.width = ((x1 - x0) * scaleX) + '%';
            bbox.style.height = ((y1 - y0) * scaleY) + '%';
            
            bbox.innerHTML = `<div class="bbox-label">${chunk.type}</div>`;
            page.appendChild(bbox);
        }

        viewer.appendChild(page);
        document.getElementById('doc-id-label').innerText = `[${docId}]`;
    };

    // --- Helpers ---
    function log(agent, message, type="info") {
        const div = document.createElement('div');
        div.className = `log-entry ${type}`;
        div.innerHTML = `<span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span><span class="log-agent">${agent}</span><span class="log-message">${message}</span>`;
        elements.terminal.prepend(div);
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

    // Start
    init();
});
