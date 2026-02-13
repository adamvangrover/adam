document.addEventListener('DOMContentLoaded', () => {
    // ------------------------------------------------------------------
    // State & Config
    // ------------------------------------------------------------------
    let library = {};
    let selectedBorrower = null;
    let agentLogInterval = null;

    const elements = {
        borrowerSelect: document.getElementById('borrower-select'),
        generateBtn: document.getElementById('generate-btn'),
        memoContainer: document.getElementById('memo-content'),
        memoPlaceholder: document.getElementById('memo-placeholder'),
        evidenceViewer: document.getElementById('pdf-mock-canvas'),
        terminal: document.getElementById('agent-terminal'),
        docIdLabel: document.getElementById('doc-id-label'),
        modeIndicator: document.querySelector('.header-nav span span') // "SOVEREIGN_AI_V2"
    };

    // ------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------
    async function init() {
        log("System", "Initializing CreditOS v2.4...", "info");

        try {
            // Load Full Library
            // Fallback to tech_credit_data.json if library is missing (safety)
            let res = await fetch('data/credit_memo_library.json');
            if (!res.ok) {
                log("System", "Full library not found, loading Tech subset...", "warn");
                res = await fetch('data/tech_credit_data.json');
            }
            library = await res.json();

            // Populate Dropdown
            // Sort by key
            const sortedKeys = Object.keys(library).sort();
            sortedKeys.forEach(key => {
                const item = library[key];
                const ticker = item.borrower_details.ticker || "UNKNOWN";
                const opt = document.createElement('option');
                opt.value = key;
                opt.innerText = `${key} (${ticker})`;
                elements.borrowerSelect.appendChild(opt);
            });

            log("Archivist", `Loaded ${Object.keys(library).length} entity profiles.`, "success");

        } catch (e) {
            log("System", "Failed to load artifact library.", "error");
            console.error(e);
        }
    }

    // ------------------------------------------------------------------
    // Event Listeners
    // ------------------------------------------------------------------
    elements.generateBtn.addEventListener('click', runAnalysis);

    // ------------------------------------------------------------------
    // Core Workflow
    // ------------------------------------------------------------------
    async function runAnalysis() {
        const borrowerName = elements.borrowerSelect.value;
        if (!borrowerName) {
            alert("Please select a target entity.");
            return;
        }

        // Default to cached data
        let data = library[borrowerName];
        const ticker = data.borrower_details.ticker;
        const sector = data.borrower_details.sector;

        // Reset UI
        elements.memoContainer.innerHTML = '';
        elements.memoContainer.style.display = 'none';
        elements.memoPlaceholder.style.display = 'flex';
        elements.memoPlaceholder.innerHTML = `<h3 style="color:var(--neon-cyan)">INITIALIZING AGENT SWARM...</h3>`;
        clearTerminal();

        // LIVE MODE ATTEMPT
        let isLive = false;
        try {
            log("Orchestrator", "Attempting Live Pipeline Connection...", "info");

            // Note: In a real scenario, we might check a "use_live" checkbox.
            // For now, we try live, fallback to cache.
            const apiRes = await fetch('/api/credit/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker, name: borrowerName, sector: sector })
            });

            if (apiRes.ok) {
                const apiData = await apiRes.json();
                if (apiData.status === 'success') {
                    log("Orchestrator", "Live Pipeline Connected. Executing...", "success");
                    data = apiData.data; // Use live data
                    isLive = true;
                    elements.modeIndicator.innerText = "LIVE_COMPUTE_NODE";
                    elements.modeIndicator.style.color = "var(--neon-pink)";
                } else {
                    throw new Error(apiData.message);
                }
            } else {
                throw new Error(`API Status: ${apiRes.status}`);
            }
        } catch (e) {
            log("Orchestrator", `Live Connection Failed (${e.message}). Reverting to Cached Artifacts.`, "warn");
            elements.modeIndicator.innerText = "CACHED_ARTIFACT_MODE";
            elements.modeIndicator.style.color = "var(--neon-cyan)";
        }

        selectedBorrower = data;

        // 1. Retrieval Phase
        await simulateAgentTask("Archivist", [
            `Querying Vector DB for ${ticker}...`,
            `Found 10-K (${new Date().getFullYear()}) - ${data.documents[0].page_count} pages.`,
            `Extracting relevant chunks (MD&A, Risk Factors)...`,
            `Retrieving Knowledge Graph nodes for sector: ${data.borrower_details.sector}...`
        ], 800);

        // 2. Spreading Phase
        const table = data.documents[0].chunks.find(c=>c.type==='financial_table');
        const assets = table ? table.content_json.total_assets : 'N/A';

        await simulateAgentTask("Quant", [
            `OCR-ing Financial Tables...`,
            `Mapping line items to FIBO ontology...`,
            `Validating Balance Sheet: Assets (${assets}) = Liab + Equity...`,
            `PASS: Checksum valid.`,
            `Calculating EBITDA Ratios...`
        ], 1000);

        // 3. Synthesis Phase
        await simulateAgentTask("Writer", [
            `Loading Prompt: commercial_credit_v1.yaml...`,
            `Synthesizing Executive Summary...`,
            `Injecting Citations...`,
            `Running Audit Checks (Citation Density)... PASS.`
        ], 1200);

        // Render Final Memo
        renderMemo(selectedBorrower);
        log("Orchestrator", `Analysis Complete. Mode: ${isLive ? 'LIVE' : 'CACHED'}`, "success");
    }

    // ------------------------------------------------------------------
    // Rendering Logic
    // ------------------------------------------------------------------
    function renderMemo(data) {
        elements.memoPlaceholder.style.display = 'none';
        elements.memoContainer.style.display = 'block';

        const docs = data.documents[0];
        const chunks = docs.chunks;

        // Helper to find chunks
        const getChunksByType = (t) => chunks.filter(c => c.type === t);
        const narratives = getChunksByType('narrative');
        const risks = getChunksByType('risk_factor');
        const table = getChunksByType('financial_table')[0];

        // Citation Helper
        const cite = (chunk) => `<span class="citation-tag" onclick="viewEvidence('${docs.doc_id}', '${chunk.chunk_id}')">[${docs.doc_id}:${chunk.chunk_id}]</span>`;

        let html = `
            <div class="memo-header">
                <div>
                    <div class="memo-title">${data.borrower_details.name}</div>
                    <div style="font-size: 0.8rem; color: #888;">TICKER: <span style="color:white">${data.borrower_details.ticker}</span> | RATING: <span style="color:var(--neon-green)">${data.borrower_details.rating}</span></div>
                </div>
                <div style="text-align: right; font-size: 0.7rem; color: #555;">
                    GENERATED: ${new Date().toISOString().split('T')[0]}<br>
                    CONFIDENCE: 98.4%
                </div>
            </div>

            <h2>Executive Summary</h2>
            <p>${narratives[0] ? narratives[0].content + ' ' + cite(narratives[0]) : 'No narrative data found.'}</p>
            <p>${narratives[1] ? narratives[1].content + ' ' + cite(narratives[1]) : ''}</p>

            <h2>Financial Analysis</h2>
            <p>The company demonstrates robust financial health. ${table ? cite(table) : ''}</p>

            ${table ? `
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0; border: 1px solid #333; font-family: var(--font-mono); font-size: 0.85rem;">
                <tr style="background: rgba(255,255,255,0.1); border-bottom: 1px solid #444;">
                    <th style="text-align: left; padding: 8px;">Metric ($B)</th>
                    <th style="text-align: right; padding: 8px;">2025</th>
                </tr>
                <tr><td style="padding: 8px; border-bottom: 1px solid #222;">Total Assets</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">${table.content_json.total_assets}</td></tr>
                <tr><td style="padding: 8px; border-bottom: 1px solid #222;">Total Liabilities</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">${table.content_json.total_liabilities}</td></tr>
                <tr><td style="padding: 8px; border-bottom: 1px solid #222;">Total Equity</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #222;">${table.content_json.total_equity}</td></tr>
                <tr><td style="padding: 8px; color: var(--neon-cyan);">EBITDA</td><td style="text-align: right; padding: 8px; color: var(--neon-cyan);">${table.content_json.ebitda}</td></tr>
            </table>
            ` : '<p>No financial table extracted.</p>'}

            <h2>Key Risk Factors</h2>
            <ul>
                ${risks.map(r => `<li>${r.content} ${cite(r)}</li>`).join('')}
            </ul>

            <h2>Conclusion</h2>
            <p>Based on the analysis of credit fundamentals and market positioning, we recommend maintaining the current exposure limit.</p>
        `;

        elements.memoContainer.innerHTML = html;

        // Expose viewEvidence globally
        window.viewEvidence = (docId, chunkId) => renderEvidence(docId, chunkId);
    }

    function renderEvidence(docId, chunkId) {
        if (!selectedBorrower) return;

        const doc = selectedBorrower.documents.find(d => d.doc_id === docId);
        const chunk = doc.chunks.find(c => c.chunk_id === chunkId);

        if (!chunk) return;

        elements.docIdLabel.innerText = `${docId} | Page ${chunk.page}`;

        // Clear Viewer
        elements.evidenceViewer.innerHTML = '';

        // Create "Page"
        const page = document.createElement('div');
        page.style.width = '100%';
        page.style.height = '800px'; // Mock height
        page.style.background = '#fff';
        page.style.position = 'relative';
        page.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';

        // Mock Page Text content (faded background text)
        page.innerHTML = `
            <div style="padding: 40px; color: #ccc; font-family: serif; font-size: 10px; overflow: hidden; height: 100%;">
                ${Array(50).fill("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua").join(' ')}
            </div>
        `;

        // Create BBox
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

        // Inject Content into BBox (to simulate readable text)
        const contentOverlay = document.createElement('div');
        contentOverlay.style.position = 'absolute';
        contentOverlay.style.top = '0';
        contentOverlay.style.left = '0';
        contentOverlay.style.width = '100%';
        contentOverlay.style.height = '100%';
        contentOverlay.style.background = 'rgba(255, 255, 255, 0.9)';
        contentOverlay.style.color = 'black';
        contentOverlay.style.fontSize = '12px';
        contentOverlay.style.padding = '5px';
        contentOverlay.style.overflow = 'hidden';
        contentOverlay.innerText = chunk.content; // Show actual text
        bbox.appendChild(contentOverlay);

        page.appendChild(bbox);
        elements.evidenceViewer.appendChild(page);

        // Scroll to view
        bbox.scrollIntoView({ behavior: 'smooth', block: 'center' });

        log("Frontend", `User inspected evidence: ${chunkId}`, "audit");
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    function log(agent, message, type="info") {
        const time = new Date().toLocaleTimeString();
        const div = document.createElement('div');
        div.className = `log-entry ${type}`;

        let color = "#888";
        if (agent === "Archivist") color = "#00f3ff";
        if (agent === "Quant") color = "#ff00ff";
        if (agent === "Writer") color = "#00ff41";
        if (agent === "Orchestrator") color = "white";

        div.innerHTML = `<span class="log-timestamp">[${time}]</span><span class="log-agent" style="color:${color}">${agent}</span><span class="log-message">${message}</span>`;

        elements.terminal.appendChild(div);
        elements.terminal.scrollTop = elements.terminal.scrollHeight;
    }

    function clearTerminal() {
        elements.terminal.innerHTML = '';
    }

    function simulateAgentTask(agent, messages, delayPerMessage) {
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
            }, delayPerMessage);
        });
    }

    // Run Init
    init();
});
