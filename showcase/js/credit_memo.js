document.addEventListener('DOMContentLoaded', () => {
    let mockData = {};
    const elements = {
        generateBtn: document.getElementById('generate-btn'),
        memoPanel: document.getElementById('memo-panel'),
        memoContent: document.getElementById('memo-content'),
        memoPlaceholder: document.getElementById('memo-placeholder'),
        evidencePanel: document.getElementById('evidence-panel'),
        pdfViewer: document.getElementById('pdf-viewer'),
        closeEvidenceBtn: document.getElementById('close-evidence'),
        progressContainer: document.getElementById('progress-container'),
        progressFill: document.getElementById('progress-fill'),
        progressText: document.getElementById('progress-text'),
        agentStatus: document.getElementById('agent-status'),
        auditLogPanel: document.getElementById('audit-log-panel'),
        auditLogContent: document.getElementById('audit-log-content'),
        toggleAuditLogBtn: document.getElementById('toggle-audit-log'),
        borrowerSelect: document.getElementById('borrower-select')
    };

    // Load Mock Data
    fetch('data/credit_mock_data.json')
        .then(response => response.json())
        .then(data => {
            mockData = data;
            logAudit("System", "Data Loaded", "Loaded credit_mock_data.json");
        })
        .catch(err => console.error("Failed to load mock data", err));

    // Event Listeners
    elements.generateBtn.addEventListener('click', startGeneration);
    elements.closeEvidenceBtn.addEventListener('click', hideEvidence);
    elements.toggleAuditLogBtn.addEventListener('click', () => {
        elements.auditLogPanel.style.display = elements.auditLogPanel.style.display === 'none' ? 'block' : 'none';
    });

    function startGeneration() {
        const borrower = elements.borrowerSelect.value;
        if (!mockData[borrower]) {
            alert("Borrower data not found!");
            return;
        }

        // UI Reset
        elements.memoContent.style.display = 'none';
        elements.memoPlaceholder.style.display = 'none';
        elements.progressContainer.style.display = 'block';
        elements.evidencePanel.classList.remove('active');
        elements.evidencePanel.style.width = '0'; // manual override

        // Simulate Agent Workflow
        simulateAgent("Archivist", "Retrieving documents...", 0, 1000)
            .then(() => simulateAgent("Quant", "Spreading financial tables...", 33, 1000))
            .then(() => simulateAgent("Risk Officer", "Checking covenants...", 66, 1000))
            .then(() => simulateAgent("Writer", "Synthesizing memo...", 90, 1500))
            .then(() => {
                elements.progressContainer.style.display = 'none';
                renderMemo(borrower);
                logAudit("Orchestrator", "Complete", `Memo generated for ${borrower}`);
            });
    }

    function simulateAgent(agentName, statusText, progressStart, duration) {
        return new Promise(resolve => {
            elements.progressText.innerText = `${agentName} Active`;
            elements.agentStatus.innerText = statusText;
            elements.progressFill.style.width = `${progressStart}%`;

            logAudit(agentName, "Start", statusText);

            setTimeout(() => {
                logAudit(agentName, "Complete", "Task finished");
                resolve();
            }, duration);
        });
    }

    function renderMemo(borrowerName) {
        const data = mockData[borrowerName];
        const docs = data.documents[0]; // Assume 1 doc for now
        const chunks = docs.chunks;

        // Helper to find chunk text
        const getChunk = (id) => chunks.find(c => c.chunk_id === id);

        // Construct Memo HTML (Simulation of Writer Agent)
        const revenueChunk = getChunk("chunk_456");
        const tableChunk = getChunk("chunk_789");
        const riskChunk = getChunk("chunk_999");

        // Generate Citations
        const cite = (chunk) => `<span class="citation-tag" onclick="showEvidence('${docs.doc_id}', '${chunk.chunk_id}')">[${docs.doc_id}:${chunk.chunk_id}]</span>`;

        let html = `<h1>Credit Memo: ${data.borrower_details.name}</h1>`;
        html += `<div style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 20px;">
                    Rating: ${data.borrower_details.rating} | Sector: ${data.borrower_details.industry} | Status: <span style="color: #0f0">Active</span>
                 </div>`;

        html += `<h2>1. Executive Summary</h2>`;
        html += `<p>${revenueChunk.content} ${cite(revenueChunk)}</p>`;
        html += `<p>The company maintains a strong market position despite sectoral headwinds.</p>`;

        html += `<h2>2. Financial Analysis</h2>`;
        html += `<p>The consolidated balance sheet indicates a stable capital structure. ${cite(tableChunk)}</p>`;

        // Render a mini table
        const fin = tableChunk.content_json;
        html += `<table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em;">
                    <tr style="border-bottom: 1px solid #333;"><th style="text-align: left;">Metric</th><th style="text-align: right;">Value ($M)</th></tr>
                    <tr><td>Total Assets</td><td style="text-align: right;">${fin.total_assets}</td></tr>
                    <tr><td>Total Liabilities</td><td style="text-align: right;">${fin.total_liabilities}</td></tr>
                    <tr><td>EBITDA</td><td style="text-align: right;">${fin.ebitda}</td></tr>
                    <tr><td>Leverage</td><td style="text-align: right;">${(fin.total_debt/fin.ebitda).toFixed(1)}x</td></tr>
                 </table>`;
        html += `<div style="margin-top: 5px; font-size: 0.8em; color: #0f0;">✔ Validation Passed: Assets = Liab + Equity</div>`;

        html += `<h2>3. Key Risks</h2>`;
        html += `<p>${riskChunk.content} ${cite(riskChunk)}</p>`;

        html += `<h2>4. Recommendation</h2>`;
        html += `<p><strong>APPROVE</strong> based on strong cash flow coverage and strategic alignment.</p>`;

        elements.memoContent.innerHTML = html;
        elements.memoContent.style.display = 'block';

        // Expose showEvidence to global scope for onclick
        window.showEvidence = (docId, chunkId) => renderEvidence(borrowerName, docId, chunkId);
    }

    function renderEvidence(borrowerName, docId, chunkId) {
        const data = mockData[borrowerName];
        const doc = data.documents.find(d => d.doc_id === docId);
        const chunk = doc.chunks.find(c => c.chunk_id === chunkId);

        if (!doc || !chunk) return;

        // Show Panel
        elements.evidencePanel.classList.add('active');
        elements.evidencePanel.style.width = '50%'; // Expanded

        // Render "PDF Page"
        const pageNum = chunk.page;
        const pageId = `page-${pageNum}`;

        // Clear Viewer
        elements.pdfViewer.innerHTML = '';

        // Create Page Container
        const pageDiv = document.createElement('div');
        pageDiv.className = 'pdf-page';
        pageDiv.id = pageId;
        pageDiv.style.position = 'relative';
        pageDiv.style.height = '1000px'; // Fixed height for 0-1000 coord system
        pageDiv.style.background = '#f0f0f0';
        pageDiv.style.border = '1px solid #ccc';
        pageDiv.style.color = 'black';
        pageDiv.style.padding = '40px';

        // Mock Page Content (Text)
        pageDiv.innerHTML = `<h4 style="color: #666;">Page ${pageNum}</h4>`;

        // Simulate text content on page (just scattering the chunk content for visual)
        // In a real app, this would be the rendered PDF image or canvas
        const contentDiv = document.createElement('div');
        contentDiv.innerText = chunk.content; // Just show the relevant text for clarity
        contentDiv.style.marginTop = '100px';
        contentDiv.style.fontSize = '14px';
        contentDiv.style.fontFamily = 'serif';
        pageDiv.appendChild(contentDiv);

        // Create BBox Highlight
        const [x0, y0, x1, y1] = chunk.bbox;
        const highlight = document.createElement('div');
        highlight.className = 'bbox-highlight';
        highlight.style.left = `${x0}px`;
        highlight.style.top = `${y0}px`;
        highlight.style.width = `${x1 - x0}px`;
        highlight.style.height = `${y1 - y0}px`;
        highlight.title = chunk.type;

        // Add label to highlight
        const label = document.createElement('div');
        label.innerText = chunk.type.toUpperCase();
        label.style.background = 'orange';
        label.style.color = 'black';
        label.style.fontSize = '10px';
        label.style.position = 'absolute';
        label.style.top = '-15px';
        label.style.left = '0';
        label.style.padding = '2px';
        highlight.appendChild(label);

        pageDiv.appendChild(highlight);
        elements.pdfViewer.appendChild(pageDiv);

        // Scroll into view
        highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });

        logAudit("Frontend", "Verify", `User viewed evidence ${docId}:${chunkId}`);
    }

    function hideEvidence() {
        elements.evidencePanel.classList.remove('active');
        elements.evidencePanel.style.width = '0';
    }

    function logAudit(actor, action, details) {
        const time = new Date().toISOString().split('T')[1].split('.')[0];
        const line = document.createElement('div');
        line.innerHTML = `<span style="color: #888;">[${time}]</span> <strong style="color: var(--accent-color);">${actor}:</strong> ${action} - ${details}`;
        elements.auditLogContent.appendChild(line);
        elements.auditLogContent.scrollTop = elements.auditLogContent.scrollHeight;
    }
});
