let currentMemoData = null;
let isEditMode = false;
let currentChart = null;
let riskChart = null;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadBorrowers();
        await loadAuditLogs();

        document.getElementById('generate-btn').addEventListener('click', handleGenerate);
        document.getElementById('edit-btn').addEventListener('click', toggleEditMode);
        document.getElementById('export-btn').addEventListener('click', handleExport);
    } catch (e) {
        console.error("Initialization failed:", e);
    }
});

async function loadBorrowers() {
    const res = await fetch('data/credit_memo_library.json');
    if (!res.ok) throw new Error("Failed to load library");
    const library = await res.json();

    const select = document.getElementById('borrower-select');
    library.forEach(item => {
        const option = document.createElement('option');
        option.value = item.file;
        option.textContent = `${item.borrower_name} (Risk: ${item.risk_score})`;
        select.appendChild(option);
    });
}

async function loadAuditLogs() {
    const res = await fetch('data/credit_memo_audit_log.json');
    if (!res.ok) throw new Error("Failed to load audit log");
    const logs = await res.json();

    const terminal = document.getElementById('agent-terminal');
    terminal.innerHTML = '';

    // Show latest logs
    logs.slice().reverse().slice(0, 50).forEach(log => {
        const line = document.createElement('div');
        line.style.marginBottom = '4px';
        const time = new Date(log.timestamp).toLocaleTimeString();
        const color = log.validation_status === 'PASS' ? '#00ff41' : (log.validation_status === 'FAIL' ? 'red' : 'yellow');

        line.innerHTML = `
            <span style="opacity: 0.5">[${time}]</span>
            <span style="color: ${color}">${log.action}</span>
            <span style="opacity: 0.7">:: ${log.transaction_id.substring(0,8)}</span>
        `;
        terminal.appendChild(line);
    });
}

async function handleGenerate() {
    const select = document.getElementById('borrower-select');
    const filename = select.value;

    if (!filename) {
        alert("Please select a target entity.");
        return;
    }

    const btn = document.getElementById('generate-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> GENERATING...';
    btn.disabled = true;

    setTimeout(async () => {
        try {
            await loadMemo(filename);
        } catch (e) {
            console.error(e);
            alert("Error generating analysis.");
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }, 1500);
}

async function loadMemo(filename) {
    const path = filename.includes('/') ? filename : `data/${filename}`;
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load memo: ${path}`);
    const memo = await res.json();
    currentMemoData = memo; // Store for models

    renderMemo(memo);
    renderRiskQuant(memo);
}

function renderMemo(memo) {
    const placeholder = document.getElementById('memo-placeholder');
    const contentDiv = document.getElementById('memo-content');

    placeholder.style.display = 'none';
    contentDiv.style.display = 'block';
    contentDiv.innerHTML = '';

    // Header
    const header = document.createElement('div');
    const riskColor = memo.risk_score < 60 ? 'red' : (memo.risk_score < 80 ? 'orange' : 'green');

    header.innerHTML = `
        <h1 class="editable-content" style="margin-top: 0;">${memo.borrower_name}</h1>
        <div style="display: flex; justify-content: space-between; margin-bottom: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #666;">
            <span>DATE: ${new Date(memo.report_date).toLocaleDateString()}</span>
            <span>RISK SCORE: <b style="color: ${riskColor}">${memo.risk_score}/100</b></span>
            <span>ID: ${memo.borrower_name.substring(0,3).toUpperCase()}-REQ-${Math.floor(Math.random()*10000)}</span>
        </div>
        <hr style="border: 0; border-top: 2px solid #000; margin-bottom: 30px;">
    `;
    contentDiv.appendChild(header);

    // Financial Snapshot (Annex A condensed)
    if (memo.financial_ratios) {
        const finDiv = document.createElement('div');
        finDiv.style.marginBottom = '30px';
        finDiv.style.backgroundColor = '#f9f9f9';
        finDiv.style.padding = '15px';
        finDiv.style.border = '1px solid #ddd';

        finDiv.innerHTML = `
            <h3 style="margin-top: 0; font-family: 'Arial'; font-size: 0.9rem; text-transform: uppercase;">Financial Snapshot</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                <div><b>Leverage:</b> ${memo.financial_ratios.leverage_ratio?.toFixed(2)}x</div>
                <div><b>EBITDA:</b> ${formatCurrency(memo.financial_ratios.ebitda)}</div>
                <div><b>Revenue:</b> ${formatCurrency(memo.financial_ratios.revenue)}</div>
                <div><b>DSCR:</b> ${memo.financial_ratios.dscr?.toFixed(2)}x</div>
                <div><b>Net Income:</b> ${formatCurrency(memo.financial_ratios.net_income)}</div>
                <div><b>Current Ratio:</b> ${memo.financial_ratios.current_ratio?.toFixed(2)}x</div>
            </div>
            <div style="margin-top: 20px; height: 200px; width: 100%;">
                <canvas id="finChart"></canvas>
            </div>
        `;
        contentDiv.appendChild(finDiv);

        // Validation Logic
        const assets = memo.historical_financials?.[0]?.total_assets || 0;
        const liabs = memo.historical_financials?.[0]?.total_liabilities || 0;
        const equity = memo.historical_financials?.[0]?.total_equity || 0;

        if (Math.abs(assets - (liabs + equity)) > 1) { // 1m tolerance
            const alert = document.createElement('div');
            alert.style.color = 'red';
            alert.style.fontSize = '0.8rem';
            alert.style.marginTop = '10px';
            alert.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Accounting Mismatch: Assets (${assets}) != Liabs (${liabs}) + Equity (${equity})`;
            finDiv.appendChild(alert);
        }
    }

    // Sections
    memo.sections.forEach(section => {
        const sectionDiv = document.createElement('div');
        sectionDiv.style.marginBottom = '20px';

        let contentHtml = section.content.replace(/\n/g, '<br>');

        // Citations
        contentHtml = contentHtml.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
             return `<a href="#" class="citation-pin" onclick="viewEvidence('${docId}'); return false;">${docId}</a>`;
        });

        sectionDiv.innerHTML = `
            <h2 class="editable-content">${section.title}</h2>
            <div class="editable-content" style="text-align: justify;">${contentHtml}</div>
        `;
        contentDiv.appendChild(sectionDiv);
    });

    // Annexes (Tables)
    if (memo.historical_financials) {
        const annexADiv = document.createElement('div');
        annexADiv.innerHTML = `<h2>Annex A: Historical Financials</h2>`;

        const table = document.createElement('table');
        table.className = 'fin-table';

        // Headers
        const periods = memo.historical_financials.map(h => h.period);
        let thead = `<thead><tr><th>Metric</th>${periods.map(p => `<th style="text-align: right;">${p}</th>`).join('')}</tr></thead>`;
        table.innerHTML = thead;

        const tbody = document.createElement('tbody');
        const metrics = [
            { key: 'revenue', label: 'Revenue' },
            { key: 'ebitda', label: 'EBITDA' },
            { key: 'net_income', label: 'Net Income' },
            { key: 'total_assets', label: 'Total Assets' },
            { key: 'total_liabilities', label: 'Total Liabilities' },
            { key: 'leverage_ratio', label: 'Leverage', fmt: v => v?.toFixed(2) + 'x' }
        ];

        metrics.forEach(m => {
            let row = `<td>${m.label}</td>`;
            memo.historical_financials.forEach(h => {
                let val = h[m.key];
                if (val && !m.fmt) val = formatCurrency(val);
                else if (val && m.fmt) val = m.fmt(val);
                else val = '-';
                row += `<td class="num">${val}</td>`;
            });
            const tr = document.createElement('tr');
            tr.innerHTML = row;
            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        annexADiv.appendChild(table);
        contentDiv.appendChild(annexADiv);
    }

    // DCF (Interactive)
    if (memo.dcf_analysis) {
        const dcfDiv = document.createElement('div');
        dcfDiv.innerHTML = `<h2>Annex B: Valuation (Interactive)</h2>`;

        const controls = document.createElement('div');
        controls.className = 'dcf-controls';
        controls.style.backgroundColor = '#f0f0f0';
        controls.style.padding = '15px';
        controls.style.marginBottom = '20px';
        controls.style.border = '1px dashed #999';

        const dcf = memo.dcf_analysis;

        controls.innerHTML = `
            <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 10px;">
                <label>WACC (%): <input type="number" id="dcf-wacc" class="model-input" value="${(dcf.wacc*100).toFixed(1)}" step="0.1" onchange="recalculateDCF()"></label>
                <label>Growth (%): <input type="number" id="dcf-growth" class="model-input" value="${(dcf.growth_rate*100).toFixed(1)}" step="0.1" onchange="recalculateDCF()"></label>
                <label>Exit Multiple: <input type="number" id="dcf-exit" class="model-input" value="12.0" step="0.5" onchange="recalculateDCF()"></label>
            </div>
            <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 1rem;">
                <div>Implied Share Price: <span id="dcf-share-price" style="font-weight: bold; color: green;">$${dcf.share_price.toFixed(2)}</span></div>
                <div>Enterprise Value: <span id="dcf-ev" style="font-weight: bold; color: blue;">${formatCurrency(dcf.enterprise_value)}</span></div>
            </div>
        `;

        dcfDiv.appendChild(controls);
        contentDiv.appendChild(dcfDiv);
    }

    // Render Chart
    if (memo.historical_financials && document.getElementById('finChart')) {
        renderFinChart(memo.historical_financials);
    }

    // Apply Edit Mode if active
    if (isEditMode) {
        applyEditMode();
    }
}

function renderFinChart(data) {
    const ctx = document.getElementById('finChart').getContext('2d');
    if (currentChart) currentChart.destroy();

    // Reverse for chronological order if needed (assuming input is newest first)
    const chartData = [...data].reverse();

    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.map(d => d.period),
            datasets: [
                {
                    label: 'Revenue',
                    data: chartData.map(d => d.revenue),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'EBITDA',
                    data: chartData.map(d => d.ebitda),
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function recalculateDCF() {
    if (!currentMemoData || !currentMemoData.dcf_analysis) return;

    const wacc = parseFloat(document.getElementById('dcf-wacc').value) / 100;
    const growth = parseFloat(document.getElementById('dcf-growth').value) / 100;
    // const exitMult = parseFloat(document.getElementById('dcf-exit').value); // Not used in simple logic yet, just placeholder

    // Simple mock recalculation logic preserving the original structure
    // In real app, re-run full NPV logic. Here we scale based on WACC change.

    const originalWacc = currentMemoData.dcf_analysis.wacc;
    const basePrice = currentMemoData.dcf_analysis.share_price;
    const baseEV = currentMemoData.dcf_analysis.enterprise_value;

    // Sensitivity: Higher WACC -> Lower Price
    // Simple inverse relation factor for demo
    const factor = originalWacc / wacc;
    // Growth factor
    const growthFactor = (1 + growth) / (1 + currentMemoData.dcf_analysis.growth_rate);

    const newPrice = basePrice * factor * growthFactor;
    const newEV = baseEV * factor * growthFactor;

    document.getElementById('dcf-share-price').textContent = `$${newPrice.toFixed(2)}`;
    document.getElementById('dcf-ev').textContent = formatCurrency(newEV);
}

function renderRiskQuant(memo) {
    const container = document.getElementById('risk-quant-panel');
    container.innerHTML = '';

    const div = document.createElement('div');
    div.className = 'quant-card';

    // Simple PD Model (Merton-ish)
    // Assets, Volatility (mock), Debt
    const debt = memo.historical_financials?.[0]?.total_liabilities || 1000;
    const equity = memo.historical_financials?.[0]?.total_equity || 2000;
    const assets = debt + equity;
    const leverage = debt / assets;

    // Mock PD based on leverage
    let pd = leverage * 0.05;
    if (memo.risk_score < 50) pd += 0.05;

    // LGD (Loss Given Default) - Unsecured assumption
    const lgd = 0.45;

    // EL (Expected Loss)
    const el = debt * pd * lgd;

    div.innerHTML = `
        <h4><i class="fas fa-calculator"></i> Credit Risk Model</h4>
        <div style="font-size: 0.75rem; color: #ccc;">
            <div style="margin-bottom: 5px;">Probability of Default (PD): <span style="color: var(--neon-cyan); float: right;">${(pd*100).toFixed(2)}%</span></div>
            <div style="margin-bottom: 5px;">Loss Given Default (LGD): <span style="color: var(--neon-cyan); float: right;">${(lgd*100).toFixed(0)}%</span></div>
            <div style="margin-bottom: 10px;">Exposure at Default (EAD): <span style="color: var(--neon-cyan); float: right;">${formatCurrency(debt)}</span></div>
            <div style="border-top: 1px solid #555; padding-top: 5px; margin-top: 5px;">
                Expected Loss (EL): <span style="color: var(--neon-magenta); float: right; font-weight: bold;">${formatCurrency(el)}</span>
            </div>
        </div>

        <h4 style="margin-top: 20px;"><i class="fas fa-sliders-h"></i> Sensitivity</h4>
        <div class="slider-container">
            <span>Asset Volatility</span>
            <input type="range" min="10" max="90" value="30">
        </div>
    `;

    container.appendChild(div);
}

function toggleEditMode() {
    isEditMode = !isEditMode;
    const btn = document.getElementById('edit-btn');

    if (isEditMode) {
        btn.classList.add('editing-active');
        btn.innerHTML = '<i class="fas fa-save"></i> SAVE CHANGES';
        applyEditMode();
    } else {
        btn.classList.remove('editing-active');
        btn.innerHTML = '<i class="fas fa-pen"></i> EDIT MODE';
        disableEditMode();
    }
}

function applyEditMode() {
    document.querySelectorAll('.editable-content').forEach(el => {
        el.contentEditable = "true";
        el.classList.add('editing-active-area'); // Add CSS class for visuals
    });
}

function disableEditMode() {
    document.querySelectorAll('.editable-content').forEach(el => {
        el.contentEditable = "false";
        el.classList.remove('editing-active-area');
    });
}

function handleExport() {
    window.print();
}

function formatCurrency(val) {
    if (val === undefined || val === null) return '-';
    if (Math.abs(val) >= 1000) return '$' + (val / 1000).toFixed(1) + 'B';
    return '$' + val.toFixed(0) + 'M';
}

window.viewEvidence = function(docId) {
    const label = document.getElementById('doc-id-label');
    label.textContent = `[${docId}]`;

    const canvas = document.getElementById('pdf-mock-canvas');
    canvas.innerHTML = '';

    const page = document.createElement('div');
    page.style.padding = '20px';
    page.style.fontSize = '8px';
    page.style.color = '#333';
    page.style.fontFamily = 'Times New Roman, serif';
    page.style.lineHeight = '1.2';
    page.style.opacity = '0.6';

    let text = "";
    for(let i=0; i<30; i++) text += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    page.textContent = text;

    const highlight = document.createElement('div');
    highlight.className = 'highlight-box';
    let hash = 0;
    for (let i = 0; i < docId.length; i++) hash = docId.charCodeAt(i) + ((hash << 5) - hash);
    const top = 10 + (Math.abs(hash) % 50);

    highlight.style.top = `${top}%`;
    highlight.style.left = '10%';
    highlight.style.width = '80%';
    highlight.style.height = '10%';

    const hlLabel = document.createElement('div');
    hlLabel.textContent = "MATCH 98%";
    hlLabel.style.backgroundColor = "yellow";
    hlLabel.style.color = "black";
    hlLabel.style.fontSize = "8px";
    hlLabel.style.fontWeight = "bold";
    hlLabel.style.position = "absolute";
    hlLabel.style.top = "-12px";
    hlLabel.style.left = "0";
    highlight.appendChild(hlLabel);

    canvas.appendChild(page);
    canvas.appendChild(highlight);
}
