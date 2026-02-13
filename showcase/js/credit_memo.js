document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadLibrary();
        await loadAuditLog();
    } catch (e) {
        console.error("Initialization failed:", e);
    }
});

// Helper for formatting millions/billions
const formatCurrency = (val) => {
    if (val === undefined || val === null) return 'N/A';
    if (Math.abs(val) >= 1000) return '$' + (val / 1000).toFixed(1) + 'B';
    return '$' + val.toFixed(0) + 'M';
};

// Helper for formatting currency no suffix
const formatCurrencyRaw = (val) => {
    if (val === undefined || val === null) return 'N/A';
    return '$' + val.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
};

async function loadLibrary() {
    const res = await fetch('data/credit_memo_library.json');
    if (!res.ok) throw new Error("Failed to load library");
    const library = await res.json();

    const listContainer = document.getElementById('library-list');
    listContainer.innerHTML = '';

    library.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = "p-3 border-b border-slate-800 hover:bg-slate-800/50 cursor-pointer transition group relative";

        div.onclick = () => {
             loadCreditMemo(item.file);
             // Visual selection state
             document.querySelectorAll('#library-list > div').forEach(d => d.classList.remove('bg-slate-800/80', 'border-l-2', 'border-blue-500'));
             div.classList.add('bg-slate-800/80', 'border-l-2', 'border-blue-500');
        };

        // Auto-load first item
        if (index === 0) {
             loadCreditMemo(item.file);
             div.classList.add('bg-slate-800/80', 'border-l-2', 'border-blue-500');
        }

        div.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-xs font-bold text-slate-300 group-hover:text-blue-400 transition truncate pr-2">${item.borrower_name}</span>
                <span class="text-[9px] text-slate-500 whitespace-nowrap">${new Date(item.report_date).toLocaleDateString()}</span>
            </div>
            <div class="text-[10px] text-slate-500 line-clamp-2 leading-tight">${item.summary}</div>
             <div class="mt-2 flex justify-between items-center">
                <span class="text-[9px] font-mono ${item.risk_score < 60 ? 'text-red-500' : 'text-emerald-500'} font-bold">Risk: ${item.risk_score}/100</span>
                <i class="fas fa-chevron-right text-[10px] text-slate-600 group-hover:text-blue-500 opacity-0 group-hover:opacity-100 transition"></i>
            </div>
        `;
        listContainer.appendChild(div);
    });
}

// Global Memo Object to hold state for tabs
let currentMemo = null;

async function loadCreditMemo(filename) {
    const path = filename.includes('/') ? filename : `data/${filename}`;

    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load memo: ${path}`);
    const memo = await res.json();
    currentMemo = memo; // Store for tab access if needed, though we render all at once usually

    // 1. Render Memo Tab
    renderMemoTab(memo);

    // 2. Render Annex A (Financials)
    renderAnnexA(memo);

    // 3. Render Annex B (DCF)
    renderAnnexB(memo);

    // 4. Render Annex C (Cap Structure)
    renderAnnexC(memo);

    // Reset to Memo Tab
    switchTab('memo');
}

function renderMemoTab(memo) {
    const container = document.getElementById('memo-container');
    container.innerHTML = '';

    // Header
    const header = document.createElement('div');
    header.className = "border-b-2 border-slate-100 pb-6 mb-8";

    // Calculate color for risk
    const scoreColor = memo.risk_score < 60 ? 'text-red-600' : (memo.risk_score < 80 ? 'text-yellow-600' : 'text-emerald-600');

    // Rating HTML
    let ratingHtml = '';
    if (memo.credit_ratings && memo.credit_ratings.length > 0) {
         ratingHtml = `<div class="mt-2 text-right space-y-1">`;
         memo.credit_ratings.forEach(r => {
             ratingHtml += `<div class="text-[10px] font-mono text-slate-400"><span class="font-bold text-slate-300">${r.agency}:</span> <span class="${r.rating.startsWith('A') ? 'text-emerald-400' : 'text-yellow-400'}">${r.rating}</span> (${r.outlook})</div>`;
         });
         ratingHtml += `</div>`;
    }

    header.innerHTML = `
        <div class="flex justify-between items-start">
            <h1 class="text-3xl font-bold font-serif text-slate-900 mb-2">${memo.borrower_name}</h1>
            <div class="text-right">
                <div class="text-xs text-slate-500 font-mono uppercase tracking-wider">Risk Score</div>
                <div class="text-2xl font-bold font-mono ${scoreColor}">${memo.risk_score}/100</div>
                ${ratingHtml}
            </div>
        </div>
        <div class="flex justify-between text-sm text-slate-500 font-mono mt-2">
            <span>Report Date: ${new Date(memo.report_date).toLocaleDateString()}</span>
            <span>ID: ${memo.borrower_name.substring(0,3).toUpperCase()}-${Math.floor(Math.random()*1000)}</span>
        </div>

        <!-- Key Metrics Grid -->
        <div class="grid grid-cols-3 gap-4 mt-6">
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Leverage</div>
                <div class="text-lg font-mono font-bold text-slate-700">${memo.financial_ratios.leverage_ratio.toFixed(2)}x</div>
             </div>
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">DSCR</div>
                <div class="text-lg font-mono font-bold text-slate-700">${memo.financial_ratios.dscr > 100 ? '>100x' : memo.financial_ratios.dscr.toFixed(2) + 'x'}</div>
             </div>
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Current Ratio</div>
                <div class="text-lg font-mono font-bold text-slate-700">${memo.financial_ratios.current_ratio.toFixed(2)}x</div>
             </div>
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Revenue</div>
                <div class="text-lg font-mono font-bold text-slate-700">${formatCurrency(memo.financial_ratios.revenue)}</div>
             </div>
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">EBITDA</div>
                <div class="text-lg font-mono font-bold text-slate-700">${formatCurrency(memo.financial_ratios.ebitda)}</div>
             </div>
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Net Income</div>
                <div class="text-lg font-mono font-bold text-slate-700">${formatCurrency(memo.financial_ratios.net_income)}</div>
             </div>
        </div>
    `;
    container.appendChild(header);

    // Sections
    memo.sections.forEach(section => {
        const secDiv = document.createElement('div');
        secDiv.className = "mb-8";

        let contentHtml = section.content.replace(/\n/g, '<br>');

        // Regex for citations [Ref: doc_id]
        contentHtml = contentHtml.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
            const displayId = docId.length > 15 ? docId.substring(0, 12) + '...' : docId;
            return `<span class="citation-pin bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-[10px] font-mono cursor-pointer hover:bg-blue-200 transition border border-blue-200 align-middle ml-1" onclick="viewEvidence('${docId}')" title="View Source: ${docId}"><i class="fas fa-search mr-1 text-[8px]"></i>${displayId}</span>`;
        });

        // Enhance list items
        contentHtml = contentHtml.replace(/- (.*?)(<br>|$)/g, '<li class="ml-4 mb-2 text-slate-700 list-disc">$1</li>');

        // Agent Badge Logic
        const agentColors = {
            "Writer": "bg-purple-100 text-purple-700 border-purple-200",
            "Risk Officer": "bg-red-100 text-red-700 border-red-200",
            "Quant": "bg-emerald-100 text-emerald-700 border-emerald-200",
            "Archivist": "bg-amber-100 text-amber-700 border-amber-200"
        };
        const agent = section.author_agent || "Writer";
        const badgeClass = agentColors[agent] || agentColors["Writer"];
        const badgeHtml = `<span class="ml-2 text-[9px] px-1.5 py-0.5 rounded border ${badgeClass} uppercase font-mono tracking-wider opacity-80"><i class="fas fa-robot mr-1"></i>${agent}</span>`;

        secDiv.innerHTML = `
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-200 pb-2 mb-4 flex items-center">
                <i class="fas fa-caret-right text-blue-500 mr-2"></i> ${section.title}
                ${badgeHtml}
            </h2>
            <div class="text-sm text-slate-600 leading-relaxed font-serif text-justify">
                ${contentHtml}
            </div>
        `;
        container.appendChild(secDiv);
    });
}

function renderAnnexA(memo) {
    const table = document.getElementById('financials-table');
    table.innerHTML = '';

    if (!memo.historical_financials || memo.historical_financials.length === 0) {
        table.innerHTML = '<tr><td class="p-4 text-slate-400 italic">No historical data available.</td></tr>';
        return;
    }

    // Sort by Period (FY23, FY24, FY25) - usually reverse order in list
    // Let's assume input order is correct (FY25, FY24, FY23)
    const data = memo.historical_financials;
    const columns = data.map(d => d.period);

    // Header Row
    const thead = document.createElement('thead');
    thead.className = "text-slate-500 border-b border-slate-700";
    let headerHtml = '<th class="p-3 w-1/4">Metric</th>';
    columns.forEach(col => headerHtml += `<th class="p-3 text-right">${col}</th>`);
    thead.innerHTML = `<tr>${headerHtml}</tr>`;
    table.appendChild(thead);

    // Rows
    const metrics = [
        { key: "revenue", label: "Revenue" },
        { key: "ebitda", label: "EBITDA" },
        { key: "net_income", label: "Net Income" },
        { key: "total_assets", label: "Total Assets" },
        { key: "total_liabilities", label: "Total Liabilities" },
        { key: "total_equity", label: "Total Equity" },
        { key: "leverage_ratio", label: "Leverage (x)", format: (v) => v.toFixed(2) + 'x' },
        { key: "current_ratio", label: "Current Ratio (x)", format: (v) => v.toFixed(2) + 'x' }
    ];

    const tbody = document.createElement('tbody');
    metrics.forEach(metric => {
        const tr = document.createElement('tr');
        tr.className = "border-b border-slate-800/50 hover:bg-slate-800/30";

        let rowHtml = `<td class="p-3 font-bold text-slate-300">${metric.label}</td>`;

        data.forEach(periodData => {
            const val = periodData[metric.key];
            let displayVal = 'N/A';
            if (val !== undefined && val !== null) {
                if (metric.format) {
                    displayVal = metric.format(val);
                } else {
                    displayVal = formatCurrency(val);
                }
            }
            rowHtml += `<td class="p-3 text-right text-slate-400">${displayVal}</td>`;
        });

        tr.innerHTML = rowHtml;
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
}

function renderAnnexB(memo) {
    const container = document.getElementById('dcf-container');
    container.innerHTML = '';

    if (!memo.dcf_analysis) {
        container.innerHTML = '<div class="text-slate-400 italic">No DCF analysis available.</div>';
        return;
    }

    const dcf = memo.dcf_analysis;

    // 1. Assumptions & Output Cards
    const assumptionsHtml = `
        <div class="grid grid-cols-4 gap-4">
             <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">WACC</div>
                <div class="text-xl font-bold text-white font-mono">${(dcf.wacc * 100).toFixed(1)}%</div>
             </div>
             <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Term Growth</div>
                <div class="text-xl font-bold text-white font-mono">${(dcf.growth_rate * 100).toFixed(1)}%</div>
             </div>
             <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Implied Share Price</div>
                <div class="text-xl font-bold text-emerald-400 font-mono">$${dcf.share_price.toFixed(2)}</div>
             </div>
             <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Enterprise Value</div>
                <div class="text-xl font-bold text-blue-400 font-mono">${formatCurrency(dcf.enterprise_value)}</div>
             </div>
        </div>
    `;

    // 2. Projection Table
    const tableHtml = `
        <div class="mt-8">
            <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Projected Free Cash Flow (5-Year)</h3>
            <table class="w-full text-left font-mono text-xs border-collapse">
                <thead class="text-slate-500 border-b border-slate-700">
                    <tr>
                        <th class="p-2">Period</th>
                        <th class="p-2 text-right">Year 1</th>
                        <th class="p-2 text-right">Year 2</th>
                        <th class="p-2 text-right">Year 3</th>
                        <th class="p-2 text-right">Year 4</th>
                        <th class="p-2 text-right">Year 5</th>
                    </tr>
                </thead>
                <tbody class="text-slate-300">
                    <tr class="border-b border-slate-800/50">
                        <td class="p-2 font-bold">Unlevered FCF</td>
                        ${dcf.free_cash_flow.map(v => `<td class="p-2 text-right">${formatCurrency(v)}</td>`).join('')}
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    // 3. Terminal Value
    const terminalHtml = `
        <div class="mt-8 p-4 bg-slate-800/30 rounded border border-slate-700 flex justify-between items-center">
             <div>
                <div class="text-sm font-bold text-slate-300">Terminal Value Calculation</div>
                <div class="text-[10px] text-slate-500 font-mono mt-1">Exit Multiple / Gordon Growth Method</div>
             </div>
             <div class="text-right">
                <div class="text-2xl font-bold text-white font-mono">${formatCurrency(dcf.terminal_value)}</div>
                <div class="text-[10px] text-slate-500">Present Value: ${formatCurrency(dcf.terminal_value / Math.pow(1+dcf.wacc, 5))}</div>
             </div>
        </div>
    `;

    container.innerHTML = assumptionsHtml + tableHtml + terminalHtml;
}

function renderAnnexC(memo) {
    const container = document.getElementById('cap-structure-container');
    container.innerHTML = '';

    // 1. Equity Market Data
    if (memo.equity_data) {
        const eq = memo.equity_data;
        const equityHtml = `
            <div class="mb-8">
                 <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Equity Market Data</h3>
                 <div class="grid grid-cols-3 gap-4">
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Market Cap</div>
                        <div class="text-xl font-bold text-white font-mono">${formatCurrency(eq.market_cap)}</div>
                     </div>
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Share Price</div>
                        <div class="text-xl font-bold text-emerald-400 font-mono">$${eq.share_price.toFixed(2)}</div>
                     </div>
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">P/E Ratio</div>
                        <div class="text-xl font-bold text-blue-400 font-mono">${eq.pe_ratio.toFixed(1)}x</div>
                     </div>
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Dividend Yield</div>
                        <div class="text-xl font-bold text-white font-mono">${eq.dividend_yield.toFixed(2)}%</div>
                     </div>
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Beta</div>
                        <div class="text-xl font-bold text-white font-mono">${eq.beta.toFixed(2)}</div>
                     </div>
                     <div class="bg-slate-800/50 p-4 rounded border border-slate-700">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Avg Vol (30d)</div>
                        <div class="text-xl font-bold text-slate-300 font-mono">${formatCurrencyRaw(eq.volume_avg_30d)}</div>
                     </div>
                </div>
            </div>
        `;
        container.innerHTML += equityHtml;
    }

    // 2. Debt Facilities
    if (memo.debt_facilities && memo.debt_facilities.length > 0) {
        const debtHtml = `
            <div>
                <h3 class="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Debt Facilities</h3>
                <table class="w-full text-left font-mono text-xs border-collapse">
                    <thead class="text-slate-500 border-b border-slate-700">
                        <tr>
                            <th class="p-2 w-1/4">Facility Type</th>
                            <th class="p-2 text-right">Committed</th>
                            <th class="p-2 text-right">Drawn</th>
                            <th class="p-2 text-right">Rate</th>
                            <th class="p-2 text-right">Maturity</th>
                            <th class="p-2 text-center">SNC Rating</th>
                            <th class="p-2 text-center">DRC</th>
                            <th class="p-2 text-center">LTV</th>
                            <th class="p-2 text-center">Conviction</th>
                        </tr>
                    </thead>
                    <tbody class="text-slate-300">
                        ${memo.debt_facilities.map(d => {
                            // Color logic for SNC
                            let sncClass = "bg-emerald-500/20 text-emerald-400 border-emerald-500/50";
                            if (d.snc_rating === "Special Mention") sncClass = "bg-yellow-500/20 text-yellow-400 border-yellow-500/50";
                            else if (d.snc_rating === "Substandard") sncClass = "bg-orange-500/20 text-orange-400 border-orange-500/50";
                            else if (d.snc_rating === "Doubtful" || d.snc_rating === "Loss") sncClass = "bg-red-500/20 text-red-400 border-red-500/50";

                            // DRC Bar
                            const drcPct = (d.drc || 0) * 100;
                            const drcColor = drcPct > 80 ? 'bg-emerald-500' : (drcPct > 50 ? 'bg-yellow-500' : 'bg-red-500');

                            // LTV Bar (Inverse logic usually, lower is better, but let's just show raw)
                            const ltvPct = (d.ltv || 0) * 100;
                            const ltvColor = ltvPct < 60 ? 'bg-emerald-500' : (ltvPct < 80 ? 'bg-yellow-500' : 'bg-red-500');

                            // Conviction Bar
                            const convPct = (d.conviction_score || 0) * 100;
                             const convColor = convPct > 80 ? 'bg-blue-500' : (convPct > 50 ? 'bg-purple-500' : 'bg-slate-500');

                            return `
                            <tr class="border-b border-slate-800/50 hover:bg-slate-800/30">
                                <td class="p-2 font-bold text-slate-200">${d.facility_type}</td>
                                <td class="p-2 text-right">${formatCurrency(d.amount_committed)}</td>
                                <td class="p-2 text-right text-slate-400">${formatCurrency(d.amount_drawn)}</td>
                                <td class="p-2 text-right text-emerald-400">${d.interest_rate}</td>
                                <td class="p-2 text-right">${d.maturity_date}</td>
                                <td class="p-2 text-center">
                                    <span class="px-2 py-0.5 rounded border text-[9px] uppercase font-bold tracking-wider ${sncClass}">${d.snc_rating}</span>
                                </td>
                                <td class="p-2 align-middle">
                                    <div class="w-16 bg-slate-700 h-1.5 rounded-full ml-auto mr-auto relative group cursor-help">
                                        <div class="${drcColor} h-1.5 rounded-full" style="width: ${drcPct}%"></div>
                                        <span class="absolute -top-6 left-1/2 -translate-x-1/2 bg-black text-white text-[9px] px-1 py-0.5 rounded opacity-0 group-hover:opacity-100 transition whitespace-nowrap">${drcPct.toFixed(0)}% Capacity</span>
                                    </div>
                                </td>
                                <td class="p-2 align-middle">
                                    <div class="w-16 bg-slate-700 h-1.5 rounded-full ml-auto mr-auto relative group cursor-help">
                                        <div class="${ltvColor} h-1.5 rounded-full" style="width: ${ltvPct}%"></div>
                                         <span class="absolute -top-6 left-1/2 -translate-x-1/2 bg-black text-white text-[9px] px-1 py-0.5 rounded opacity-0 group-hover:opacity-100 transition whitespace-nowrap">${ltvPct.toFixed(0)}% LTV</span>
                                    </div>
                                </td>
                                <td class="p-2 align-middle">
                                    <div class="w-16 bg-slate-700 h-1.5 rounded-full ml-auto mr-auto relative group cursor-help">
                                        <div class="${convColor} h-1.5 rounded-full" style="width: ${convPct}%"></div>
                                         <span class="absolute -top-6 left-1/2 -translate-x-1/2 bg-black text-white text-[9px] px-1 py-0.5 rounded opacity-0 group-hover:opacity-100 transition whitespace-nowrap">${convPct.toFixed(0)}% Conviction</span>
                                    </div>
                                </td>
                            </tr>
                        `}).join('')}
                    </tbody>
                </table>
            </div>
        `;
        container.innerHTML += debtHtml;
    } else {
        container.innerHTML += '<div class="text-slate-400 italic">No debt facility data available.</div>';
    }
}

// Tab Switching Logic
window.switchTab = function(tabName) {
    // Hide all
    ['memo', 'annex-a', 'annex-b', 'annex-c'].forEach(t => {
        document.getElementById(`tab-${t}`).classList.add('hidden');
        const btn = document.getElementById(`btn-tab-${t}`);
        if(btn) {
            btn.classList.remove('border-blue-500', 'text-blue-400');
            btn.classList.add('border-transparent', 'text-slate-400');
        }
    });

    // Show target
    document.getElementById(`tab-${tabName}`).classList.remove('hidden');
    const activeBtn = document.getElementById(`btn-tab-${tabName}`);
    activeBtn.classList.remove('border-transparent', 'text-slate-400');
    activeBtn.classList.add('border-blue-500', 'text-blue-400');
};

async function loadAuditLog() {
    const res = await fetch('data/credit_memo_audit_log.json');
    if (!res.ok) throw new Error("Failed to load audit log");
    const logs = await res.json();

    const tbody = document.getElementById('audit-table-body');
    tbody.innerHTML = '';

    // Reverse to show latest first, limit to 20
    logs.slice().reverse().slice(0, 20).forEach(log => {
        const tr = document.createElement('tr');
        tr.className = "hover:bg-slate-800/30 transition group cursor-default";

        const statusColor = log.validation_status === 'PASS' ? 'text-emerald-400' :
                          (log.validation_status === 'FAIL' ? 'text-red-400' : 'text-yellow-400');

        tr.innerHTML = `
            <td class="p-2 border-b border-slate-800/50 whitespace-nowrap text-slate-500 text-[10px] font-mono">${new Date(log.timestamp).toLocaleTimeString()}</td>
            <td class="p-2 border-b border-slate-800/50">
                <div class="text-slate-300 font-bold text-[10px]">${log.action}</div>
                <div class="text-slate-600 text-[9px] font-mono">Tx: ${log.transaction_id.substring(0,8)}</div>
            </td>
            <td class="p-2 border-b border-slate-800/50 ${statusColor} font-bold text-[10px] font-mono text-right">${log.validation_status}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Global function for onclick
window.viewEvidence = function(docId) {
    const viewer = document.getElementById('pdf-viewer');
    const mockPage = document.getElementById('mock-pdf-page');
    const docTitle = document.getElementById('doc-title');
    const highlight = document.getElementById('highlight-box');

    viewer.querySelector('.text-center').classList.add('hidden');
    mockPage.classList.remove('hidden');

    // Determine Type Badge
    let type = "DOC";
    let color = "bg-slate-500";
    if (docId.includes("10-K") || docId.includes("10-Q")) { type = "FILING"; color = "bg-blue-500"; }
    else if (docId.includes("Credit_Agreement") || docId.includes("Revolver")) { type = "LEGAL"; color = "bg-purple-500"; }
    else if (docId.includes("Risk") || docId.includes("Report")) { type = "RESEARCH"; color = "bg-amber-500"; }
    else if (docId.includes("Earnings")) { type = "TRANSCRIPT"; color = "bg-emerald-500"; }

    docTitle.innerHTML = `<span class="${color} text-white px-1 rounded mr-2 text-[9px] font-bold">${type}</span>${docId}`;

    // Update Snippet Text based on docId context (Simulated Retrieval)
    const pageText = mockPage.querySelectorAll('p');
    if (docId.includes("Credit_Agreement")) {
        pageText[0].textContent = "SECTION 6.01. Financial Covenants. (a) Consolidated Leverage Ratio. The Borrower will not permit the Consolidated Leverage Ratio as of the last day of any fiscal quarter to exceed 3.50 to 1.00.";
        pageText[1].textContent = "(b) Consolidated Interest Coverage Ratio. The Borrower will not permit the Consolidated Interest Coverage Ratio as of the last day of any fiscal quarter to be less than 3.00 to 1.00.";
        pageText[2].textContent = "SECTION 6.02. Liens. The Borrower will not, and will not permit any Subsidiary to, create, incur, assume or permit to exist any Lien on any property or asset now owned or hereafter acquired by it, or assign or sell any income or revenues (including accounts receivable) or rights in respect of any thereof.";
    } else if (docId.includes("10-K") || docId.includes("10-Q")) {
        pageText[0].textContent = "ITEM 1A. RISK FACTORS. Our business, financial condition and results of operations could be materially and adversely affected by a number of factors, including the following: Global economic conditions and geopolitical tensions could adversely affect our business.";
        pageText[1].textContent = "We operate in highly competitive markets and subject to rapid technological change. If we are unable to compete effectively, our financial results could be adversely affected. ";
        pageText[2].textContent = "MANAGEMENT'S DISCUSSION AND ANALYSIS. Net sales increased 11% year-over-year, primarily driven by strong performance in our Services segment and higher demand in emerging markets.";
    } else {
        pageText[0].textContent = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        pageText[1].textContent = "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.";
        pageText[2].textContent = "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.";
    }


    // Simulate different highlight positions based on docId simple hash
    let hash = 0;
    for (let i = 0; i < docId.length; i++) {
        hash = docId.charCodeAt(i) + ((hash << 5) - hash);
    }

    const randomTop = 10 + (Math.abs(hash) % 40); // Keep it higher up
    const randomLeft = 10 + (Math.abs(hash) % 40);

    highlight.style.top = `${randomTop}%`;
    highlight.style.left = `${randomLeft}%`;
    highlight.style.width = '60%';
    highlight.style.height = '15%';

    mockPage.classList.add('ring-2', 'ring-blue-500');
    setTimeout(() => mockPage.classList.remove('ring-2', 'ring-blue-500'), 500);
};
