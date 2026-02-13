document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadLibrary();
        await loadAuditLog();
    } catch (e) {
        console.error("Initialization failed:", e);
    }
});

let currentMemoData = null; // Store for PDF export

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
                <span class="text-[9px] font-mono ${item.risk_score < 60 ? 'text-red-500' : 'text-emerald-500'} font-bold">Risk: ${item.risk_score.toFixed(0)}/100</span>
                <i class="fas fa-chevron-right text-[10px] text-slate-600 group-hover:text-blue-500 opacity-0 group-hover:opacity-100 transition"></i>
            </div>
        `;
        listContainer.appendChild(div);
    });
}

function getAttributionHTML(attribution) {
    if (!attribution) return '';
    return `
        <div class="attribution-tooltip group-hover:block hidden absolute bg-slate-900 text-slate-200 text-[10px] p-3 rounded shadow-xl border border-slate-700 w-64 z-50 left-1/2 -translate-x-1/2 top-full mt-2 text-left">
            <div class="absolute w-2 h-2 bg-slate-900 border-t border-l border-slate-700 transform rotate-45 left-1/2 -ml-1 -top-1.5"></div>
            <div class="font-bold text-blue-400 mb-1 flex justify-between">
                <span>${attribution.agent_id}</span>
                <span class="text-slate-500 text-[9px]">${attribution.model_version}</span>
            </div>
            <div class="mb-2 italic text-slate-400 border-b border-slate-700 pb-1">"${attribution.justification}"</div>
            <ul class="list-disc pl-3 text-slate-300">
                ${attribution.key_factors.map(f => `<li>${f}</li>`).join('')}
            </ul>
        </div>
    `;
}

async function loadCreditMemo(filename) {
    const path = filename.includes('/') ? filename : `data/${filename}`;

    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load memo: ${path}`);
    const memo = await res.json();
    currentMemoData = memo; // Store for PDF

    const container = document.getElementById('memo-container');
    container.innerHTML = ''; // Clear

    // Header
    const header = document.createElement('div');
    header.className = "border-b-2 border-slate-100 pb-6 mb-8";

    // Calculate color for risk
    const scoreColor = memo.risk_score < 60 ? 'text-red-600' : (memo.risk_score < 80 ? 'text-yellow-600' : 'text-emerald-600');

    // Format Price
    const priceText = memo.price_target ? `$${memo.price_target.toFixed(2)}` : "N/A";
    const priceLevel = memo.price_level || "Neutral";
    const levelColor = priceLevel === "Undervalued" ? "text-emerald-600" : (priceLevel === "Overvalued" ? "text-red-600" : "text-slate-600");

    // Format Market Cap
    const formatLargeNum = (num) => {
        if (!num) return "N/A";
        if (num > 1000000) return `$${(num/1000000).toFixed(1)}T`;
        if (num > 1000) return `$${(num/1000).toFixed(1)}B`;
        return `$${num.toFixed(1)}M`;
    };
    const marketCapText = formatLargeNum(memo.market_cap);

    // Attributions
    const attributions = memo.score_attributions || {};

    header.innerHTML = `
        <div class="flex justify-between items-start">
            <div>
                <h1 class="text-3xl font-bold font-serif text-slate-900 mb-2">${memo.borrower_name}</h1>
                <div class="flex gap-4 text-xs font-mono uppercase tracking-wider text-slate-500">
                    <span>${memo.credit_rating} RATED</span>
                    <span>|</span>
                    <span class="${levelColor}">${priceLevel.toUpperCase()}</span>
                </div>
            </div>
            <div class="text-right group relative cursor-help">
                <div class="text-xs text-slate-500 font-mono uppercase tracking-wider border-b border-dashed border-slate-300 inline-block">Risk Score</div>
                <div class="text-2xl font-bold font-mono ${scoreColor}">${memo.risk_score.toFixed(1)}/100</div>
                ${getAttributionHTML(attributions['Risk Score'])}
            </div>
        </div>

        <div class="flex justify-between text-sm text-slate-500 font-mono mt-4 border-t border-slate-100 pt-2">
            <span>Report Date: ${new Date(memo.report_date).toLocaleDateString()}</span>
            <span>ID: ${memo.borrower_name.substring(0,3).toUpperCase()}-${Math.floor(Math.random()*1000)}</span>
        </div>

        <!-- Key Metrics Grid -->
        <div class="grid grid-cols-4 gap-4 mt-6">
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center flex flex-col items-center justify-center relative group">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Total Market Value</div>
                <div class="text-lg font-mono font-bold text-slate-700">${marketCapText}</div>
                ${getAttributionHTML(attributions['Valuation'])}
             </div>

             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center flex flex-col items-center justify-center relative group cursor-help">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 border-b border-dashed border-slate-300">Price Target</div>
                <div class="text-lg font-mono font-bold ${levelColor}">${priceText}</div>
                ${getAttributionHTML(attributions['Valuation'])}
             </div>

             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center flex flex-col items-center justify-center relative group cursor-help">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 border-b border-dashed border-slate-300">Sentiment</div>
                <div class="text-lg font-mono font-bold text-slate-700">${memo.sentiment_score.toFixed(0)}%</div>
                ${getAttributionHTML(attributions['Sentiment'])}
             </div>

             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center flex flex-col items-center justify-center relative group cursor-help">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 border-b border-dashed border-slate-300">Conviction</div>
                <div class="text-lg font-mono font-bold text-slate-700">${memo.conviction_score.toFixed(0)}%</div>
                 ${getAttributionHTML(attributions['System Two'] || attributions['Sentiment'])}
             </div>
        </div>

        <!-- Debt Facilities Table -->
        ${memo.debt_ratings && memo.debt_ratings.length > 0 ? `
            <div class="mt-8 mb-8">
                <h3 class="text-xs font-bold text-slate-400 uppercase tracking-widest border-b border-slate-200 pb-2 mb-4">Debt Structure & Ratings</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm font-mono">
                        <thead>
                            <tr class="bg-slate-50 text-slate-500 border-b border-slate-200">
                                <th class="p-2 font-normal">Facility</th>
                                <th class="p-2 font-normal">Rating</th>
                                <th class="p-2 font-normal">Recovery</th>
                                <th class="p-2 font-normal text-right">Outstanding</th>
                            </tr>
                        </thead>
                        <tbody class="text-slate-700">
                            ${memo.debt_ratings.map(d => `
                                <tr class="border-b border-slate-100 last:border-0">
                                    <td class="p-2 font-bold">${d.facility_type}</td>
                                    <td class="p-2"><span class="bg-slate-100 px-1.5 py-0.5 rounded text-xs border border-slate-300">${d.rating}</span></td>
                                    <td class="p-2 text-slate-500 text-xs">${d.recovery_rating || '-'}</td>
                                    <td class="p-2 text-right">${formatLargeNum(d.amount_outstanding)}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        ` : ''}

        <!-- System Two Notes -->
        ${memo.system_two_notes ? `
        <div class="mt-4 bg-slate-100 border-l-4 border-slate-400 p-2 text-[10px] font-mono text-slate-600 italic group relative cursor-help">
            <i class="fas fa-brain mr-1"></i> ${memo.system_two_notes}
             ${getAttributionHTML(attributions['System Two'])}
        </div>` : ''}
    `;
    container.appendChild(header);

    // Sections
    memo.sections.forEach(section => {
        const secDiv = document.createElement('div');
        secDiv.className = "mb-8";

        let contentHtml = section.content.replace(/\n/g, '<br>');

        // Regex for citations [Ref: doc_id]
        contentHtml = contentHtml.replace(/\[Ref:\s*(.*?)\]/g, (match, docId) => {
            const citation = section.citations.find(c => c.doc_id === docId);
            const bboxStr = citation && citation.bbox ? JSON.stringify(citation.bbox) : null;
            const displayId = docId.length > 15 ? docId.substring(0, 12) + '...' : docId;

            return `<span class="citation-pin bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-[10px] font-mono cursor-pointer hover:bg-blue-200 transition border border-blue-200 align-middle ml-1"
                    onclick='viewEvidence("${docId}", ${bboxStr})'
                    title="View Source: ${docId}"><i class="fas fa-search mr-1 text-[8px]"></i>${displayId}</span>`;
        });

        // Enhance list items
        contentHtml = contentHtml.replace(/- (.*?)(<br>|$)/g, '<li class="ml-4 mb-2 text-slate-700 list-disc">$1</li>');

        secDiv.innerHTML = `
            <h2 class="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-200 pb-2 mb-4 flex items-center gap-2">
                <i class="fas fa-caret-right text-blue-500"></i> ${section.title}
            </h2>
            <div class="text-sm text-slate-600 leading-relaxed font-serif text-justify">
                ${contentHtml}
            </div>
        `;
        container.appendChild(secDiv);
    });
}

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

// PDF Export Function
window.exportPDF = function() {
    window.print();
};

// Global function for onclick
window.viewEvidence = function(docId, bbox) {
    const viewer = document.getElementById('pdf-viewer');
    const mockPage = document.getElementById('mock-pdf-page');
    const docTitle = document.getElementById('doc-title');
    const highlight = document.getElementById('highlight-box');

    viewer.querySelector('.text-center').classList.add('hidden');
    mockPage.classList.remove('hidden');
    docTitle.textContent = docId;

    if (bbox) {
        // Use real BBox [x0, y0, x1, y1] (0-1 normalized)
        const [x0, y0, x1, y1] = bbox;

        highlight.style.left = `${x0 * 100}%`;
        highlight.style.top = `${y0 * 100}%`;
        highlight.style.width = `${(x1 - x0) * 100}%`;
        highlight.style.height = `${(y1 - y0) * 100}%`;

        // Add label
        highlight.innerHTML = `
            <div class="absolute -top-6 left-0 bg-blue-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-t shadow-sm">
                SYSTEM 2 VERIFIED
            </div>
        `;
        highlight.className = "absolute border-2 border-blue-500/80 bg-blue-400/20 transition-all duration-500";

    } else {
        // Fallback random
        highlight.style.top = '10%';
        highlight.style.left = '10%';
        highlight.style.width = '80%';
        highlight.style.height = '10%';
    }

    mockPage.classList.add('ring-2', 'ring-blue-500');
    setTimeout(() => mockPage.classList.remove('ring-2', 'ring-blue-500'), 500);
};

// Bind Export Button
document.addEventListener('DOMContentLoaded', () => {
    const exportBtn = document.querySelector('button i.fa-download').parentElement;
    if(exportBtn) {
        exportBtn.onclick = window.exportPDF;
    }
});
