document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadLibrary();
        await loadAuditLog();
    } catch (e) {
        console.error("Initialization failed:", e);
    }
});

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

async function loadCreditMemo(filename) {
    // Assuming filename is relative to data/
    // The library generator produces filenames like credit_memo_TechCorp_Inc.json
    // But the fetch needs correct path
    const path = filename.includes('/') ? filename : `data/${filename}`;

    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load memo: ${path}`);
    const memo = await res.json();

    const container = document.getElementById('memo-container');
    container.innerHTML = ''; // Clear

    // Header
    const header = document.createElement('div');
    header.className = "border-b-2 border-slate-100 pb-6 mb-8";

    // Calculate color for risk
    const scoreColor = memo.risk_score < 60 ? 'text-red-600' : (memo.risk_score < 80 ? 'text-yellow-600' : 'text-emerald-600');

    header.innerHTML = `
        <div class="flex justify-between items-start">
            <h1 class="text-3xl font-bold font-serif text-slate-900 mb-2">${memo.borrower_name}</h1>
            <div class="text-right">
                <div class="text-xs text-slate-500 font-mono uppercase tracking-wider">Risk Score</div>
                <div class="text-2xl font-bold font-mono ${scoreColor}">${memo.risk_score}/100</div>
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
                <div class="text-lg font-mono font-bold text-slate-700">${memo.financial_ratios.dscr.toFixed(2)}x</div>
             </div>
             <div class="bg-slate-50 p-3 rounded border border-slate-200 text-center">
                <div class="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">Current Ratio</div>
                <div class="text-lg font-mono font-bold text-slate-700">${memo.financial_ratios.current_ratio.toFixed(2)}x</div>
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
            // Truncate docId if too long
            const displayId = docId.length > 15 ? docId.substring(0, 12) + '...' : docId;
            return `<span class="citation-pin bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-[10px] font-mono cursor-pointer hover:bg-blue-200 transition border border-blue-200 align-middle ml-1" onclick="viewEvidence('${docId}')" title="View Source: ${docId}"><i class="fas fa-search mr-1 text-[8px]"></i>${displayId}</span>`;
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

// Global function for onclick
window.viewEvidence = function(docId) {
    const viewer = document.getElementById('pdf-viewer');
    const mockPage = document.getElementById('mock-pdf-page');
    const docTitle = document.getElementById('doc-title');
    const highlight = document.getElementById('highlight-box');

    viewer.querySelector('.text-center').classList.add('hidden');
    mockPage.classList.remove('hidden');
    docTitle.textContent = docId;

    // Simulate different highlight positions based on docId simple hash
    let hash = 0;
    for (let i = 0; i < docId.length; i++) {
        hash = docId.charCodeAt(i) + ((hash << 5) - hash);
    }

    const randomTop = 10 + (Math.abs(hash) % 70);
    const randomLeft = 10 + (Math.abs(hash) % 40);

    highlight.style.top = `${randomTop}%`;
    highlight.style.left = `${randomLeft}%`;
    highlight.style.width = '40%';
    highlight.style.height = '10%';

    mockPage.classList.add('ring-2', 'ring-blue-500');
    setTimeout(() => mockPage.classList.remove('ring-2', 'ring-blue-500'), 500);
};
