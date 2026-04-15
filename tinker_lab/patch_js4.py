import re

with open("showcase/js/comprehensive_credit_dashboard.js", "r") as f:
    js = f.read()

# PD/LGD
js = js.replace("""
    // PD/LGD
    let pdHtml = '';
    if (memo.pd_model) {""", """
    // PD/LGD
    let pdHtml = '';
    if (memo.regulatoryAnalysis && memo.regulatoryAnalysis.facilityRatings) {
        const pd = memo.regulatoryAnalysis.facilityRatings[0];
        pdHtml += `
            <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2">
                <div>
                    <div class="text-xs text-slate-400">Internal Rating</div>
                    <div class="text-xl font-bold ${pd.internalRating?.includes('A') ? 'text-green-400' : 'text-orange-400'}">${pd.internalRating || 'N/A'}</div>
                </div>
                <div class="text-right">
                    <div class="text-xs text-slate-400">Recovery Rating</div>
                    <div class="text-xl font-bold text-white mono">${pd.rr || '--'}</div>
                </div>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs mb-3">
                <div class="bg-slate-800 p-2 rounded"><span class="text-slate-400 block mb-1">PD</span> <span class="font-bold">${((pd.pd || 0) * 100).toFixed(4)}%</span></div>
                <div class="bg-slate-800 p-2 rounded"><span class="text-slate-400 block mb-1">LGD</span> <span class="font-bold">${((pd.lgd || 0) * 100).toFixed(1)}%</span></div>
            </div>
            <div class="text-xs font-bold text-slate-300 uppercase tracking-widest mb-2 mt-2 border-t border-slate-700 pt-2">EL Simulation</div>
            <div class="grid grid-cols-2 gap-2 text-xs">
                <div class="bg-slate-800 p-2 rounded border border-orange-900/50"><span class="text-slate-400 block mb-1">Expected Loss</span> <span class="font-bold text-orange-400 mono">${((pd.el || 0) * 100).toFixed(4)}%</span></div>
                <div class="bg-slate-800 p-2 rounded border border-orange-900/50"><span class="text-slate-400 block mb-1">RWA Impact</span> <span class="font-bold text-orange-400 text-[10px] leading-tight">${memo.regulatoryAnalysis.basel_iii_rwa_impact || 'N/A'}</span></div>
            </div>
        `;
    } else if (memo.pd_model) {""")

# Debt Facilities
js = js.replace("""
    // Debt Facilities
    let debtHtml = '';
    if (memo.debt_facilities && memo.debt_facilities.length > 0) {""", """
    // Debt Facilities
    let debtHtml = '';
    if (memo.regulatoryAnalysis && memo.regulatoryAnalysis.facilityRatings) {
        debtHtml = `<div class="space-y-2 max-h-48 overflow-y-auto custom-scrollbar pr-2">`;
        memo.regulatoryAnalysis.facilityRatings.forEach(fac => {
            debtHtml += `
                <div class="bg-slate-800/50 border border-slate-700 p-2 rounded">
                    <div class="flex justify-between items-center mb-1">
                        <span class="font-bold text-slate-200 text-xs">${fac.facility || 'Facility'}</span>
                        <span class="text-[10px] text-green-400 border border-green-800 px-1 rounded">${fac.internalRating || 'N/A'}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-[10px]">
                        <div><span class="text-slate-500 block">PD</span><span class="mono">${(fac.pd * 100).toFixed(4)}%</span></div>
                        <div><span class="text-slate-500 block">LGD</span><span class="mono">${(fac.lgd * 100).toFixed(1)}%</span></div>
                        <div><span class="text-slate-500 block">EL</span><span class="mono">${(fac.el * 100).toFixed(4)}%</span></div>
                    </div>
                </div>
            `;
        });
        debtHtml += `</div>`;
    } else if (memo.debt_facilities && memo.debt_facilities.length > 0) {""")

# Peer Comps
js = js.replace("""
    // Peer Comps
    let peerHtml = '';
    if (memo.peer_comps && memo.peer_comps.length > 0) {""", """
    // Peer Comps
    let peerHtml = '';
    if (memo.peers && memo.peers.length > 0) {
        memo.peers.forEach(peer => {
            peerHtml += `
                <tr class="border-b border-slate-800 hover:bg-slate-800/50">
                    <td class="p-2 text-cyan-400">${peer}</td>
                    <td class="p-2 text-slate-300 truncate max-w-[120px]">--</td>
                    <td class="p-2">--</td>
                    <td class="p-2">--</td>
                    <td class="p-2">--</td>
                    <td class="p-2">--</td>
                </tr>
            `;
        });
    } else if (memo.peer_comps && memo.peer_comps.length > 0) {""")

with open("showcase/js/comprehensive_credit_dashboard.js", "w") as f:
    f.write(js)
