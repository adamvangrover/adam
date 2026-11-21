import re

with open("showcase/js/comprehensive_credit_dashboard.js", "r") as f:
    js = f.read()

# DCF
js = js.replace("""
    // DCF
    let dcfHtml = '';
    if (memo.dcf_analysis) {""", """
    // DCF
    let dcfHtml = '';
    if (memo.valuation) {
        const dcf = memo.valuation;
        dcfHtml += `<div class="grid grid-cols-2 gap-2 text-xs mb-3">
            <div class="bg-slate-800 p-2 rounded border border-slate-700"><span class="text-slate-400 block mb-1">Base Case EV</span> <span class="text-white font-bold mono">$${dcf.baseCaseEV.toLocaleString()}M</span></div>
            <div class="bg-slate-800 p-2 rounded border border-slate-700"><span class="text-slate-400 block mb-1">Implied Value</span> <span class="text-blue-400 font-bold mono">$${dcf.dcfSensitivityMatrix[1]?.implied_price || '--'}</span></div>
        </div>`;
    } else if (memo.dcf_analysis) {""")

# Sensitivity Matrix
js = js.replace("""
    // Sensitivity Matrix
    let sensHtml = '';
    if (memo.dcf_analysis && memo.dcf_analysis.sensitivity) {""", """
    // Sensitivity Matrix
    let sensHtml = '';
    if (memo.valuation && memo.valuation.dcfSensitivityMatrix) {
        sensHtml = `<table class="w-full text-center text-xs mono border-collapse">`;
        sensHtml += `<thead><tr><th class="p-2 border border-slate-700 bg-slate-800 text-slate-400">WACC \\ Growth</th>`;

        // Extract unique TGRs
        const tgrs = [...new Set(memo.valuation.dcfSensitivityMatrix.map(item => item.tgr))];
        const waccs = [...new Set(memo.valuation.dcfSensitivityMatrix.map(item => item.wacc))];

        tgrs.forEach(g => {
            sensHtml += `<th class="p-2 border border-slate-700 bg-slate-800 text-slate-300">${(g*100).toFixed(1)}%</th>`;
        });
        sensHtml += `</tr></thead><tbody>`;

        waccs.forEach(w => {
            sensHtml += `<tr><td class="p-2 border border-slate-700 bg-slate-800 text-slate-300 font-bold">${(w*100).toFixed(1)}%</td>`;
            tgrs.forEach(g => {
                const item = memo.valuation.dcfSensitivityMatrix.find(i => i.wacc === w && i.tgr === g);
                sensHtml += `<td class="p-2 border border-slate-700 text-cyan-400 font-bold">$${item ? item.implied_price.toFixed(2) : '--'}</td>`;
            });
            sensHtml += `</tr>`;
        });
        sensHtml += `</tbody></table>`;
    } else if (memo.dcf_analysis && memo.dcf_analysis.sensitivity) {""")

with open("showcase/js/comprehensive_credit_dashboard.js", "w") as f:
    f.write(js)
