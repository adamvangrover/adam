import re

with open("showcase/js/comprehensive_credit_dashboard.js", "r") as f:
    js = f.read()

# Core Metrics
core_metrics_replace = """
    // Core Metrics
    let metricsHtml = '';
    if (memo.financials && memo.financials.historicals) {
        const h = memo.financials.historicals;
        metricsHtml += `
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">Net Debt / EBITDA</div><div class="text-lg text-white mono">${h.net_debt_to_ebitda !== undefined ? h.net_debt_to_ebitda.toFixed(2) + 'x' : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">EBITDA Margin</div><div class="text-lg text-white mono">${h.ebitda_margin !== undefined ? (h.ebitda_margin * 100).toFixed(1) + '%' : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">FCF Conversion</div><div class="text-lg text-white mono">${h.fcf_conversion !== undefined ? (h.fcf_conversion * 100).toFixed(1) + '%' : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">Revenue 2024 (M)</div><div class="text-lg text-white mono">$${h.revenue_2024 ? h.revenue_2024.toLocaleString() : 'N/A'}</div></div>
        `;
    } else if (memo.financial_ratios) {
"""

js = js.replace("""    // Core Metrics
    let metricsHtml = '';
    if (memo.financial_ratios) {""", core_metrics_replace)


# Sections
# Ensure charts inside sections are initialized. We can hook into the section rendering loop.
section_rendering_replace = """
    // Sections
    let sectionsHtml = '';
    let sectionChartIds = [];
    if (memo.sections && memo.sections.length > 0) {
        memo.sections.forEach(sec => {
            let icon = 'fa-file-alt';
            let color = 'text-slate-500';
            if (sec.title.includes('Valuation')) { icon = 'fa-calculator'; color = 'text-blue-500'; }
            if (sec.title.includes('Regulatory')) { icon = 'fa-balance-scale'; color = 'text-green-500'; }
            if (sec.title.includes('System 2') || sec.title.includes('System 2 Critique')) { icon = 'fa-brain'; color = 'text-purple-500'; }
            if (sec.title.includes('Risk')) { icon = 'fa-exclamation-triangle'; color = 'text-orange-500'; }

            // Extract canvas IDs for later chart rendering
            const canvasMatch = sec.content.match(/<canvas id=['"]([^'"]+)['"]/);
            if (canvasMatch) {
                sectionChartIds.push(canvasMatch[1]);
            }

            sectionsHtml += `
                <div class="section-box">
                    <h3 class="text-sm font-bold text-slate-300 uppercase tracking-widest mb-3 flex items-center gap-2">
                        <i class="fas ${icon} ${color}"></i> ${sec.title}
                    </h3>
                    <div class="text-sm text-slate-400 whitespace-pre-wrap">${sec.content}</div>
                </div>
            `;
        });
    }
    document.getElementById('mc-sections').innerHTML = sectionsHtml;
"""
# Find start of Sections
# Regex replace
js = re.sub(
    r"    // Sections\s+let sectionsHtml = '';\s+if \(memo\.sections.*?innerHTML = sectionsHtml;",
    section_rendering_replace.strip('\n'),
    js,
    flags=re.DOTALL
)

with open("showcase/js/comprehensive_credit_dashboard.js", "w") as f:
    f.write(js)
