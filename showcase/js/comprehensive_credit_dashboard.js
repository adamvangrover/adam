let memosData = [];
let chartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    fetch('data/unified_credit_memos.json')
        .then(response => response.json())
        .then(data => {
            memosData = data;
            renderSidebar();

            // Search filter
            document.getElementById('search-input').addEventListener('input', (e) => {
                const term = e.target.value.toLowerCase();
                const filtered = memosData.filter(m => {
                    const name = (m.borrower_name || '').toLowerCase();
                    const ticker = (m._metadata?.ticker || '').toLowerCase();
                    return name.includes(term) || ticker.includes(term);
                });
                renderSidebar(filtered);
            });

            // Auto-select first if available
            if (memosData.length > 0) {
                selectMemo(memosData[0]);
            }
        })
        .catch(err => {
            console.error("Failed to load memos:", err);
            document.getElementById('company-list').innerHTML = `<div class="text-red-400 text-xs p-4">Error loading data.</div>`;
        });
});

function renderSidebar(data = memosData) {
    const list = document.getElementById('company-list');
    list.innerHTML = '';

    if (data.length === 0) {
        list.innerHTML = `<div class="text-slate-500 text-xs text-center mt-4">No entities found.</div>`;
        return;
    }

    data.forEach(memo => {
        const ticker = memo._metadata?.ticker || memo.borrower_name;
        const name = memo.borrower_name;

        const item = document.createElement('div');
        item.className = 'nav-item px-3 py-2 cursor-pointer hover:bg-slate-800 rounded transition text-sm flex justify-between items-center';
        item.innerHTML = `
            <span class="truncate pr-2">${name}</span>
            <span class="text-xs mono text-slate-500 bg-slate-800 px-1.5 rounded">${ticker}</span>
        `;

        item.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            item.classList.add('active');
            selectMemo(memo);
        });

        list.appendChild(item);
    });
}

function selectMemo(memo) {
    document.getElementById('placeholder').classList.add('hidden');
    document.getElementById('memo-container').classList.remove('hidden');

    // Header
    document.getElementById('mc-name').textContent = memo.borrower_name || 'Unknown';
    document.getElementById('mc-ticker').textContent = memo._metadata?.ticker || 'N/A';
    document.getElementById('mc-sector').textContent = memo._metadata?.sector || memo.borrower_details?.sector || 'Unknown';
    document.getElementById('mc-risk-score').textContent = memo._metadata?.risk_score || memo.risk_score || '--';

    // Exec Summary
    document.getElementById('mc-summary').innerHTML = (memo.executive_summary || 'No summary available.').replace(/\n/g, '<br>');

    // System 2
    const sys2 = memo.system_two_critique;
    let sys2Html = '';
    if (sys2) {
        sys2Html += `<div class="mb-2"><span class="text-slate-400">Conviction:</span> <span class="text-purple-300 font-bold">${(sys2.conviction_score * 100).toFixed(0)}%</span> | <span class="text-slate-400">Status:</span> <span class="${sys2.verification_status === 'PASS' ? 'text-green-400' : 'text-orange-400'}">${sys2.verification_status}</span></div>`;
        sys2Html += `<ul class="list-disc pl-4 space-y-1">`;
        (sys2.critique_points || []).forEach(pt => {
            sys2Html += `<li>${pt}</li>`;
        });
        sys2Html += `</ul>`;
    } else {
        sys2Html = '<div class="italic text-slate-500">No System 2 analysis available for this entity.</div>';
    }
    document.getElementById('mc-system2').innerHTML = sys2Html;

    // Core Metrics
    let metricsHtml = '';
    if (memo.financial_ratios) {
        const fr = memo.financial_ratios;
        metricsHtml += `
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">Leverage Ratio</div><div class="text-lg text-white mono">${fr.leverage_ratio ? fr.leverage_ratio.toFixed(2) + 'x' : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">DSCR</div><div class="text-lg text-white mono">${fr.dscr ? fr.dscr.toFixed(2) + 'x' : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">EBITDA (M)</div><div class="text-lg text-white mono">$${fr.ebitda ? fr.ebitda.toLocaleString() : 'N/A'}</div></div>
        `;
    } else if (memo.historical_financials && memo.historical_financials.length > 0) {
        const latest = memo.historical_financials[0];
        metricsHtml += `
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">Leverage Ratio</div><div class="text-lg text-white mono">${latest.leverage_ratio ? latest.leverage_ratio.toFixed(2) + 'x' : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">Revenue (M)</div><div class="text-lg text-white mono">$${latest.revenue ? latest.revenue.toLocaleString() : 'N/A'}</div></div>
            <div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">EBITDA (M)</div><div class="text-lg text-white mono">$${latest.ebitda ? latest.ebitda.toLocaleString() : 'N/A'}</div></div>
        `;
    }
    // Add LGD if available
    if (memo.lgd_analysis) {
        metricsHtml += `<div class="metric-card"><div class="text-slate-500 text-[10px] uppercase font-bold mb-1">Loss Given Default</div><div class="text-lg text-orange-400 mono">${(memo.lgd_analysis.loss_given_default * 100).toFixed(1)}%</div></div>`;
    }
    document.getElementById('mc-core-metrics').innerHTML = metricsHtml;

    // Financials Chart
    renderChart(memo);

    // DCF
    let dcfHtml = '';
    if (memo.dcf_analysis) {
        const dcf = memo.dcf_analysis;
        dcfHtml = `
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div><span class="text-slate-400 text-xs">Implied Share Price:</span> <span class="text-green-400 font-bold ml-2">$${dcf.share_price ? dcf.share_price.toFixed(2) : '--'}</span></div>
                <div><span class="text-slate-400 text-xs">Enterprise Value:</span> <span class="text-white ml-2">$${dcf.enterprise_value ? (dcf.enterprise_value / 1000).toFixed(1) + 'B' : '--'}</span></div>
            </div>

            <div class="text-xs font-bold text-slate-300 uppercase tracking-widest mb-2 mt-4 border-t border-slate-700 pt-2">Model Assumptions</div>
            <div class="grid grid-cols-2 gap-2 mb-4 text-xs">
        `;

        if (memo.assumptions) {
             for (const [key, value] of Object.entries(memo.assumptions)) {
                  dcfHtml += `<div><span class="text-slate-500">${key}:</span> <span class="text-white ml-1">${value}</span></div>`;
             }
        } else {
             dcfHtml += `
                <div><span class="text-slate-500">WACC:</span> <span class="text-white ml-1">${((dcf.wacc || 0.09) * 100).toFixed(1)}%</span></div>
                <div><span class="text-slate-500">Term. Growth Rate:</span> <span class="text-white ml-1">${((dcf.growth_rate || 0.03) * 100).toFixed(1)}%</span></div>
             `;
        }
        dcfHtml += `</div>`;

        dcfHtml += `
            <div class="text-xs font-bold text-slate-300 uppercase tracking-widest mb-2 mt-2 border-t border-slate-700 pt-2">Consensus Estimates</div>
            <div class="grid grid-cols-2 gap-2 mb-4 text-xs">
        `;
        if (memo.consensus_data) {
             for (const [key, value] of Object.entries(memo.consensus_data)) {
                  dcfHtml += `<div><span class="text-slate-500">${key}:</span> <span class="text-cyan-400 font-bold ml-1">${value}</span></div>`;
             }
        } else {
            dcfHtml += `<div class="italic text-slate-500 col-span-2">No consensus data available.</div>`;
        }
        dcfHtml += `</div>`;

        dcfHtml += `
            <div class="text-xs font-bold text-slate-300 uppercase tracking-widest mb-2 mt-2 border-t border-slate-700 pt-2">Free Cash Flow Projections (M)</div>
            <div class="flex gap-2 mt-1 overflow-x-auto custom-scrollbar pb-2">
                ${(dcf.free_cash_flow || [1000,1100,1200]).slice(0, 5).map(fcf => `<div class="bg-blue-900/20 px-2 py-1 rounded border border-blue-800/50">$${Math.round(fcf)}</div>`).join('')}
            </div>
        `;
    } else {
         dcfHtml = '<div class="italic text-slate-500">No DCF model available.</div>';
    }
    document.getElementById('mc-dcf-content').innerHTML = dcfHtml;

    // PD/LGD
    let pdHtml = '';
    if (memo.pd_model) {
        const pd = memo.pd_model;
        pdHtml += `
            <div class="flex justify-between items-center mb-3 border-b border-slate-700 pb-2">
                <div>
                    <div class="text-xs text-slate-400">Implied Rating</div>
                    <div class="text-xl font-bold ${pd.implied_rating?.includes('A') ? 'text-green-400' : 'text-orange-400'}">${pd.implied_rating || pd.regulatory_rating || 'N/A'}</div>
                </div>
                <div class="text-right">
                    <div class="text-xs text-slate-400">Model Score</div>
                    <div class="text-xl font-bold text-white mono">${pd.model_score || pd.z_score || '--'}</div>
                </div>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs">
                <div class="bg-slate-800 p-2 rounded"><span class="text-slate-400 block mb-1">1Y PD</span> <span class="font-bold">${((pd.one_year_pd || pd.pd_1yr || 0) * 100).toFixed(2)}%</span></div>
                <div class="bg-slate-800 p-2 rounded"><span class="text-slate-400 block mb-1">5Y PD</span> <span class="font-bold">${((pd.five_year_pd || 0) * 100).toFixed(2)}%</span></div>
            </div>
        `;
    } else {
        pdHtml = '<div class="italic text-slate-500">No Regulatory PD model available.</div>';
    }
    document.getElementById('mc-pd-lgd-content').innerHTML = pdHtml;

    // Debt Facilities
    let debtHtml = '';
    if (memo.debt_facilities && memo.debt_facilities.length > 0) {
        debtHtml = `<div class="space-y-2 max-h-48 overflow-y-auto custom-scrollbar pr-2">`;
        memo.debt_facilities.forEach(fac => {
            debtHtml += `
                <div class="bg-slate-800/50 border border-slate-700 p-2 rounded">
                    <div class="flex justify-between items-center mb-1">
                        <span class="font-bold text-slate-200 text-xs">${fac.facility_type || 'Facility'}</span>
                        <span class="text-[10px] ${fac.snc_rating === 'Pass' ? 'text-green-400' : 'text-red-400'} border ${fac.snc_rating === 'Pass' ? 'border-green-800' : 'border-red-800'} px-1 rounded">${fac.snc_rating || 'N/A'}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-[10px]">
                        <div><span class="text-slate-500 block">Committed</span><span class="mono">$${fac.amount_committed || fac.amount || 0}M</span></div>
                        <div><span class="text-slate-500 block">Rate</span><span class="mono">${fac.interest_rate || 'N/A'}</span></div>
                        <div><span class="text-slate-500 block">Maturity</span><span class="mono">${fac.maturity_date || 'N/A'}</span></div>
                    </div>
                </div>
            `;
        });
        debtHtml += `</div>`;
    } else {
        debtHtml = '<div class="italic text-slate-500">No distinct debt facilities reported.</div>';
    }
    document.getElementById('mc-debt-content').innerHTML = debtHtml;

    // Peer Comps
    let peerHtml = '';
    if (memo.peer_comps && memo.peer_comps.length > 0) {
        memo.peer_comps.forEach(peer => {
            peerHtml += `
                <tr class="border-b border-slate-800 hover:bg-slate-800/50">
                    <td class="p-2 text-cyan-400">${peer.ticker}</td>
                    <td class="p-2 text-slate-300 truncate max-w-[120px]">${peer.name}</td>
                    <td class="p-2">${peer.ev_ebitda ? peer.ev_ebitda.toFixed(1) + 'x' : '--'}</td>
                    <td class="p-2">${peer.pe_ratio ? peer.pe_ratio.toFixed(1) + 'x' : '--'}</td>
                    <td class="p-2">${peer.leverage_ratio ? peer.leverage_ratio.toFixed(1) + 'x' : '--'}</td>
                    <td class="p-2">$${peer.market_cap ? (peer.market_cap / 1000).toFixed(1) + 'B' : '--'}</td>
                </tr>
            `;
        });
    } else {
        peerHtml = `<tr><td colspan="6" class="p-4 text-center text-slate-500 italic border-none">No peer set analysis available.</td></tr>`;
    }
    document.getElementById('mc-peers-table').innerHTML = peerHtml;
}

function renderChart(memo) {
    const ctx = document.getElementById('financialsChart');
    if (chartInstance) {
        chartInstance.destroy();
    }

    if (!memo.historical_financials || memo.historical_financials.length === 0) {
        ctx.style.display = 'none';
        return;
    }
    ctx.style.display = 'block';

    // Sort chronologically
    const hist = [...memo.historical_financials].reverse();
    const labels = hist.map(h => h.period || h.year || 'Unknown');
    const revs = hist.map(h => h.revenue || 0);
    const ebitdas = hist.map(h => h.ebitda || 0);

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Revenue',
                    data: revs,
                    backgroundColor: 'rgba(34, 211, 238, 0.2)',
                    borderColor: 'rgba(34, 211, 238, 1)',
                    borderWidth: 1,
                    order: 2
                },
                {
                    label: 'EBITDA',
                    data: ebitdas,
                    type: 'line',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono' } } }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(51, 65, 85, 0.5)' },
                    ticks: { color: '#94a3b8', callback: function(value) { return '$' + value; } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}
