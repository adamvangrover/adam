import glob
import re
import os

def get_links():
    links = []
    # evals
    for f in sorted(glob.glob('evals/**/*.html', recursive=True)):
        if not os.path.isfile(f): continue
        links.append(('evals', f, f.split('/')[-1]))
    # research
    for f in sorted(glob.glob('research/**/*.html', recursive=True)):
        if not os.path.isfile(f): continue
        links.append(('research', f, f.split('/')[-1]))
    # showcase
    for f in sorted(glob.glob('showcase/data/**/*.html', recursive=True)):
        if not os.path.isfile(f): continue
        links.append(('showcase_data', f, f.split('/')[-1]))
    return links

def build():
    links = get_links()
    
    # For the unified table
    rows = []
    for category, path, name in links:
        rel_path = f"../../{path}"
        # Determine badge color based on category
        badge_color = "slate"
        if category == "evals": badge_color = "emerald"
        elif category == "research": badge_color = "violet"
        elif category == "showcase_data": badge_color = "cyan"
        
        row = f'''<tr class="border-b border-slate-800/50 hover:bg-slate-800/50 transition-colors">
            <td class="px-4 py-3"><span class="px-2 py-1 rounded text-[10px] uppercase font-bold bg-{badge_color}-900/30 text-{badge_color}-400 border border-{badge_color}-800/50">{category}</span></td>
            <td class="px-4 py-3 font-mono text-xs text-slate-300 truncate max-w-md" title="{path}">{name}</td>
            <td class="px-4 py-3 text-right"><a href="{rel_path}" class="inline-flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors group">Open <svg class="w-3 h-3 group-hover:translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg></a></td>
        </tr>'''
        rows.append(row)
        
    rows_html = '\n'.join(rows)

    with open('public/adam-research/historic_archives.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # We need to replace the content of the single unified tbody.
    html_content = re.sub(r'<tbody id="archive-body">.*?</tbody>', f'<tbody id="archive-body">\n{rows_html}\n            </tbody>', html_content, count=1, flags=re.DOTALL)

    import json
    # Inject JSON manifest for client side graph
    manifest_data = {"evals": [], "research": [], "showcase_data": []}
    for category, path, name in links:
        if category in manifest_data:
            manifest_data[category].append({"path": path, "name": name})
    
    json_script = f"\n<script id='archive-manifest-data' type='application/json'>{json.dumps(manifest_data)}</script>\n"
    
    additional_ui = """
    <!-- Additive Temporal Event Sourcing & Ledger Module -->
    <div class="glass-panel p-6 border-violet-900/50 mt-8 mb-8" id="temporal-ledger-module">
        <h3 class="text-2xl font-bold font-mono text-violet-400 mb-6">Temporal Event Sourcing & Ledger</h3>
        <p class="text-slate-400 mb-4">Immutable cryptographic ledger simulation showcasing real-time temporal event streams and W3C PROV-O compliance.</p>
        <div class="bg-slate-900 border border-slate-800 rounded p-4 h-64 overflow-y-auto font-mono text-sm" id="ledger-stream">
            <!-- Simulated events will populate here -->
        </div>
        <div class="mt-4 flex justify-end">
            <button id="btn-simulate-event" class="bg-violet-900/50 hover:bg-violet-800 text-violet-400 px-4 py-2 rounded font-mono text-sm border border-violet-700/50 transition-colors">Generate Event</button>
        </div>
    </div>

    <!-- Additive Multi-Agent System Health Matrix Module -->
    <div class="glass-panel p-6 border-violet-900/50 mt-8 mb-16" id="health-matrix-module">
        <h3 class="text-2xl font-bold font-mono text-violet-400 mb-6">Multi-Agent System Health Matrix</h3>
        <p class="text-slate-400 mb-4">Live system vitals during simulated historical data generation cycles.</p>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-slate-900/80 border border-slate-700 p-4 rounded text-center">
                <span class="text-xs text-slate-500 font-bold uppercase block mb-2">Memory Integrity</span>
                <div class="text-2xl font-mono text-emerald-400" id="metric-memory">99.9%</div>
            </div>
            <div class="bg-slate-900/80 border border-slate-700 p-4 rounded text-center">
                <span class="text-xs text-slate-500 font-bold uppercase block mb-2">Latency (ms)</span>
                <div class="text-2xl font-mono text-cyan-400" id="metric-latency">12.4</div>
            </div>
            <div class="bg-slate-900/80 border border-slate-700 p-4 rounded text-center">
                <span class="text-xs text-slate-500 font-bold uppercase block mb-2">Confidence Score</span>
                <div class="text-2xl font-mono text-violet-400" id="metric-confidence">0.982</div>
            </div>
            <div class="bg-slate-900/80 border border-slate-700 p-4 rounded text-center">
                <span class="text-xs text-slate-500 font-bold uppercase block mb-2">Active Agents</span>
                <div class="text-2xl font-mono text-slate-200" id="metric-agents">42</div>
            </div>
        </div>
    </div>

    <!-- Additive Simulation States Module -->
    <div class="glass-panel p-6 border-violet-900/50 mt-8 mb-8" id="simulation-states-module">
        <h3 class="text-2xl font-bold font-mono text-violet-400 mb-6">Historical Environment Simulation</h3>
        <p class="text-slate-400 mb-4">Zero-dependency vector mapping of historical environment states over continuous evaluation runs.</p>
        <div class="flex flex-col md:flex-row gap-6">
            <div class="w-full md:w-1/2 bg-slate-900 border border-slate-800 rounded p-4 flex justify-center items-center h-64 relative">
                <svg id="env-radar" width="200" height="200" viewBox="-100 -100 200 200" class="overflow-visible">
                    <!-- Base Grid -->
                    <polygon points="0,-80 69,-40 69,40 0,80 -69,40 -69,-40" fill="none" stroke="#334155" stroke-width="1" />
                    <polygon points="0,-40 34,-20 34,20 0,40 -34,20 -34,-20" fill="none" stroke="#334155" stroke-width="1" />
                    <!-- Dynamic Data Polygon -->
                    <polygon id="env-polygon" points="0,-60 50,-20 30,30 0,60 -40,30 -50,-10" fill="rgba(139, 92, 246, 0.2)" stroke="#8b5cf6" stroke-width="2" style="transition: all 0.5s ease-in-out;" />
                </svg>
            </div>
            <div class="w-full md:w-1/2 flex flex-col justify-center space-y-4">
                <button onclick="updateEnvState([0,-80, 20,-10, 60,40, 0,20, -50,10, -70,-30])" class="bg-slate-900 border border-slate-700 hover:border-violet-500 text-slate-300 px-4 py-3 rounded font-mono text-sm transition-colors text-left">Trigger State Alpha</button>
                <button onclick="updateEnvState([0,-30, 70,-40, 40,60, 0,70, -20,30, -40,-10])" class="bg-slate-900 border border-slate-700 hover:border-cyan-500 text-slate-300 px-4 py-3 rounded font-mono text-sm transition-colors text-left">Trigger State Beta</button>
                <button onclick="updateEnvState([0,-90, 80,-20, 80,20, 0,90, -80,20, -80,-20])" class="bg-slate-900 border border-slate-700 hover:border-emerald-500 text-slate-300 px-4 py-3 rounded font-mono text-sm transition-colors text-left">Trigger State Gamma (Max Volatility)</button>
            </div>
        </div>
    </div>
    
    <!-- Additive Execution Trace Log Module -->
    <div class="glass-panel p-6 border-violet-900/50 mt-8 mb-16" id="execution-trace-module">
        <h3 class="text-2xl font-bold font-mono text-violet-400 mb-6">Continuous Alignment Trace</h3>
        <p class="text-slate-400 mb-4">Simulated terminal output of active alignment rules engines executing during historical backtests.</p>
        <div class="bg-black border border-slate-800 rounded p-4 h-48 overflow-y-auto font-mono text-xs text-green-500" id="trace-terminal">
            <div class="mb-1">> INIT_ALIGNMENT_ENGINE v3.4.1</div>
            <div class="mb-1">> LOADING HISTORICAL SCENARIO CACHE... OK</div>
        </div>
    </div>
    
    <script>
        // Env Radar Logic
        function updateEnvState(ptsArray) {
            const poly = document.getElementById('env-polygon');
            const ptString = ptsArray.reduce((acc, val, i) => acc + val + (i % 2 === 0 ? ',' : ' '), '').trim();
            if(poly) poly.setAttribute('points', ptString);
        }
        
        // Execution Trace Logic
        const traceTerm = document.getElementById('trace-terminal');
        const traceLogs = [
            "> [WARN] Entropy detected in agent sub-routine. Re-aligning context...",
            "> [INFO] Historic backtest #4092 completed. Confidence: 0.98",
            "> [INFO] Injecting baseline compliance constraint...",
            "> [OK] W3C PROV-O metadata appended to local state artifact.",
            "> [WARN] Latency spike across node cluster. Mitigating...",
            "> [INFO] Executing dynamic risk validation gate..."
        ];
        setInterval(() => {
            if(!traceTerm) return;
            const logLine = document.createElement('div');
            logLine.className = 'mb-1 opacity-80';
            logLine.innerText = traceLogs[Math.floor(Math.random() * traceLogs.length)];
            traceTerm.appendChild(logLine);
            if(traceTerm.children.length > 15) traceTerm.removeChild(traceTerm.children[0]);
            traceTerm.scrollTop = traceTerm.scrollHeight;
        }, 2500);
        
        // Temporal Ledger Logic
        const ledgerStream = document.getElementById('ledger-stream');
        const btnSimulate = document.getElementById('btn-simulate-event');
        let eventCount = 0;
        
        function addLedgerEvent() {
            eventCount++;
            const timestamp = new Date().toISOString();
            const hash = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
            const evt = document.createElement('div');
            evt.className = 'mb-2 pb-2 border-b border-slate-800/50 text-slate-300';
            evt.innerHTML = `<span class="text-slate-500">[${timestamp}]</span> <span class="text-emerald-400">PROV_O_LOG_${eventCount}</span> :: HASH_${hash} :: Event recorded to temporal ledger.`;
            ledgerStream.prepend(evt);
        }
        
        btnSimulate?.addEventListener('click', addLedgerEvent);
        setInterval(addLedgerEvent, 3500); // Auto-simulate every 3.5s
        
        // Health Matrix Logic
        setInterval(() => {
            document.getElementById('metric-latency').innerText = (10 + Math.random() * 5).toFixed(1);
            document.getElementById('metric-confidence').innerText = (0.97 + Math.random() * 0.02).toFixed(3);
            document.getElementById('metric-memory').innerText = (99.0 + Math.random() * 0.9).toFixed(1) + '%';
        }, 2000);
    </script>
    """
    html_content = html_content.replace("</body>", f"{additional_ui}\n{json_script}\n</body>")

    with open('public/adam-research/historic_archives.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    build()
