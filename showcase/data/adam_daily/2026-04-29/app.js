
let modules = [];
let chartDefinitions = [];
let currentModuleId = 1;
let isAIOpen = false;
let isDefcon = false;


        // Global Telemetry State Object
        ;

        function applyChartDefaults() {
            Chart.defaults.color = '#64748b';
            Chart.defaults.font.family = "'Fira Code', monospace";
            Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(9, 15, 26, 0.95)';
            Chart.defaults.plugins.tooltip.titleFont = { family: "'Oswald', sans-serif", size: 14 };
            Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.1)';
            Chart.defaults.plugins.tooltip.borderWidth = 1;
            Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
        }
        
        let probChartInstance = null;
        let radarChartInstance = null;
        let gammaChartInstance = null;
        let bifurcationChartInstance = null;
        let baseRadarData = [99, 88, 95, 95, 80];

        // Helper for Chart colors
        window.getRGBA = function(varName, alpha=1) {
            const rgb = getComputedStyle(document.body).getPropertyValue(varName).trim();
            return `rgba(${rgb.split(' ').join(',')}, ${alpha})`;
        };

        // ---------------------------------------------------------
        // DATA MODULES (24 Modules Total)
        // ---------------------------------------------------------
        ;

        // ---------------------------------------------------------
        // STATE & INITIALIZATION
        // ---------------------------------------------------------
        let currentModuleId = 1;
        const navEl = document.getElementById('moduleNav');
        const mobileNavEl = document.getElementById('mobileNav');
        const contentEl = document.getElementById('contentArea');
        const aiPanel = document.getElementById('aiPanel');
        const chatLog = document.getElementById('chatLog');
        const aiInput = document.getElementById('aiInput');
        const rootStyle = getComputedStyle(document.body);
        let defconState = false;

        // Interactive Methods Implementation
        window.triggerPreset = function(val) {
            if(!val) return;
            const input = document.getElementById('aiInput');
            if(input) {
                input.value = val;
                if(document.getElementById('aiPanel').classList.contains('translate-x-full')) {
                    toggleAI();
                }
                handleAIQuery({ preventDefault: () => {} });
            }
        };

        window.toggleAI = function() {
            const panel = document.getElementById('aiPanel');
            if(panel.classList.contains('translate-x-full')) {
                panel.classList.remove('translate-x-full');
                panel.classList.add('translate-x-0');
                document.getElementById('aiInput').focus();
            } else {
                panel.classList.add('translate-x-full');
                panel.classList.remove('translate-x-0');
            }
        };

        window.toggleDefcon = function() {
            defconState = !defconState;
            if(defconState) {
                document.body.classList.add('defcon-active');
            } else {
                document.body.classList.remove('defcon-active');
            }
        };

        window.changeTheme = function(val) {
            document.body.className = `h-screen w-screen flex flex-col selection:bg-primary selection:text-black relative text-sm ${val}`;
            if(defconState) document.body.classList.add('defcon-active');
        };

        window.generateEmail = function() {
            const output = document.getElementById('emailCodeOutput');
            const preview = document.getElementById('emailPreviewFrame');
            const copyBtn = document.getElementById('copyEmailBtn');
            if(!output || !preview || !copyBtn) return;
            
            const m = modules.find(x => x.id === currentModuleId) || modules[0];
            const html = `
            <!DOCTYPE html>
            <html>
            <head><style>body { font-family: monospace; background: #010308; color: #cbd5e1; padding: 20px; }</style></head>
            <body>
                <h2 style="color: #00e5ff;">ADAM OMNIPRESENCE DAILY BRIEF</h2>
                <div style="border-left: 4px solid #ffaa00; padding-left: 15px; margin-bottom: 20px;">
                    <p><strong>SYSTEM SPX:</strong> ${window.systemTelemetry.spx.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p>
                    <p><strong>10Y YIELD:</strong> ${window.systemTelemetry.yield.toFixed(3)}%</p>
                    <p><strong>BRENT CRUDE:</strong> $${window.systemTelemetry.brent.toFixed(2)}</p>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 5px;">
                    <h3 style="color: #00e5ff; margin-top: 0;">Featured Intel: ${m.title}</h3>
                    <p>${m.subtitle}</p>
                </div>
                <p style="font-size: 10px; color: #475569; margin-top: 30px;">AUTO-GENERATED BY ADAM SYSTEM // DATE: 2026.04.29</p>
            </body>
            </html>`;
            output.value = html;
            preview.srcdoc = html;
            copyBtn.disabled = false;
        };
        
        window.copyEmailHTML = function() {
            const output = document.getElementById('emailCodeOutput');
            if(output) {
                navigator.clipboard.writeText(output.value);
                const btn = document.getElementById('copyEmailBtn');
                if(btn) {
                    btn.innerText = 'COPIED!';
                    setTimeout(() => btn.innerText = '2. Copy Raw HTML', 2000);
                }
            }
        };

        window.renderTicker = function() {
            const tickerText = document.getElementById('tickerText');
            if(tickerText) {
                tickerText.innerHTML = `
                    <span>[SYS] NORMAL OPERATIONS</span> • 
                    <span class="text-primary">SPX 7,138.80 (-0.49%)</span> • 
                    <span class="text-accent">10Y YIELD 4.415% (+6.3bps)</span> • 
                    <span>BRENT $111.78</span> • 
                    <span class="text-primary">BTC $75,935</span> • 
                    <span>VIX 18.60</span> • 
                    <span class="text-secondary">OAS 3.05%</span> • 
                    <span>HORMUZ: 0 TRANSITS (24HR)</span> • 
                    <span class="text-accent">DEFCON RATING: MODERATE</span>
                `.repeat(5);
            }
        };

        window.initCharts = function() {
            // Probability Chart (Bar)
            const ctxProb = document.getElementById('probabilityChart');
            if(ctxProb) {
                probChartInstance = new Chart(ctxProb, {
                    type: 'bar',
                    data: {
                        labels: ['Dovish Pivot', 'H4L', 'Inflation Shock', 'Recession'],
                        datasets: [{
                            label: 'Probability %',
                            data: [12, 45, 33, 10],
                            backgroundColor: [
                                'rgba(0, 229, 255, 0.6)', 
                                'rgba(255, 170, 0, 0.6)', 
                                'rgba(255, 30, 56, 0.6)', 
                                'rgba(181, 60, 255, 0.6)'
                            ],
                            borderColor: [
                                'rgba(0, 229, 255, 1)', 
                                'rgba(255, 170, 0, 1)', 
                                'rgba(255, 30, 56, 1)', 
                                'rgba(181, 60, 255, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }

            // Radar Chart
            const ctxRadar = document.getElementById('radarChart');
            if(ctxRadar) {
                radarChartInstance = new Chart(ctxRadar, {
                    type: 'radar',
                    data: {
                        labels: ['Geopolitics', 'Credit Risk', 'Liquidity', 'Valuation', 'Policy'],
                        datasets: [{
                            label: 'Systemic Threat Level',
                            data: baseRadarData,
                            backgroundColor: 'rgba(255, 30, 56, 0.2)',
                            borderColor: 'rgba(255, 30, 56, 1)',
                            pointBackgroundColor: 'rgba(255, 30, 56, 1)',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255,255,255,0.1)' },
                                grid: { color: 'rgba(255,255,255,0.1)' },
                                pointLabels: { color: '#94a3b8', font: { family: "'Fira Code', monospace", size: 10 } },
                                min: 0, max: 100,
                                ticks: { display: false }
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }
        };

        window.scrambleChart = function() {
            if(probChartInstance) {
                probChartInstance.data.datasets[0].data = [
                    Math.floor(Math.random() * 30),
                    Math.floor(Math.random() * 50) + 20,
                    Math.floor(Math.random() * 60) + 10,
                    Math.floor(Math.random() * 20)
                ];
                probChartInstance.update();
            }
        };

        window.updateRadarChart = function(index, value) {
            if(radarChartInstance) {
                radarChartInstance.data.datasets[0].data[index] = value * 100;
                radarChartInstance.update();
            }
        };

        window.recalibrateModel = function() {
            const els = ['eqBar1', 'eqBar2', 'eqBar3', 'dbBar1', 'dbBar2', 'dbBar3'];
            els.forEach(id => {
                const el = document.getElementById(id);
                if(el) {
                    el.style.width = '0%';
                    setTimeout(() => {
                        el.style.width = el.getAttribute('data-target');
                    }, 500);
                }
            });
        };

        window.simulateLiveKPIs = function() {
            setInterval(() => {
                const spxEl = document.getElementById('kpiSpx');
                if(spxEl) {
                    const change = (Math.random() - 0.5) * 5;
                    window.systemTelemetry.spx += change;
                    spxEl.innerText = window.systemTelemetry.spx.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    spxEl.classList.remove('flash-pos', 'flash-neg');
                    void spxEl.offsetWidth; // trigger reflow
                    spxEl.classList.add(change >= 0 ? 'flash-pos' : 'flash-neg');
                }
            }, 3500);
        };

        window.runBootSequence = function() {
            const bootLogs = document.getElementById('bootLogs');
            const bootScreen = document.getElementById('bootScreen');
            if(!bootLogs || !bootScreen) return;
            
            const logs = [
                "INITIALIZING ADAM OMNIPRESENCE V50.0...",
                "LOADING NEURO-SYMBOLIC ROUTER...",
                "SYNCING MACRO TELEMETRY...",
                "ESTABLISHING SECURE CONNECTION...",
                "BYPASSING MEATSPACE LIMITERS...",
                "SYSTEM READY."
            ];
            
            let delay = 0;
            logs.forEach((log) => {
                setTimeout(() => {
                    const line = document.createElement('div');
                    line.className = 'boot-text-line';
                    line.innerText = `> ${log}`;
                    bootLogs.appendChild(line);
                }, delay);
                delay += 400 + Math.random() * 400;
            });
            
            setTimeout(() => {
                bootScreen.style.opacity = '0';
                setTimeout(() => bootScreen.style.display = 'none', 800);
            }, delay + 500);
        };

        // Action for Module 23
        window.pushTelemetryData = function() {
            window.systemTelemetry.spx = parseFloat(document.getElementById('input_spx').value) || window.systemTelemetry.spx;
            window.systemTelemetry.yield = parseFloat(document.getElementById('input_yield').value) || window.systemTelemetry.yield;
            window.systemTelemetry.brent = parseFloat(document.getElementById('input_brent').value) || window.systemTelemetry.brent;
            window.systemTelemetry.btc = parseFloat(document.getElementById('input_btc').value) || window.systemTelemetry.btc;
            window.systemTelemetry.vix = parseFloat(document.getElementById('input_vix').value) || window.systemTelemetry.vix;
            window.systemTelemetry.oas = parseFloat(document.getElementById('input_oas').value) || window.systemTelemetry.oas;
            
            document.getElementById('kpiSpx').innerText = window.systemTelemetry.spx.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('kpiYield').innerText = window.systemTelemetry.yield.toFixed(3) + '%';
            document.getElementById('kpiBrent').innerText = '$' + window.systemTelemetry.brent.toFixed(2);
            document.getElementById('kpiBrent').setAttribute('data-text', '$' + window.systemTelemetry.brent.toFixed(2));
            document.getElementById('kpiBtc').innerText = window.systemTelemetry.btc.toLocaleString('en-US');
            document.getElementById('kpiVix').innerText = window.systemTelemetry.vix.toFixed(2);
            document.getElementById('kpiOas').innerText = window.systemTelemetry.oas.toFixed(2) + '%';

            const container = document.getElementById('toastContainer');
            if(container) {
                const toast = document.createElement('div');
                toast.className = `bg-panel/90 border border-primary/50 p-3 shadow-[0_5px_15px_rgba(0,0,0,0.5)] flex items-start gap-3 backdrop-blur transform transition-all duration-500 clip-path-toast w-full text-white`;
                toast.innerHTML = `<span class="font-mono text-xs text-primary">> Core Telemetry Data Overwritten Successfully.</span>`;
                container.appendChild(toast);
                setTimeout(() => toast.remove(), 3000);
            }
        }

        function initApp() {
            applyChartDefaults();
            renderTicker();
            initCharts();
            renderNav();
            loadModule(1);
            
            setInterval(() => {
                const d = new Date();
                document.getElementById('sysClock').innerText = 
                    d.getUTCHours().toString().padStart(2, '0') + ':' + 
                    d.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                    d.getUTCSeconds().toString().padStart(2, '0') + ' UTC';
            }, 1000);

            simulateLiveKPIs();
            runBootSequence();
            initDraggable();

            document.getElementById('prevBtn').onclick = () => { if(currentModuleId > 1) loadModule(currentModuleId - 1); };
            document.getElementById('nextBtn').onclick = () => { if(currentModuleId < modules.length) loadModule(currentModuleId + 1); };
        }

        // Initialize HTML5 Drag and Drop for widgets
        function initDraggable() {
            let draggedEl = null;
            
            const makeDraggable = (containerId) => {
                const container = document.getElementById(containerId);
                if(!container) return;
                
                const widgets = container.querySelectorAll('.bento-draggable');
                widgets.forEach(widget => {
                    widget.setAttribute('draggable', true);
                    
                    widget.addEventListener('dragstart', (e) => {
                        draggedEl = widget;
                        e.dataTransfer.effectAllowed = 'move';
                        widget.style.opacity = '0.4';
                        e.stopPropagation();
                    });
                    
                    widget.addEventListener('dragend', (e) => {
                        widget.style.opacity = '1';
                        draggedEl = null;
                        document.querySelectorAll('.bento-placeholder').forEach(el => el.classList.remove('bento-placeholder'));
                    });
                    
                    widget.addEventListener('dragover', (e) => {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = 'move';
                        if (draggedEl && draggedEl !== widget && draggedEl.parentNode === widget.parentElement) {
                            widget.classList.add('bento-placeholder');
                        }
                        return false;
                    });
                    
                    widget.addEventListener('dragleave', (e) => {
                        widget.classList.remove('bento-placeholder');
                    });
                    
                    widget.addEventListener('drop', (e) => {
                        e.preventDefault();
                        widget.classList.remove('bento-placeholder');
                        
                        if (draggedEl && draggedEl !== widget && draggedEl.parentNode === widget.parentElement) {
                            const parent = widget.parentElement;
                            const allChildren = Array.from(parent.children);
                            const draggedIndex = allChildren.indexOf(draggedEl);
                            const targetIndex = allChildren.indexOf(widget);
                            
                            if (draggedIndex < targetIndex) {
                                parent.insertBefore(draggedEl, widget.nextSibling);
                            } else {
                                parent.insertBefore(draggedEl, widget);
                            }
                        }
                        return false;
                    });
                });
            };

            makeDraggable('kpiGrid');
            makeDraggable('chartGrid');
        }

        function renderNav() {
            navEl.innerHTML = ''; mobileNavEl.innerHTML = ''; document.getElementById('paginationDots').innerHTML = '';
            let currentGroup = '';
            
            modules.forEach(m => {
                if (m.group && m.group !== currentGroup) {
                    currentGroup = m.group;
                    const groupHeader = document.createElement('div');
                    groupHeader.className = 'text-[9px] text-primary/50 font-mono mt-6 mb-2 px-4 uppercase tracking-[0.2em] font-bold';
                    groupHeader.innerText = currentGroup;
                    navEl.appendChild(groupHeader);
                    
                    const optGroup = document.createElement('optgroup');
                    optGroup.label = currentGroup;
                    mobileNavEl.appendChild(optGroup);
                }
                
                const btn = document.createElement('button');
                btn.className = `nav-item w-full text-left px-4 py-3 rounded flex flex-col mb-1 relative overflow-hidden group cursor-pointer ${m.id === currentModuleId ? 'bg-primary/10 border-l-2 border-primary text-white' : 'text-slate-400 border-l-2 border-transparent hover:bg-white/5'}`;
                btn.onclick = () => loadModule(m.id);
                
                btn.innerHTML = `
                    <div class="pl-2 transition-all duration-200 w-full">
                        <span class="font-bold text-[13px] font-display tracking-wide uppercase group-hover:text-white block">
                            <span class="text-[10px] font-mono opacity-50 group-hover:text-primary transition-colors pr-1">${m.id.toString().padStart(2,'0')}</span> ${m.title}
                        </span>
                        <span class="opacity-60 text-[10px] truncate block font-mono mt-0.5">${m.subtitle}</span>
                    </div>`;
                navEl.appendChild(btn);
                
                const opt = document.createElement('option');
                opt.value = m.id; opt.innerText = `MOD ${m.id.toString().padStart(2,'0')} - ${m.title}`;
                if(m.id === currentModuleId) opt.selected = true;
                const lastOptGroup = mobileNavEl.querySelector('optgroup:last-of-type');
                if (lastOptGroup) lastOptGroup.appendChild(opt); else mobileNavEl.appendChild(opt);

                const footDot = document.createElement('div');
                footDot.className = `w-1.5 h-1.5 rounded-full cursor-pointer transition-all duration-300 m-[1px] ${m.id === currentModuleId ? 'bg-primary scale-150 shadow-[0_0_8px_rgba(var(--color-primary),1)]' : 'bg-slate-600 hover:bg-slate-400'}`;
                footDot.onclick = () => loadModule(m.id);
                document.getElementById('paginationDots').appendChild(footDot);
            });
        }

        function loadModule(id) {
            currentModuleId = id;
            const m = modules.find(x => x.id === id);
            if (!m) return;
            
            document.getElementById('prevBtn').disabled = id === 1;
            document.getElementById('nextBtn').disabled = id === modules.length;
            document.getElementById('moduleTitleDisplay').innerText = m.title;
            document.getElementById('moduleIndicator').innerText = `MOD ${id.toString().padStart(2,'0')}/${modules.length.toString().padStart(2,'0')}`;
            
            contentEl.style.opacity = 0; 
            contentEl.style.transform = 'translateY(15px)';
            
            if(window.weightAnim) clearInterval(window.weightAnim);
            if(window.activeAnimFrame) cancelAnimationFrame(window.activeAnimFrame);
            window.activeAnimFrame = null;
            
            setTimeout(() => {
                contentEl.innerHTML = m.content;
                contentEl.style.opacity = 1; 
                contentEl.style.transform = 'translateY(0)';
                renderNav(); 
                document.getElementById('articleScrollBox').scrollTo({top: 0, behavior: 'smooth'});
                
                if(m.onLoad) m.onLoad();
            }, 300);
        }

        const aiResponses = {
            "defcon": "DEFCON 1 override active. The system has shifted into a Hostile Event framework. All macro analysis logic routing is now assuming kinetic military escalation.",
            "dark pool": "FINRA off-exchange volume confirms institutional distribution. The SPY block trades show heavy selling into the retail rally. The Order Flow Toxicity index is flashing at 58%. Routed to Module 8.",
            "prob curves": "The Monte Carlo simulation ran 10,000 paths weighting the OAS spread widening (credit risk) and physical Hormuz blockades. The resulting distribution shows a severe left-tail skew. Routed to Module 14.",
            "activist": "Radoff-JEC's bid for SEER was rejected. IMKTA's vote is in 24 hours. ASTS continues to face immense sell pressure from Rakuten. Routed to Module 7.",
            "email": "I have routed you to Module 22. Click 'Generate' to compile the inline-CSS HTML payload based on today's intelligence.",
            "architecture": "The new Enterprise AI Blueprint leverages a Neuro-Symbolic Multi-Agent Framework. It pairs Llama 3 8B strictly as a semantic router. Routed to Module 16.",
            "economics": "I have logged the system terminology and token economics in Module 21: System Appendix.",
            "judge": "Module 19 features the LLM-as-a-Judge scorecard. A completely disconnected Llama-3-70B model retroactively scores Adam's 8B router predictions.",
            "weights": "I have routed you to Module 20: SLM Weights. You can actively adjust the geopolitical, credit, and policy attention bias sliders here.",
            "network": "Routing to Module 17. The Neural Knowledge Graph is dynamically mapping geopolitical vectors against financial flows.",
            "supply": "Routing to Module 11. Analyzing global critical asset depletion. Note the extreme degradation in Neon Gas synthesis affecting lithography.",
            "cds": "Routing to Module 10. Sovereign CDS spreads are widening globally.",
            "gpu": "Routing to Module 12. The Sovereign Compute Matrix indicates US Hyperclusters are at 82% utilization, burning $4.2B/wk in capex.",
            "earnings": "Routing to Module 3/4. I have compiled the Q1 2026 capital expenditure, AI monetization metrics, and foundational model shifts for Alphabet, Microsoft, Amazon, Meta.",
            "dcf": "Routing to Module 24. I have compiled the $600B+ forward capex projections into explicit Discounted Cash Flow models for the Magnificent Seven.",
            "valuation": "Routing to Module 24: Big Tech DCF Valuations. AAPL is priced for perfection while AMZN faces a liquidity deficit.",
            "trial": "Routing to Module 20: The Musk v. Altman Trial. The litigation seeks $150 billion in disgorgement.",
            "default": "Command not recognized. Try querying 'Pull Big Tech DCF Valuations', 'Explain the Stagflationary Trap', 'Analyze Dark Pool Liquidity', 'View Sovereign CDS Monitor', or 'Adjust SLM Weights'."
        };

        window.handleAIQuery = function(e) {
            e.preventDefault();
            const aiInput = document.getElementById('aiInput');
            const chatLog = document.getElementById('chatLog');
            const query = aiInput.value.trim();
            if(!query) return;

            chatLog.insertAdjacentHTML('beforeend', `
                <div class="flex justify-end gap-3 w-full animate-[fadeIn_0.3s_ease-out]">
                    <div class="bg-primary/10 border border-primary/30 text-white p-3 rounded-lg rounded-tr-none max-w-[85%] text-sm font-mono shadow-[0_0_10px_rgba(var(--color-primary),0.1)]">
                        ${query}
                    </div>
                </div>
            `);
            aiInput.value = '';
            chatLog.scrollTop = chatLog.scrollHeight;

            const typingId = 'typing-' + Date.now();
            chatLog.insertAdjacentHTML('beforeend', `
                <div id="${typingId}" class="flex gap-3 w-full animate-[fadeIn_0.3s_ease-out]">
                    <div class="w-7 h-7 rounded border border-primary/50 bg-primary/10 flex items-center justify-center shrink-0">
                        <span class="text-[10px] font-display font-bold text-primary">A</span>
                    </div>
                    <div class="bg-panel border border-white/10 p-3.5 rounded-lg rounded-tl-none flex items-center">
                        <div class="typing-indicator"><span></span><span></span><span></span></div>
                    </div>
                </div>
            `);
            chatLog.scrollTop = chatLog.scrollHeight;

            let response = aiResponses["default"];
            const lowerQ = query.toLowerCase();
            for (const key in aiResponses) {
                if (lowerQ.includes(key)) { response = aiResponses[key]; break; }
            }

            setTimeout(() => {
                const typingEl = document.getElementById(typingId);
                if (typingEl) typingEl.remove();
                
                chatLog.insertAdjacentHTML('beforeend', `
                    <div class="flex gap-3 w-full animate-[fadeIn_0.3s_ease-out]">
                        <div class="w-7 h-7 rounded border border-primary/50 bg-primary/10 flex items-center justify-center shrink-0 shadow-[0_0_10px_rgba(var(--color-primary),0.2)]">
                            <span class="text-[10px] font-display font-bold text-primary">A</span>
                        </div>
                        <div class="bg-panel p-4 rounded-lg rounded-tl-none border-l-2 border-l-primary text-slate-300 max-w-[90%] shadow-lg">
                            <p class="leading-relaxed m-0">${response}</p>
                        </div>
                    </div>
                `);
                chatLog.scrollTop = chatLog.scrollHeight;
                
                // Auto-route
                if(lowerQ.includes('dark pool')) loadModule(6);
                else if(lowerQ.includes('eval') || lowerQ.includes('judge')) loadModule(19);
                else if(lowerQ.includes('weight')) loadModule(20);
                else if(lowerQ.includes('email')) loadModule(22);
                else if(lowerQ.includes('network') || lowerQ.includes('graph')) loadModule(17);
                else if(lowerQ.includes('supply') || lowerQ.includes('choke')) loadModule(14);
                else if(lowerQ.includes('architecture') || lowerQ.includes('neuro')) loadModule(16);
                else if(lowerQ.includes('cds')) loadModule(7);
                else if(lowerQ.includes('gpu')) loadModule(15);
                else if(lowerQ.includes("musk") || lowerQ.includes("trial")) loadModule(1);
                else if(lowerQ.includes("earnings") || lowerQ.includes("mag 7")) loadModule(3);
                else if(lowerQ.includes("dcf") || lowerQ.includes("valuation") || lowerQ.includes("apple")) loadModule(24);

            }, 800 + Math.random() * 500);
        };

        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.code === 'Space') {
                e.preventDefault();
                toggleAI();
            }
        });

        
    

async function loadDashboard() {
    try {
        const response = await fetch('data.json?v=20260429');
        const data = await response.json();
        window.systemTelemetry = data.systemTelemetry;
        modules = data.modules;
        if (data.chartDefinitions) {
            chartDefinitions = data.chartDefinitions;
        }
    } catch (e) {
        console.error("Failed to load data:", e);
    }
    
    // Inject shell HTML
    document.getElementById('dashboard-shell').innerHTML = `<!-- BOOT SCREEN -->
    <div id="bootScreen" class="fixed inset-0 z-[10000] bg-void flex flex-col items-start justify-end p-10 font-mono text-xs text-primary shadow-[inset_0_0_150px_rgba(0,0,0,1)]">
        <div id="bootLogs" class="space-y-2 w-full max-w-4xl flex flex-col items-start"></div>
    </div>

    <!-- TOAST NOTIFICATION CONTAINER -->
    <div id="toastContainer" class="fixed bottom-14 right-4 z-[9995] flex flex-col gap-3 items-end pointer-events-none w-80"></div>

    <div class="crt-overlay"></div>
    <div class="vignette"></div>
    <div class="absolute inset-0 bg-hex-pattern opacity-30 z-0 pointer-events-none"></div>

    <!-- HEADER / COMMAND BAR -->
    <header class="h-16 border-b border-core-border bg-panel/90 backdrop-blur-md flex items-center justify-between px-4 shrink-0 z-30 relative shadow-2xl transition-colors duration-500" id="mainHeader">
        <div class="absolute bottom-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-primary/50 to-transparent"></div>
        
        <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded border border-primary/50 bg-primary/10 flex items-center justify-center box-glow-primary shadow-[0_0_15px_rgba(var(--color-primary),0.4)] transition-all cursor-crosshair">
                <span class="font-display font-bold text-primary text-xl leading-none tracking-tighter animate-pulse-slow">A50</span>
            </div>
            <div class="flex flex-col justify-center">
                <h1 class="font-display font-bold text-white text-lg uppercase leading-none tracking-widest m-0">Adam Omnipresence</h1>
                <span class="font-mono text-[10px] text-primary uppercase tracking-widest mt-0.5" id="headerSubtitle">v50.0 // ML Reinforcement Core</span>
            </div>
        </div>

        <div class="hidden md:flex flex-1 max-w-xl mx-8 relative group">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg class="h-4 w-4 text-primary/70 group-focus-within:text-primary transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </div>
            <input type="text" id="commandInput" class="block w-full pl-10 pr-3 py-1.5 border border-core-border rounded bg-void/50 text-white font-mono text-xs placeholder-slate-500 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all shadow-inner" placeholder="Query macro matrix or execute override (e.g., /route PLTR)..." onkeypress="if(event.key==='Enter') triggerPreset(this.value)">
            <div class="absolute inset-y-0 right-0 pr-2 flex items-center gap-2">
                <span class="text-[9px] font-mono text-primary/80 uppercase px-1.5 border border-primary/30 rounded bg-primary/10">Ctrl+Space</span>
            </div>
        </div>

        <div class="flex items-center gap-4">
            
            <button onclick="loadModule(23)" class="hidden xl:flex items-center gap-2 bg-void border border-secondary/50 text-secondary hover:bg-secondary hover:text-black px-3 py-1.5 font-mono text-[10px] font-bold transition-all uppercase cursor-pointer rounded">
                Inventory / Control
            </button>

            <!-- DEFCON TOGGLE -->
            <button onclick="toggleDefcon()" id="defconBtn" class="hidden xl:flex items-center gap-2 bg-void border border-accent/50 text-accent hover:bg-accent hover:text-black px-3 py-1.5 font-mono text-[10px] font-bold transition-all uppercase cursor-pointer rounded">
                <span class="w-2 h-2 rounded-full bg-accent animate-pulse-fast"></span> DEFCON Override
            </button>

            <div class="hidden lg:flex items-center gap-3">
                <span class="text-[10px] font-mono text-slate-400">SKIN:</span>
                <select id="themeSelector" onchange="changeTheme(this.value)" class="bg-void border border-core-border text-primary font-mono text-xs p-1.5 rounded outline-none focus:border-primary cursor-pointer transition-colors hover:border-primary/50">
                    <option value="theme-cyan">Sys: Cyber Cyan</option>
                    <option value="theme-red">Sys: Terminal Red</option>
                    <option value="theme-green">Sys: Matrix Green</option>
                    <option value="theme-purple">Sys: Neon Purple</option>
                    <option value="theme-gold">Sys: Aurum Gold</option>
                    <option value="theme-cobalt">Sys: Cobalt Abyss</option>
                </select>
            </div>

            <div class="hidden lg:flex items-center gap-4 font-mono text-[10px]">
                <div class="flex flex-col text-right">
                    <span class="text-slate-400">SYS_CLOCK</span>
                    <span class="text-white font-bold tracking-wider" id="sysClock">00:00:00 UTC</span>
                </div>
                <div class="h-8 w-px bg-core-border"></div>
                <div class="flex flex-col text-right">
                    <span class="text-slate-400" id="rlhfLabel">RLHF_STATUS</span>
                    <span class="text-primary font-bold tracking-wider animate-pulse glow-primary" id="rlhfStatus">OPTIMIZING</span>
                </div>
            </div>
            
            <button onclick="toggleAI()" class="clip-path-angled bg-primary/10 border border-primary text-primary hover:bg-primary hover:text-black px-6 py-2 font-display text-sm font-bold transition-all box-glow-primary tracking-widest uppercase cursor-pointer relative z-50">
                Engage AI
            </button>
        </div>
    </header>

    <!-- MAIN BENTO GRID LAYOUT -->
    <main class="flex-1 flex overflow-hidden relative z-10 p-3 md:p-4 gap-4">
        
        <!-- LEFT COLUMN: Navigation & Telemetry -->
        <aside class="w-72 flex flex-col gap-4 hidden xl:flex shrink-0 h-full">
            <!-- Nav Routing -->
            <div class="glass-panel rounded flex-1 border-t-2 border-t-primary flex flex-col overflow-hidden shadow-lg bento-widget min-h-[300px]">
                <div class="p-3 border-b border-core-border bg-black/40 flex justify-between items-center cursor-pointer" onclick="loadModule(23)" title="Click for System Control">
                    <h2 class="font-mono text-[10px] text-slate-400 tracking-widest uppercase m-0 hover:text-primary transition-colors">Data Topography ⚙️</h2>
                    <span class="w-1.5 h-1.5 bg-primary rounded-full animate-pulse shadow-[0_0_8px_rgba(var(--color-primary),1)]"></span>
                </div>
                <nav id="moduleNav" class="flex-1 p-2 space-y-1 overflow-y-auto scrollbar-hide"></nav>
            </div>

            <!-- Live Telemetry Feed -->
            <div class="glass-panel rounded h-[35%] min-h-[200px] border-t-2 border-t-secondary flex flex-col overflow-hidden relative shrink-0 shadow-lg bento-widget">
                <div class="absolute top-0 right-0 w-24 h-24 bg-secondary/10 blur-2xl rounded-full pointer-events-none"></div>
                <div class="p-3 border-b border-core-border bg-black/40 flex justify-between items-center z-10">
                    <h2 class="font-mono text-[10px] text-secondary tracking-widest uppercase flex items-center gap-2 m-0">
                        <span class="w-1.5 h-1.5 bg-secondary rounded-full animate-pulse"></span> Terminal Telemetry
                    </h2>
                </div>
                <div class="flex-1 bg-void p-3 overflow-hidden relative z-10 border-t border-black shadow-inner flex flex-col justify-end">
                    <div id="liveTelemetryBox" class="font-mono text-[9px] leading-relaxed space-y-2 h-full flex flex-col justify-end overflow-hidden">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div>
        </aside>

        <!-- CENTER GRID: Dashboard Bento Box -->
        <section class="flex-1 flex flex-col gap-4 overflow-y-auto pr-2 scrollbar-hide relative z-10" id="mainScrollArea">
            
            <!-- Top Row: Key Metrics (Flex to allow dynamic sizing & Drag/Drop) -->
            <div class="flex flex-wrap gap-3 shrink-0" id="kpiGrid">
                <div class="glass-panel interactive-panel p-4 rounded border-b-2 border-b-accent flex flex-col justify-between bento-widget bento-draggable flex-1 min-w-[150px]" onclick="loadModule(1)">
                    <span class="font-mono text-[10px] text-slate-400 uppercase tracking-widest mb-1 flex justify-between pointer-events-none">SPX 500 <span class="text-accent">▼</span></span>
                    <div class="font-display text-2xl xl:text-3xl font-bold text-white tracking-tight pointer-events-none" id="kpiSpx">7,138.80</div>
                    <div class="font-mono text-[10px] text-accent mt-1 pointer-events-none">-0.49%</div>
                </div>
                <div class="glass-panel interactive-panel p-4 rounded border-b-2 border-b-accent flex flex-col justify-between box-glow-accent bento-widget bento-draggable flex-1 min-w-[150px]" onclick="loadModule(2)">
                    <span class="font-mono text-[10px] text-slate-400 uppercase tracking-widest mb-1 flex justify-between pointer-events-none">10Y Backbone <span class="text-accent">▲</span></span>
                    <div class="font-display text-2xl xl:text-3xl font-bold text-accent glow-accent tracking-tight pointer-events-none" id="kpiYield">4.415%</div>
                    <div class="font-mono text-[10px] text-accent mt-1 pointer-events-none">+6.3 bps (SURGE)</div>
                </div>
                <div class="glass-panel interactive-panel p-4 rounded border-b-2 border-b-secondary flex flex-col justify-between box-glow-secondary bento-widget bento-draggable flex-1 min-w-[150px]" onclick="loadModule(13)">
                    <span class="font-mono text-[10px] text-slate-400 uppercase tracking-widest mb-1 flex justify-between pointer-events-none">Brent Entropy <span class="text-secondary">▲</span></span>
                    <div class="font-display text-2xl xl:text-3xl font-bold text-secondary tracking-tight glow-secondary glitch-text pointer-events-none" id="kpiBrent" data-text="$111.78">$111.78</div>
                    <div class="font-mono text-[10px] text-secondary mt-1 pointer-events-none">+0.47% (8th DAY)</div>
                </div>
                <div class="glass-panel interactive-panel p-4 rounded border-b-2 border-b-primary flex flex-col justify-between bento-widget bento-draggable flex-1 min-w-[150px]" onclick="loadModule(2)">
                    <span class="font-mono text-[10px] text-slate-400 uppercase tracking-widest mb-1 flex justify-between pointer-events-none">BTC Tether <span class="text-primary">▲</span></span>
                    <div class="font-display text-2xl xl:text-3xl font-bold text-white tracking-tight pointer-events-none" id="kpiBtc">75,935</div>
                    <div class="font-mono text-[10px] text-primary mt-1 pointer-events-none">-0.50% (LIQ DRAIN)</div>
                </div>
                <div class="glass-panel interactive-panel p-4 rounded border-b-2 border-b-primary flex flex-col justify-between bento-widget bento-draggable flex-1 min-w-[150px]" onclick="loadModule(2)">
                    <span class="font-mono text-[10px] text-slate-400 uppercase tracking-widest mb-1 flex justify-between pointer-events-none">VIX Daemon <span class="text-white">▲</span></span>
                    <div class="font-display text-2xl xl:text-3xl font-bold text-white tracking-tight pointer-events-none" id="kpiVix">18.60</div>
                    <div class="font-mono text-[10px] text-primary mt-1 pointer-events-none">+4.0% (WAKING)</div>
                </div>
                <div class="glass-panel interactive-panel p-4 rounded border-b-2 border-b-accent flex flex-col justify-between box-glow-accent bento-widget bento-draggable flex-1 min-w-[150px]" onclick="loadModule(8)">
                    <span class="font-mono text-[10px] text-slate-400 uppercase tracking-widest mb-1 flex justify-between pointer-events-none">HY OAS Spread <span class="text-accent">▲</span></span>
                    <div class="font-display text-2xl xl:text-3xl font-bold text-white tracking-tight pointer-events-none" id="kpiOas">3.05%</div>
                    <div class="font-mono text-[10px] text-accent mt-1 pointer-events-none">+9 bps (LEAKING)</div>
                </div>
            </div>

            <!-- Middle Row: Persistent Charts -->
            <div class="flex flex-wrap gap-4 shrink-0 min-h-[250px]" id="chartGrid">
                <div class="glass-panel rounded border border-core-border flex flex-col relative overflow-hidden shadow-lg group bento-widget bento-draggable flex-[2] min-w-[300px]">
                    <div class="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                    <div class="p-2.5 border-b border-core-border bg-black/30 flex justify-between items-center z-10 shrink-0 pointer-events-none">
                        <div class="flex items-center gap-3">
                            <h2 class="font-display text-sm text-white tracking-wide uppercase m-0">Macro Convergence Matrix (T+7)</h2>
                            <span class="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                        </div>
                        <button onclick="scrambleChart(); event.stopPropagation();" class="text-[9px] font-mono text-primary px-2 py-1 rounded bg-primary/10 border border-primary/30 hover:bg-primary hover:text-black transition-all cursor-pointer pointer-events-auto">UPDATE MATRIX</button>
                    </div>
                    <div class="flex-1 p-2 relative z-10 bg-void/50 w-full h-full min-h-[200px] pointer-events-none">
                        <div class="chart-container"><canvas id="probabilityChart"></canvas></div>
                    </div>
                </div>

                <div class="glass-panel rounded border border-core-border flex flex-col relative shadow-lg group bento-widget bento-draggable flex-1 min-w-[250px]">
                    <div class="absolute inset-0 bg-accent/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                    <div class="p-2.5 border-b border-core-border bg-black/30 z-10 shrink-0 flex justify-between items-center pointer-events-none">
                        <h2 class="font-display text-sm text-white tracking-wide uppercase m-0">Threat Topology</h2>
                        <span class="text-[9px] font-mono text-accent animate-pulse">REAL-TIME</span>
                    </div>
                    <div class="flex-1 p-2 relative z-10 flex items-center justify-center bg-void/50 w-full h-full min-h-[200px] pointer-events-none">
                        <div class="chart-container"><canvas id="radarChart"></canvas></div>
                    </div>
                </div>
            </div>

            <!-- Bottom Row: Dynamic Content Module Area -->
            <div class="flex flex-col flex-1 min-h-[600px] bento-widget w-full">
                
                <select id="mobileNav" onchange="loadModule(parseInt(this.value))" class="xl:hidden w-full bg-panel border border-core-border text-primary font-bold text-sm rounded p-3 mb-4 font-mono outline-none shadow-[0_0_10px_rgba(var(--color-primary),0.1)] focus:border-primary appearance-none cursor-pointer"></select>

                <div class="glass-panel rounded border border-core-border flex flex-col shadow-[0_15px_40px_rgba(0,0,0,0.8)] flex-1 overflow-hidden relative transition-all duration-300 h-full" id="moduleContainer">
                    <div class="p-4 border-b border-core-border bg-black/60 flex flex-wrap gap-2 justify-between items-center shrink-0 sticky top-0 z-20">
                        <h2 class="font-display text-2xl text-white tracking-widest uppercase glow-primary m-0 leading-none" id="moduleTitleDisplay">Loading Matrix...</h2>
                        <span class="font-mono text-[10px] text-primary border border-primary/30 px-3 py-1 rounded bg-primary/10 box-glow-primary shrink-0 transition-all" id="moduleIndicator">MOD --/--</span>
                    </div>
                    
                    <div class="flex-1 p-6 md:p-10 bg-gradient-to-b from-void/80 to-panel/90 overflow-y-auto relative z-10 scrollbar-hide h-full" id="articleScrollBox">
                        <article id="contentArea" class="prose prose-invert prose-slate max-w-4xl mx-auto transition-all duration-500 ease-out opacity-0 translate-y-4">
                            <!-- Module content injected here -->
                        </article>
                    </div>

                    <!-- Footer Pagination -->
                    <div class="p-4 border-t border-core-border bg-black/90 flex justify-between items-center shrink-0 z-20 relative">
                        <button id="prevBtn" class="group px-4 py-2 bg-void border border-core-border text-slate-300 rounded text-xs font-mono font-bold hover:text-white hover:border-primary/50 transition-all disabled:opacity-20 disabled:cursor-not-allowed flex items-center gap-2 cursor-pointer">
                            <span class="group-hover:-translate-x-1 transition-transform">←</span> PREV
                        </button>
                        <div class="flex gap-1.5 flex-wrap justify-center max-w-[50%]" id="paginationDots"></div>
                        <button id="nextBtn" class="group px-6 py-2 bg-primary text-black rounded text-xs font-mono font-bold hover:bg-white hover:shadow-[0_0_15px_rgba(var(--color-primary),0.8)] transition-all disabled:opacity-20 disabled:cursor-not-allowed flex items-center gap-2 clip-path-angled cursor-pointer">
                            NEXT <span class="group-hover:translate-x-1 transition-transform">→</span>
                        </button>
                    </div>
                </div>
            </div>

        </section>

        <!-- RIGHT SLIDE-OUT: Adam AI Console -->
        <aside id="aiPanel" class="fixed right-0 top-16 bottom-8 w-full md:w-[480px] glass-panel border-l border-primary/30 transform translate-x-full transition-transform duration-500 ease-[cubic-bezier(0.19,1,0.22,1)] z-50 flex flex-col shadow-[-20px_0_50px_rgba(0,0,0,0.8)] bg-panel/95 backdrop-blur-xl">
            <div class="p-4 border-b border-primary/20 flex justify-between items-center bg-black/60 shrink-0">
                <div class="flex items-center gap-3">
                    <div class="relative">
                        <div class="w-3 h-3 rounded-full bg-primary animate-pulse glow-primary"></div>
                        <div class="absolute inset-0 bg-primary rounded-full animate-ping opacity-40"></div>
                    </div>
                    <div>
                        <h2 class="font-display text-base font-bold text-white tracking-widest uppercase leading-none m-0">Adam Terminal</h2>
                        <p class="text-[9px] text-primary font-mono uppercase tracking-wider mt-1 m-0">Neuro-Symbolic Agent</p>
                    </div>
                </div>
                <button onclick="toggleAI()" class="text-slate-400 hover:text-white p-1 transition-colors cursor-pointer bg-transparent border-none text-xl">
                    ✕
                </button>
            </div>
            
            <div id="chatLog" class="flex-1 overflow-y-auto p-5 flex flex-col gap-6 text-sm font-sans scroll-smooth bg-void/80 scrollbar-hide pb-8">
                <div class="flex gap-3">
                    <div class="w-7 h-7 rounded border border-primary/50 bg-primary/10 flex items-center justify-center shrink-0 shadow-[0_0_10px_rgba(var(--color-primary),0.2)]">
                        <span class="text-[10px] font-display font-bold text-primary">A</span>
                    </div>
                    <div class="bg-panel p-4 rounded-lg rounded-tl-none border border-white/10 text-slate-300 max-w-[90%] shadow-lg relative">
                        <div class="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-primary to-transparent rounded-t-lg"></div>
                        <p class="text-[10px] font-mono text-primary mb-2 uppercase tracking-widest font-bold">Sys_Ready</p>
                        <p class="leading-relaxed mb-3">Model parameters synched for <strong class="text-white">2026.04.29</strong>. RLHF Fine-Tuning Active. Ask me to pull the Big Tech DCFs, explain the Stagflationary Trap, view the Musk trial, or execute a macro override.</p>
                    </div>
                </div>
            </div>

            <div class="px-5 py-3 border-t border-white/5 bg-black/90 shrink-0">
                <div class="flex flex-wrap gap-2">
                    <button onclick="triggerPreset('Pull Big Tech DCF Valuations')" class="px-2 py-1.5 bg-panel border border-white/10 hover:border-primary hover:text-primary text-[9px] font-mono rounded text-slate-300 transition-all cursor-pointer">Big Tech DCF</button>
                    <button onclick="triggerPreset('Explain the Stagflationary Trap & Yield Surge')" class="px-2 py-1.5 bg-panel border border-white/10 hover:border-secondary hover:text-secondary text-[9px] font-mono rounded text-slate-300 transition-all cursor-pointer">Stagflation Trap</button>
                    <button onclick="triggerPreset('Summarize Musk v Altman Trial')" class="px-2 py-1.5 bg-panel border border-white/10 hover:border-accent hover:text-accent text-[9px] font-mono rounded text-slate-300 transition-all cursor-pointer">Musk Trial</button>
                    <button onclick="triggerPreset('Analyze GPU Compute Topology')" class="px-2 py-1.5 bg-panel border border-white/10 hover:border-core-green hover:text-core-green text-[9px] font-mono rounded text-slate-300 transition-all cursor-pointer">GPU Topology</button>
                </div>
            </div>

            <div class="p-4 border-t border-primary/20 bg-panel shrink-0 relative">
                <form id="chatForm" class="flex gap-2" onsubmit="handleAIQuery(event)">
                    <input type="text" id="aiInput" placeholder="Command Adam..." class="flex-1 bg-void border border-white/10 rounded px-4 py-2.5 text-sm text-white font-mono focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all placeholder:text-slate-600 shadow-inner" autocomplete="off">
                    <button type="submit" class="bg-primary text-black hover:bg-white px-4 py-2.5 rounded font-display text-sm font-bold transition-all shadow-[0_0_15px_rgba(var(--color-primary),0.3)] uppercase tracking-wider clip-path-angled cursor-pointer border-none">Run</button>
                </form>
            </div>
        </aside>

    </main>

    <!-- BOTTOM TICKER -->
    <footer class="h-8 border-t border-core-border bg-panel flex items-center relative z-40 shrink-0 shadow-[0_-5px_15px_rgba(0,0,0,0.5)]">
        <div class="absolute left-0 top-0 h-full bg-primary text-black z-10 px-4 flex items-center font-mono text-[10px] font-bold tracking-widest clip-path-angled shadow-[10px_0_20px_#010308] cursor-crosshair hover:bg-white transition-colors">
            GLOBAL FEED
        </div>
        <div class="w-full overflow-hidden ml-32">
            <div class="animate-ticker whitespace-nowrap text-[11px] font-mono text-slate-400 tracking-wider flex items-center gap-4" id="tickerText">
                <!-- Populated via JS -->
            </div>
        </div>
    </footer>

    <!-- LOGIC -->`;
    
    // Run initialization
    initApp();
}

document.addEventListener('DOMContentLoaded', loadDashboard);
