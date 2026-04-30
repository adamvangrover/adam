<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMNIPRESENCE // ADAM V50.0 // DYNAMIC SHELL</title>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;700&family=Inter:wght@300;400;600;800&family=Oswald:wght@300;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['Fira Code', 'monospace'],
                        display: ['Oswald', 'sans-serif'],
                    },
                    colors: {
                        void: '#010308',
                        panel: '#090f1a',
                        primary: 'rgb(var(--color-primary) / <alpha-value>)',
                        secondary: 'rgb(var(--color-secondary) / <alpha-value>)',
                        accent: 'rgb(var(--color-accent) / <alpha-value>)',
                        core: { cyan: '#00e5ff', red: '#ff1e38', amber: '#ffaa00', green: '#00fa9a', purple: '#b53cff', muted: '#475569', border: '#1e293b' }
                    },
                    backgroundImage: {
                        'hex-pattern': 'linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)',
                        'scanline': 'linear-gradient(to bottom, transparent 50%, rgba(0, 0, 0, 0.3) 51%)'
                    },
                    animation: {
                        'ticker': 'ticker 40s linear infinite',
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'flash-red': 'flashRed 2s infinite',
                        'glitch': 'glitch 2s linear infinite'
                    },
                    keyframes: {
                        ticker: { '0%': { transform: 'translateX(100vw)' }, '100%': { transform: 'translateX(-100%)' } },
                        flashRed: { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.1 } },
                        glitch: { '2%, 64%': { transform: 'translate(2px,0) skew(0deg)' }, '4%, 60%': { transform: 'translate(-2px,0) skew(0deg)' }, '62%': { transform: 'translate(0,0) skew(5deg)' } }
                    }
                }
            }
        }
    </script>
    
    <style>
        :root { --color-primary: 0 229 255; --color-secondary: 255 170 0; --color-accent: 255 30 56; }
        body.theme-cyan { --color-primary: 0 229 255; --color-secondary: 255 170 0; --color-accent: 255 30 56; }
        body { background-color: #010308; color: #cbd5e1; overflow: hidden; }
        /* Add the rest of your custom Adam CSS classes here (.glass-panel, .crt-overlay, .bento-widget, etc.) */
        
        .bento-widget { resize: both; overflow: auto; min-width: 150px; min-height: 80px; }
        .bento-widget::-webkit-scrollbar { display: none; }
    </style>
</head>
<body class="h-screen w-screen flex flex-col selection:bg-primary selection:text-black relative theme-cyan text-sm" id="appBody">
    
    <div class="crt-overlay"></div>
    <div class="vignette"></div>
    <div class="absolute inset-0 bg-hex-pattern opacity-30 z-0 pointer-events-none"></div>

    <header class="h-16 border-b border-core-border bg-panel/90 backdrop-blur-md flex items-center justify-between px-4 shrink-0 z-30 relative shadow-2xl">
        <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded border border-primary/50 bg-primary/10 flex items-center justify-center box-glow-primary text-primary font-display font-bold text-xl animate-pulse-slow">A50</div>
            <div>
                <h1 class="font-display font-bold text-white text-lg uppercase leading-none m-0" id="sysTitle">Adam Omnipresence</h1>
                <span class="font-mono text-[10px] text-primary uppercase mt-0.5 block" id="sysSubtitle">Initializing...</span>
            </div>
        </div>
        <div class="flex items-center gap-4 font-mono text-[10px]">
            <div class="text-right"><span class="text-slate-400">SYS_CLOCK</span><br><span class="text-white font-bold" id="sysClock">00:00:00 UTC</span></div>
            <button onclick="fetchDailyState()" class="bg-primary/10 border border-primary text-primary hover:bg-primary hover:text-black px-4 py-1.5 font-display text-xs font-bold transition-all uppercase cursor-pointer">Force Sync</button>
        </div>
    </header>

    <main class="flex-1 flex overflow-hidden relative z-10 p-3 md:p-4 gap-4">
        
        <aside class="w-72 flex flex-col gap-4 hidden xl:flex shrink-0 h-full">
            <div class="glass-panel rounded flex-1 border-t-2 border-t-primary flex flex-col overflow-hidden shadow-lg bento-widget">
                <div class="p-3 border-b border-core-border bg-black/40">
                    <h2 class="font-mono text-[10px] text-slate-400 tracking-widest uppercase m-0">Data Topography</h2>
                </div>
                <nav id="moduleNav" class="flex-1 p-2 space-y-1 overflow-y-auto scrollbar-hide"></nav>
            </div>
        </aside>

        <section class="flex-1 flex flex-col gap-4 overflow-y-auto pr-2 scrollbar-hide relative z-10" id="mainScrollArea">
            
            <div class="flex flex-wrap gap-3 shrink-0" id="kpiGrid">
                </div>

            <div class="flex flex-col flex-1 min-h-[600px] w-full">
                <div class="glass-panel rounded border border-core-border flex flex-col shadow-[0_15px_40px_rgba(0,0,0,0.8)] flex-1 overflow-hidden relative h-full">
                    <div class="p-4 border-b border-core-border bg-black/60 flex justify-between items-center z-20">
                        <h2 class="font-display text-2xl text-white tracking-widest uppercase glow-primary m-0" id="moduleTitleDisplay">Loading Matrix...</h2>
                        <span class="font-mono text-[10px] text-primary border border-primary/30 px-3 py-1 rounded bg-primary/10" id="moduleIndicator">MOD --/--</span>
                    </div>
                    
                    <div class="flex-1 p-6 md:p-10 bg-gradient-to-b from-void/80 to-panel/90 overflow-y-auto relative z-10 scrollbar-hide h-full">
                        <article id="contentArea" class="prose prose-invert prose-slate max-w-4xl mx-auto transition-all duration-500 ease-out">
                            </article>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="h-8 border-t border-core-border bg-panel flex items-center relative z-40 shrink-0">
        <div class="absolute left-0 top-0 h-full bg-primary text-black z-10 px-4 flex items-center font-mono text-[10px] font-bold">GLOBAL FEED</div>
        <div class="w-full overflow-hidden ml-32">
            <div class="animate-ticker whitespace-nowrap text-[11px] font-mono text-slate-400 tracking-wider flex items-center gap-4" id="tickerText">
                </div>
        </div>
    </footer>

    <script>
        let appState = {
            metadata: {},
            kpis: [],
            modules: [],
            currentModuleId: null
        };

        // 1. Fetch the state from a JSON file
        async function fetchDailyState() {
            try {
                // In production, you might append a timestamp query param to bust cache: `daily_state.json?v=${Date.now()}`
                const response = await fetch('daily_state.json');
                const data = await response.json();
                
                appState = data;
                hydrateShell();
                loadModule(appState.modules[0]?.id); // Load the first module automatically
            } catch (error) {
                console.error("Failed to load Adam state:", error);
                document.getElementById('moduleTitleDisplay').innerText = "SYSTEM FAILURE: OFFLINE";
            }
        }

        // 2. Hydrate the DOM with JSON data
        function hydrateShell() {
            // Meta
            document.getElementById('sysSubtitle').innerText = appState.metadata.version_string;
            document.title = `OMNIPRESENCE // ${appState.metadata.date}`;

            // KPIs
            const kpiGrid = document.getElementById('kpiGrid');
            kpiGrid.innerHTML = '';
            appState.kpis.forEach(kpi => {
                const colorClass = kpi.trend === 'up' ? 'text-core-green' : 'text-accent';
                const html = `
                    <div class="glass-panel p-4 rounded border-b-2 border-b-${kpi.theme_color} flex flex-col justify-between bento-widget bento-draggable flex-1 min-w-[150px]">
                        <span class="font-mono text-[10px] text-slate-400 uppercase mb-1">${kpi.label}</span>
                        <div class="font-display text-2xl xl:text-3xl font-bold text-white">${kpi.value}</div>
                        <div class="font-mono text-[10px] ${colorClass} mt-1">${kpi.delta}</div>
                    </div>
                `;
                kpiGrid.insertAdjacentHTML('beforeend', html);
            });

            // Navigation
            const nav = document.getElementById('moduleNav');
            nav.innerHTML = '';
            let currentGroup = '';
            
            appState.modules.forEach(m => {
                if (m.group !== currentGroup) {
                    currentGroup = m.group;
                    nav.insertAdjacentHTML('beforeend', `<div class="text-[9px] font-mono text-slate-500 uppercase tracking-widest mt-4 mb-1 px-3">${currentGroup}</div>`);
                }
                const navItem = document.createElement('div');
                navItem.className = 'nav-item p-2 rounded hover:bg-white/5 cursor-pointer text-xs font-sans text-slate-300 border-l-2 border-transparent hover:border-primary flex flex-col gap-1';
                navItem.innerHTML = `<span class="font-bold text-white">${m.title}</span><span class="text-[10px] text-slate-500 font-mono">${m.subtitle}</span>`;
                navItem.onclick = () => loadModule(m.id);
                nav.appendChild(navItem);
            });

            // Ticker
            const ticker = document.getElementById('tickerText');
            ticker.innerHTML = appState.ticker_items.map(t => `<span class="text-white">${t.asset}</span> ${t.price} <span class="${t.dir === 'up' ? 'text-core-green' : 'text-accent'}">${t.dir === 'up' ? '▲' : '▼'}</span>`).join(' &nbsp;&nbsp;&nbsp; ');
        }

        // 3. Render a specific module
        function loadModule(id) {
            if (!id) return;
            appState.currentModuleId = id;
            const mod = appState.modules.find(m => m.id === id);
            
            document.getElementById('moduleTitleDisplay').innerText = mod.title;
            document.getElementById('moduleIndicator').innerText = `MOD ${String(id).padStart(2, '0')} // ${mod.group}`;
            
            const contentArea = document.getElementById('contentArea');
            contentArea.style.opacity = 0;
            
            setTimeout(() => {
                // If the JSON contains raw HTML string, inject it. 
                // Alternatively, if it contains an object, you can map it to a JS template function here.
                contentArea.innerHTML = mod.content_html; 
                contentArea.style.opacity = 1;
                
                // Execute any lifecycle hooks required by the module (e.g., rendering charts)
                if(mod.chart_data) {
                    renderChart(mod.chart_data);
                }
            }, 300);
        }

        // Clock loop
        setInterval(() => {
            document.getElementById('sysClock').innerText = new Date().toISOString().substring(11,19) + ' UTC';
        }, 1000);

        // Boot
        window.onload = fetchDailyState;
    </script>
</body>
</html>
