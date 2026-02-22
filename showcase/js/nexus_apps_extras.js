// Nexus Apps: Extras
// Adds Nexus Hub, Log Viewer, and Theme Studio

(function() {
    const EXTRAS_CONFIG = [
        {
            name: 'NexusHub',
            icon: 'https://img.icons8.com/color/48/000000/module.png',
            title: 'Nexus Hub',
            width: 900,
            height: 650
        },
        {
            name: 'LogViewer',
            icon: 'https://img.icons8.com/color/48/000000/console.png',
            title: 'Log Viewer',
            width: 800,
            height: 500
        },
        {
            name: 'ThemeStudio',
            icon: 'https://img.icons8.com/color/48/000000/paint-palette.png',
            title: 'Theme Studio',
            width: 400,
            height: 550
        }
    ];

    function initExtras() {
        if (!window.officeOS) {
            setTimeout(initExtras, 500);
            return;
        }

        patchAppRegistry();
        injectIcons();
        console.log("Nexus Extras: Loaded.");
    }

    function patchAppRegistry() {
        const originalLaunch = window.officeOS.appRegistry.launch.bind(window.officeOS.appRegistry);

        window.officeOS.appRegistry.launch = function(appName, args) {
            if (appName === 'NexusHub') {
                launchNexusHub(args);
            } else if (appName === 'LogViewer') {
                launchLogViewer(args);
            } else if (appName === 'ThemeStudio') {
                launchThemeStudio(args);
            } else {
                originalLaunch(appName, args);
            }
        };
    }

    function injectIcons() {
        // Add all to Start Menu
        const startMenuGrid = document.querySelector('.start-menu-grid');
        if (startMenuGrid) {
            EXTRAS_CONFIG.forEach(app => {
                const el = document.createElement('div');
                el.className = 'start-menu-item';
                el.innerHTML = `<img src="${app.icon}"><span>${app.title}</span>`;
                el.addEventListener('click', () => {
                    window.officeOS.appRegistry.launch(app.name);
                    window.officeOS.toggleStartMenu();
                });
                startMenuGrid.appendChild(el);
            });
        }
    }

    // --- Nexus Hub ---
    function launchNexusHub(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Nexus Command Hub',
            icon: EXTRAS_CONFIG[0].icon,
            width: EXTRAS_CONFIG[0].width,
            height: EXTRAS_CONFIG[0].height,
            app: 'NexusHub'
        });

        const container = document.createElement('div');
        container.style.padding = '30px';
        container.style.height = '100%';
        container.style.overflowY = 'auto';
        container.style.background = 'linear-gradient(135deg, #1a1c29 0%, #0d0e15 100%)';
        container.style.color = '#fff';

        const dashboards = [
            { id: 'mission', title: 'Mission Control', desc: 'Central operational dashboard monitoring all active agents and system alerts.', file: 'mission_control.html', icon: 'https://img.icons8.com/color/96/000000/monitor.png', color: '#00bcd4' },
            { id: 'war', title: 'War Room', desc: 'Strategic command center for high-stakes decision making and crisis management.', file: 'war_room.html', icon: 'https://img.icons8.com/color/96/000000/strategy-board.png', color: '#f44336' },
            { id: 'neural', title: 'Neural Deck', desc: 'Immersive 3D visualization of system neural pathways and data flow.', file: 'neural_deck.html', icon: 'https://img.icons8.com/color/96/000000/augmented-reality.png', color: '#9c27b0' },
            { id: 'holodeck', title: 'Holodeck', desc: 'Virtual reality environment for market data simulation and visualization.', file: 'holodeck.html', icon: 'https://img.icons8.com/color/96/000000/virtual-reality.png', color: '#4caf50' },
            { id: 'archive', title: 'Market Archive', desc: 'Comprehensive historical records of market mayhem events.', file: 'market_mayhem_archive.html', icon: 'https://img.icons8.com/color/96/000000/archive.png', color: '#ff9800' },
            { id: 'sovereign', title: 'Sovereign DB', desc: 'Financial sovereignty and portfolio management suite.', file: 'sovereign_dashboard.html', icon: 'https://img.icons8.com/color/96/000000/museum.png', color: '#ffd700' }
        ];

        let gridHtml = `<h2 style="text-align:center; margin-bottom:30px; font-weight:300; letter-spacing:2px;">CORE SYSTEMS</h2>
                        <div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap:20px;">`;

        dashboards.forEach(d => {
            gridHtml += `
                <div class="hub-card" onclick="window.officeOS.appRegistry.launch('Browser', {url: '${d.file}', name: '${d.title}'})"
                     style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:20px; cursor:pointer; transition:all 0.2s; position:relative; overflow:hidden;">
                    <div style="position:absolute; top:0; left:0; width:4px; height:100%; background:${d.color};"></div>
                    <div style="text-align:center; margin-bottom:15px;">
                        <img src="${d.icon}" style="width:64px; height:64px; filter:drop-shadow(0 0 10px ${d.color});">
                    </div>
                    <h3 style="margin:0 0 10px 0; color:${d.color}; font-size:18px;">${d.title}</h3>
                    <p style="margin:0; font-size:13px; color:#aaa; line-height:1.4;">${d.desc}</p>
                </div>
            `;
        });
        gridHtml += `</div>`;

        // Add hover effect via JS since inline styles are limited
        container.innerHTML = gridHtml;
        window.officeOS.windowManager.setWindowContent(winId, container);

        // Add simple hover listeners
        const cards = container.querySelectorAll('.hub-card');
        cards.forEach(c => {
            c.addEventListener('mouseenter', () => {
                c.style.background = 'rgba(255,255,255,0.1)';
                c.style.transform = 'translateY(-5px)';
                c.style.boxShadow = '0 10px 20px rgba(0,0,0,0.5)';
            });
            c.addEventListener('mouseleave', () => {
                c.style.background = 'rgba(255,255,255,0.05)';
                c.style.transform = 'translateY(0)';
                c.style.boxShadow = 'none';
            });
        });
    }

    // --- Log Viewer ---
    function launchLogViewer(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: args.name || 'Log Viewer',
            icon: EXTRAS_CONFIG[1].icon,
            width: EXTRAS_CONFIG[1].width,
            height: EXTRAS_CONFIG[1].height,
            app: 'LogViewer'
        });

        const container = document.createElement('div');
        container.style.backgroundColor = '#111';
        container.style.color = '#ccc';
        container.style.fontFamily = 'Consolas, monospace';
        container.style.padding = '15px';
        container.style.height = '100%';
        container.style.overflowY = 'auto';
        container.style.whiteSpace = 'pre-wrap';
        container.style.fontSize = '13px';
        container.innerHTML = 'Loading log...';

        window.officeOS.windowManager.setWindowContent(winId, container);

        if (args.path) {
            fetch(args.path)
                .then(res => res.text())
                .then(text => {
                    // Colorize
                    const lines = text.split('\n');
                    const colorized = lines.map(line => {
                        if (line.includes('ERROR') || line.includes('CRITICAL')) return `<span style="color:#ff5252">${line}</span>`;
                        if (line.includes('WARN') || line.includes('WARNING')) return `<span style="color:#ffd740">${line}</span>`;
                        if (line.includes('INFO')) return `<span style="color:#69f0ae">${line}</span>`;
                        if (line.includes('DEBUG')) return `<span style="color:#40c4ff">${line}</span>`;
                        return line;
                    }).join('\n');
                    container.innerHTML = colorized;
                })
                .catch(e => container.innerHTML = `<span style="color:red">Error loading log: ${e}</span>`);
        } else {
            container.innerHTML = 'No log file specified.';
        }
    }

    // --- Theme Studio ---
    function launchThemeStudio(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Theme Studio',
            icon: EXTRAS_CONFIG[2].icon,
            width: EXTRAS_CONFIG[2].width,
            height: EXTRAS_CONFIG[2].height,
            app: 'ThemeStudio'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.style.color = 'var(--text-color)';

        const vars = [
            { label: 'Background Color', name: '--bg-color', type: 'color' },
            { label: 'Window Background', name: '--window-bg', type: 'color' },
            { label: 'Title Bar', name: '--title-bar-bg', type: 'color' },
            { label: 'Highlight Color', name: '--highlight-color', type: 'color' },
            { label: 'Text Color', name: '--text-color', type: 'color' }
        ];

        let html = `<h3>Customize Theme</h3><div style="display:flex; flex-direction:column; gap:15px;">`;

        vars.forEach(v => {
            // Get computed style to set initial value
            const val = getComputedStyle(document.documentElement).getPropertyValue(v.name).trim();
            // Simple check if it's hex or rgb, input type color needs hex
            // This is a basic implementation
            html += `
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <label>${v.label}</label>
                    <input type="text" data-var="${v.name}" value="${val}" style="width:150px; padding:5px; background:rgba(0,0,0,0.2); border:1px solid #555; color:white;">
                </div>
            `;
        });

        html += `</div>
            <div style="margin-top:20px; padding-top:10px; border-top:1px solid #444;">
                <button id="apply-theme-${winId}" class="cyber-btn" style="width:100%; padding:8px; background:var(--highlight-color); border:none; color:black; cursor:pointer; font-weight:bold;">Apply Changes</button>
                <button id="reset-theme-${winId}" class="cyber-btn" style="width:100%; padding:8px; background:transparent; border:1px solid #555; color:#aaa; cursor:pointer; margin-top:10px;">Reset to Default</button>
            </div>
        `;

        container.innerHTML = html;
        window.officeOS.windowManager.setWindowContent(winId, container);

        // Apply Logic
        container.querySelector(`#apply-theme-${winId}`).addEventListener('click', () => {
            const inputs = container.querySelectorAll('input[data-var]');
            inputs.forEach(input => {
                document.documentElement.style.setProperty(input.getAttribute('data-var'), input.value);
            });
            alert('Theme updated! (Not saved persistently in this demo)');
        });

        container.querySelector(`#reset-theme-${winId}`).addEventListener('click', () => {
             // Reload page or re-apply default theme via ThemeManager
             if(window.officeOS.themeManager) {
                 window.officeOS.themeManager.applyTheme('cyberpunk'); // Assuming default
             }
        });
    }

    initExtras();
})();
