// Nexus Apps: System Utilities
// Adds System Health Monitor and Documentation Viewer

(function() {
    const APP_CONFIG = [
        {
            name: 'SystemHealth',
            icon: 'https://img.icons8.com/color/48/000000/system-task.png',
            title: 'System Health',
            width: 800,
            height: 600
        },
        {
            name: 'Documentation',
            icon: 'https://img.icons8.com/color/48/000000/help--v1.png',
            title: 'Documentation',
            width: 1000,
            height: 800
        }
    ];

    function initSystemApps() {
        if (!window.officeOS) {
            setTimeout(initSystemApps, 500);
            return;
        }

        patchAppRegistry();
        injectIcons();
        console.log("Nexus System Utilities: Loaded.");
    }

    function patchAppRegistry() {
        const originalLaunch = window.officeOS.appRegistry.launch.bind(window.officeOS.appRegistry);

        window.officeOS.appRegistry.launch = function(appName, args) {
            if (appName === 'SystemHealth') {
                launchSystemHealth(args);
            } else if (appName === 'Documentation') {
                launchDocumentation(args);
            } else {
                originalLaunch(appName, args);
            }
        };
    }

    function injectIcons() {
        // Desktop
        const desktop = document.getElementById('desktop');
        if (desktop) {
            APP_CONFIG.forEach(app => {
                const el = document.createElement('div');
                el.className = 'desktop-icon';
                el.innerHTML = `<img src="${app.icon}"><span>${app.title}</span>`;
                el.addEventListener('dblclick', () => window.officeOS.appRegistry.launch(app.name));
                desktop.appendChild(el);
            });
        }

        // Start Menu
        const startMenuGrid = document.querySelector('.start-menu-grid');
        if (startMenuGrid) {
            APP_CONFIG.forEach(app => {
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

    // --- System Health ---
    function launchSystemHealth(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'System Health Monitor',
            icon: APP_CONFIG[0].icon,
            width: APP_CONFIG[0].width,
            height: APP_CONFIG[0].height,
            app: 'SystemHealth'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.style.backgroundColor = '#1e1e1e';
        container.style.color = '#fff';
        container.style.height = '100%';
        container.style.overflowY = 'auto';

        // Mock Data for now, ideally fetch from /api/system/health if backend existed
        const mockData = {
            cpu: Math.floor(Math.random() * 30) + 10,
            memory: Math.floor(Math.random() * 40) + 20,
            disk: 45,
            agents: [
                { name: 'SystemHealthAgent', status: 'Active', latency: '12ms' },
                { name: 'FinancialAgent', status: 'Idle', latency: '-' },
                { name: 'NewsAgent', status: 'Active', latency: '145ms' }
            ]
        };

        container.innerHTML = `
            <h2 style="margin-top:0;">System Status</h2>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:15px; margin-bottom:20px;">
                <div style="background:#333; padding:15px; border-radius:5px; text-align:center;">
                    <div style="font-size:24px; color:#4caf50;">${mockData.cpu}%</div>
                    <div style="font-size:12px; color:#aaa;">CPU Usage</div>
                </div>
                <div style="background:#333; padding:15px; border-radius:5px; text-align:center;">
                    <div style="font-size:24px; color:#2196f3;">${mockData.memory}%</div>
                    <div style="font-size:12px; color:#aaa;">Memory Usage</div>
                </div>
                <div style="background:#333; padding:15px; border-radius:5px; text-align:center;">
                    <div style="font-size:24px; color:#ff9800;">${mockData.disk}%</div>
                    <div style="font-size:12px; color:#aaa;">Disk Usage</div>
                </div>
            </div>

            <h3>Active Agents</h3>
            <table style="width:100%; border-collapse:collapse; text-align:left;">
                <tr style="border-bottom:1px solid #444; color:#aaa;">
                    <th style="padding:10px;">Agent Name</th>
                    <th style="padding:10px;">Status</th>
                    <th style="padding:10px;">Latency</th>
                </tr>
                ${mockData.agents.map(a => `
                <tr style="border-bottom:1px solid #333;">
                    <td style="padding:10px;">${a.name}</td>
                    <td style="padding:10px;"><span style="color:${a.status === 'Active' ? '#4caf50' : '#888'}">●</span> ${a.status}</td>
                    <td style="padding:10px;">${a.latency}</td>
                </tr>
                `).join('')}
            </table>

            <div style="margin-top:20px; font-size:12px; color:#666;">
                Backend connectivity simulated. To enable real-time metrics, ensure SystemHealthAgent is active.
            </div>
        `;

        window.officeOS.windowManager.setWindowContent(winId, container);
    }

    // --- Documentation Viewer ---
    function launchDocumentation(args) {
        const winId = window.officeOS.windowManager.createWindow({
            title: 'Documentation Viewer',
            icon: APP_CONFIG[1].icon,
            width: APP_CONFIG[1].width,
            height: APP_CONFIG[1].height,
            app: 'Documentation'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.height = '100%';
        container.style.backgroundColor = '#fff';
        container.style.color = '#333';

        const toolbar = document.createElement('div');
        toolbar.style.padding = '10px';
        toolbar.style.borderBottom = '1px solid #ddd';
        toolbar.style.backgroundColor = '#f5f5f5';
        toolbar.innerHTML = `
            <button class="cyber-btn" id="reload-doc-${winId}" style="margin-right:10px; cursor:pointer;">Reload</button>
            <span id="doc-status-${winId}" style="color:#666; font-size:12px;">Loading...</span>
        `;

        const contentArea = document.createElement('div');
        contentArea.style.flex = '1';
        contentArea.style.overflow = 'auto';
        contentArea.style.padding = '20px';
        contentArea.style.fontFamily = 'monospace';
        contentArea.style.whiteSpace = 'pre-wrap';
        contentArea.id = `doc-content-${winId}`;

        container.appendChild(toolbar);
        container.appendChild(contentArea);
        window.officeOS.windowManager.setWindowContent(winId, container);

        const loadDoc = async () => {
             const status = document.getElementById(`doc-status-${winId}`);
             const area = document.getElementById(`doc-content-${winId}`);
             status.innerText = 'Fetching...';

             try {
                 // Try to fetch the tutorial
                 let text = "";

                 // Try multiple paths
                 const paths = ['../docs/TUTORIAL_OFFICE_NEXUS.md', 'docs/TUTORIAL_OFFICE_NEXUS.md', '/docs/TUTORIAL_OFFICE_NEXUS.md'];
                 let found = false;

                 for(let p of paths) {
                     try {
                         const res = await fetch(p);
                         if(res.ok) {
                             text = await res.text();
                             found = true;
                             break;
                         }
                     } catch(e) {}
                 }

                 if(found) {
                     area.innerText = text;
                     status.innerText = 'Loaded: TUTORIAL_OFFICE_NEXUS.md';
                 } else {
                     throw new Error("File not found in any expected location.");
                 }

             } catch(e) {
                 area.innerText = `Error loading documentation: ${e}\n\nMake sure the file exists at docs/TUTORIAL_OFFICE_NEXUS.md`;
                 status.innerText = 'Error';
             }
        };

        document.getElementById(`reload-doc-${winId}`).addEventListener('click', loadDoc);
        loadDoc();
    }

    initSystemApps();
})();
