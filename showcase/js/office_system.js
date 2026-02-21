/**
 * Office Nexus System Logic
 * Mimics a desktop environment for navigating the repository.
 */

class FileSystem {
    constructor() {
        this.manifest = null;
        this.root = [];
        this.index = {}; // Path -> Node map
    }

    async init() {
        try {
            if (window.FILESYSTEM_MANIFEST) {
                this.root = window.FILESYSTEM_MANIFEST;
                console.log('Loaded manifest from global variable (Offline Mode).');
            } else {
                const response = await fetch('data/filesystem_manifest.json');
                this.root = await response.json();
                console.log('Loaded manifest from JSON fetch.');
            }
            this.buildIndex(this.root);
            console.log('FileSystem initialized with ' + Object.keys(this.index).length + ' files.');
        } catch (e) {
            console.error('Failed to load filesystem manifest:', e);
            // Fallback for demo purposes if both fail
            this.root = [];
            alert('Error loading filesystem. Please run scripts/generate_filesystem_manifest.py');
        }
    }

    buildIndex(nodes) {
        nodes.forEach(node => {
            this.index[node.path] = node;
            if (node.children) {
                this.buildIndex(node.children);
            }
        });
    }

    readDir(path) {
        // Root case
        if (path === './' || path === '.') {
            return this.root;
        }
        const node = this.index[path];
        if (node && node.type === 'directory') {
            return node.children || [];
        }
        return [];
    }

    stat(path) {
        return this.index[path] || null;
    }

    findFiles(extension) {
        const results = [];
        const traverse = (nodes) => {
            nodes.forEach(node => {
                if (node.type === 'file' && node.name.toLowerCase().endsWith(extension.toLowerCase())) {
                    results.push(node);
                } else if (node.type === 'directory' && node.children) {
                    traverse(node.children);
                }
            });
        };
        traverse(this.root);
        return results;
    }
}

class ThemeManager {
    constructor(os) {
        this.os = os;
        this.currentThemeId = 'cyberpunk'; // Default
    }

    init() {
        const saved = localStorage.getItem('nexus_theme');
        if (saved && window.THEME_LIBRARY && window.THEME_LIBRARY[saved]) {
            this.applyTheme(saved);
        } else {
            // Apply default from library if available, else CSS fallback
            if (window.THEME_LIBRARY) this.applyTheme('cyberpunk');
        }
    }

    applyTheme(themeId) {
        if (!window.THEME_LIBRARY || !window.THEME_LIBRARY[themeId]) {
            console.error(`Theme ${themeId} not found.`);
            return;
        }

        const theme = window.THEME_LIBRARY[themeId];
        const root = document.documentElement;

        for (const [key, value] of Object.entries(theme.variables)) {
            root.style.setProperty(key, value);
        }

        this.currentThemeId = themeId;
        localStorage.setItem('nexus_theme', themeId);
        console.log(`Applied theme: ${theme.name}`);
    }

    exportTheme() {
        if (!window.THEME_LIBRARY) return "{}";
        const theme = window.THEME_LIBRARY[this.currentThemeId];
        return JSON.stringify(theme, null, 2);
    }
}

class WindowManager {
    constructor(os) {
        this.os = os;
        this.windows = [];
        this.activeWindow = null;
        this.container = document.getElementById('desktop');
        this.zIndexCounter = 100;
    }

    createWindow(options) {
        const id = 'win-' + Date.now();
        const winConfig = {
            id: id,
            title: options.title || 'Untitled',
            icon: options.icon || 'https://img.icons8.com/color/48/000000/application-window.png',
            width: options.width || 600,
            height: options.height || 400,
            x: options.x || 50 + (this.windows.length * 20),
            y: options.y || 50 + (this.windows.length * 20),
            content: options.content || '',
            app: options.app || null,
            state: 'normal' // normal, minimized, maximized
        };

        const winEl = document.createElement('div');
        winEl.className = 'window';
        winEl.id = id;
        winEl.style.width = winConfig.width + 'px';
        winEl.style.height = winConfig.height + 'px';
        winEl.style.left = winConfig.x + 'px';
        winEl.style.top = winConfig.y + 'px';
        winEl.style.zIndex = ++this.zIndexCounter;

        winEl.innerHTML = `
            <div class="title-bar">
                <div class="title-bar-title">
                    <img src="${winConfig.icon}">
                    <span>${winConfig.title}</span>
                </div>
                <div class="title-bar-controls">
                    <button class="control-btn minimize-btn">_</button>
                    <button class="control-btn maximize-btn">□</button>
                    <button class="control-btn close-btn">✕</button>
                </div>
            </div>
            <div class="window-content">
                ${winConfig.content}
            </div>
        `;

        this.container.appendChild(winEl);

        // Attach Event Listeners
        const titleBar = winEl.querySelector('.title-bar');

        // Dragging
        let isDragging = false;
        let startX, startY, initialLeft, initialTop;

        titleBar.addEventListener('mousedown', (e) => {
            if (e.target.closest('.control-btn')) return; // Don't drag if clicking buttons

            this.focusWindow(id);
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            initialLeft = winEl.offsetLeft;
            initialTop = winEl.offsetTop;

            // Prevent text selection
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging && winConfig.state !== 'maximized') {
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                winEl.style.left = (initialLeft + dx) + 'px';
                winEl.style.top = (initialTop + dy) + 'px';
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Controls
        winEl.querySelector('.close-btn').addEventListener('click', () => this.closeWindow(id));
        winEl.querySelector('.minimize-btn').addEventListener('click', () => this.minimizeWindow(id));
        winEl.querySelector('.maximize-btn').addEventListener('click', () => this.maximizeWindow(id));
        winEl.addEventListener('mousedown', () => this.focusWindow(id));

        this.windows.push({ id, el: winEl, config: winConfig });
        this.os.onWindowCreated(id, winConfig);
        this.focusWindow(id);

        return id;
    }

    closeWindow(id) {
        const index = this.windows.findIndex(w => w.id === id);
        if (index !== -1) {
            const win = this.windows[index];
            win.el.remove();
            this.windows.splice(index, 1);
            this.os.onWindowClosed(id);
        }
    }

    minimizeWindow(id) {
        const win = this.windows.find(w => w.id === id);
        if (win) {
            win.el.classList.add('minimized');
            win.config.state = 'minimized';
            this.activeWindow = null;
        }
    }

    maximizeWindow(id) {
        const win = this.windows.find(w => w.id === id);
        if (win) {
            if (win.config.state === 'maximized') {
                win.el.classList.remove('maximized');
                win.config.state = 'normal';
            } else {
                win.el.classList.add('maximized');
                win.config.state = 'maximized';
            }
        }
    }

    restoreWindow(id) {
        const win = this.windows.find(w => w.id === id);
        if (win) {
            win.el.classList.remove('minimized');
            if (win.config.state === 'minimized') win.config.state = 'normal';
            this.focusWindow(id);
        }
    }

    focusWindow(id) {
        const win = this.windows.find(w => w.id === id);
        if (win) {
            win.el.style.zIndex = ++this.zIndexCounter;
            this.activeWindow = id;

            // Unfocus others visually if needed (optional)
            this.windows.forEach(w => {
                if(w.id !== id) {
                    w.el.querySelector('.title-bar').style.backgroundColor = '#f0f0f0';
                    w.el.querySelector('.title-bar-title').style.color = '#888';
                }
            });
            win.el.querySelector('.title-bar').style.backgroundColor = '#0078d7';
            win.el.querySelector('.title-bar-title').style.color = '#fff';
        }
    }

    setWindowContent(id, contentElement) {
        const win = this.windows.find(w => w.id === id);
        if (win) {
            const contentArea = win.el.querySelector('.window-content');
            contentArea.innerHTML = '';
            contentArea.appendChild(contentElement);
        }
    }
}

class AppRegistry {
    constructor(os) {
        this.os = os;
    }

    launch(appName, args) {
        switch (appName) {
            case 'Explorer':
                this.launchExplorer(args);
                break;
            case 'Browser':
                this.launchBrowser(args);
                break;
            case 'Notepad':
                this.launchNotepad(args);
                break;
            case 'ImageViewer':
                this.launchImageViewer(args);
                break;
            case 'Terminal':
                this.launchTerminal(args);
                break;
            case 'MarketMonitor':
                this.launchMarketMonitor(args);
                break;
            case 'ReportGenerator':
                this.launchReportGenerator(args);
                break;
            case 'CreditSentinel':
                this.launchCreditSentinel(args);
                break;
            case 'Spreadsheet':
                this.launchSpreadsheet(args);
                break;
            case 'Settings':
                this.launchSettings(args);
                break;
            default:
                console.error('Unknown app:', appName);
        }
    }

    launchSettings(args) {
        const winId = this.os.windowManager.createWindow({
            title: 'System Settings',
            icon: 'https://img.icons8.com/color/48/000000/settings.png',
            width: 500,
            height: 400,
            app: 'Settings'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.style.height = '100%';
        container.style.color = 'var(--text-color)';

        let content = `<h2>Theme Selection</h2><div style="display:flex; flex-direction:column; gap:10px; margin-bottom:20px;">`;

        if (window.THEME_LIBRARY) {
            Object.keys(window.THEME_LIBRARY).forEach(key => {
                const theme = window.THEME_LIBRARY[key];
                content += `
                    <button class="cyber-btn theme-btn" data-theme="${key}" style="text-align:left; padding:10px; background:rgba(255,255,255,0.05); border:1px solid var(--window-border); color:var(--text-color); cursor:pointer;">
                        <strong>${theme.name}</strong><br>
                        <span style="font-size:11px; opacity:0.7;">${theme.description}</span>
                    </button>
                `;
            });
        } else {
            content += `<p>Theme library not loaded.</p>`;
        }

        content += `</div>`;
        content += `
            <div style="border-top:1px solid var(--window-border); padding-top:15px;">
                <h3>Theme Actions</h3>
                <button id="export-theme-${winId}" class="cyber-btn" style="padding:8px 15px; background:var(--highlight-color); color:var(--bg-color); border:none; cursor:pointer;">Export Current Theme</button>
            </div>
        `;

        container.innerHTML = content;
        this.os.windowManager.setWindowContent(winId, container);

        // Event Listeners
        const themeBtns = container.querySelectorAll('.theme-btn');
        themeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const themeId = btn.getAttribute('data-theme');
                this.os.themeManager.applyTheme(themeId);
            });
        });

        const exportBtn = container.querySelector(`#export-theme-${winId}`);
        exportBtn.addEventListener('click', () => {
            const json = this.os.themeManager.exportTheme();
            navigator.clipboard.writeText(json).then(() => {
                alert('Theme JSON copied to clipboard!');
            });
        });
    }

    launchMarketMonitor(args) {
        const winId = this.os.windowManager.createWindow({
            title: 'Market Monitor',
            icon: 'https://img.icons8.com/color/48/000000/line-chart.png',
            width: 1000,
            height: 600,
            app: 'MarketMonitor'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.style.backgroundColor = '#1e1e1e';
        container.style.color = '#fff';
        container.style.height = '100%';
        container.style.overflowY = 'auto';
        container.innerHTML = '<h3>Loading Market Data...</h3>';
        this.os.windowManager.setWindowContent(winId, container);

        const renderData = (data) => {
             let html = `
                    <h2 style="margin-top:0;">S&P 500 Market Monitor</h2>
                    <table style="width:100%; border-collapse:collapse; text-align:left; font-size:14px;">
                        <tr style="border-bottom:1px solid #444;">
                            <th style="padding:10px;">Ticker</th>
                            <th style="padding:10px;">Company</th>
                            <th style="padding:10px;">Price</th>
                            <th style="padding:10px;">Change</th>
                            <th style="padding:10px;">P/E</th>
                            <th style="padding:10px;">Div %</th>
                            <th style="padding:10px;">Rating</th>
                            <th style="padding:10px;">Conviction</th>
                        </tr>
                `;
                data.forEach(item => {
                    const color = item.change_pct >= 0 ? '#4caf50' : '#f44336';
                    const rating = item.outlook ? item.outlook.consensus : 'N/A';
                    const conviction = item.outlook ? item.outlook.conviction : 'N/A';

                    html += `
                        <tr style="border-bottom:1px solid #333; cursor:pointer;" onclick="window.officeOS.appRegistry.launch('Browser', {url: 'data/equity_reports/${item.ticker}_Equity_Report.html', name: '${item.ticker} Equity Report'})">
                            <td style="padding:10px; font-weight:bold;">${item.ticker}</td>
                            <td style="padding:10px;">${item.name}</td>
                            <td style="padding:10px;">$${item.current_price}</td>
                            <td style="padding:10px; color:${color}">${item.change_pct}%</td>
                            <td style="padding:10px;">${item.pe_ratio || '-'}</td>
                            <td style="padding:10px;">${item.dividend_yield || '-'}%</td>
                            <td style="padding:10px;">${rating}</td>
                            <td style="padding:10px;">${conviction}</td>
                        </tr>
                    `;
                });
                html += '</table>';
                container.innerHTML = html;
        };

        if (window.MARKET_DATA) {
            renderData(window.MARKET_DATA);
        } else {
            fetch('data/sp500_market_data.json')
                .then(res => res.json())
                .then(data => renderData(data))
                .catch(err => {
                    container.innerHTML = `<h3 style="color:red">Error loading market data: ${err}</h3><p>Make sure scripts/generate_sp500_micro_build.py has been run.</p>`;
                });
        }
    }

    launchReportGenerator(args) {
        const winId = this.os.windowManager.createWindow({
            title: 'Report Generator',
            icon: 'https://img.icons8.com/color/48/000000/print.png',
            width: 500,
            height: 400,
            app: 'ReportGenerator'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';

        // Build options dynamically if MARKET_DATA exists
        let optionsHtml = '';
        if(window.MARKET_DATA) {
             window.MARKET_DATA.sort((a,b) => a.ticker.localeCompare(b.ticker)).forEach(c => {
                 optionsHtml += `<option value="${c.ticker}">${c.name} (${c.ticker})</option>`;
             });
        } else {
            // Fallback
             optionsHtml = `
                    <option value="AAPL">Apple Inc. (AAPL)</option>
                    <option value="MSFT">Microsoft Corp. (MSFT)</option>
                    <option value="GOOGL">Alphabet Inc. (GOOGL)</option>
            `;
        }

        container.innerHTML = `
            <h2>Generate New Report</h2>
            <div style="margin-bottom:15px;">
                <label style="display:block; margin-bottom:5px;">Select Company:</label>
                <select id="report-ticker-${winId}" style="width:100%; padding:8px;">
                    ${optionsHtml}
                </select>
            </div>
            <div style="margin-bottom:15px;">
                <label style="display:block; margin-bottom:5px;">Report Type:</label>
                <select id="report-type-${winId}" style="width:100%; padding:8px;">
                    <option value="equity">Equity Research Report</option>
                    <option value="credit">Credit Memo</option>
                </select>
            </div>
            <button id="generate-btn-${winId}" class="cyber-btn" style="width:100%; padding:10px; background:#0078d7; color:white; border:none; cursor:pointer;">Generate Report</button>
            <p id="status-${winId}" style="margin-top:10px; color:#666;"></p>
        `;
        this.os.windowManager.setWindowContent(winId, container);

        const btn = container.querySelector(`#generate-btn-${winId}`);
        btn.addEventListener('click', () => {
            const ticker = container.querySelector(`#report-ticker-${winId}`).value;
            const type = container.querySelector(`#report-type-${winId}`).value;
            const status = container.querySelector(`#status-${winId}`);

            status.innerText = 'Generating...';
            setTimeout(() => {
                status.innerText = 'Done! Opening report...';
                let path = '';
                if(type === 'equity') path = `data/equity_reports/${ticker}_Equity_Report.html`;
                else path = `data/credit_reports/${ticker}_Credit_Memo.html`;

                this.os.appRegistry.launch('Browser', { url: path, name: `${ticker} Report` });
                this.os.windowManager.closeWindow(winId);
            }, 1000);
        });
    }

    launchCreditSentinel(args) {
        const winId = this.os.windowManager.createWindow({
            title: 'Credit Sentinel',
            icon: 'https://img.icons8.com/color/48/000000/security-checked--v1.png',
            width: 800,
            height: 600,
            app: 'CreditSentinel'
        });

        const container = document.createElement('div');
        container.style.padding = '20px';
        container.style.backgroundColor = '#f5f5f5';
        container.style.height = '100%';
        container.style.overflowY = 'auto';
        container.innerHTML = '<h3>Loading Credit Data...</h3>';
        this.os.windowManager.setWindowContent(winId, container);

        const renderCreditData = (data) => {
                let html = `
                    <h2 style="margin-top:0; color:#333;">Credit Risk Monitor</h2>
                    <div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap:15px;">
                `;
                data.sort((a,b) => a.risk_score - b.risk_score).forEach(item => {
                    const riskColor = item.risk_score > 90 ? 'green' : item.risk_score > 80 ? 'orange' : 'red';
                    // Check if new fields exist (backward compatibility)
                    const pd = item.credit && item.credit.pd_rating ? item.credit.pd_rating : 'N/A';
                    const reg = item.credit && item.credit.regulatory_rating ? item.credit.regulatory_rating : 'N/A';

                    html += `
                        <div style="background:white; padding:15px; border-radius:4px; border:1px solid #ddd; box-shadow:0 2px 5px rgba(0,0,0,0.05); cursor:pointer;"
                             onclick="window.officeOS.appRegistry.launch('Browser', {url: 'data/credit_reports/${item.ticker}_Credit_Memo.html', name: '${item.ticker} Credit Memo'})">
                            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                                <span style="font-weight:bold; font-size:16px;">${item.ticker}</span>
                                <span style="font-weight:bold; color:${riskColor}; font-size:16px;">${item.risk_score}</span>
                            </div>
                            <div style="font-size:12px; color:#666; margin-bottom:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${item.name}</div>
                            <div style="font-size:11px; background:#f0f0f0; padding:5px; border-radius:3px;">
                                <div style="display:flex; justify-content:space-between;"><span>PD:</span> <strong>${pd}</strong></div>
                                <div style="display:flex; justify-content:space-between; margin-top:3px;"><span>Reg:</span> <strong>${reg}</strong></div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                container.innerHTML = html;
        };

        if (window.MARKET_DATA) {
            renderCreditData(window.MARKET_DATA);
        } else {
            fetch('data/sp500_market_data.json')
                .then(res => res.json())
                .then(data => renderCreditData(data))
                .catch(err => {
                    container.innerHTML = `<h3 style="color:red">Error: ${err}</h3>`;
                });
        }
    }

    launchSpreadsheet(args) {
        const winId = this.os.windowManager.createWindow({
            title: args.name || 'Spreadsheet',
            icon: 'https://img.icons8.com/color/48/000000/microsoft-excel-2019.png',
            width: 900,
            height: 600,
            app: 'Spreadsheet'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.height = '100%';

        container.innerHTML = `
            <div style="background:#217346; color:white; padding:5px 10px; font-size:12px;">Spreadsheet View - ${args.name || 'Untitled'}</div>
            <div id="sheet-content-${winId}" style="flex-grow:1; overflow:auto; background:white; padding:10px;">Loading...</div>
        `;
        this.os.windowManager.setWindowContent(winId, container);

        fetch(args.path)
            .then(res => res.text())
            .then(text => {
                const contentDiv = container.querySelector(`#sheet-content-${winId}`);
                try {
                    // Try JSON first
                    const data = JSON.parse(text);
                    if(Array.isArray(data)) {
                        this.renderJsonTable(contentDiv, data);
                    } else {
                        contentDiv.innerText = JSON.stringify(data, null, 2);
                    }
                } catch(e) {
                    // Fallback to basic CSV rendering (very simple)
                    const rows = text.split('\n');
                    let html = '<table style="border-collapse:collapse; width:100%;">';
                    rows.forEach((row, i) => {
                        html += '<tr>';
                        const cells = row.split(',');
                        cells.forEach(cell => {
                            if(i === 0) html += `<th style="border:1px solid #ddd; padding:5px; background:#f0f0f0;">${cell}</th>`;
                            else html += `<td style="border:1px solid #ddd; padding:5px;">${cell}</td>`;
                        });
                        html += '</tr>';
                    });
                    html += '</table>';
                    contentDiv.innerHTML = html;
                }
            });
    }

    renderJsonTable(container, data) {
        if(data.length === 0) {
            container.innerText = 'Empty Data';
            return;
        }
        const headers = Object.keys(data[0]);
        let html = '<table style="border-collapse:collapse; width:100%; font-family: sans-serif; font-size:12px;">';

        // Header
        html += '<thead><tr>';
        headers.forEach(h => html += `<th style="border:1px solid #ccc; padding:6px; background:#f3f3f3; text-align:left;">${h}</th>`);
        html += '</tr></thead><tbody>';

        // Body
        data.forEach(row => {
            html += '<tr>';
            headers.forEach(h => {
                let val = row[h];
                if(typeof val === 'object') val = JSON.stringify(val);
                html += `<td style="border:1px solid #ccc; padding:6px;">${val}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    }

    launchExplorer(args) {
        const path = args ? args.path : './';
        const winId = this.os.windowManager.createWindow({
            title: 'Nexus Explorer',
            icon: 'https://img.icons8.com/color/48/000000/folder-invoices--v1.png',
            width: 800,
            height: 500,
            app: 'Explorer'
        });

        const container = document.createElement('div');
        container.className = 'explorer-container';

        // Toolbar
        const toolbar = document.createElement('div');
        toolbar.className = 'explorer-toolbar';
        toolbar.innerHTML = `
            <button class="cyber-btn" style="margin-right:10px;" id="up-btn-${winId}">Up</button>
            <input type="text" value="${path}" id="path-input-${winId}">
        `;
        container.appendChild(toolbar);

        // Body
        const body = document.createElement('div');
        body.className = 'explorer-body';

        // Sidebar (Tree)
        const sidebar = document.createElement('div');
        sidebar.className = 'explorer-sidebar';
        sidebar.innerHTML = '<div style="padding:10px; color:#666;">Quick Access<br><br>Desktop<br>Documents<br>Downloads</div>';
        body.appendChild(sidebar);

        // Main View
        const main = document.createElement('div');
        main.className = 'explorer-main';
        main.id = `explorer-main-${winId}`;
        body.appendChild(main);

        container.appendChild(body);
        this.os.windowManager.setWindowContent(winId, container);

        // Logic
        this.renderExplorerView(winId, path);

        // Listeners
        const upBtn = toolbar.querySelector(`#up-btn-${winId}`);
        const pathInput = toolbar.querySelector(`#path-input-${winId}`);

        upBtn.addEventListener('click', () => {
            const currentPath = pathInput.value;
            const parts = currentPath.split('/');
            if (parts.length > 1) {
                parts.pop();
                let newPath = parts.join('/');
                if (newPath === '.') newPath = './';
                pathInput.value = newPath;
                this.renderExplorerView(winId, newPath);
            }
        });

        // Handle input enter
        pathInput.addEventListener('keydown', (e) => {
            if(e.key === 'Enter') {
                this.renderExplorerView(winId, pathInput.value);
            }
        });
    }

    renderExplorerView(winId, path) {
        const main = document.getElementById(`explorer-main-${winId}`);
        const pathInput = document.getElementById(`path-input-${winId}`);
        main.innerHTML = '';

        // Normalize path
        if(!path.startsWith('./') && !path.startsWith('data/')) {
             if(path === '.') path = './';
             else if(!path.startsWith('./')) path = './' + path;
        }

        pathInput.value = path;

        const contents = this.os.fs.readDir(path);
        const list = document.createElement('div');
        list.className = 'file-list';

        if (contents.length === 0) {
            main.innerHTML = '<div style="padding:20px; color:#888;">Empty folder</div>';
            return;
        }

        contents.forEach(item => {
            const itemEl = document.createElement('div');
            itemEl.className = 'file-item';

            let icon = 'https://img.icons8.com/color/48/000000/file.png';
            if (item.type === 'directory') icon = 'https://img.icons8.com/color/48/000000/folder-invoices--v1.png';
            else if (item.name.endsWith('.html')) icon = 'https://img.icons8.com/color/48/000000/html-5--v1.png';
            else if (item.name.endsWith('.json')) icon = 'https://img.icons8.com/color/48/000000/json--v1.png';
            else if (item.name.endsWith('.py')) icon = 'https://img.icons8.com/color/48/000000/python--v1.png';
            else if (item.name.endsWith('.png') || item.name.endsWith('.jpg')) icon = 'https://img.icons8.com/color/48/000000/image-file.png';

            itemEl.innerHTML = `
                <img src="${icon}">
                <span>${item.name}</span>
            `;

            itemEl.addEventListener('dblclick', () => {
                if (item.type === 'directory') {
                    this.renderExplorerView(winId, item.path);
                } else {
                    this.os.openFile(item);
                }
            });

            // Single click select
            itemEl.addEventListener('click', () => {
                document.querySelectorAll(`#explorer-main-${winId} .file-item`).forEach(el => el.classList.remove('selected'));
                itemEl.classList.add('selected');
            });

            list.appendChild(itemEl);
        });

        main.appendChild(list);
    }

    launchBrowser(args) {
        const url = args.path || args.url;
        const winId = this.os.windowManager.createWindow({
            title: args.name || 'Browser',
            icon: 'https://img.icons8.com/color/48/000000/internet.png',
            width: 1000,
            height: 700,
            app: 'Browser'
        });

        const iframe = document.createElement('iframe');
        iframe.src = url;
        iframe.className = 'iframe-viewer';
        this.os.windowManager.setWindowContent(winId, iframe);
    }

    launchNotepad(args) {
        const winId = this.os.windowManager.createWindow({
            title: args.name || 'Notepad',
            icon: 'https://img.icons8.com/color/48/000000/notepad.png',
            width: 600,
            height: 400,
            app: 'Notepad'
        });

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.height = '100%';

        const toolbar = document.createElement('div');
        toolbar.style.padding = '5px';
        toolbar.style.borderBottom = '1px solid #ddd';
        toolbar.style.backgroundColor = '#f9f9f9';
        toolbar.style.display = 'flex';
        toolbar.style.gap = '5px';

        const saveBtn = document.createElement('button');
        saveBtn.innerText = 'Save';
        saveBtn.className = 'cyber-btn';
        saveBtn.style.padding = '2px 10px';
        saveBtn.style.fontSize = '12px';

        // Template Dropdown
        const templateSelect = document.createElement('select');
        templateSelect.style.padding = '2px';
        templateSelect.style.fontSize = '12px';
        templateSelect.innerHTML = '<option value="">Load Template...</option>';
        if (window.TEMPLATE_LIBRARY) {
            Object.keys(window.TEMPLATE_LIBRARY).forEach(key => {
                templateSelect.innerHTML += `<option value="${key}">${window.TEMPLATE_LIBRARY[key].name}</option>`;
            });
        }

        // Prompt Dropdown
        const promptSelect = document.createElement('select');
        promptSelect.style.padding = '2px';
        promptSelect.style.fontSize = '12px';
        promptSelect.innerHTML = '<option value="">Insert Prompt...</option>';
        if (window.PROMPT_LIBRARY) {
            Object.keys(window.PROMPT_LIBRARY).forEach(key => {
                promptSelect.innerHTML += `<option value="${key}">${window.PROMPT_LIBRARY[key].name}</option>`;
            });
        }

        const status = document.createElement('span');
        status.style.marginLeft = '10px';
        status.style.fontSize = '11px';
        status.style.color = '#888';
        status.style.alignSelf = 'center';

        toolbar.appendChild(saveBtn);
        toolbar.appendChild(templateSelect);
        toolbar.appendChild(promptSelect);
        toolbar.appendChild(status);

        const textarea = document.createElement('textarea');
        textarea.style.flexGrow = '1';
        textarea.style.width = '100%';
        textarea.style.border = 'none';
        textarea.style.padding = '10px';
        textarea.style.resize = 'none';
        textarea.style.fontFamily = 'Consolas, monospace';
        textarea.style.fontSize = '14px';
        textarea.style.outline = 'none';
        textarea.value = 'Loading...';

        container.appendChild(toolbar);
        container.appendChild(textarea);

        this.os.windowManager.setWindowContent(winId, container);

        if(args.path) {
            fetch(args.path)
                .then(res => res.text())
                .then(text => {
                    textarea.value = text;
                })
                .catch(err => {
                    textarea.value = 'Error loading file: ' + err;
                });
        } else {
            textarea.value = args.content || '';
        }

        saveBtn.addEventListener('click', () => {
            console.log(`Saving file ${args.name || 'Untitled'}:`, textarea.value);
            status.innerText = 'Saved to console.';
            setTimeout(() => status.innerText = '', 2000);

            // Mock file system write
            // In a real app, this would POST to a backend or update the FS object
        });

        templateSelect.addEventListener('change', () => {
            const key = templateSelect.value;
            if (key && window.TEMPLATE_LIBRARY[key]) {
                if (confirm('Overwrite current content with template?')) {
                    textarea.value = window.TEMPLATE_LIBRARY[key].content;
                    // Simple placeholder replacement for Date
                    textarea.value = textarea.value.replace('[Date]', new Date().toLocaleDateString());
                }
                templateSelect.value = ""; // Reset
            }
        });

        promptSelect.addEventListener('change', () => {
            const key = promptSelect.value;
            if (key && window.PROMPT_LIBRARY[key]) {
                const promptText = window.PROMPT_LIBRARY[key].text;
                // Insert at cursor position
                const startPos = textarea.selectionStart;
                const endPos = textarea.selectionEnd;
                textarea.value = textarea.value.substring(0, startPos)
                    + promptText
                    + textarea.value.substring(endPos, textarea.value.length);
                promptSelect.value = ""; // Reset
            }
        });
    }

    launchImageViewer(args) {
        const winId = this.os.windowManager.createWindow({
            title: args.name || 'Photos',
            icon: 'https://img.icons8.com/color/48/000000/image-file.png',
            width: 600,
            height: 500,
            app: 'ImageViewer'
        });

        const img = document.createElement('img');
        img.src = args.path;
        img.style.maxWidth = '100%';
        img.style.maxHeight = '100%';
        img.style.display = 'block';
        img.style.margin = 'auto';

        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.alignItems = 'center';
        container.style.justifyContent = 'center';
        container.style.height = '100%';
        container.style.backgroundColor = '#222';

        container.appendChild(img);
        this.os.windowManager.setWindowContent(winId, container);
    }

    launchTerminal(args) {
        const winId = this.os.windowManager.createWindow({
            title: 'Terminal',
            icon: 'https://img.icons8.com/color/48/000000/console.png',
            width: 700,
            height: 450,
            app: 'Terminal'
        });

        const term = document.createElement('div');
        term.style.backgroundColor = 'black';
        term.style.color = '#0f0';
        term.style.fontFamily = 'Consolas, monospace';
        term.style.padding = '10px';
        term.style.height = '100%';
        term.style.overflowY = 'auto';
        term.style.display = 'flex';
        term.style.flexDirection = 'column';

        const history = document.createElement('div');
        history.innerHTML = 'AdamOS Kernel v25.0<br>Copyright (c) 2026 Omega Corp. All rights reserved.<br><br>';
        term.appendChild(history);

        const inputLine = document.createElement('div');
        inputLine.style.display = 'flex';
        inputLine.innerHTML = '<span style="margin-right: 5px;">user@nexus:~$</span>';

        const input = document.createElement('input');
        input.type = 'text';
        input.style.backgroundColor = 'transparent';
        input.style.border = 'none';
        input.style.color = '#0f0';
        input.style.fontFamily = 'inherit';
        input.style.flexGrow = '1';
        input.style.outline = 'none';

        inputLine.appendChild(input);
        term.appendChild(inputLine);

        this.os.windowManager.setWindowContent(winId, term);
        input.focus();

        // Keep focus on click
        term.addEventListener('click', () => input.focus());

        let currentDir = './';

        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                const cmd = input.value.trim();
                const line = document.createElement('div');
                line.textContent = `user@nexus:~$ ${cmd}`;
                history.appendChild(line);
                input.value = '';

                if (cmd) {
                    const output = document.createElement('div');
                    output.style.whiteSpace = 'pre-wrap';
                    output.style.marginBottom = '10px';
                    output.style.color = '#ccc';

                    const args = cmd.split(' ');
                    const command = args[0].toLowerCase();

                    switch (command) {
                        case 'help':
                            output.textContent = 'Available commands: help, ls, clear, echo, cat, open, whoami, date';
                            break;
                        case 'clear':
                            history.innerHTML = '';
                            output.remove(); // Don't append empty output
                            break;
                        case 'ls':
                            const files = this.os.fs.readDir(currentDir);
                            if (files.length === 0) {
                                output.textContent = '(empty)';
                            } else {
                                output.innerHTML = files.map(f => {
                                    const color = f.type === 'directory' ? '#4e94ce' : '#fff';
                                    return `<span style="color:${color}; margin-right: 15px;">${f.name}${f.type === 'directory' ? '/' : ''}</span>`;
                                }).join('');
                            }
                            break;
                        case 'echo':
                            output.textContent = args.slice(1).join(' ');
                            break;
                        case 'whoami':
                            output.textContent = 'Administrator (Level 10 Clearance)';
                            break;
                        case 'date':
                            output.textContent = new Date().toString();
                            break;
                        case 'cat':
                            if (args[1]) {
                                const target = this.os.fs.index[args[1]] || this.os.fs.index['./' + args[1]];
                                if (target && target.type === 'file') {
                                    try {
                                        // Adjust path for fetch if needed
                                        let fetchPath = target.path;
                                        if (fetchPath.startsWith('./')) {
                                             fetchPath = '../' + fetchPath.substring(2);
                                        }
                                        const res = await fetch(fetchPath);
                                        if (res.ok) {
                                            output.textContent = await res.text();
                                        } else {
                                            output.style.color = 'red';
                                            output.textContent = `Error reading file: ${res.status}`;
                                        }
                                    } catch (err) {
                                        output.style.color = 'red';
                                        output.textContent = `Error: ${err.message}`;
                                    }
                                } else {
                                    output.style.color = 'red';
                                    output.textContent = `File not found: ${args[1]}`;
                                }
                            } else {
                                output.textContent = 'Usage: cat <filename>';
                            }
                            break;
                        case 'open':
                            if (args[1]) {
                                const targetFile = this.os.fs.index[args[1]] || this.os.fs.index['./' + args[1]];
                                if (targetFile) {
                                    this.os.openFile(targetFile);
                                    output.textContent = `Opening ${args[1]}...`;
                                } else {
                                    output.style.color = 'red';
                                    output.textContent = `File not found: ${args[1]}`;
                                }
                            } else {
                                output.textContent = 'Usage: open <filename>';
                            }
                            break;
                        default:
                            output.style.color = 'red';
                            output.textContent = `Command not found: ${command}`;
                    }
                    if(command !== 'clear') history.appendChild(output);
                }

                // Scroll to bottom
                term.scrollTop = term.scrollHeight;
            }
        });
    }
}

class OfficeOS {
    constructor() {
        this.fs = new FileSystem();
        this.windowManager = new WindowManager(this);
        this.appRegistry = new AppRegistry(this);
        this.themeManager = new ThemeManager(this);
        this.taskbarItems = {}; // WinID -> Element
    }

    async boot() {
        // Show loading screen
        const loading = document.getElementById('loading');

        // Initialize FS
        await this.fs.init();

        // Initialize Theme
        this.themeManager.init();

        // Setup UI
        this.setupTaskbar();
        this.setupStartMenu();
        this.setupDesktop();

        // Hide loading
        loading.style.display = 'none';

        // Play Startup Sound (Optional/Mock)
        console.log('OS Booted');
    }

    setupTaskbar() {
        const startBtn = document.getElementById('start-button');
        startBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleStartMenu();
        });

        // Update Clock
        setInterval(() => {
            const now = new Date();
            document.getElementById('clock').innerText = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) + '\n' + now.toLocaleDateString();
        }, 1000);

        // Start menu click away
        document.addEventListener('click', (e) => {
            const startMenu = document.getElementById('start-menu');
            if(startMenu.classList.contains('open') && !startMenu.contains(e.target) && e.target.id !== 'start-button') {
                startMenu.classList.remove('open');
            }
        });
    }

    toggleStartMenu() {
        const menu = document.getElementById('start-menu');
        menu.classList.toggle('open');
    }

    setupStartMenu() {
        const grid = document.querySelector('.start-menu-grid');
        // Add apps
        const apps = [
            { name: 'Explorer', icon: 'https://img.icons8.com/color/48/000000/folder-invoices--v1.png', action: () => this.appRegistry.launch('Explorer') },
            { name: 'Terminal', icon: 'https://img.icons8.com/color/48/000000/console.png', action: () => this.appRegistry.launch('Terminal') },
            { name: 'Notepad', icon: 'https://img.icons8.com/color/48/000000/notepad.png', action: () => this.appRegistry.launch('Notepad', {name:'Untitled', content:''}) },
            { name: 'Market Monitor', icon: 'https://img.icons8.com/color/48/000000/line-chart.png', action: () => this.appRegistry.launch('MarketMonitor') },
            { name: 'Credit Sentinel', icon: 'https://img.icons8.com/color/48/000000/security-checked--v1.png', action: () => this.appRegistry.launch('CreditSentinel') },
            { name: 'Report Gen', icon: 'https://img.icons8.com/color/48/000000/print.png', action: () => this.appRegistry.launch('ReportGenerator') },
            { name: 'Settings', icon: 'https://img.icons8.com/color/48/000000/settings.png', action: () => this.appRegistry.launch('Settings') },
            // Add shortcut to specific dashboards
            { name: 'Mission Ctrl', icon: 'https://img.icons8.com/color/48/000000/monitor.png', action: () => this.appRegistry.launch('Browser', {url:'mission_control.html', name:'Mission Control'}) },
            { name: 'Archive', icon: 'https://img.icons8.com/color/48/000000/archive.png', action: () => this.appRegistry.launch('Browser', {url:'market_mayhem_archive.html', name:'Archive'}) },
            { name: 'Neural Deck', icon: 'https://img.icons8.com/color/48/000000/augmented-reality.png', action: () => this.appRegistry.launch('Browser', {url:'neural_deck.html', name:'Neural Deck'}) },
            { name: 'Holodeck', icon: 'https://img.icons8.com/color/48/000000/virtual-reality.png', action: () => this.appRegistry.launch('Browser', {url:'holodeck.html', name:'Holodeck'}) },
            { name: 'Sovereign', icon: 'https://img.icons8.com/color/48/000000/museum.png', action: () => this.appRegistry.launch('Browser', {url:'sovereign_dashboard.html', name:'Sovereign'}) }
        ];

        apps.forEach(app => {
            const el = document.createElement('div');
            el.className = 'start-menu-item';
            el.innerHTML = `<img src="${app.icon}"><span>${app.name}</span>`;
            el.addEventListener('click', () => {
                app.action();
                this.toggleStartMenu();
            });
            grid.appendChild(el);
        });
    }

    setupDesktop() {
        const desktop = document.getElementById('desktop');
        desktop.innerHTML = ''; // Clear existing

        const icons = [
            { name: 'My Computer', icon: 'https://img.icons8.com/color/48/000000/workstation.png', action: () => this.appRegistry.launch('Explorer', {path: './'}) },
            { name: 'Market Monitor', icon: 'https://img.icons8.com/color/48/000000/line-chart.png', action: () => this.appRegistry.launch('MarketMonitor') },
            { name: 'Credit Sentinel', icon: 'https://img.icons8.com/color/48/000000/security-checked--v1.png', action: () => this.appRegistry.launch('CreditSentinel') },
            { name: 'Report Generator', icon: 'https://img.icons8.com/color/48/000000/print.png', action: () => this.appRegistry.launch('ReportGenerator') },
        ];

        // Dynamic App Discovery
        const htmlFiles = this.fs.findFiles('.html');

        // Priority Apps mapping
        const appMap = {
            'neural_deck.html': { name: 'Neural Deck', icon: 'https://img.icons8.com/color/48/000000/augmented-reality.png' },
            'holodeck.html': { name: 'Holodeck', icon: 'https://img.icons8.com/color/48/000000/virtual-reality.png' },
            'war_room_v2.html': { name: 'War Room', icon: 'https://img.icons8.com/color/48/000000/strategy-board.png' },
            'mission_control.html': { name: 'Mission Ctrl', icon: 'https://img.icons8.com/color/48/000000/monitor.png' },
            'sovereign_dashboard.html': { name: 'Sovereign DB', icon: 'https://img.icons8.com/color/48/000000/museum.png' },
            'quantum_search.html': { name: 'Q-Search', icon: 'https://img.icons8.com/color/48/000000/search--v1.png' },
            'system_brain.html': { name: 'System Brain', icon: 'https://img.icons8.com/color/48/000000/brain--v1.png' },
            'unified_dashboard.html': { name: 'Unified DB', icon: 'https://img.icons8.com/color/48/000000/dashboard.png' },
            'nexus_explorer.html': { name: 'Nexus Explorer', icon: 'https://img.icons8.com/color/48/000000/galaxy.png' }
        };

        htmlFiles.forEach(file => {
            if(appMap[file.name]) {
                const config = appMap[file.name];
                let path = file.path;
                if (path.startsWith('./')) {
                    path = '../' + path.substring(2);
                }
                icons.push({
                    name: config.name,
                    icon: config.icon,
                    action: () => this.appRegistry.launch('Browser', {url: path, name: config.name})
                });
            }
        });

        // Add System logs folder
        icons.push({ name: 'System Logs', icon: 'https://img.icons8.com/color/48/000000/txt.png', action: () => this.appRegistry.launch('Explorer', {path: './logs'}) });

        icons.forEach(icon => {
            const el = document.createElement('div');
            el.className = 'desktop-icon';
            el.innerHTML = `<img src="${icon.icon}"><span>${icon.name}</span>`;
            el.addEventListener('dblclick', icon.action);
            el.addEventListener('click', (e) => {
                e.stopPropagation();
                document.querySelectorAll('.desktop-icon').forEach(i => i.classList.remove('selected'));
                el.classList.add('selected');
            });
            desktop.appendChild(el);
        });

        desktop.addEventListener('click', () => {
             document.querySelectorAll('.desktop-icon').forEach(i => i.classList.remove('selected'));
        });
    }

    onWindowCreated(id, config) {
        const bar = document.getElementById('taskbar-apps');
        const item = document.createElement('div');
        item.className = 'taskbar-item active';
        item.id = 'taskbar-' + id;
        item.innerHTML = `<img src="${config.icon}"><span>${config.title}</span>`;
        item.addEventListener('click', () => {
            const win = this.windowManager.windows.find(w => w.id === id);
            if (win.el.classList.contains('minimized')) {
                this.windowManager.restoreWindow(id);
            } else if (this.windowManager.activeWindow === id) {
                this.windowManager.minimizeWindow(id);
            } else {
                this.windowManager.focusWindow(id);
            }
        });
        bar.appendChild(item);
        this.taskbarItems[id] = item;
    }

    onWindowClosed(id) {
        const item = this.taskbarItems[id];
        if (item) item.remove();
        delete this.taskbarItems[id];
    }

    openFile(file) {
        let path = file.path;
        if (path.startsWith('./')) {
            // Adjust for being in showcase/ subdirectory
            path = '../' + path.substring(2);
        }

        const ext = file.name.split('.').pop().toLowerCase();
        if (['html', 'htm'].includes(ext)) {
            this.appRegistry.launch('Browser', { url: path, name: file.name });
        } else if (['csv', 'xls', 'xlsx'].includes(ext)) {
             this.appRegistry.launch('Spreadsheet', { path: path, name: file.name });
        } else if (['txt', 'md', 'json', 'py', 'js', 'css', 'yaml', 'yml', 'xml', 'log'].includes(ext)) {
            // Check if json is meant for spreadsheet
            if(ext === 'json' && (file.name.includes('data') || file.name.includes('market'))) {
                this.appRegistry.launch('Spreadsheet', { path: path, name: file.name });
            } else {
                this.appRegistry.launch('Notepad', { path: path, name: file.name });
            }
        } else if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(ext)) {
            this.appRegistry.launch('ImageViewer', { path: path, name: file.name });
        } else {
            this.appRegistry.launch('Notepad', { path: path, name: file.name });
        }
    }
}

// Start
window.officeOS = new OfficeOS();
document.addEventListener('DOMContentLoaded', () => {
    window.officeOS.boot();
});
