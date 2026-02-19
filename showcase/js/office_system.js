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
            default:
                console.error('Unknown app:', appName);
        }
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

        const contentDiv = document.createElement('div');
        contentDiv.className = 'text-viewer-content';
        contentDiv.innerText = 'Loading...';
        this.os.windowManager.setWindowContent(winId, contentDiv);

        fetch(args.path)
            .then(res => res.text())
            .then(text => {
                contentDiv.innerText = text;
            })
            .catch(err => {
                contentDiv.innerText = 'Error loading file: ' + err;
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
        term.style.fontFamily = 'monospace';
        term.style.padding = '10px';
        term.style.height = '100%';
        term.style.overflowY = 'auto';
        term.innerHTML = 'Microsoft Windows [Version 10.0.19045.3693]<br>(c) Microsoft Corporation. All rights reserved.<br><br>C:\\Users\\Admin>';

        this.os.windowManager.setWindowContent(winId, term);
    }
}

class OfficeOS {
    constructor() {
        this.fs = new FileSystem();
        this.windowManager = new WindowManager(this);
        this.appRegistry = new AppRegistry(this);
        this.taskbarItems = {}; // WinID -> Element
    }

    async boot() {
        // Show loading screen
        const loading = document.getElementById('loading');

        // Initialize FS
        await this.fs.init();

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
            // Add shortcut to specific dashboards
            { name: 'Mission Ctrl', icon: 'https://img.icons8.com/color/48/000000/monitor.png', action: () => this.appRegistry.launch('Browser', {url:'showcase/mission_control.html', name:'Mission Control'}) },
            { name: 'Archive', icon: 'https://img.icons8.com/color/48/000000/archive.png', action: () => this.appRegistry.launch('Browser', {url:'showcase/market_mayhem_archive.html', name:'Archive'}) }
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
        const icons = [
            { name: 'My Computer', icon: 'https://img.icons8.com/color/48/000000/workstation.png', action: () => this.appRegistry.launch('Explorer', {path: './'}) },
            { name: 'Market Monitor', icon: 'https://img.icons8.com/color/48/000000/line-chart.png', action: () => this.appRegistry.launch('MarketMonitor') },
            { name: 'Credit Sentinel', icon: 'https://img.icons8.com/color/48/000000/security-checked--v1.png', action: () => this.appRegistry.launch('CreditSentinel') },
            { name: 'Report Generator', icon: 'https://img.icons8.com/color/48/000000/print.png', action: () => this.appRegistry.launch('ReportGenerator') },
            { name: 'System Logs', icon: 'https://img.icons8.com/color/48/000000/txt.png', action: () => this.appRegistry.launch('Explorer', {path: './logs'}) },
            { name: 'Showcase', icon: 'https://img.icons8.com/color/48/000000/presentation.png', action: () => this.appRegistry.launch('Explorer', {path: './showcase'}) }
        ];

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
        const ext = file.name.split('.').pop().toLowerCase();
        if (['html', 'htm'].includes(ext)) {
            this.appRegistry.launch('Browser', { url: file.path, name: file.name });
        } else if (['csv', 'xls', 'xlsx'].includes(ext)) {
             this.appRegistry.launch('Spreadsheet', { path: file.path, name: file.name });
        } else if (['txt', 'md', 'json', 'py', 'js', 'css', 'yaml', 'yml', 'xml', 'log'].includes(ext)) {
            // Check if json is meant for spreadsheet
            if(ext === 'json' && (file.name.includes('data') || file.name.includes('market'))) {
                this.appRegistry.launch('Spreadsheet', { path: file.path, name: file.name });
            } else {
                this.appRegistry.launch('Notepad', { path: file.path, name: file.name });
            }
        } else if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(ext)) {
            this.appRegistry.launch('ImageViewer', { path: file.path, name: file.name });
        } else {
            this.appRegistry.launch('Notepad', { path: file.path, name: file.name });
        }
    }
}

// Start
window.officeOS = new OfficeOS();
document.addEventListener('DOMContentLoaded', () => {
    window.officeOS.boot();
});
