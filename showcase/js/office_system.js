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
            default:
                console.error('Unknown app:', appName);
        }
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
            { name: 'Recycle Bin', icon: 'https://img.icons8.com/color/48/000000/trash.png', action: () => alert('Recycle Bin is empty') },
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
        } else if (['txt', 'md', 'json', 'py', 'js', 'css', 'yaml', 'yml', 'xml', 'log'].includes(ext)) {
            this.appRegistry.launch('Notepad', { path: file.path, name: file.name });
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
