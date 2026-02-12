class WindowManager {
    constructor() {
        this.windows = {};
        this.zIndexCounter = 100;
        this.desktop = document.getElementById('os-desktop');
        this.taskbarApps = document.getElementById('taskbar-apps');
    }

    openWindow(id, title, url, width, height) {
        if (this.windows[id]) {
            this.bringToFront(id);
            return;
        }

        const win = document.createElement('div');
        win.className = 'os-window';
        win.id = `win-${id}`;
        win.style.width = `${width}px`;
        win.style.height = `${height}px`;
        win.style.zIndex = ++this.zIndexCounter;
        
        // Center the window initially
        const left = (window.innerWidth - width) / 2 + (Math.random() * 40 - 20);
        const top = (window.innerHeight - height) / 2 + (Math.random() * 40 - 20);
        win.style.left = `${Math.max(0, left)}px`;
        win.style.top = `${Math.max(0, top)}px`;

        win.innerHTML = `
            <div class="window-header" onmousedown="wm.startDrag(event, '${id}')">
                <div class="window-title"><i class="fas fa-terminal mr-2"></i>${title}</div>
                <div class="window-controls">
                    <div class="window-btn min" onclick="wm.minimizeWindow('${id}')" title="Minimize"></div>
                    <div class="window-btn max" onclick="wm.maximizeWindow('${id}')" title="Maximize"></div>
                    <div class="window-btn close" onclick="wm.closeWindow('${id}')" title="Close"></div>
                </div>
            </div>
            <div class="window-content">
                <iframe src="${url}"></iframe>
            </div>
        `;

        win.onmousedown = () => this.bringToFront(id);

        this.desktop.appendChild(win);
        this.windows[id] = { element: win, title: title, minimized: false };
        this.updateTaskbar();
    }

    closeWindow(id) {
        if (this.windows[id]) {
            this.windows[id].element.remove();
            delete this.windows[id];
            this.updateTaskbar();
        }
    }

    minimizeWindow(id) {
        const win = this.windows[id];
        if (win) {
            win.element.style.display = 'none';
            win.minimized = true;
            this.updateTaskbar();
        }
    }

    restoreWindow(id) {
        const win = this.windows[id];
        if (win) {
            win.element.style.display = 'flex';
            win.minimized = false;
            this.bringToFront(id);
            this.updateTaskbar();
        }
    }

    toggleWindow(id) {
        if (this.windows[id].minimized) {
            this.restoreWindow(id);
        } else {
            this.bringToFront(id); // If already open and not minimized, just focus
        }
    }

    maximizeWindow(id) {
        const win = this.windows[id].element;
        if (win.style.width === '100%' && win.style.height === 'calc(100% - 48px)') {
            // Restore
            win.style.width = win.dataset.prevWidth || '800px';
            win.style.height = win.dataset.prevHeight || '600px';
            win.style.top = win.dataset.prevTop || '50px';
            win.style.left = win.dataset.prevLeft || '50px';
        } else {
            // Maximize
            win.dataset.prevWidth = win.style.width;
            win.dataset.prevHeight = win.style.height;
            win.dataset.prevTop = win.style.top;
            win.dataset.prevLeft = win.style.left;

            win.style.width = '100%';
            win.style.height = 'calc(100% - 48px)'; // Subtract taskbar height
            win.style.top = '0';
            win.style.left = '0';
        }
        this.bringToFront(id);
    }

    bringToFront(id) {
        if (this.windows[id]) {
            this.windows[id].element.style.zIndex = ++this.zIndexCounter;
        }
    }

    startDrag(e, id) {
        e.preventDefault();
        const win = this.windows[id].element;
        this.bringToFront(id);

        // Check if maximized
        if (win.style.width === '100%') return;

        let startX = e.clientX;
        let startY = e.clientY;
        let startLeft = win.offsetLeft;
        let startTop = win.offsetTop;

        const mouseMoveHandler = (ev) => {
            const dx = ev.clientX - startX;
            const dy = ev.clientY - startY;
            win.style.left = `${startLeft + dx}px`;
            win.style.top = `${startTop + dy}px`;
        };

        const mouseUpHandler = () => {
            document.removeEventListener('mousemove', mouseMoveHandler);
            document.removeEventListener('mouseup', mouseUpHandler);
        };

        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
    }

    updateTaskbar() {
        this.taskbarApps.innerHTML = '';
        Object.keys(this.windows).forEach(id => {
            const win = this.windows[id];
            const item = document.createElement('div');
            item.className = `taskbar-item ${win.minimized ? '' : 'active'}`;
            item.innerHTML = `<i class="fas fa-window-maximize text-cyan-500"></i><span>${win.title}</span>`;
            item.onclick = () => this.toggleWindow(id);
            this.taskbarApps.appendChild(item);
        });
    }
}

// Global instance
const wm = new WindowManager();

// Boot Sequence & Clock
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Start Menu
    if (typeof StartMenu !== 'undefined') {
        window.startMenu = new StartMenu(wm);
        
        const startBtn = document.getElementById('start-btn');
        if (startBtn) {
            startBtn.onclick = (e) => {
                e.stopPropagation();
                window.startMenu.toggle();
            };
        }

        // Close start menu when clicking on desktop
        const desktop = document.getElementById('os-desktop');
        if (desktop) {
            desktop.addEventListener('mousedown', () => {
                if (window.startMenu && window.startMenu.isOpen) window.startMenu.close();
            });
        }
    }

    // Clock
    setInterval(() => {
        const now = new Date();
        const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
        document.getElementById('sys-clock').innerText = time;
    }, 1000);

    // Boot Animation
    const bootScreen = document.getElementById('boot-screen');
    if (bootScreen) {
        setTimeout(() => {
            bootScreen.style.opacity = '0';
            setTimeout(() => {
                bootScreen.style.display = 'none';
                // Auto-open Welcome/Nexus or just sound
            }, 1000);
        }, 2500);
    }
});
