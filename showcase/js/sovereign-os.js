/*
=========================================================
ADAM Sovereign OS Core JS Components
=========================================================
Handles Window Management, Dock logic, and system functions.
*/

document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    let zIndexCounter = 100;

    // System Clock Updater
    function updateClock() {
        const now = new Date();
        const clockEl = document.getElementById('clock');
        if (clockEl) {
            clockEl.innerText = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Window Management System
    window.openWindow = function(id, title, src, w, h) {
        let win = document.getElementById(`win-${id}`);
        if (win) {
            bringToFront(win);
            return;
        }

        win = document.createElement('div');
        win.id = `win-${id}`;
        win.className = 'os-window active';
        win.style.width = w + 'px';
        win.style.height = h + 'px';

        const count = document.querySelectorAll('.os-window').length;
        win.style.top = (50 + (count * 30)) + 'px';
        win.style.left = (50 + (count * 30)) + 'px';
        win.style.zIndex = ++zIndexCounter;

        // Data hook for AI ingestion
        win.setAttribute('data-ai-state', 'window-active');
        win.setAttribute('data-ai-component', id);

        win.innerHTML = `
            <div class="window-bar" onmousedown="startDrag(event, '${id}')">
                <div class="flex items-center gap-2">
                     <i data-lucide="app-window" class="w-3 h-3 text-gray-500"></i>
                     <span class="text-xs font-bold text-gray-300 tracking-wider">${title}</span>
                </div>
                <div class="flex gap-2 items-center">
                    <button onclick="minimizeWindow('${id}')" class="w-3 h-3 rounded-full bg-yellow-500/50 hover:bg-yellow-500 cursor-pointer border border-yellow-500/30 transition-colors" title="Minimize" aria-label="Minimize Window"></button>
                    <button onclick="closeWindow('${id}')" class="w-3 h-3 rounded-full bg-red-500/50 hover:bg-red-500 cursor-pointer border border-red-500/30 transition-colors" title="Close" aria-label="Close Window"></button>
                </div>
            </div>
            <div class="window-content">
                <iframe src="${src}" title="${title}"></iframe>
            </div>
        `;

        const desktop = document.getElementById('desktop');
        if(desktop) desktop.appendChild(win);

        const dot = document.getElementById(`dot-${id}`);
        if(dot) dot.classList.add('running');

        win.onmousedown = () => bringToFront(win);
        lucide.createIcons();
    };

    window.closeWindow = function(id) {
        const win = document.getElementById(`win-${id}`);
        if(win) win.remove();
        const dot = document.getElementById(`dot-${id}`);
        if(dot) dot.classList.remove('running');
    };

    window.minimizeWindow = function(id) {
        const win = document.getElementById(`win-${id}`);
        if(win) {
            win.style.display = 'none';
            win.setAttribute('data-ai-state', 'window-minimized');
        }
    };

    window.bringToFront = function(el) {
        if(el.style.display === 'none') {
            el.style.display = 'flex';
            el.setAttribute('data-ai-state', 'window-active');
        }
        document.querySelectorAll('.os-window').forEach(w => w.classList.remove('active'));
        el.classList.add('active');
        el.style.zIndex = ++zIndexCounter;
    };

    // Drag System
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };
    let currentWindow = null;

    window.startDrag = function(e, id) {
        isDragging = true;
        currentWindow = document.getElementById(`win-${id}`);
        bringToFront(currentWindow);

        const rect = currentWindow.getBoundingClientRect();
        dragOffset.x = e.clientX - rect.left;
        dragOffset.y = e.clientY - rect.top;

        const overlay = document.createElement('div');
        overlay.id = 'drag-overlay';
        overlay.style.position = 'fixed';
        overlay.style.inset = '0';
        overlay.style.zIndex = '9999';
        document.body.appendChild(overlay);
    };

    document.addEventListener('mousemove', (e) => {
        if (!isDragging || !currentWindow) return;

        let x = e.clientX - dragOffset.x;
        let y = e.clientY - dragOffset.y;

        // Snapping to top bar
        if(y < 32) y = 32;

        const maxW = window.innerWidth;
        const maxH = window.innerHeight;
        const winW = currentWindow.offsetWidth;
        const winH = currentWindow.offsetHeight;

        // Edge Snapping
        const snapThreshold = 20;
        if (x < snapThreshold) x = 0;
        if (x + winW > maxW - snapThreshold) x = maxW - winW;
        if (y + winH > maxH - snapThreshold) y = maxH - winH;

        currentWindow.style.left = x + 'px';
        currentWindow.style.top = y + 'px';
    });

    document.addEventListener('mouseup', () => {
        if(isDragging) {
            isDragging = false;
            currentWindow = null;
            const overlay = document.getElementById('drag-overlay');
            if(overlay) overlay.remove();
        }
    });

    // Start Menu Toggle
    window.toggleStartMenu = function() {
        const menu = document.getElementById('start-menu');
        if (!menu) return;

        if (menu.classList.contains('hidden')) {
            menu.classList.remove('hidden');
            void menu.offsetWidth; // trigger reflow
            menu.classList.remove('opacity-0', 'translate-y-4');
            menu.classList.add('opacity-100', 'translate-y-0');
            menu.setAttribute('data-ai-state', 'menu-open');
        } else {
            menu.classList.remove('opacity-100', 'translate-y-0');
            menu.classList.add('opacity-0', 'translate-y-4');
            menu.setAttribute('data-ai-state', 'menu-closed');
            setTimeout(() => {
                menu.classList.add('hidden');
            }, 200);
        }
    };

    // Close start menu when clicking outside
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('start-menu');
        const dockItem = document.querySelector('[data-title="System Matrix"]');

        if (menu && !menu.classList.contains('hidden') &&
            !menu.contains(e.target) &&
            !(dockItem && dockItem.contains(e.target))) {
            toggleStartMenu();
        }
    });

    // Fullscreen Toggle
    window.toggleFullScreen = function() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    };
});
