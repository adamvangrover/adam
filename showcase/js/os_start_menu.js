class StartMenu {
    constructor(windowManager) {
        this.wm = windowManager;
        this.isOpen = false;
        this.menuElement = null;
        this.init();
    }

    init() {
        // Create the menu DOM
        this.menuElement = document.createElement('div');
        this.menuElement.id = 'start-menu';
        this.menuElement.className = 'hidden absolute bottom-12 left-2 w-64 bg-black/90 border border-cyan-500/50 backdrop-blur-md z-[9999] shadow-[0_0_20px_rgba(0,255,255,0.2)]';
        
        // Menu Content
        this.menuElement.innerHTML = `
            <div class="p-2 border-b border-cyan-500/30 flex items-center gap-2 bg-cyan-900/20">
                <i class="fas fa-user-astronaut text-cyan-400"></i>
                <span class="text-sm font-bold text-white tracking-widest">ADAM USER</span>
            </div>
            <div class="p-2 grid gap-1">
                <div class="menu-item" onclick="wm.openWindow('agent_status', 'Agent Status', 'apps/agent_feed.html', 400, 600); startMenu.close()">
                    <i class="fas fa-network-wired w-6 text-center text-green-400"></i> Agent Feed
                </div>
                <div class="menu-item" onclick="wm.openWindow('sys_mon', 'System Monitor', 'widgets/sys_monitor.html', 500, 300); startMenu.close()">
                    <i class="fas fa-microchip w-6 text-center text-red-400"></i> System Monitor
                </div>
                <div class="menu-item" onclick="wm.openWindow('files', 'File Explorer', 'apps/file_explorer.html', 800, 600); startMenu.close()">
                    <i class="fas fa-folder-open w-6 text-center text-yellow-400"></i> File Explorer
                </div>
                 <div class="h-px bg-gray-700 my-1"></div>
                <div class="menu-item" onclick="wm.openWindow('nexus', 'Nexus Simulation', 'nexus.html', 1000, 600); startMenu.close()">
                    <i class="fas fa-globe-americas w-6 text-center text-amber-500"></i> Nexus
                </div>
                <div class="menu-item" onclick="wm.openWindow('warroom', 'War Room', 'war_room_v2.html', 1200, 800); startMenu.close()">
                    <i class="fas fa-crosshairs w-6 text-center text-red-500"></i> War Room
                </div>
                <div class="menu-item" onclick="wm.openWindow('cortex', 'System Cortex', 'system_knowledge_graph.html', 1100, 700); startMenu.close()">
                    <i class="fas fa-brain w-6 text-center text-purple-500"></i> Cortex
                </div>
                <div class="h-px bg-gray-700 my-1"></div>
                <div class="menu-item hover:bg-red-900/50" onclick="alert('System Shutdown Sequence Initiated... (Just Kidding)'); startMenu.close()">
                    <i class="fas fa-power-off w-6 text-center text-red-600"></i> Shutdown
                </div>
            </div>
        `;

        document.body.appendChild(this.menuElement);

        // Add CSS styles for menu items
        const style = document.createElement('style');
        style.textContent = `
            .menu-item {
                display: flex;
                align-items: center;
                padding: 8px;
                cursor: pointer;
                color: #ccc;
                font-family: 'Share Tech Mono', monospace;
                font-size: 0.9rem;
                transition: all 0.2s;
            }
            .menu-item:hover {
                background: rgba(0, 243, 255, 0.1);
                color: #fff;
                padding-left: 12px; /* Slight slide effect */
            }
        `;
        document.head.appendChild(style);
    }

    toggle() {
        this.isOpen = !this.isOpen;
        if (this.isOpen) {
            this.menuElement.classList.remove('hidden');
            this.menuElement.classList.add('block');
        } else {
            this.menuElement.classList.remove('block');
            this.menuElement.classList.add('hidden');
        }
    }

    close() {
        this.isOpen = false;
        this.menuElement.classList.remove('block');
        this.menuElement.classList.add('hidden');
    }
}

// Attach to window
window.StartMenu = StartMenu;
