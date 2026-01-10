/**
 * ADAM v23.0 Shell Logic
 * Handles routing, terminal simulation, data bridging, and global UI state.
 */

class Shell {
    constructor() {
        this.iframe = document.getElementById('content-frame');
        this.terminalPane = document.getElementById('terminal-pane');
        this.terminalContent = document.getElementById('terminal-content');
        this.sidebarLinks = document.querySelectorAll('.sidebar-link');
        this.searchInput = document.getElementById('global-search');
        this.searchResults = document.getElementById('search-results');

        this.liveMode = false;

        this.routes = {
            '#mission_control': 'mission_control.html',
            '#neural_dashboard': 'neural_dashboard.html',
            '#deep_dive': 'deep_dive.html',
            '#financial_twin': 'financial_twin.html',
            '#agents': 'agents.html',
            '#reports': 'reports.html',
            '#prompts': 'prompts.html',
            '#data': 'data.html',
            '#graph': 'graph.html',
            '#repository': 'repository.html'
        };

        this.repoIndex = { files: [] };

        this.init();
    }

    async init() {
        // Load Repo Index
        try {
            const res = await fetch('data/repo_full_index.json');
            if(res.ok) {
                this.repoIndex = await res.json();
                console.log(`[Shell] Loaded index with ${this.repoIndex.files.length} items.`);
            }
        } catch (e) {
            console.warn("[Shell] Failed to load repo index.", e);
        }

        // Router Init
        window.addEventListener('hashchange', () => this.handleRoute());
        this.handleRoute();

        // Event Listeners
        document.getElementById('toggle-terminal').addEventListener('click', () => this.toggleTerminal());
        document.getElementById('close-terminal').addEventListener('click', () => this.toggleTerminal(false));
        document.getElementById('clear-terminal').addEventListener('click', () => this.clearTerminal());

        // Mode Toggle
        document.getElementById('mode-toggle').addEventListener('click', () => this.toggleMode());

        this.searchInput.addEventListener('input', (e) => this.handleSearch(e.target.value));
        this.searchInput.addEventListener('focus', () => { if(this.searchInput.value) this.searchResults.classList.remove('hidden'); });
        document.addEventListener('click', (e) => {
            if (!this.searchInput.contains(e.target) && !this.searchResults.contains(e.target)) {
                this.searchResults.classList.add('hidden');
            }
        });

        document.getElementById('terminal-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processCommand(e.target.value);
                e.target.value = '';
            }
        });

        window.addEventListener('message', (event) => this.handleMessage(event));

        this.log("ADAM v23.5 Adaptive Shell Online.", "SYSTEM");
    }

    handleRoute() {
        const hash = window.location.hash || '#mission_control';
        const page = this.routes[hash];

        if (page) {
            this.iframe.src = page;
            this.sidebarLinks.forEach(link => {
                if (link.getAttribute('href') === hash) link.classList.add('active');
                else link.classList.remove('active');
            });
            this.log(`Switched context to ${hash}`, 'SYSTEM');
        } else {
            this.log(`Module not found: ${hash}`, 'ERROR');
        }
    }

    toggleMode() {
        this.liveMode = !this.liveMode;
        const btn = document.getElementById('mode-toggle');
        const indicator = document.getElementById('mode-indicator');

        if (this.liveMode) {
            btn.innerHTML = '<i class="fas fa-satellite-dish mr-1"></i> LIVE';
            btn.classList.add('text-green-400');
            indicator.classList.remove('bg-slate-500');
            indicator.classList.add('bg-green-500', 'animate-pulse');
            this.log("Connection Mode: LIVE API (Hybrid)", "SYSTEM");
        } else {
            btn.innerHTML = '<i class="fas fa-cube mr-1"></i> STATIC';
            btn.classList.remove('text-green-400');
            indicator.classList.add('bg-slate-500');
            indicator.classList.remove('bg-green-500', 'animate-pulse');
            this.log("Connection Mode: STATIC SNAPSHOT", "SYSTEM");
        }

        // Notify iframe
        if(this.iframe.contentWindow) {
            this.iframe.contentWindow.postMessage({ type: 'MODE_CHANGE', mode: this.liveMode ? 'LIVE' : 'STATIC' }, '*');
        }
    }

    // Terminal
    toggleTerminal(forceState = null) {
        const currentHeight = this.terminalPane.style.height;
        const isOpen = currentHeight === '300px';
        const shouldOpen = forceState !== null ? forceState : !isOpen;
        this.terminalPane.style.height = shouldOpen ? '300px' : '0px';
        const toggleBtn = document.getElementById('toggle-terminal');
        toggleBtn.classList.toggle('text-cyan-400', shouldOpen);
    }

    log(msg, source = 'SYSTEM') {
        const div = document.createElement('div');
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        let colorClass = 'text-green-400';
        if (source === 'ERROR') colorClass = 'text-red-400';
        if (source === 'WARN') colorClass = 'text-amber-400';
        if (source === 'AGENT') colorClass = 'text-blue-400';
        if (source === 'LLM') colorClass = 'text-purple-400';

        div.innerHTML = `<span class="text-slate-500 mr-2">[${time}]</span> <span class="font-bold text-slate-400 w-16 inline-block text-right mr-2">[${source}]</span> <span class="${colorClass}">${msg}</span>`;
        this.terminalContent.appendChild(div);
        this.terminalContent.scrollTop = this.terminalContent.scrollHeight;
    }

    clearTerminal() { this.terminalContent.innerHTML = ''; }

    async processCommand(cmd) {
        this.log(`$ ${cmd}`, 'USER');
        const args = cmd.trim().split(' ');
        const command = args[0].toLowerCase();

        switch (command) {
            case 'help':
                this.log('CMDS: help, clear, status, agent [name], query [text], open [file]', 'INFO');
                break;
            case 'clear':
                this.clearTerminal();
                break;
            case 'query':
                // Simulate LLM
                this.log("Thinking...", "LLM");
                setTimeout(() => {
                    const response = this.liveMode ?
                        "Simulated Live Response: Based on current market conditions..." :
                        "Static Response: Unable to connect to inference engine. Using cached knowledge graph.";
                    this.log(response, "LLM");
                }, 1000);
                break;
            case 'open':
                if(args[1]) this.searchAndOpen(args[1]);
                else this.log("Usage: open [filename]", "WARN");
                break;
            default:
                this.log(`Unknown command: ${command}`, 'ERROR');
        }
    }

    searchAndOpen(filename) {
        const file = this.repoIndex.files.find(f => f.name === filename || f.path.endsWith(filename));
        if(file) this.openViewer(file.path, file);
        else this.log(`File not found: ${filename}`, 'ERROR');
    }

    // Search
    handleSearch(query) {
        if (!query || query.length < 2) {
            this.searchResults.classList.add('hidden');
            return;
        }
        const results = this.searchIndex(query);
        this.renderSearchResults(results);
    }

    searchIndex(query) {
        const q = query.toLowerCase();
        const matches = [];

        // Search Real Repo Index
        this.repoIndex.files.slice(0, 5000).forEach(f => {
            if (f.name.toLowerCase().includes(q)) {
                matches.push({ type: 'file', title: f.name, subtitle: f.path, data: f });
            }
        });

        // Search Mock Agents (just in case)
        if(window.MOCK_DATA && window.MOCK_DATA.agents) {
            window.MOCK_DATA.agents.forEach(a => {
                if (a.name.toLowerCase().includes(q)) matches.push({ type: 'agent', title: a.name, subtitle: a.role, data: a });
            });
        }

        return matches.slice(0, 10);
    }

    renderSearchResults(results) {
        this.searchResults.innerHTML = '';
        if (results.length === 0) {
            this.searchResults.classList.add('hidden');
            return;
        }

        results.forEach(res => {
            const el = document.createElement('div');
            el.className = 'px-4 py-3 hover:bg-slate-800 cursor-pointer border-b border-slate-800 last:border-0 group';

            let icon = 'fa-file-code text-cyan-500';
            if (res.type === 'agent') icon = 'fa-robot text-amber-400';
            if (res.subtitle.includes('.md')) icon = 'fa-book text-blue-400';
            if (res.subtitle.includes('.json')) icon = 'fa-database text-yellow-400';

            el.innerHTML = `
                <div class="flex items-center">
                    <i class="fas ${icon} w-6 text-center mr-3 group-hover:scale-110 transition"></i>
                    <div class="overflow-hidden">
                        <div class="text-sm font-bold text-slate-200 truncate">${res.title}</div>
                        <div class="text-xs text-slate-500 font-mono truncate">${res.subtitle}</div>
                    </div>
                </div>
            `;
            el.addEventListener('click', () => {
                this.openItem(res);
                this.searchResults.classList.add('hidden');
                this.searchInput.value = '';
            });
            this.searchResults.appendChild(el);
        });

        this.searchResults.classList.remove('hidden');
    }

    openItem(item) {
        if (item.type === 'file') this.openViewer(item.subtitle, item.data);
        else if (item.type === 'agent') window.location.hash = '#agents';
    }

    // Viewer
    async openViewer(path, fileData) {
        const modal = document.getElementById('universal-viewer');
        const title = document.getElementById('viewer-title');
        const meta = document.getElementById('viewer-meta');
        const body = document.getElementById('viewer-body');

        title.textContent = path.split('/').pop();
        meta.textContent = `${path} • ${fileData.size || 0} bytes`;
        body.innerHTML = '<div class="flex items-center justify-center h-full text-cyan-500"><i class="fas fa-circle-notch fa-spin text-4xl"></i></div>';

        modal.classList.remove('hidden');

        try {
            // Attempt to fetch from relative root
            // If we are at showcase/index.html, repo root is ../
            const res = await fetch(`../${path}`);
            if(!res.ok) throw new Error("Fetch failed");
            let content = await res.text();

            const ext = path.split('.').pop().toLowerCase();
            if(['json', 'js', 'py', 'html', 'css', 'md', 'txt'].includes(ext)) {
                if(ext === 'json') content = JSON.stringify(JSON.parse(content), null, 2); // Prettify

                let lang = ext;
                if(ext === 'js') lang = 'javascript';
                if(ext === 'py') lang = 'python';
                if(ext === 'md') {
                    body.innerHTML = `<div class="markdown-body prose prose-invert max-w-none p-4">${marked.parse(content)}</div>`;
                } else {
                    body.innerHTML = `<pre><code class="language-${lang}">${this.escapeHtml(content)}</code></pre>`;
                    Prism.highlightAllUnder(body);
                }
            } else {
                body.innerHTML = `<div class="p-8 text-center text-slate-500">Binary or unsupported file type.<br><br><a href="../${path}" download class="text-cyan-400 hover:underline">Download File</a></div>`;
            }

        } catch (e) {
            body.innerHTML = `<div class="p-8 text-center text-red-400">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i><br>
                Unable to load file content.<br>
                <span class="text-xs text-slate-500">This usually happens in static preview without a local server or if the file is restricted.</span>
            </div>`;
        }
    }

    closeViewer() { document.getElementById('universal-viewer').classList.add('hidden'); }

    escapeHtml(unsafe) {
        return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    handleMessage(event) {
        if (event.data?.type === 'LOG') this.log(event.data.message, event.data.source);
    }
}

window.shell = new Shell();
