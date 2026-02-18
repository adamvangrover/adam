class DataManager {
    constructor() {
        this.useApi = localStorage.getItem('adam_live_mode') === 'true';
        this.apiBase = '/api';
        this.staticBase = 'js/mock_data.js';
        this.data = null;
    }

    async init() {
        // Try to connect to API if in live mode
        if (this.useApi) {
            try {
                const response = await fetch(`${this.apiBase}/state`);
                if (response.ok) {
                    const status = await response.json();
                    console.log("Connected to Live API:", status);

                    // Load basic structure from Mock Data (files, agents list)
                    // but override status
                    this.loadMock();
                    if(this.data) {
                        this.data.stats.version = status.version;
                        this.data.stats.orchestrator = status.orchestrator;
                    }
                } else {
                    console.warn("API Error, falling back to mock.");
                    this.useApi = false;
                    this.loadMock();
                }
            } catch (e) {
                console.warn("API Unreachable, falling back to mock.");
                this.useApi = false;
                this.loadMock();
            }
        } else {
            this.loadMock();
        }
    }

    loadMock() {
        if (window.MOCK_DATA) {
            this.data = window.MOCK_DATA;
            console.log("Loaded data from window.MOCK_DATA");
        } else {
            this.data = this.getEmptyState();
        }
    }

    getEmptyState() {
        return {
            stats: { cpu_usage: 0, memory_usage: 0, active_tasks: 0, version: "Offline" },
            files: [],
            agents: [],
            reports: [],
            prompts: [],
            strategies: [],
            training_sets: []
        };
    }

    // Accessors
    getStats() { return this.data?.stats || {}; }
    getSystemStats() { return this.getStats(); }
    getAgents() { return this.data?.agents || []; }
    getReports() { return this.data?.reports || []; }
    getPrompts() { return this.data?.prompts || []; }
    getStrategies() { return this.data?.strategies || []; }
    getTrainingSets() { return this.data?.training_sets || []; }

    toggleApiMode() {
        this.useApi = !this.useApi;
        localStorage.setItem('adam_live_mode', this.useApi);
        window.location.reload();
    }

    async sendQuery(query) {
        if (!this.useApi) {
            // Mock Response
            console.log("Mock Query:", query);
            return new Promise(resolve => setTimeout(() => {
                resolve({
                    status: "Mock Response",
                    human_readable_status: "This is a mock response. Enable Live Mode (Top Right) to query the real Adam AI.",
                    v23_knowledge_graph: {
                        meta: { target: "Mock Target" },
                        nodes: {
                            strategic_synthesis: {
                                conviction: "High",
                                investment_thesis: "This is a simulation. Connect to the backend for real analysis."
                            }
                        }
                    }
                });
            }, 1500));
        }

        const response = await fetch(`${this.apiBase}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "API Request Failed");
        }

        return await response.json();
    }
}

class SettingsManager {
    constructor() {
        this.theme = localStorage.getItem('adam_theme') || 'cyber';
        this.applyTheme();
    }

    toggleTheme() {
        this.theme = this.theme === 'cyber' ? 'minimal' : 'cyber';
        localStorage.setItem('adam_theme', this.theme);
        this.applyTheme();
    }

    applyTheme() {
        if (this.theme === 'minimal') {
            document.body.classList.add('minimal-theme');
            document.body.classList.remove('cyber-theme');
        } else {
            document.body.classList.add('cyber-theme');
            document.body.classList.remove('minimal-theme');
        }
    }
}

class GlobalSearch {
    constructor(dataManager) {
        this.dm = dataManager;
    }

    search(query) {
        if (!query || query.length < 2) return [];
        query = query.toLowerCase();

        const results = [];

        // Agents
        this.dm.getAgents().forEach(a => {
            if (a.name.toLowerCase().includes(query) || (a.docstring && a.docstring.toLowerCase().includes(query))) {
                results.push({ type: 'Agent', title: a.name, subtitle: a.type, link: 'agents.html' });
            }
        });

        // Reports
        this.dm.getReports().forEach(r => {
            if (r.title.toLowerCase().includes(query)) {
                results.push({ type: 'Report', title: r.title, subtitle: 'Deep Dive', link: 'reports.html' });
            }
        });

        // Strategies
        this.dm.getStrategies().forEach(s => {
            if (s.name.toLowerCase().includes(query)) {
                results.push({ type: 'Strategy', title: s.name, subtitle: s.risk_level, link: 'financial_twin.html' });
            }
        });

        // Repo Files
        if (window.REPO_DATA && window.REPO_DATA.nodes) {
             window.REPO_DATA.nodes.forEach(node => {
                if (node.label.toLowerCase().includes(query)) {
                    results.push({ type: 'File', title: node.label, subtitle: node.path, link: 'data.html?file=' + encodeURIComponent(node.path) });
                }
             });
        }

        return results.slice(0, 10); // Limit results
    }
}

// Initialize
window.dataManager = new DataManager();
window.settingsManager = new SettingsManager();
window.dataManager.init();
window.globalSearch = new GlobalSearch(window.dataManager);

// Bind to UI if present
document.addEventListener('DOMContentLoaded', () => {
    // Settings Toggles
    const themeBtn = document.getElementById('theme-toggle');
    if(themeBtn) themeBtn.addEventListener('click', () => window.settingsManager.toggleTheme());

    const apiBtn = document.getElementById('api-toggle');
    if(apiBtn) apiBtn.addEventListener('click', () => window.dataManager.toggleApiMode());
});
