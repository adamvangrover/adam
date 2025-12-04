class DataManager {
    constructor() {
        this.useApi = localStorage.getItem('adam_live_mode') === 'true';
        this.apiBase = '/api';
        this.staticBase = 'js/mock_data.js'; // We load via script tag usually, but fetch fallback
        this.data = null;
    }

    async init() {
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
