class DataManager {
    constructor() {
        this.useApi = true;
        this.apiBase = '/api';
        this.staticBase = 'data/ui_data.json';
        this.data = null;
    }

    async init() {
        try {
            await this.fetchData();
        } catch (e) {
            console.warn("API failed, falling back to static data", e);
            this.useApi = false;
            await this.fetchData();
        }
    }

    async fetchData() {
        if (this.useApi) {
            try {
                const response = await fetch(`${this.apiBase}/state`);
                if (!response.ok) throw new Error("API Error");
                this.data = await response.json();
            } catch (e) {
                this.useApi = false;
                return this.fetchData();
            }
        } else {
            // Check global mock data first (safe for file://)
            if (window.MOCK_DATA) {
                this.data = window.MOCK_DATA;
                console.log("Loaded data from window.MOCK_DATA");
                return this.data;
            }

            try {
                const response = await fetch(this.staticBase);
                if(response.ok) {
                    this.data = await response.json();
                } else {
                    throw new Error("Static fetch failed");
                }
            } catch (e) {
                console.warn("Failed to fetch static JSON:", e);
                // Last resort fallback
                this.data = {
                    system_stats: { cpu_usage: 0, memory_usage: 0, active_tasks: 0 },
                    files: [],
                    agents: []
                };
            }
        }
        return this.data;
    }

    getSystemStats() {
        return this.data?.system_stats || {};
    }

    getFiles() {
        return this.data?.files || [];
    }

    getAgents() {
        return this.data?.agents || [];
    }

    async getFileContent(path) {
        if (this.useApi) {
            try {
                const res = await fetch(`/${path}`); // Serve static file from root
                if (!res.ok) throw new Error("File not found");
                return await res.text();
            } catch(e) {
                return `Error loading file via API: ${e.message}`;
            }
        } else {
            // Attempt to fetch relatively if running statically
            try {
                // If path is absolute from repo root like 'README.md', and we are in 'showcase/', we need '../README.md'
                const relPath = `../${path}`;
                const res = await fetch(relPath);
                if (res.ok) {
                    return await res.text();
                }
            } catch (e) {
                console.warn("Failed to fetch static file:", path, e);
            }
            return "File content viewing is only available when running a local server (e.g. 'python -m http.server').\n\nRun 'python services/ui_backend.py' to enable full API features.";
        }
    }
}

// Global instance
window.dataManager = new DataManager();
