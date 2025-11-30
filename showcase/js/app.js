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
            const response = await fetch(this.staticBase);
            this.data = await response.json();
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
            const res = await fetch(`/${path}`); // Serve static file from root
            if (!res.ok) return "Error loading file.";
            return await res.text();
        } else {
            // Attempt to fetch relatively if running statically
            try {
                // Assuming index.html is in showcase/ and path is relative to repo root (e.g. README.md)
                // We need to go up one level.
                const res = await fetch(`../${path}`);
                if (res.ok) {
                    return await res.text();
                }
            } catch (e) {
                console.warn("Failed to fetch static file:", path, e);
            }
            return "File content viewing is only available when running the UI Backend server or if files are hosted statically.\n\nRun 'python services/ui_backend.py' to enable full features.";
        }
    }
}

// Global instance
window.dataManager = new DataManager();
