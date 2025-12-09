/**
 * ADAM v23.5 Data Loader
 * Handles asynchronous fetching of "hardened" JSON data.
 * Mimics API latency and handles errors gracefully.
 */

const CONFIG = {
    basePath: '../data/', // Adjust based on deployment relative path
    latency: 300 // Simulate slight API delay for realism
};

export class DataLoader {
    static async fetchJSON(filename) {
        const url = `${CONFIG.basePath}${filename}`;
        
        try {
            // Simulate network latency
            await new Promise(resolve => setTimeout(resolve, CONFIG.latency));

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`[System Error] Failed to load ${filename}:`, error);
            return null; // Return null so UI can handle empty state
        }
    }

    // specific getters for ease of use
    static async getSystemStatus() {
        return this.fetchJSON('system_status.json');
    }

    static async getMarketSnapshot() {
        return this.fetchJSON('market_snapshot.json');
    }

    static async getDeepDiveReports() {
        return this.fetchJSON('deep_dive_reports.json');
    }

    static async getDataVaultFiles() {
        return this.fetchJSON('data_vault_files.json');
    }

    static async getDeploymentLog() {
        return this.fetchJSON('deployment_log.json');
    }
}
