/**
 * ADAM v23.5 Modular Data Manager
 * -----------------------------------------------------------------------------
 * Context:   Part of the "Optional Data Subsets" initiative.
 * Strategy:  Allows granular loading of large datasets (Reports, Credit Memos, File Index).
 * Usage:     window.modularDataManager.loadReports().then(...)
 * -----------------------------------------------------------------------------
 */

class ModularDataManager {
    constructor() {
        this.basePath = 'data/';
        this.cache = {
            reports: null,
            creditMemos: null,
            files: null
        };
    }

    /**
     * Generic loader helper
     */
    async _loadResource(key, filename) {
        if (this.cache[key]) {
            console.log(`[ModularDataManager] Returning cached ${key}.`);
            return this.cache[key];
        }

        console.log(`[ModularDataManager] Fetching ${key} from ${filename}...`);
        try {
            const res = await fetch(`${this.basePath}${filename}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this.cache[key] = data;
            return data;
        } catch (e) {
            console.error(`[ModularDataManager] Failed to load ${key}:`, e);
            throw e;
        }
    }

    /**
     * Loads the "Reports" subset (formerly part of mock_data.js)
     */
    async loadReports() {
        return this._loadResource('reports', 'seed_reports.json');
    }

    /**
     * Loads the "Credit Memos" subset
     */
    async loadCreditMemos() {
        return this._loadResource('creditMemos', 'seed_credit_memos.json');
    }

    /**
     * Loads the "File Index" subset (huge list of repo files)
     */
    async loadFiles() {
        return this._loadResource('files', 'seed_file_index.json');
    }

    /**
     * Clears local cache to force re-fetch
     */
    clearCache() {
        this.cache = {
            reports: null,
            creditMemos: null,
            files: null
        };
        console.log("[ModularDataManager] Cache cleared.");
    }
}

// Global Instance
window.modularDataManager = new ModularDataManager();
