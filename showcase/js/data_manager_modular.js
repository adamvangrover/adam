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
            files: null,
            agents: null,
            prompts: null,
            trainingData: null,
            architecture: null,
            artisanalData: null
        };

        // State Management: Subscribers
        this.subscribers = {};

        // Performance Metrics
        this.metrics = {
            loads: {}, // key: { count, totalTime, avgTime }
            errors: 0
        };
    }

    /**
     * Subscribe to data changes for a specific key.
     * @param {string} key - The data key (e.g., 'reports')
     * @param {function} callback - Function to call when data is loaded/updated
     */
    subscribe(key, callback) {
        if (!this.subscribers[key]) {
            this.subscribers[key] = [];
        }
        this.subscribers[key].push(callback);

        // If data is already cached, notify immediately
        if (this.cache[key]) {
            callback(this.cache[key]);
        }
    }

    /**
     * Notify subscribers of a data update.
     */
    notify(key, data) {
        if (this.subscribers[key]) {
            this.subscribers[key].forEach(cb => cb(data));
        }
    }

    /**
     * Record performance metric
     */
    _recordMetric(key, duration) {
        if (!this.metrics.loads[key]) {
            this.metrics.loads[key] = { count: 0, totalTime: 0, avgTime: 0 };
        }
        const m = this.metrics.loads[key];
        m.count++;
        m.totalTime += duration;
        m.avgTime = m.totalTime / m.count;
        console.log(`[ModularDataManager] Metric: ${key} loaded in ${duration.toFixed(2)}ms (Avg: ${m.avgTime.toFixed(2)}ms)`);
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
        const startTime = performance.now();

        try {
            const res = await fetch(`${this.basePath}${filename}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            this.cache[key] = data;

            const endTime = performance.now();
            this._recordMetric(key, endTime - startTime);

            this.notify(key, data);

            return data;
        } catch (e) {
            console.error(`[ModularDataManager] Failed to load ${key}:`, e);
            this.metrics.errors++;
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
     * Loads the "Agents" subset
     */
    async loadAgents() {
        return this._loadResource('agents', 'seed_agents.json');
    }

    /**
     * Loads the "Prompts" subset
     */
    async loadPrompts() {
        return this._loadResource('prompts', 'seed_prompts.json');
    }

    /**
     * Loads the "Training Data" subset
     */
    async loadTrainingData() {
        return this._loadResource('trainingData', 'seed_training_data.json');
    }

    /**
     * Loads the "Architecture" subset
     */
    async loadArchitecture() {
        return this._loadResource('architecture', 'system_knowledge_graph.json');
    }

    /**
     * Loads the "Artisanal Data" subset
     */
    async loadArtisanalData() {
        return this._loadResource('artisanalData', 'artisanal/synthetic_batch_001.json');
    }

    /**
     * Clears local cache to force re-fetch
     */
    clearCache() {
        this.cache = {
            reports: null,
            creditMemos: null,
            files: null,
            agents: null,
            prompts: null,
            trainingData: null,
            architecture: null,
            artisanalData: null
        };
        console.log("[ModularDataManager] Cache cleared.");
    }
}

// Global Instance
window.modularDataManager = new ModularDataManager();
