/**
 * ADAM v24.0 Modular Data Loader
 * -----------------------------------------------------------------------------
 * Provides asynchronous, on-demand loading of data modules to reduce initial
 * page load weight and memory footprint.
 *
 * Supports:
 * - Lazy loading of specific datasets (reports, credit_memos, etc.)
 * - Hybrid fallback (uses window.MOCK_DATA if available)
 * - Error handling and retries
 * -----------------------------------------------------------------------------
 */

class ModularLoader {
    constructor(basePath = 'data/modular/') {
        // Adjust base path if running in a subdirectory
        this.basePath = this._resolveBasePath(basePath);
        this.cache = new Map();
        this.manifest = null;
        console.log(`[ModularLoader] Initialized with base path: ${this.basePath}`);
    }

    _resolveBasePath(path) {
        // Simple logic to handle relative paths in exported modules
        if (window.location.pathname.includes('/exports/')) {
            // If we are deep in an export, we might need to go up or stay relative
            // Assuming the standard export structure puts data/modular adjacent to index.html
            return './data/modular/';
        }
        return path;
    }

    async init() {
        try {
            const response = await fetch(`${this.basePath}manifest.json`);
            if (response.ok) {
                this.manifest = await response.json();
                console.log("[ModularLoader] Manifest loaded:", this.manifest);
            } else {
                console.warn("[ModularLoader] Manifest not found. Proceeding in fallback mode.");
            }
        } catch (e) {
            console.warn("[ModularLoader] Failed to load manifest:", e);
        }
    }

    /**
     * Loads a specific data module.
     * @param {string} moduleName - The key name (e.g., 'reports', 'credit_memos')
     * @returns {Promise<any>} - The requested data
     */
    async load(moduleName) {
        // 1. Check Cache
        if (this.cache.has(moduleName)) {
            console.log(`[ModularLoader] Returning cached: ${moduleName}`);
            return this.cache.get(moduleName);
        }

        // 2. Check Global Fallback (Legacy)
        if (window.MOCK_DATA && window.MOCK_DATA[moduleName]) {
            console.log(`[ModularLoader] Returning legacy global: ${moduleName}`);
            this.cache.set(moduleName, window.MOCK_DATA[moduleName]);
            return window.MOCK_DATA[moduleName];
        }

        // 3. Fetch from Modular JSON
        const filename = `${moduleName}.json`; // Simple mapping, could be dynamic from manifest
        const url = `${this.basePath}${filename}`;

        console.log(`[ModularLoader] Fetching: ${url}`);

        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            this.cache.set(moduleName, data);

            // Optional: Hydrate global scope for compatibility with legacy scripts
            if (!window.MOCK_DATA) window.MOCK_DATA = {};
            window.MOCK_DATA[moduleName] = data;

            return data;
        } catch (error) {
            console.error(`[ModularLoader] Failed to load module '${moduleName}':`, error);
            return null;
        }
    }

    /**
     * Helper to load multiple modules in parallel
     */
    async loadAll(moduleNames) {
        return Promise.all(moduleNames.map(name => this.load(name)));
    }
}

// Attach to window for global access
window.ModularLoader = ModularLoader;
