/**
 * ADAM v23.5 Universal Data Loader
 * -----------------------------------------------------------------------------
 * Architect: System Core
 * Context:   Centralized data fetching for Credit & Sovereign applications.
 * Strategy:  Network First -> Mock Data Fallback
 * -----------------------------------------------------------------------------
 */

class UniversalLoader {
    constructor() {
        this.mockData = window.MOCK_DATA || {};
        this.basePath = 'data/';
    }

    /**
     * Loads the Credit Memo Library (list of available reports)
     */
    async loadLibrary() {
        try {
            const res = await fetch(`${this.basePath}credit_memo_library.json`);
            if (res.ok) return await res.json();
            throw new Error("Network fetch failed");
        } catch (e) {
            console.warn("[UniversalLoader] Falling back to mock library");
            // Check if credit_library exists in mockData, otherwise synthesize from credit_memos
            if (this.mockData.credit_library) {
                return this.mockData.credit_library;
            } else if (this.mockData.credit_memos) {
                // Synthesize library from memo details
                return Object.keys(this.mockData.credit_memos).map(key => {
                    const m = this.mockData.credit_memos[key];
                    return {
                        id: key.replace(/ /g, '_'),
                        borrower_name: m.borrower_details?.name || key,
                        risk_score: m.risk_score || 50,
                        file: `credit_memo_${key.replace(/ /g, '_')}.json`,
                        summary: m.summary || "Mock data entry."
                    };
                });
            }
            return [];
        }
    }

    /**
     * Loads a specific Credit Memo by filename or ID
     */
    async loadCreditMemo(identifier) {
        // identifier can be a filename (credit_memo_Apple_Inc.json) or an ID (Apple_Inc)
        let filename = identifier;
        // Clean identifier for searching
        const cleanId = identifier.replace('.json', '').replace('credit_memo_', '');

        try {
            // 1. Try Exact Match
            let path = filename.includes('/') ? filename : `${this.basePath}${filename}`;
            let res = await fetch(path);
            if (res.ok) return await res.json();

            // 2. Try Constructing Filename (if ID provided)
            if (!filename.endsWith('.json')) {
                // Try RAG first
                path = `${this.basePath}credit_memo_${filename}_RAG.json`;
                res = await fetch(path);
                if (res.ok) return await res.json();

                // Try Standard
                path = `${this.basePath}credit_memo_${filename}.json`;
                res = await fetch(path);
                if (res.ok) return await res.json();
            }

            throw new Error(`File not found: ${identifier}`);
        } catch (e) {
            console.warn(`[UniversalLoader] Falling back for ${filename}`);

            const memos = this.mockData.credit_memos || {};

            // 1. Exact match on key
            if (memos[cleanId]) return memos[cleanId];
            if (memos[identifier]) return memos[identifier];

            // 2. Fuzzy match (spaces vs underscores)
            const fuzzyKey = Object.keys(memos).find(k =>
                k.replace(/ /g, '_') === cleanId ||
                cleanId.includes(k.replace(/ /g, '_'))
            );

            if (fuzzyKey) return memos[fuzzyKey];

            // 3. Fallback to first available (demo mode)
            const firstKey = Object.keys(memos)[0];
            if (firstKey) {
                console.log(`[UniversalLoader] Returning default mock for ${identifier}`);
                return memos[firstKey];
            }

            return null;
        }
    }

    /**
     * Loads Sovereign Artifacts (Spread, Memo, Audit)
     * Returns an object { spread, memo, audit }
     */
    async loadSovereignData(ticker, useRag = false) {
        const artifacts = ['spread', 'memo', 'audit'];
        const results = {};
        let failed = false;

        // Try Network
        await Promise.all(artifacts.map(async (type) => {
            try {
                // Construct filename: AAPL_spread.json or AAPL_rag_spread.json
                const filename = useRag ? `${ticker}_rag_${type}.json` : `${ticker}_${type}.json`;
                const res = await fetch(`${this.basePath}sovereign_artifacts/${filename}`);
                if (res.ok) {
                    results[type] = await res.json();
                } else {
                    failed = true;
                }
            } catch (e) {
                failed = true;
            }
        }));

        if (!failed) return results;

        console.warn(`[UniversalLoader] Falling back for Sovereign ${ticker}`);

        // Try Mock Data
        if (this.mockData.sovereign_data && this.mockData.sovereign_data[ticker]) {
            return this.mockData.sovereign_data[ticker];
        }

        // Return partials or null if completely missing
        return failed && Object.keys(results).length > 0 ? results : null;
    }

    /**
     * Loads Market Mayhem Data
     */
    async loadMarketMayhem() {
        try {
            const res = await fetch(`${this.basePath}market_mayhem.json`);
            if (res.ok) return await res.json();
            throw new Error("Network fetch failed");
        } catch (e) {
            console.warn("[UniversalLoader] Falling back for Market Mayhem");
            // Check global variable first (legacy support from inline scripts)
            if (window.MARKET_DATA) return window.MARKET_DATA;
            // Check mockData
            if (this.mockData.market_mayhem) return this.mockData.market_mayhem;

            // Minimal fallback structure to prevent crashes
            return {
                v23_knowledge_graph: {
                    nodes: {
                        macro_ecosystem: { regime_classification: { status: "OFFLINE", inflation_vector: "N/A" } },
                        strategic_synthesis: { final_verdict: { recommendation: "HOLD", conviction_level: 5 } }
                    }
                }
            };
        }
    }
}

// Global Instance
window.universalLoader = new UniversalLoader();
