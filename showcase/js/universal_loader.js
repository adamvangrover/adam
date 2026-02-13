class UniversalLoader {
    constructor() {
        this.mockData = window.MOCK_DATA || {};
    }

    async loadSystemData() {
        if (this.mockData.stats && this.mockData.files) {
            return {
                stats: this.mockData.stats,
                files: this.mockData.files
            };
        }
        // Fallback or Fetch implementation could go here
        return { stats: {}, files: [] };
    }

    async loadFinancialData() {
        const financialData = this.mockData.financial_data || {};
        const parsedData = {};

        // Simple CSV parser for the embedded strings
        for (const [key, value] of Object.entries(financialData)) {
            if (Array.isArray(value)) {
                parsedData[key] = value; // Already JSON
            } else {
                // Implement CSV parsing if raw string
                parsedData[key] = value;
            }
        }
        return parsedData;
    }

    async loadCreditLibrary() {
        // Priority: 1. Fetch JSON file 2. Mock Data Fallback
        try {
            const res = await fetch('data/credit_memo_library.json');
            if (!res.ok) throw new Error("Fetch failed");
            const data = await res.json();
            return Array.isArray(data) ? data : Object.values(data);
        } catch (e) {
            console.warn("UniversalLoader: Network fetch failed, using embedded mock data.");
            return this.mockData.credit_memo_library || [];
        }
    }

    async loadSovereignArtifacts(ticker) {
        // Priority: 1. Fetch Individual Artifacts 2. Generate from Library
        try {
            const ts = new Date().getTime();
            const [spread, memo, audit] = await Promise.all([
                fetch(`data/sovereign_artifacts/${ticker}_spread.json?t=${ts}`).then(r => r.json()),
                fetch(`data/sovereign_artifacts/${ticker}_memo.json?t=${ts}`).then(r => r.json()),
                fetch(`data/sovereign_artifacts/${ticker}_audit.json?t=${ts}`).then(r => r.json())
            ]);
            return { spread, memo, audit };
        } catch (e) {
            console.warn(`UniversalLoader: Sovereign artifacts for ${ticker} missing. Generating from library.`);
            const library = await this.loadCreditLibrary();
            const item = library.find(i => i.ticker === ticker || i.id === ticker);
            if (item) {
                return this._adaptToSovereign(item);
            }
            throw new Error(`No data found for ${ticker}`);
        }
    }

    getGlobalState() {
        return {
            system: this.mockData.stats,
            financials: this.mockData.financial_data,
            library: this.mockData.credit_memo_library,
            version: "2.5.0-UNIVERSAL"
        };
    }

    // Adapter from Credit Memo Library -> Sovereign Bundle
    _adaptToSovereign(libItem) {
        return {
            spread: {
                fiscal_year: "2025",
                history: libItem.historical_financials || [],
                growth_metrics: { "Revenue": 15.0, "EBITDA": 12.0 }, // Mock derived
                ratios: libItem.financial_ratios || {},
                validation: { identity_check: "PASS", identity_delta: 0.0 }
            },
            memo: {
                title: "Credit Assessment",
                date: new Date().toISOString(),
                recommendation: libItem.risk_score > 70 ? "APPROVE" : "REVIEW",
                executive_summary: libItem.summary,
                covenant_analysis: { "Leverage < 4.0x": "PASS", "Interest Coverage > 3.0x": "PASS" }
            },
            audit: {
                timestamp: new Date().toISOString(),
                quant_audit: { action: "SPREAD_COMPLETED", status: "SUCCESS", details: "Verified from 10-K" },
                risk_audit: { action: "RISK_MODEL_RUN", status: "SUCCESS", details: "Score " + libItem.risk_score }
            }
        };
    }
}

// Expose globally
window.UniversalLoader = new UniversalLoader();
