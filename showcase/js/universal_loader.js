/**
 * ADAM v23.5 Universal Data Loader
 * -----------------------------------------------------------------------------
 * Architect: System Core
 * Context:   Centralized data fetching for Credit & Sovereign applications.
 * Strategy:  Network First -> Mock Data Fallback -> Synthetic Generation
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

            // 3. Fallback to Synthetic Generation (Runtime Simulation)
            console.log(`[UniversalLoader] Generating synthetic memo for ${cleanId}`);
            return this.generateSyntheticMemo(cleanId);
        }
    }

    /**
     * Generates a synthetic credit memo for any unknown ticker/name.
     * Mimics backend logic for client-side demo.
     */
    generateSyntheticMemo(name) {
        const cleanName = name.replace(/_/g, ' ');
        // Deterministic-ish random based on name length
        const seed = name.length;
        const riskScore = 60 + (seed % 30); // 60-90

        const sector = seed % 2 === 0 ? "Technology" : "Industrial";
        const revenueBase = 1000 * seed;

        const hist = [
            { period: "2023", revenue: revenueBase, ebitda: revenueBase * 0.2, net_income: revenueBase * 0.1, total_debt: revenueBase * 0.5 },
            { period: "2024", revenue: revenueBase * 1.05, ebitda: revenueBase * 0.22, net_income: revenueBase * 0.12, total_debt: revenueBase * 0.45 },
            { period: "2025", revenue: revenueBase * 1.1, ebitda: revenueBase * 0.25, net_income: revenueBase * 0.15, total_debt: revenueBase * 0.4 }
        ].reverse(); // Descending order usually

        return {
            "borrower_name": cleanName,
            "borrower_details": { "name": cleanName, "sector": sector },
            "report_date": new Date().toISOString(),
            "risk_score": riskScore,
            "historical_financials": hist,
            "sections": [
                {
                    "title": "Executive Summary",
                    "content": `Synthetic analysis generated for ${cleanName}. The company operates in the ${sector} sector with stable margins.\n\nKey Metrics:\n- Revenue: $${(revenueBase/1000).toFixed(1)}B\n- EBITDA Margin: ~20%`,
                    "citations": [],
                    "author_agent": "Client-Side Sim"
                },
                {
                    "title": "Risk Analysis",
                    "content": "Automated Risk Assessment:\n1. Market Competition\n2. Supply Chain dependencies\n3. Rate sensitivity",
                    "citations": [],
                    "author_agent": "Risk Assessment Agent"
                }
            ],
            "key_strengths": ["Projected Growth", "Market Position"],
            "key_weaknesses": ["Leverage Ratio", "Sector Volatility"],
            "dcf_analysis": {
                "enterprise_value": revenueBase * 2.5,
                "share_price": 120.50,
                "wacc": 0.09,
                "growth_rate": 0.03,
                "terminal_value": revenueBase * 3.0,
                "free_cash_flow": [revenueBase*0.1, revenueBase*0.11, revenueBase*0.12, revenueBase*0.13, revenueBase*0.14]
            },
            "pd_model": {
                "model_score": riskScore,
                "implied_rating": riskScore > 80 ? "A-" : (riskScore > 60 ? "BBB" : "BB"),
                "one_year_pd": 0.02,
                "five_year_pd": 0.08,
                "input_factors": { "Leverage": "2.5x", "Z-Score": "2.8" }
            },
            "system_two_critique": {
                "critique_points": ["Synthetic generation successful.", "Data is simulated."],
                "conviction_score": 0.75,
                "verification_status": "PASS",
                "author_agent": "System 2"
            },
            "equity_data": {
                "share_price": 120.50,
                "market_cap": revenueBase * 2.0,
                "beta": 1.1,
                "pe_ratio": 20.0
            },
            "debt_facilities": [
                 {"facility_type": "Revolver", "amount_committed": revenueBase * 0.2, "amount_drawn": 0, "interest_rate": "S+200", "snc_rating": "Pass", "ltv": 0.0},
                 {"facility_type": "Term Loan", "amount_committed": revenueBase * 0.5, "amount_drawn": revenueBase * 0.5, "interest_rate": "S+350", "snc_rating": "Pass", "ltv": 0.4}
            ],
            "repayment_schedule": [
                {"year": "2026", "amount": revenueBase * 0.1, "source": "Amortization"},
                {"year": "2027", "amount": revenueBase * 0.4, "source": "Maturity"}
            ]
        };
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
