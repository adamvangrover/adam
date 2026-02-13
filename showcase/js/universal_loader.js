/**
 * ADAM Universal Data Loader (v24.1)
 *
 * Unified Data Access Layer for:
 * - Sovereign Credit Artifacts (JSON files in sovereign_artifacts/)
 * - Legacy Credit Library (credit_memo_library.json)
 * - Mock Data Generation (Graceful Fallback)
 *
 * Schema Normalization:
 * Returns a consistent 'CreditMemo' object regardless of source.
 */

class UniversalLoader {
    constructor() {
        this.library = {};
        this.sovereignTickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"];
        this.libraryLoaded = false;
    }

    async init() {
        if (this.libraryLoaded) return;

        try {
            const res = await fetch('data/credit_memo_library.json');
            if (res.ok) {
                const data = await res.json();
                // Normalize array to object map
                if (Array.isArray(data)) {
                    data.forEach(item => {
                        this.library[item.ticker || item.id] = item;
                    });
                } else {
                    this.library = data;
                }
                console.log("[UniversalLoader] Library loaded:", Object.keys(this.library).length, "entries");
            }
        } catch (e) {
            console.warn("[UniversalLoader] Failed to load library index:", e);
        }
        this.libraryLoaded = true;
    }

    getLibrary() {
        return Object.values(this.library).map(item => ({
            id: item.ticker || item.id,
            name: item.borrower_name || item.name,
            ticker: item.ticker || "UNK",
            risk_score: item.risk_score || 50,
            sector: item.sector || "General",
            source: this.sovereignTickers.includes(item.ticker) ? "SOVEREIGN" : "LIBRARY"
        }));
    }

    async loadEntity(ticker) {
        if (!this.libraryLoaded) await this.init();

        // 1. Try Sovereign Artifacts (Highest Fidelity)
        if (this.sovereignTickers.includes(ticker)) {
            try {
                // Fetch in parallel
                const [memoRes, spreadRes, auditRes] = await Promise.all([
                    fetch(`data/sovereign_artifacts/${ticker}_memo.json`),
                    fetch(`data/sovereign_artifacts/${ticker}_spread.json`),
                    fetch(`data/sovereign_artifacts/${ticker}_audit.json`)
                ]);

                if (memoRes.ok && spreadRes.ok) { // Audit is optional
                    const memo = await memoRes.json();
                    const spread = await spreadRes.json();
                    const audit = auditRes.ok ? await auditRes.json() : null;
                    console.log(`[UniversalLoader] Loaded Sovereign Artifacts for ${ticker}`);
                    return this._normalizeSovereign(ticker, memo, spread, audit);
                }
            } catch (e) {
                console.warn(`[UniversalLoader] Sovereign load failed for ${ticker}, falling back...`, e);
            }
        }

        // 2. Try Library / Legacy Data
        const libEntry = this.library[ticker];
        if (libEntry) {
            console.log(`[UniversalLoader] Loaded Library Entry for ${ticker}`);
            return this._normalizeLibrary(libEntry);
        }

        // 3. Graceful Fallback (Mock)
        console.warn(`[UniversalLoader] No data found for ${ticker}. Generating mock.`);
        return this._generateMock(ticker);
    }

    validateAndNormalize(data) {
        if (!data || typeof data !== 'object') {
            console.error("[UniversalLoader] Invalid data format");
            return null;
        }

        // Ensure core structure matches Unified Schema
        return {
            id: data.id || data.ticker || "UNKNOWN",
            name: data.name || "Unknown Entity",
            ticker: data.ticker || "UNK",
            sector: data.sector || "General",
            rating: data.rating || "N/A",
            risk_score: data.risk_score || 50,
            report_date: data.report_date || new Date().toISOString(),
            summary: data.summary || "",

            sections: Array.isArray(data.sections) ? data.sections : [],

            financials: {
                history: Array.isArray(data.financials?.history) ? data.financials.history : [],
                ratios: data.financials?.ratios || {},
                growth_metrics: data.financials?.growth_metrics || {},
                validation: data.financials?.validation || { identity_check: "UNK", identity_delta: 0 }
            },

            valuation: {
                dcf: {
                    enterprise_value: data.valuation?.dcf?.enterprise_value || 0,
                    share_price: data.valuation?.dcf?.share_price || 0,
                    wacc: data.valuation?.dcf?.wacc || 0.08,
                    growth_rate: data.valuation?.dcf?.growth_rate || 0.02
                }
            },

            audit_log: Array.isArray(data.audit_log) ? data.audit_log : [],
            documents: Array.isArray(data.documents) ? data.documents : []
        };
    }

    // --- Normalization Logic ---

    _normalizeSovereign(ticker, memo, spread, audit) {
        return {
            id: ticker,
            name: memo.title ? memo.title.replace(" Credit Memo", "").replace(" Credit Assessment", "") : ticker,
            ticker: ticker,
            sector: "Technology", // Sovereign pipeline focuses on tech for now
            rating: memo.recommendation === "APPROVE" ? "AAA" : (memo.recommendation === "WATCH" ? "BBB" : "CCC"), // Infer
            risk_score: memo.recommendation === "APPROVE" ? 95 : 75,
            report_date: memo.date,
            summary: memo.executive_summary,

            // Sections
            sections: [
                { title: "Executive Summary", content: memo.executive_summary },
                { title: "Risk Factors", content: memo.risk_factors ? memo.risk_factors.map(r => `- ${r}`).join('\n') : "No specific risk factors listed." },
                { title: "Covenant Analysis", content: JSON.stringify(memo.covenant_analysis, null, 2) }
            ],

            // Financials
            financials: {
                history: spread.history || [], // { fiscal_year, revenue, ebitda, ... }
                ratios: spread.ratios || {},
                growth_metrics: spread.growth_metrics || {},
                validation: spread.validation || { identity_check: "PASS", identity_delta: 0 }
            },

            // Valuation
            valuation: {
                dcf: {
                    enterprise_value: spread.valuation?.enterprise_value || 0,
                    share_price: spread.valuation?.implied_share_price || 0,
                    wacc: spread.valuation?.wacc || 0.08,
                    growth_rate: spread.valuation?.terminal_growth || 0.02
                }
            },

            // Audit
            audit_log: audit ? [
                { timestamp: audit.timestamp, action: "INIT", status: "SUCCESS" },
                { timestamp: audit.timestamp, action: "QUANT_CHECK", status: audit.quant_audit?.status || "UNK" },
                { timestamp: audit.timestamp, action: "RISK_CHECK", status: audit.risk_audit?.status || "UNK" }
            ] : [],

            // Documents (Evidence)
            documents: [] // Sovereign doesn't map chunks to bbox in the same way, need to adapt if possible.
            // Future: Inject mock chunks if needed for UI demo
        };
    }

    _normalizeLibrary(item) {
        // Map credit_memo_library.json structure to Unified Schema
        return {
            id: item.ticker || item.id,
            name: item.borrower_name,
            ticker: item.ticker || "UNK",
            sector: item.sector,
            rating: item.borrower_details?.rating || "N/A",
            risk_score: item.risk_score || 50,
            report_date: item.report_date,
            summary: item.summary,

            sections: item.documents?.[0]?.chunks?.filter(c => c.type === 'narrative').map(c => ({
                title: "Narrative Extraction",
                content: c.content
            })) || [],

            financials: {
                // Try to find financial table chunk
                history: this._extractFinancialsFromChunks(item.documents),
                ratios: { "Leverage": 2.5, "Interest Coverage": 8.0 }, // Mock ratios
                growth_metrics: { "Revenue Growth": 5.0, "EBITDA Growth": 3.2 }, // Mock growth
                validation: { identity_check: "PASS", identity_delta: 0.0 }
            },

            valuation: {
                dcf: {
                    enterprise_value: 0,
                    share_price: 0,
                    wacc: 0.08,
                    growth_rate: 0.02
                }
            },

            audit_log: [],

            documents: item.documents || [] // Keep original structure for bbox rendering
        };
    }

    _extractFinancialsFromChunks(docs) {
        if (!docs) return [];
        const tableChunk = docs[0]?.chunks?.find(c => c.type === 'financial_table');
        if (tableChunk && tableChunk.content_json) {
            // Convert single year snapshot to array
            return [{
                period: "FY25", // Mock year
                revenue: tableChunk.content_json.revenue || 0,
                ebitda: tableChunk.content_json.ebitda || 0,
                total_assets: tableChunk.content_json.total_assets || 0,
                total_liabilities: tableChunk.content_json.total_liabilities || 0,
                total_equity: tableChunk.content_json.total_equity || 0
            }];
        }
        return [];
    }

    _generateMock(ticker) {
        const baseRevenue = 10000;
        const growth = 0.05;
        const history = [];
        for (let i = 0; i < 5; i++) {
            const year = 2021 + i;
            const rev = baseRevenue * Math.pow(1 + growth, i);
            history.push({
                period: `FY${year}`,
                fiscal_year: year,
                revenue: rev,
                ebitda: rev * 0.25,
                net_income: rev * 0.15,
                total_assets: rev * 0.8,
                total_debt: rev * 0.3,
                total_equity: rev * 0.5
            });
        }

        return {
            id: ticker,
            name: `${ticker} Corp (Simulated)`,
            ticker: ticker,
            sector: "Simulated Technology",
            rating: "BBB",
            risk_score: 65,
            report_date: new Date().toISOString(),
            summary: `This is a simulated credit memo for ${ticker}. The entity demonstrates stable growth characteristics but lacks primary source data in the repository. Analysis is based on synthetic projection models.`,

            sections: [
                {
                    title: "Executive Summary",
                    content: `The company has shown consistent revenue growth of 5.0% CAGR over the last 5 years. Margins remain healthy at 25% EBITDA. Leverage is manageable at 1.2x Net Debt/EBITDA. Primary risk factors include market volatility and simulated data uncertainty.`
                },
                {
                    title: "Business Overview",
                    content: "Simulated Corp operates in the high-tech sector, providing synthetic data services. The business model is recurring revenue based."
                },
                {
                    title: "Risk Factors",
                    content: "- **Model Risk:** Data is generated by fallback engine.\n- **Market Risk:** High sensitivity to interest rate changes.\n- **Operational:** No real operational history available."
                }
            ],

            financials: {
                history: history,
                ratios: {
                    "Leverage": 1.2,
                    "Interest Coverage": 8.5,
                    "Current Ratio": 1.5,
                    "ROE": 18.2
                },
                growth_metrics: {
                    "Revenue Growth": 5.0,
                    "EBITDA Growth": 5.2
                },
                validation: { identity_check: "WARN", identity_delta: 0.00 }
            },

            valuation: {
                dcf: {
                    enterprise_value: history[4].revenue * 4.5, // ~4.5x Revenue
                    share_price: 145.20,
                    wacc: 0.085,
                    growth_rate: 0.025
                }
            },

            audit_log: [
                { timestamp: new Date().toISOString(), action: "INIT", status: "SUCCESS" },
                { timestamp: new Date().toISOString(), action: "MOCK_GEN", status: "WARNING" }
            ],

            documents: []
        };
    }
}

// Expose globally
if (typeof window !== 'undefined') {
    window.UniversalLoader = new UniversalLoader();
} else if (typeof module !== 'undefined' && module.exports) {
    module.exports = UniversalLoader;
}
