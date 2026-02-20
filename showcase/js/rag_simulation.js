/**
 * ADAM v23.5 RAG SIMULATION ENGINE
 * -----------------------------------------------------------------------------
 * Enhances the 'Evidence Viewer' with high-fidelity mock 10-K document chunks.
 * Replaces generic placeholders with ticker-specific legal/financial text.
 * -----------------------------------------------------------------------------
 */

class RagEngine {
    constructor() {
        this.init();
        this.chunkDatabase = this.buildDatabase();
    }

    init() {
        console.log("[RagEngine] Initializing...");

        // Overwrite the global viewEvidence function from credit_memo.js
        // We wrap it to ensure we capture calls
        const originalView = window.viewEvidence;
        window.viewEvidence = (docId, chunkId, pageNum) => {
            console.log(`[RagEngine] Intercepted view request: ${docId}, ${chunkId}`);
            this.renderDocument(docId, chunkId, pageNum);
        };
    }

    buildDatabase() {
        return {
            "AAPL": {
                "risk": `
                    <div class="doc-page">
                        <div class="doc-header">APPLE INC. | FORM 10-K | FISCAL YEAR 2025</div>
                        <h4 class="doc-title">Item 1A. Risk Factors</h4>
                        <p class="doc-para">The Company’s business, results of operations, financial condition, and stock price have been and may continue to be adversely affected by various factors, including those described below.</p>
                        <p class="doc-para"><span class="highlight-rag">Global Supply Chain Concentration.</span> Substantially all of the Company’s manufacturing is performed in whole or in part by outsourcing partners located primarily in Asia, including mainland China, India, and Vietnam. <span class="highlight-focus">A significant concentration of this manufacturing is currently performed by a small number of outsourcing partners, often in single locations.</span> Political stability in these regions, including trade disputes or conflicts, could disrupt the supply chain.</p>
                        <p class="doc-para">The Company has invested heavily in diversifying its manufacturing footprint; however, this transition entails significant execution risk and capital expenditure.</p>
                    </div>
                `,
                "financial": `
                    <div class="doc-page">
                        <div class="doc-header">APPLE INC. | FORM 10-K | MANAGEMENT DISCUSSION</div>
                        <h4 class="doc-title">Liquidity and Capital Resources</h4>
                        <p class="doc-para">As of September 27, 2025, the Company had $162.1 billion in cash, cash equivalents, and marketable securities. The Company believes its existing balances of cash and cash equivalents, along with commercial paper programs and access to capital markets, will be sufficient to satisfy its working capital needs, capital asset purchases, dividends, and share repurchases for at least the next 12 months.</p>
                    </div>
                `
            },
            "MSFT": {
                "risk": `
                    <div class="doc-page">
                        <div class="doc-header">MICROSOFT CORP | FORM 10-K | FY2025</div>
                        <h4 class="doc-title">Risk Factors: AI and Cloud Computing</h4>
                        <p class="doc-para">We face intense competition across all markets for our products and services. <span class="highlight-rag">The Generative AI sector is characterized by rapid technological change and aggressive entry by competitors.</span> Our significant investments in OpenAI and internal AI infrastructure (Azure Maia) may not yield the expected return on investment if enterprise adoption slows or regulatory barriers emerge.</p>
                    </div>
                `
            },
            "TSLA": {
                "risk": `
                    <div class="doc-page">
                        <div class="doc-header">TESLA, INC. | FORM 10-K | FY2025</div>
                        <h4 class="doc-title">Item 1A. Risks Related to Our Business</h4>
                        <p class="doc-para"><span class="highlight-rag">Key Man Risk.</span> We are highly dependent on the services of Elon Musk, our Technoking and Chief Executive Officer. Although Mr. Musk spends significant time with Tesla and is highly active in our management, he does not devote his full time and attention to Tesla.</p>
                        <p class="doc-para"><span class="highlight-focus">Automotive Gross Margins.</span> Recent price adjustments to adapt to a high-interest rate environment have compressed margins. Continued pricing pressure from competitors in China may further impact profitability.</p>
                    </div>
                `
            },
            "NVDA": {
                "risk": `
                    <div class="doc-page">
                        <div class="doc-header">NVIDIA CORP | FORM 10-K | FY2026</div>
                        <h4 class="doc-title">Risk Factors</h4>
                        <p class="doc-para">Failure to meet the evolving needs of our customers could adversely affect our business. <span class="highlight-rag">We derive a significant portion of our revenue from a limited number of Hyperscale Cloud Providers.</span> Changes in their capital expenditure cycles could result in significant volatility in our Data Center revenue.</p>
                        <p class="doc-para">Furthermore, geopolitical restrictions on the export of high-performance GPUs to certain regions remain a material headwind to total addressable market expansion.</p>
                    </div>
                `
            }
        };
    }

    getTickerContext() {
        // Attempt to find ticker in the DOM
        const title = document.querySelector('#memo-container h1');
        if (title) {
            const text = title.innerText.toUpperCase();
            if (text.includes("APPLE") || text.includes("AAPL")) return "AAPL";
            if (text.includes("MICROSOFT") || text.includes("MSFT")) return "MSFT";
            if (text.includes("TESLA") || text.includes("TSLA")) return "TSLA";
            if (text.includes("NVIDIA") || text.includes("NVDA")) return "NVDA";
        }
        // Fallback checks
        if (document.body.innerText.includes("AAPL")) return "AAPL";
        return "AAPL"; // Default mock
    }

    renderDocument(docId, chunkId, pageNum) {
        const viewer = document.getElementById('pdf-viewer');

        if (!viewer) {
            console.error("[RagEngine] Viewer elements missing");
            return;
        }

        // Determine Content
        const ticker = this.getTickerContext();
        const db = this.chunkDatabase[ticker] || this.chunkDatabase["AAPL"];

        // Simple keyword matching for content type (Risk vs Financial)
        let contentHtml = db.financial; // Default
        if (String(docId).toLowerCase().includes('risk') || String(chunkId).includes('risk')) {
            contentHtml = db.risk;
        }

        // Render (simplified for sidebar context)
        viewer.innerHTML = `
            <div style="background:#525659; padding:10px; height:100%; overflow-y:auto; font-size: 0.8rem;">
                <div style="background:white; color:black; width:100%; min-height:600px; padding:20px; box-shadow:0 0 5px rgba(0,0,0,0.5); font-family:'Times New Roman', serif;">
                    ${contentHtml}

                    <div style="margin-top:20px; border-top:1px solid #ccc; padding-top:5px; color:#666; font-family:sans-serif; font-size:0.8em;">
                        RAG CITATION ID: ${chunkId || 'GENERIC'} | CONFIDENCE: 98.4%
                    </div>
                </div>
            </div>
        `;

        // Inject Styles
        if (!document.getElementById('rag-styles')) {
            const style = document.createElement('style');
            style.id = 'rag-styles';
            style.innerHTML = `
                .doc-header { text-align: center; font-weight: bold; border-bottom: 1px solid black; margin-bottom: 10px; padding-bottom: 5px; font-family: sans-serif; font-size: 0.9em; }
                .doc-title { font-weight: bold; text-transform: uppercase; margin-bottom: 5px; }
                .doc-para { margin-bottom: 10px; line-height: 1.4; text-align: justify; }
                .highlight-rag { background-color: rgba(255, 255, 0, 0.3); border-bottom: 2px solid #f59e0b; cursor: pointer; }
                .highlight-focus { background-color: rgba(0, 255, 157, 0.2); border: 1px dashed #00ff9d; }
            `;
            document.head.appendChild(style);
        }
    }
}

// Auto-Launch
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        window.ragEngine = new RagEngine();
    }, 1500); // Load after main scripts
});
