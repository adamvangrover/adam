/**
 * ADAM v26.0 SYSTEM 2 JUDGE ENGINE
 * -----------------------------------------------------------------------------
 * Architect: System Core
 * Context:   LLM-as-a-Judge Logic for Conviction Scoring & Critique
 * Status:    Active
 * -----------------------------------------------------------------------------
 */

class System2Judge {
    constructor() {
        this.version = "26.0.1";
        this.knowledgeBase = {
            "Industrial AI": { weight: 1.5, sentiment: "Bullish" },
            "Legacy SaaS": { weight: 0.8, sentiment: "Bearish" },
            "Energy": { weight: 1.4, sentiment: "Bullish" },
            "Crypto": { weight: 1.2, sentiment: "Volatile" }
        };
    }

    /**
     * Evaluates a trading thesis and assigns a conviction score (0-100).
     * @param {Object} data - { symbol, sector, thesis_points: [], data_sources: [] }
     * @returns {Object} - { score, tier, reasoning }
     */
    evaluateConviction(data) {
        let score = 50; // Base score

        // 1. Sector Weighting
        if (this.knowledgeBase[data.sector]) {
            score *= this.knowledgeBase[data.sector].weight;
        }

        // 2. Evidence Density (Heuristic)
        if (data.thesis_points && data.thesis_points.length > 0) {
            score += data.thesis_points.length * 5;
        }

        // 3. Source Quality (RAG Provenance)
        const highQualitySources = ["SEC", "Bloomberg", "Goldman", "Federal Reserve"];
        let sourceBonus = 0;
        if (data.data_sources) {
            data.data_sources.forEach(src => {
                if (highQualitySources.some(hq => src.includes(highQualitySources))) {
                    sourceBonus += 5;
                }
            });
        }
        score += sourceBonus;

        // Cap at 99
        score = Math.min(99, Math.floor(score));

        // Determine Tier
        let tier = "LOW";
        if (score > 85) tier = "HIGH";
        else if (score > 65) tier = "MEDIUM";

        return {
            score: score,
            tier: tier,
            reasoning: `Sector alignment (${data.sector}) + Evidence Density (${data.thesis_points ? data.thesis_points.length : 0} pts) + Source Quality (+${sourceBonus})`
        };
    }

    /**
     * Generates a "Critique" string based on the thesis and score.
     * @param {Object} convictionResult - Result from evaluateConviction
     * @param {Object} data - Original data
     */
    generateCritique(convictionResult, data) {
        const timestamp = new Date().toISOString().split('T')[0];
        let critique = `[${timestamp}] SYSTEM 2 AUDIT: `;

        if (convictionResult.score > 80) {
            critique += `STRONG CONVICTION. The thesis for ${data.symbol} is well-supported by high-quality provenance. Key driver: ${data.thesis_points[0] || 'Macro alignment'}.`;
        } else if (convictionResult.score > 60) {
            critique += `MODERATE CONVICTION. ${data.symbol} thesis is plausible but relies on inference. Recommend monitoring ${data.thesis_points[1] || 'technical levels'}.`;
        } else {
            critique += `LOW CONVICTION / SPECULATIVE. The thesis for ${data.symbol} lacks sufficient verified data. High risk of hallucination or regime shift error.`;
        }

        // Add "Hallucination Check" if no sources
        if (!data.data_sources || data.data_sources.length === 0) {
            critique += " WARNING: Zero-shot inference detected. No RAG sources provided.";
        }

        return critique;
    }

    /**
     * Returns a color code for the score.
     */
    getScoreColor(score) {
        if (score >= 90) return "#00ff9d"; // Cyber Green
        if (score >= 70) return "#00f3ff"; // Cyber Blue
        if (score >= 50) return "#ffaa00"; // Amber
        return "#ff0055"; // Red
    }
}

/**
 * ADAM v26.0 REALTIME MONITOR
 * -----------------------------------------------------------------------------
 * Architect: System Core
 * Context:   Mock Data Feed for Price Provenance
 * -----------------------------------------------------------------------------
 */
class RealtimeMonitor {
    constructor() {
        // Mock "Live" Prices (Simulating Feb 2026 data)
        this.livePrices = {
            "NVDA": 138.20,
            "MSFT": 412.50,
            "CRM": 255.40,
            "BTC": 99100.00,
            "AWAV": 44.10,
            "SPX": 6680.00,
            "DXY": 104.50
        };
    }

    /**
     * Fetches current price for a symbol.
     * In a real system, this would call an external API (Bloomberg/Polygon).
     */
    getPrice(symbol) {
        return this.livePrices[symbol] || null;
    }

    /**
     * Calculates the delta between current price and target.
     * @returns {Object} { delta, percent, status }
     */
    calculateDelta(symbol, target) {
        const current = this.getPrice(symbol);
        if (!current) return null;

        const diff = target - current;
        const percent = (diff / current) * 100;

        // Status relative to target (e.g. if target > current, it's 'Upside')
        const status = diff > 0 ? "UPSIDE" : "DOWNSIDE";

        return {
            current: current,
            delta: diff.toFixed(2),
            percent: percent.toFixed(2) + "%",
            status: status
        };
    }
}

// Attach to window
window.System2Judge = System2Judge;
window.RealtimeMonitor = RealtimeMonitor;
