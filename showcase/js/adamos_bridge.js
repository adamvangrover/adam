/**
 * AdamOS Kernel Bridge (WASM/JS Hybrid)
 * -----------------------------------------------------------------------------
 * This module acts as the interface between the frontend (The Neural Deck) and
 * the AdamOS Kernel. Ideally, it loads the compiled WASM binary.
 * In this prototype environment, it falls back to a JS implementation of the
 * financial physics engine.
 */

const WASM_SUPPORTED = false; // Set to true when .wasm is available

// --- JS FALLBACK IMPLEMENTATION ---

class AdamOSKernel {
    constructor() {
        console.log("AdamOS Kernel: Initialized (JS Mode)");
        this.agents = new Map();
        this.messageBus = [];
    }

    registerAgent(name, capabilities) {
        const id = crypto.randomUUID();
        this.agents.set(id, { name, capabilities, status: 'IDLE' });
        console.log(`AdamOS Kernel: Registered Agent ${name} (${id})`);
        return id;
    }

    calculateOptionPrice(S, K, T, r, sigma, isCall) {
        // Black-Scholes Implementation
        if (T <= 0) return Math.max(0, isCall ? S - K : K - S);

        const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        const d2 = d1 - sigma * Math.sqrt(T);

        const nd1 = this.normalCdf(d1);
        const nd2 = this.normalCdf(d2);

        if (isCall) {
            return S * nd1 - K * Math.exp(-r * T) * nd2;
        } else {
            const nnd1 = this.normalCdf(-d1);
            const nnd2 = this.normalCdf(-d2);
            return K * Math.exp(-r * T) * nnd2 - S * nnd1;
        }
    }

    // Cumulative Distribution Function for Standard Normal Distribution
    normalCdf(x) {
        var t = 1 / (1 + .2316419 * Math.abs(x));
        var d = .3989423 * Math.exp(-x * x / 2);
        var prob = d * t * (.3193815 + t * (-.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        if (x > 0) prob = 1 - prob;
        return prob;
    }
}

// Global singleton
window.AdamOS = new AdamOSKernel();

// Export for module usage if needed
if (typeof module !== 'undefined') {
    module.exports = { AdamOS: window.AdamOS };
}
