/**
 * AdamOS Kernel Bridge (WASM/JS Hybrid)
 * -----------------------------------------------------------------------------
 * This module acts as the interface between the frontend (The Neural Deck) and
 * the AdamOS Kernel. Ideally, it loads the compiled WASM binary.
 * In this prototype environment, it falls back to a JS implementation of the
 * financial physics engine if WASM is unavailable.
 */

// --- JS FALLBACK IMPLEMENTATION ---

class AdamOSKernel {
    constructor() {
        this.agents = new Map();
        this.messageBus = [];
        this.mode = 'JS_FALLBACK';
        this.wasmInstance = null;

        this.initializeWasm();
    }

    async initializeWasm() {
        console.log("AdamOS Kernel: Initializing...");
        try {
            // Attempt to load the WASM module
            // Adjust path based on your build output location
            const wasmModule = await import('./adamos_pkg/adamos_kernel.js');
            await wasmModule.default(); // Initialize WASM
            this.wasmInstance = wasmModule;
            this.mode = 'WASM_NATIVE';
            console.log("✅ AdamOS Kernel: WASM Module Loaded Successfully!");
        } catch (e) {
            console.warn("⚠️ AdamOS Kernel: WASM Module not found or failed to load. Using JS Fallback.", e);
            this.mode = 'JS_FALLBACK';
        }
    }

    registerAgent(name, capabilities) {
        if (this.mode === 'WASM_NATIVE' && this.wasmInstance && this.wasmInstance.registerAgent) {
             // Note: The WASM implementation of registerAgent might need a Kernel instance context
             // or be a method on a Kernel object.
             // For simplicity in this bridge, we assume the WASM module exports a global Kernel class
             // or we are just wrapping the static functions.
             // If we really had a Kernel struct in WASM, we'd instantiate it.
             // Let's stick to the fallback logic for complex state for now unless we fully implement the WASM binding.
             console.log("Using JS Fallback for Agent Registry (WASM Registry TODO)");
        }

        const id = crypto.randomUUID();
        this.agents.set(id, { name, capabilities, status: 'IDLE' });
        console.log(`AdamOS Kernel: Registered Agent ${name} (${id})`);
        return id;
    }

    calculateOptionPrice(S, K, T, r, sigma, isCall) {
        if (this.mode === 'WASM_NATIVE' && this.wasmInstance) {
            try {
                return this.wasmInstance.calculateOptionPrice(S, K, T, r, sigma, isCall);
            } catch (e) {
                console.error("WASM Calculation Error:", e);
                return this._calculateOptionPriceJS(S, K, T, r, sigma, isCall);
            }
        }
        return this._calculateOptionPriceJS(S, K, T, r, sigma, isCall);
    }

    // --- INTERNAL JS MATH ---

    _calculateOptionPriceJS(S, K, T, r, sigma, isCall) {
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
