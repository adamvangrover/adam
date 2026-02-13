class NexusSimulator {
    constructor() {
        this.library = null;
        this.isReady = false;
    }

    async init() {
        try {
            const response = await fetch('data/nexus_static_library.json?t=' + Date.now());
            this.library = await response.json();
            this.isReady = true;
            console.log("NexusSimulator v2 initialized:", this.library.metadata.version);
            return true;
        } catch (error) {
            console.error("Failed to initialize NexusSimulator:", error);
            return false;
        }
    }

    search(query) {
        if (!this.isReady || !query) return [];
        const lowerQuery = query.toLowerCase();

        // Search sovereigns and shadow factions
        const allEntities = [...this.library.sovereigns, ...this.library.shadow_factions];
        return allEntities.filter(s => s.toLowerCase().includes(lowerQuery));
    }

    runMonteCarlo(query, iterations = null) {
        if (!this.isReady) throw new Error("Simulator not initialized");

        const config = this.library.simulation_config;
        const count = iterations || config.default_iterations;
        const horizon = config.prediction_horizon_steps;
        const templates = this.library.event_templates;
        const causalGraph = this.library.causal_graph;

        // Use "Crisis" regime for simulation interest, or random
        const regimeName = Math.random() > 0.5 ? "Standard" : "Crisis";
        const regime = this.library.probability_sets[regimeName];

        let target = "Global System";
        const searchResults = this.search(query);
        if (searchResults.length > 0) target = searchResults[0];

        const results = {
            target: target,
            regime: regimeName,
            iterations: count,
            survived: 0,
            collapsed: 0,
            avgStability: 0,
            avgSentiment: 0,
            avgConviction: 0,
            cascadesDetected: 0,
            eventCounts: {},
            chainReactions: [] // Log unique chain reactions
        };

        let totalStability = 0;
        let totalSentiment = 0;
        let totalConviction = 0;

        // Run Loop
        for (let i = 0; i < count; i++) {
            // Initial State (randomized baseline)
            let state = {
                stability: 0.5 + (Math.random() - 0.5) * 0.1,
                sentiment: 50,
                conviction: 50,
                activeTriggers: [] // Events that happened last step
            };

            let survived = true;
            let runCascadeCount = 0;

            for (let t = 0; t < horizon; t++) {
                // 1. Determine Event Pool
                let candidateEvents = [];

                // Check triggers from previous step
                if (state.activeTriggers.length > 0) {
                    for (const triggerEvent of state.activeTriggers) {
                        const chains = causalGraph[triggerEvent];
                        if (chains) {
                            for (const [nextEvtName, prob] of chains) {
                                if (Math.random() < prob * regime.chain_multiplier) {
                                    const nextEvt = templates.find(e => e.type === nextEvtName);
                                    if (nextEvt) {
                                        candidateEvents.push(nextEvt);
                                        // Log rare chains
                                        if (state.activeTriggers.length > 0 && results.chainReactions.length < 5) {
                                            results.chainReactions.push(`${triggerEvent} -> ${nextEvtName}`);
                                        }
                                        runCascadeCount++;
                                    }
                                }
                            }
                        }
                    }
                }

                // If no chain event, pick random 1st order event
                if (candidateEvents.length === 0) {
                    // Filter for base events (mostly 1st order, check via impact or convention)
                    // For simplicity, pick any event, but weigh 1st order higher implicitly
                    const evt = templates[Math.floor(Math.random() * templates.length)];
                    candidateEvents.push(evt);
                }

                // 2. Execute Step
                const event = candidateEvents[0]; // Simplification: 1 major event per step
                results.eventCounts[event.type] = (results.eventCounts[event.type] || 0) + 1;

                // Update State
                const vol = config.volatility_factor * regime.volatility; // Base * Regime
                state.stability += event.impact + (Math.random() - 0.5) * vol;
                state.sentiment += event.sentiment;
                state.conviction += event.conviction;

                // Clamp
                state.stability = Math.max(0, Math.min(1, state.stability));
                state.sentiment = Math.max(0, Math.min(100, state.sentiment));
                state.conviction = Math.max(0, Math.min(100, state.conviction));

                // Set triggers for next step
                state.activeTriggers = [event.type];

                if (state.stability < config.stability_threshold) {
                    survived = false;
                    break;
                }
            }

            if (survived) results.survived++;
            else results.collapsed++;
            if (runCascadeCount > 0) results.cascadesDetected++;

            totalStability += state.stability;
            totalSentiment += state.sentiment;
            totalConviction += state.conviction;
        }

        results.avgStability = totalStability / count;
        results.avgSentiment = totalSentiment / count;
        results.avgConviction = totalConviction / count;

        return results;
    }

    generateScenarioText(result) {
        const survivalRate = ((result.survived / result.iterations) * 100).toFixed(1);
        const cascadeRate = ((result.cascadesDetected / result.iterations) * 100).toFixed(1);

        let sentimentBar = "||||||||||".split("");
        const sentIdx = Math.floor(result.avgSentiment / 10);
        sentimentBar = sentimentBar.map((c, i) => i < sentIdx ? "█" : "░").join("");

        let convictionBar = "||||||||||".split("");
        const convIdx = Math.floor(result.avgConviction / 10);
        convictionBar = convictionBar.map((c, i) => i < convIdx ? "█" : "░").join("");

        // Most frequent
        const topEvents = Object.entries(result.eventCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(e => `> ${e[0]}: ${e[1]}`)
            .join("\n            ");

        const chains = [...new Set(result.chainReactions)].slice(0, 3).map(c => `> ${c}`).join("\n            ");

        return `
            TARGET: ${result.target} [${result.regime.toUpperCase()}]
            ----------------------------------------
            SURVIVAL RATE   : ${survivalRate}%
            CASCADE PROB    : ${cascadeRate}%

            SENTIMENT  : [${sentimentBar}] ${result.avgSentiment.toFixed(0)}/100
            CONVICTION : [${convictionBar}] ${result.avgConviction.toFixed(0)}/100

            DOMINANT VECTORS:
            ${topEvents}

            DETECTED CHAINS (2nd/3rd Order):
            ${chains || "> None detected"}
        `;
    }
}

window.nexusSimulator = new NexusSimulator();
