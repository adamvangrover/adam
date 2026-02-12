window.AGENT_FEED_DATA = {
    agents: [
        { name: "MarketScanner", color: "text-green-400", icon: "fa-chart-line" },
        { name: "RiskGuardian", color: "text-red-400", icon: "fa-shield-alt" },
        { name: "QuantitativeAnalyst", color: "text-blue-400", icon: "fa-calculator" },
        { name: "NarrativeWeaver", color: "text-purple-400", icon: "fa-book-open" },
        { name: "Sentinel", color: "text-yellow-400", icon: "fa-eye" }
    ],
    templates: [
        "Analyzing volume spike in sector {sector}...",
        "Detected anomaly in {ticker}: deviation > 2.5 sigma.",
        "Rebalancing portfolio allocation for {strategy}.",
        "Fetching latest news for {ticker}...",
        "Sentiment analysis for {ticker}: {sentiment}.",
        "Cross-referencing {ticker} with global macro indicators.",
        "Optimizing execution path for trade #{id}.",
        "Scanning for arbitrage opportunities in {market}.",
        "Risk threshold breached for {ticker}, initiating hedge.",
        "Compiling daily briefing report...",
        "Updating knowledge graph node: {entity}."
    ],
    tickers: ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "GOOGL", "AMZN", "BTC-USD", "ETH-USD"],
    sectors: ["Tech", "Energy", "Finance", "Healthcare", "Consumer"],
    strategies: ["Momentum", "Mean Reversion", "Volatility", "Delta Neutral"],
    
    generate() {
        const agent = this.agents[Math.floor(Math.random() * this.agents.length)];
        let text = this.templates[Math.floor(Math.random() * this.templates.length)];
        
        text = text.replace("{ticker}", this.tickers[Math.floor(Math.random() * this.tickers.length)]);
        text = text.replace("{sector}", this.sectors[Math.floor(Math.random() * this.sectors.length)]);
        text = text.replace("{strategy}", this.strategies[Math.floor(Math.random() * this.strategies.length)]);
        text = text.replace("{id}", Math.floor(Math.random() * 9000) + 1000);
        text = text.replace("{sentiment}", ["Bullish", "Bearish", "Neutral"][Math.floor(Math.random() * 3)]);
        text = text.replace("{market}", ["NYSE", "NASDAQ", "Crypto", "Forex"][Math.floor(Math.random() * 4)]);
        text = text.replace("{entity}", "Entity_" + Math.floor(Math.random() * 100));

        return {
            agent: agent,
            text: text,
            timestamp: new Date().toLocaleTimeString()
        };
    }
};
