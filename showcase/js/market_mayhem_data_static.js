// Static Data Injection for Market Mayhem Archive V2
// This file is generated to ensure the dashboard works without a server (file:// protocol)

window.MARKET_MAYHEM_DATA = {};

// Strategic Command
window.MARKET_MAYHEM_DATA.strategic = {
  "meta": {
    "engine": "ConsensusEngineV2",
    "version": "2.0",
    "generated_at": "2026-02-25T02:42:11.046078"
  },
  "strategic_directives": {
    "house_view": "NEUTRAL",
    "score": 0.02,
    "narrative": "Consensus is split. Risk Officer and Growth Strategist are at odds regarding AI Technology."
  },
  "insights": {
    "total_analyzed": 6,
    "active_topics": [
      "AI Technology",
      "Semiconductors",
      "Interest Rates",
      "Energy",
      "Consumer Discretionary"
    ]
  },
  "actionable_plans": [
    {
      "action": "MAINTAIN_BALANCE",
      "target": "Current Allocation",
      "rationale": "No clear directional signal."
    },
    {
      "action": "VOLATILITY_HARVESTING",
      "target": "Options Writing",
      "rationale": "Range-bound market expected."
    }
  ]
};

// S&P 500 Market Data (Truncated for brevity in static file, critical for top lists)
window.MARKET_MAYHEM_DATA.market = [
  {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "current_price": 233.79,
    "change_pct": 1.65,
    "risk_score": 92,
    "outlook": { "conviction": "High" },
    "price_history": [228.91, 228.77, 230.45, 232.56, 231.88, 233.79]
  },
  {
    "ticker": "NVDA",
    "name": "NVIDIA Corp.",
    "sector": "Technology",
    "current_price": 139.31,
    "change_pct": -0.49,
    "risk_score": 75,
    "outlook": { "conviction": "Low" },
    "price_history": [138.98, 137.56, 138.08, 138.61, 138.22, 138.62, 139.31]
  },
  {
    "ticker": "MSFT",
    "name": "Microsoft Corp.",
    "sector": "Technology",
    "current_price": 420.0,
    "change_pct": 0.0,
    "risk_score": 95,
    "outlook": { "conviction": "Medium" },
    "price_history": [420.43, 419.31, 420.54, 421.25, 420.0]
  },
  {
    "ticker": "AMZN",
    "name": "Amazon.com Inc.",
    "sector": "Consumer Discretionary",
    "current_price": 199.28,
    "change_pct": -0.36,
    "risk_score": 85,
    "outlook": { "conviction": "Medium" },
    "price_history": [200.18, 201.04, 199.83, 199.28]
  },
  {
    "ticker": "GOOGL",
    "name": "Alphabet Inc.",
    "sector": "Technology",
    "current_price": 177.25,
    "change_pct": 1.29,
    "risk_score": 88,
    "outlook": { "conviction": "High" },
    "price_history": [176.12, 176.95, 176.17, 177.25]
  },
  {
    "ticker": "TSLA",
    "name": "Tesla Inc.",
    "sector": "Consumer Discretionary",
    "current_price": 315.02,
    "change_pct": -1.56,
    "risk_score": 60,
    "outlook": { "conviction": "Low" },
    "price_history": [318.11, 316.13, 315.36, 314.66, 315.02]
  }
];

// Archive Index (Subset for demonstration)
window.MARKET_MAYHEM_DATA.archive = [
  {
    "title": "Daily Briefing: Sovereign AI & Crypto Supercycle Update",
    "date": "2026-03-27",
    "summary": "Quick update on Sovereign AI & Crypto Supercycle and market movements.",
    "type": "DAILY_BRIEFING",
    "sentiment_score": 50,
    "conviction": 50,
    "filename": "Daily_Briefing_2026_03_27.html",
    "entities": { "keywords": ["AI", "Crypto"] }
  },
  {
    "title": "Market Pulse: Market Melts Up",
    "date": "2026-03-27",
    "summary": "Weekly analysis covering the Sovereign AI & Crypto Supercycle theme. Sentiment is currently Euphoric.",
    "type": "MARKET_PULSE",
    "sentiment_score": 50,
    "conviction": 50,
    "filename": "Market_Pulse_2026_03_27.html",
    "entities": { "keywords": ["AI", "Crypto", "Tech"] }
  },
  {
    "title": "THE GREAT BIFURCATION",
    "date": "2026-03-15",
    "summary": "Market analysis.",
    "type": "MARKET_OUTLOOK",
    "sentiment_score": 0,
    "conviction": 50,
    "filename": "newsletter_market_mayhem_mar_15_2026.html",
    "entities": { "keywords": ["AI", "Banks", "Bitcoin"] }
  },
  {
    "title": "MARKET MAYHEM",
    "date": "2026-03-15",
    "summary": "Market analysis.",
    "type": "MARKET_OUTLOOK",
    "sentiment_score": 20,
    "conviction": 50,
    "filename": "newsletter_market_mayhem_mar_2026.html",
    "entities": { "keywords": ["AI", "Bitcoin", "Energy"] }
  },
  {
    "title": "Conviction Report: Bitcoin (BTC)",
    "date": "2026-02-25",
    "summary": "Digital Gold or Digital Leverage?",
    "type": "CONVICTION_REPORT",
    "sentiment_score": 65,
    "conviction": 80,
    "filename": "conviction_btc_feb26.html",
    "entities": { "keywords": ["Analysis", "Conviction"] }
  }
];
