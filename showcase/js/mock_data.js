const recentActivities = [
    {
        type: "analysis",
        title: "Credit Risk Analysis: Apple Inc.",
        status: "Completed",
        time: "Just now",
        details: "Risk Score: 12/100 (Low Risk). Verified by 3 sub-agents. Sentiment: Positive."
    },
    {
        type: "red_team",
        title: "Red Team: Global Cyber Event",
        status: "In Progress",
        time: "2 mins ago",
        details: "Simulating 'Coordinated Ransomware Attack' on Payment Rails. Impact: High."
    },
    {
        type: "analysis",
        title: "SNC Review: Project Titan",
        status: "Review",
        time: "5 mins ago",
        details: "Syndicate structure analysis complete. Flagged 2 covenants for review."
    },
    {
        type: "monitor",
        title: "Market Watch: Semiconductor Sector",
        status: "Active",
        time: "12 mins ago",
        details: "Detected volatility spike (>3 sigma) in NVDA, AMD. Correlating with Taiwan news."
    },
    {
        type: "system",
        title: "Neuro-Symbolic Planner",
        status: "Optimization",
        time: "15 mins ago",
        details: "Optimized query path for 'ESG Impact on Credit'. Reduced hops by 40%."
    },
    {
        type: "analysis",
        title: "ESG Audit: Exxon Mobil",
        status: "Completed",
        time: "28 mins ago",
        details: "Environmental Score: 45/100. Governance Score: 82/100. Report generated."
    },
    {
        type: "system",
        title: "Knowledge Graph Update",
        status: "Completed",
        time: "1 hour ago",
        details: "Ingested 1,402 new FIBO nodes and 500 PROV-O edges from SEC filings."
    },
    {
        type: "red_team",
        title: "Stress Test: 2008 Crisis Replay",
        status: "Completed",
        time: "2 hours ago",
        details: "Portfolio resilience tested against 2008 liquidity crunch. VaR increased by 15%."
    }
];

const financialData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [{
        label: 'Portfolio Value',
        data: [12000, 19000, 3000, 5000, 2000, 3000],
        borderWidth: 1
    }]
};
