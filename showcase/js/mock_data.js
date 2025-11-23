const recentActivities = [
    {
        type: "analysis",
        title: "Credit Risk Analysis: Apple Inc.",
        status: "Completed",
        time: "2 mins ago",
        details: "Risk Score: 12/100 (Low Risk). Verified by 3 sub-agents."
    },
    {
        type: "red_team",
        title: "Adversarial Scenario Generation",
        status: "In Progress",
        time: "5 mins ago",
        details: "Simulating 'Cyber Attack + Interest Rate Hike' scenario."
    },
    {
        type: "monitor",
        title: "Market Watch: Tech Sector",
        status: "Active",
        time: "12 mins ago",
        details: "Detected volatility spike in semiconductor stocks."
    },
    {
        type: "system",
        title: "Knowledge Graph Update",
        status: "Completed",
        time: "1 hour ago",
        details: "Ingested 1,402 new FIBO nodes from SEC filings."
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
