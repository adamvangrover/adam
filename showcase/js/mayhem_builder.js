/**
 * MARKET MAYHEM BUILDER ENGINE
 * -----------------------------------------------------------------------------
 * Generates synthetic historical artifacts based on crisis scenarios.
 * -----------------------------------------------------------------------------
 */

document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('btnGenerate');
    const status = document.getElementById('statusText');
    const paper = document.getElementById('paper');
    const content = document.getElementById('contentArea');

    const SCENARIOS = {
        "1929": {
            title: "BLACK TUESDAY",
            subhead: "The Great Crash wipes out billions. The Roaring Twenties end in a scream.",
            date: "OCTOBER 29, 1929",
            body: [
                "Wall Street is in ruins. The ticker tape ran hours late as panic selling overwhelmed the system. Billions of dollars in paper wealth evaporated in a single session, shattering public confidence and signaling the end of an era.",
                "Crowds gathered outside the Exchange, their faces etched with disbelief. The prosperity that seemed endless has collided with the hard reality of margin calls and over-leverage.",
                "Bankers attempted to stem the tide, but the wave of selling was too powerful. This is not just a correction; it is a fundamental breakage of the financial machine. {TARGET} shares plummeted 40% in early trading, a bellwether for the carnage."
            ]
        },
        "1987": {
            title: "BLACK MONDAY",
            subhead: "Dow plunges 22.6% in single session. Program trading blamed for cascade.",
            date: "OCTOBER 19, 1987",
            body: [
                "The screens turned red and never looked back. In a display of volatility never before seen, the Dow Jones Industrial Average shed nearly a quarter of its value in six and a half hours.",
                "Portfolio insurance, designed to protect capital, instead acted as an accelerant, triggering automatic sell orders that overwhelmed buyers. The market mechanism itself failed. {TARGET}, previously a stalwart, saw no bids for forty-five minutes.",
                "Traders on the floor described a scene of absolute pandemonium. 'It was a free fall,' said one specialist. 'There were no bids. The market just disappeared.'"
            ]
        },
        "2000": {
            title: "DOT COM BUST",
            subhead: "Nasdaq craters as tech bubble bursts. 'New Economy' proven to be a mirage.",
            date: "MARCH 10, 2000",
            body: [
                "Gravity has returned to the markets. The tech-heavy Nasdaq, fueled by companies with no earnings and sky-high valuations, has finally peaked and begun its brutal descent.",
                "Investors who poured billions into 'clicks and eyeballs' are waking up to the reality of cash flow and burn rates. The party is over. {TARGET}, once valued at 50x revenue, is now trading near cash value.",
                "From Pets.com to Webvan, the darlings of the internet age are being decimated. It is a harsh lesson in valuation: a cool URL is not a business model."
            ]
        },
        "2008": {
            title: "SYSTEMIC FAILURE",
            subhead: "Lehman Brothers collapses. Credit markets seize up globally.",
            date: "SEPTEMBER 15, 2008",
            body: [
                "The unthinkable has happened. Lehman Brothers, a titan of Wall Street with a 158-year history, has filed for bankruptcy. The moral hazard play is over, and the fallout is catastrophic.",
                "Credit markets are frozen. Banks have stopped lending to each other. The toxic sludge of subprime mortgages has poisoned the entire global financial water supply. Counterparty risk for {TARGET} has spiked to distress levels.",
                "We are staring into the abyss of a second Great Depression. Central banks are scrambling, but confidence—the currency of the realm—has vanished."
            ]
        },
        "2020": {
            title: "THE GREAT SHUT-IN",
            subhead: "Global economy halts amid pandemic. Fastest bear market in history.",
            date: "MARCH 20, 2020",
            body: [
                "The world has stopped. In an unprecedented move to combat the pandemic, governments have locked down economies, triggering the swiftest market collapse on record.",
                "Oil has turned negative. Volatility has spiked to levels not seen since 2008. The 'dash for cash' is liquidating everything from gold to treasuries. {TARGET} has withdrawn full-year guidance citing 'extreme uncertainty'.",
                "This is not a financial crisis turned economic; it is a biological event forcing a depression-level contraction. The Fed's printing press is the only thing standing between us and total collapse."
            ]
        },
        "2026": {
            title: "THE AI RECKONING",
            subhead: "Compute bubble pops. GPU utilization rates collapse.",
            date: "FEBRUARY 14, 2026",
            body: [
                "The exponential curve has broken. After years of trillion-dollar capex spend, the ROI on Generative AI has failed to materialize, triggering a massive repricing of the tech sector.",
                "Data centers are sitting idle. The 'infinite demand' narrative for compute was a hallucination. As the hyperscalers cut guidance, the entire semiconductor supply chain is imploding. {TARGET} announced a 30% reduction in capex, sending shockwaves through the supply chain.",
                "It is the Dot Com bust with bigger numbers. The market built a Ferrari engine for a go-kart economy. Now, the bill has come due."
            ]
        }
    };

    btn.addEventListener('click', () => {
        const scenarioKey = document.getElementById('scenarioSelect').value;
        const target = document.getElementById('targetInput').value;
        const data = SCENARIOS[scenarioKey];

        // UI State
        status.innerText = "GENERATING...";
        status.style.color = "#f59e0b";
        paper.classList.add('generating');

        // Simulate Processing Delay
        setTimeout(() => {
            // Render
            let bodyText = data.body.map(p => `<p>${p}</p>`).join('');

            // Inject Target if provided
            if (target && target.trim() !== "") {
                bodyText = bodyText.replace(/{TARGET}/g, `<span style="background:rgba(255,255,0,0.2); border-bottom:1px solid yellow;">${target}</span>`);
            } else {
                bodyText = bodyText.replace(/{TARGET}/g, "The sector");
            }

            // Also replace generic terms for color
            if (target && target.trim() !== "") {
                 bodyText = bodyText.replace(/Wall Street|The market|Investors/g, (match) => `<span style="border-bottom:1px dotted #ccc;">${match}</span>`);
            }

            const html = `
                <h1 class="headline">${data.title}</h1>
                <div class="subhead">${data.subhead}</div>
                <div class="date-line">
                    <span>VOL. ${Math.floor(Math.random() * 1000)}</span>
                    <span>DATE: ${data.date}</span>
                    <span>PRICE: $${(Math.random() * 100).toFixed(2)}</span>
                </div>
                <div class="article-body">
                    ${bodyText}
                </div>
            `;

            content.innerHTML = html;

            // Reset UI
            status.innerText = "COMPLETE";
            status.style.color = "#00ff9d";
            paper.classList.remove('generating');

        }, 1500);
    });
});
