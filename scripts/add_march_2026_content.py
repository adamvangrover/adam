import json
import os
from datetime import datetime

NEWSLETTER_DATA_PATH = "showcase/data/newsletter_data.json"
INDEX_DATA_PATH = "showcase/data/market_mayhem_index.json"

new_entries = [
    {
        "date": "2026-03-08",
        "title": "🔴 SYSTEM STATUS: DEGRADED (Kinetic Conflict Injection)",
        "summary": "The simulation has entered a high-volatility state following a kinetic escalation in the Middle East over the weekend. The architecture is struggling to reconcile a 'soft landing' narrative with a sudden 'War Premium' re-render.",
        "type": "DAILY_BRIEFING",
        "filename": "Daily_Briefing_2026_03_08.html",
        "is_sourced": True,
        "full_body": """<h2>Signal Integrity: The Middle East War-Patch</h2>
<p>The simulation has entered a high-volatility state following a kinetic escalation in the Middle East over the weekend. The architecture is struggling to reconcile a "soft landing" narrative with a sudden "War Premium" re-render.</p>
<p>The S&P 500 slipped -0.43% to 6,878.88, but the headline number hides the internal packet loss. This was a classic "Gap-and-Trap" session where early losses of -1% were partially bought back, yet the underlying plumbing remains under extreme tension.</p>
<p><strong>Credit Dominance Check:</strong> We are seeing a <strong>Systemic Inversion</strong>. While equities attempted to find a floor, the <strong>10-Year Treasury Yield surged to 4.05% (+9bps)</strong>. This is a "Hawkish Flight-to-Safety" anomaly; safe-haven demand for bonds was completely overwhelmed by the fear that $90+ oil will hard-code a new wave of inflation.</p>
<p><strong>The Verdict: IT’S A TRAP.</strong> High-yield spreads (HYG/JNK) are under pressure as energy prices spike, raising the cost of carry for the entire industrial architecture. When yields jump alongside a spike in the <strong>VIX (+18.4% to 23.5 intraday, closing near 20)</strong>, the equity "bounce" is merely a liquidity artifact. The market is pricing in a "No-Cut" scenario for the foreseeable future.</p>
<h3>Artifacts</h3>
<ul>
<li><strong>Bitcoin ($69,483 | +6.3%):</strong> The "Digital Gold" render is finally operational. BTC decoupled from the Nasdaq today, reclaiming the $69k handle as it captures "Crisis Alpha" while the traditional fiat architecture glitches.</li>
<li><strong>Crude Oil (WTI | +7%):</strong> The primary disruptor. The death of the Iranian Supreme Leader and subsequent strikes have injected a massive supply-chain virus into the system.</li>
<li><strong>MicroStrategy (MSTR | +6.3%):</strong> A high-fidelity proxy for the BTC reclaim. Strategy Inc. reported another 3,015 BTC buy today, doubling down on the "Bitcoin Treasury" code.</li>
<li><strong>Airlines & Logistics:</strong> Critical system failure. Surging fuel costs are rendering these sectors' Q1 earnings projections obsolete in real-time.</li>
</ul>
<h3>The Glitch</h3>
<blockquote>"We spent years building a digital cathedral of AI and automation, only to be reminded that the entire simulation still runs on 20th-century fossil fuels. Today, the 'War Premium' deleted the 'Rate Cut' fantasy. Bitcoin at $69k is a lonely signal of trust in a system where the 10-Year yield and Oil are both screaming 'Inflation.' The Dow's 10-month winning streak is the last monument standing, but the VIX at 23 is the sound of the foundation cracking. We aren't trading cash flows anymore; we are trading the speed of the kinetic escalation."</blockquote>
<p><strong>Next Step:</strong> With the 10-year yield surging back to 4.05% and Oil at $90, would you like me to run a <strong>"Credit Default Sensitivity" scan</strong> on the major airlines and logistics firms to see whose debt-load hits the "insolvency trigger" first at these energy prices?</p>""",
        "source_priority": 3,
        "conviction": 95,
        "sentiment_score": 10
    },
    {
        "date": "2026-03-09",
        "title": "Geopolitical and Economic Reverberations of the 2026 Iranian Collapse",
        "summary": "Cascading Impacts on Global Energy Markets and United States Leveraged Credit. The abrupt escalation of military hostilities in the Middle East in March 2026 has fundamentally destabilized the global macroeconomic baseline.",
        "type": "NEWSLETTER",
        "filename": "newsletter_iran_collapse_2026.html",
        "is_sourced": True,
        "full_body": """<h2>1. Executive Summary</h2>
<p>The abrupt escalation of military hostilities in the Middle East in March 2026, culminating in direct United States and Israeli kinetic strikes on Iranian nuclear and military infrastructure, has fundamentally destabilized the global macroeconomic baseline. The subsequent retaliatory maneuvering by Iran’s Islamic Revolutionary Guard Corps (IRGC) to restrict maritime traffic through the Strait of Hormuz has paralyzed the world’s most critical artery for global energy commerce.</p>
<p>This acute geopolitical dislocation arrives at a highly precarious moment for United States financial markets, specifically the deeply interconnected $1.2 trillion broadly syndicated leveraged loan market and the rapidly expanding $1.3 trillion private credit ecosystem. The prospect of a sustained oil price shock—with Brent crude modeled to reach between $120 and $150 per barrel in severe disruption scenarios—acts as a highly regressive, systemic tax on corporate margins.</p>
<p>The transmission mechanism from the Persian Gulf to the United States leveraged finance market is highly complex. Surging energy input costs relentlessly compress operating margins, while the inflationary impulse threatens to definitively stall or reverse the Federal Reserve's easing cycle.</p>
<h2>2. The Geopolitical Catalyst: The Strait of Hormuz and Global Supply Disruption</h2>
<p>The strategic geography of the Strait of Hormuz establishes it as the ultimate maritime chokepoint in the global energy infrastructure network. The March 2026 hostilities have effectively severed the flow of millions of barrels of crude oil and billions of cubic feet of natural gas.</p>
<p>The strategic calculus for Tehran regarding the closure of the Strait is exceptionally complex and inherently paradoxical. Closing the Strait operates as a double-edged sword; while it inflicts maximum economic damage on Western economies, it simultaneously devastates Iran's own revenue streams.</p>
<h2>3. Global Energy Price Shocks: Scenario Modeling and Volatility Dynamics</h2>
<p>The market response to supply disruptions of the magnitude seen in the Strait of Hormuz is historically violent. Rather than a linear, predictable price increase, commodities markets exhibit asymmetric upside volatility.</p>
<p>Goldman Sachs Global Investment Research projections from February 2026 indicate that a sustained disruption could elevate Brent crude to a sustained $150–$180 range, with short-term spikes eclipsing $200 per barrel.</p>
<h2>4. Macroeconomic Transmission: Inflation, Monetary Policy, and Fiscal Fragility</h2>
<p>The kinetic events in the Middle East do not impact United States corporate credit in a vacuum. They intersect with a highly complex, pre-existing domestic macroeconomic environment defined by an ongoing battle against sticky services inflation, record-high peacetime sovereign debt burdens, and a newly implemented, highly aggressive protectionist trade regime (17% to 18% average tariffs).</p>
<h2>5. Structural Fragility in the United States Leveraged Loan Market</h2>
<p>The modern leveraged loan ecosystem is fundamentally different from the market that existed during the 2008 global financial crisis. The most critical structural vulnerability defining the 2026 leveraged loan market is the absolute ubiquity of covenant-lite ("cov-lite") loan structures. By late 2021, cov-lite loans accounted for more than 86% of outstanding volume, completely stripping away traditional early-warning mechanisms and creditor protections.</p>
<h2>Conclusion</h2>
<p>The intersection of a Middle Eastern kinetic conflict, a closed Strait of Hormuz, and a highly leveraged, covenant-lite United States corporate credit market creates a perfect storm of financial instability. The defining characteristic of the coming credit cycle will be the extreme friction between economic reality and loan documentation.</p>""",
        "source_priority": 4,
        "conviction": 85,
        "sentiment_score": 15
    },
    {
        "date": "2026-03-10",
        "title": "Market Mayhem: The Adam Financial System Intelligence Briefing",
        "summary": "Phase 2: Sentiment & Synthesis - The global financial ecosystem currently executes a violent rotation from artificial intelligence exuberance to aggressive risk hedging.",
        "type": "NEWSLETTER",
        "filename": "newsletter_market_mayhem_mar_10_2026.html",
        "is_sourced": True,
        "full_body": """<h2>Phase 2: Sentiment & Synthesis</h2>
<h3>The "Vibe Check"</h3>
<p>The global financial ecosystem currently executes a violent rotation from artificial intelligence exuberance to aggressive risk hedging. Synthesizing real-time cross-asset flows, options market positioning, and deep-web macroeconomic data via FinBERT indicates the market sits firmly in a "Hedging" regime. Equities take a structural beating as the semiconductor narrative collides with physical energy constraints and sovereign defense ultimatums.</p>
<p><strong>Overall Market Sentiment Score: -0.45.</strong></p>
<h2>Phase 3: Content Generation</h2>
<h3>Headlines from the Edge</h3>
<ul>
<li><strong>OpenAI's $110 Billion Gravity Well:</strong> Generative AI leader reaches an $840 billion valuation with backing from Amazon, Nvidia, and SoftBank.</li>
<li><strong>The Pentagon's 5:01 PM Ultimatum:</strong> Anthropic formally rejects the Department of Defense's demand for unrestricted military AI use.</li>
<li><strong>Shadow Banking's £930M Cockroach:</strong> UK mortgage lender Market Financial Solutions collapses amid severe "double pledging" fraud allegations.</li>
</ul>
<h3>Adam's Alpha</h3>
<h4>Theme 1: Hydrocarbon Statecraft and the Venezuelan Arbitrage</h4>
<p>The geopolitical narrative surrounding global energy markets shifted radically this week following the Trump administration's explicit goal to drive US oil prices down to $50 per barrel utilizing massive crude reserves from Venezuela. The quantitative alpha dictates going long on the equity of complex US refiners (VLO, PSX) capable of processing this heavy crude.</p>
<h4>Theme 2: Silicon Sovereignty and the Infrastructure Leviathan</h4>
<p>The artificial intelligence sector officially transitions from a speculative software boom into the most capital-intensive physical infrastructure buildout in modern economic history. The structural trade avoids purchasing OpenAI equity on the secondary market or blindly chasing Nvidia. The true alpha resides in the physical constraints of the technology: energy generation, thermal management, and data center real estate.</p>
<h4>Theme 3: Demographic Deflation and the 28% Alpha</h4>
<p>While equity markets remain fixated on the automation potential of generative AI, raw labor market data reveals a contradictory and highly profitable trend: a structural, permanent deficit in global human capital. AI skills now command a massive 28% salary premium globally.</p>
<h3>The "Macro Glitch"</h3>
<p>In any complex, highly optimized financial system, catastrophic failure rarely begins with a massive, visible explosion. It begins with a glitch. On Friday, February 27, 2026, Wall Street and City of London credit desks suffered violent jolts upon the implosion of Market Financial Solutions (MFS), a UK-based bridging and specialist property lender. The firm fell into administration following aggressive legal action from its own asset-based funding vehicles, exposing Tier-1 global investment banks to severe liquidity mismatches and "double pledging" fraud contagion.</p>""",
        "source_priority": 5,
        "conviction": 80,
        "sentiment_score": 25
    },
    {
        "date": "2026-03-11",
        "title": "The 2026 Global Intelligence Crisis: Reconciling Macroeconomic Realities",
        "summary": "Deconstructing the speculative narrative of 'Ghost GDP'. A rigorous examination of the macroeconomic fundamentals in early 2026 reveals a constructive reality despite market hysteria over AI labor displacement.",
        "type": "DEEP_DIVE",
        "filename": "deep_dive_ghost_gdp_2026.html",
        "is_sourced": True,
        "full_body": """<h2>Introduction: The Macroeconomic Paradox of 2026</h2>
<p>By the end of the first quarter of 2026, the global macroeconomic environment has reached a historical inflection point characterized by a profound and highly visible divergence between empirical economic data and speculative market narratives. Advanced economies are experiencing sturdy growth, yet a pervasive undercurrent of systemic anxiety has permeated financial markets, catalyzed by the "Global Intelligence Crisis" memo predicting an imminent "Human Intelligence Displacement Spiral."</p>
<h2>Deconstructing the Speculative Narrative: The "Ghost GDP" Hypothesis</h2>
<p>The central tenet of the 2028 crisis scenario is the emergence of a structural anomaly termed "Ghost GDP." In this theoretical framework, the rapid deployment of agentic AI systems allows corporations to aggressively substitute human labor with scalable compute. The scenario argues this severs the circular flow of macroeconomic income, causing aggregate demand to collapse while measured output rises—a dynamic that violates fundamental national income accounting identities.</p>
<h2>Empirical Labor Market Dynamics in 2026: Evidence Over Extrapolation</h2>
<p>The fundamental premise of the imminent labor collapse theory requires observable, systemic deterioration in high-skill employment data. However, the labor market of early 2026 directly contradicts the narrative. The United States unemployment rate sits at a highly resilient 4.28%, and demand for software engineers is actually rising (up 11% year-over-year according to Indeed), proving AI acts primarily as a complement that alters task composition rather than an absolute labor substitute.</p>
<h2>The Physical and Thermodynamic Boundaries of Artificial Intelligence</h2>
<p>The speculation surrounding infinite, frictionless intelligence scaling ignores the profound material realities of the physical world. Artificial intelligence is hard-bounded by silicon fabrication limits, thermodynamics, global supply chains, and the severe constraints of the physical power grid (projected to hit 945 TWh by 2030 for data centers globally). The immense capital expenditure requirement—currently $650B to $674B annually in the US—imposes a rising marginal cost of compute that serves as a definitive economic brake on total labor substitution.</p>
<h2>Macroeconomic Policy and Fiscal Stimulus: The Impact of the OBBBA</h2>
<p>Assessments of AI's economic impact frequently ignore fiscal policy. In 2026, the US economy is operating under the massive demand-side fiscal stimulus of the "One Big Beautiful Bill Act" (OBBBA). By eliminating taxes on tips and overtime and expanding the child tax credit, the federal government has intentionally counterweighted localized labor displacement, effectively plugging the demand gap and ensuring the circular flow of income remains robust.</p>
<h2>Conclusion: The Persistence of the Human Economy</h2>
<p>The "2026 Global Intelligence Crisis" is fundamentally a crisis of narrative, not of macroeconomic reality. Artificial intelligence, constrained by the immutable laws of thermodynamics, regulatory oversight, and intense competitive market forces, remains a tool of human enterprise. The future of the global economy will be determined not by the autonomous, unchecked replication of software, but by the persistent, unyielding elasticity of human aspiration.</p>""",
        "source_priority": 5,
        "conviction": 90,
        "sentiment_score": 65
    }
]

def add_entries(path):
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    existing_keys = {(item.get('date'), item.get('title')) for item in data}
    added_count = 0

    for entry in new_entries:
        if (entry['date'], entry['title']) not in existing_keys:
            data.insert(0, entry) # Add to top
            added_count += 1
            print(f"Added entry to {path} for {entry['date']}")

    # Sort by date desc
    data.sort(key=lambda x: x.get('date', ''), reverse=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Data updated in {path}. Added {added_count} new entries.")

if __name__ == "__main__":
    add_entries(NEWSLETTER_DATA_PATH)
    add_entries(INDEX_DATA_PATH)
