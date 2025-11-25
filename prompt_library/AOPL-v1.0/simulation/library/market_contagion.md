# Crisis Simulation Library: Market Contagion Scenarios

This library provides a set of user-ready scenarios focused on **Market Contagion Events**. These can be used as the `{{USER_SCENARIO_INPUT}}` in the main `crisis_simulation.md` prompt.

---

### Scenario MKT-001: "Lehman Moment" Counterparty Collapse

**Description:** A major, systemically important financial institution, with whom we have significant counterparty exposure (e.g., derivatives contracts, short-term lending facilities), is rumored to be on the verge of collapse. Regulators are silent. Credit markets freeze as every institution begins questioning the solvency of its trading partners.

**Potential Primary Impact Nodes:**
*   **R-FIN-05 (Counterparty Risk):** Our counterparty defaults on all obligations. Hedges we thought we had are now worthless. Cash margins held by the counterparty are lost.
*   **R-FIN-03 (Liquidity Risk):** Our own access to short-term funding evaporates, triggering a liquidity crisis.
*   **R-FIN-02 (Market Risk):** The entire market reprices systemic risk, causing the value of all our financial assets to plummet.

---

### Scenario MKT-002: Flash Crash

**Description:** An algorithmic trading error triggers a "flash crash" in the equity markets. The Dow Jones Industrial Average drops 10% in five minutes. Trading is halted, but panic has already spread to other asset classes. The VIX (volatility index) spikes to record levels.

**Potential Primary Impact Nodes:**
*   **R-FIN-02 (Market Risk):** Automated stop-loss orders are triggered across our portfolio, locking in massive losses.
*   **R-OPS-04 (IT & Systems Risk):** Our own trading and risk management systems may be overwhelmed by the volume and velocity of market data, leading to failures.
*   **R-STR-01 (Strategic Risk):** Confidence in the stability of market structures is shaken, impacting long-term investment strategies.

---

### Scenario MKT-003: Asset Class Correlation Shock

**Description:** A portfolio of historically uncorrelated assets (e.g., government bonds, gold, and equities) suddenly start moving in lockstep, all declining sharply. The fundamental assumptions of our diversification and hedging strategy are proven wrong in a live-fire event.

**Potential Primary Impact Nodes:**
*   **R-FIN-02 (Market Risk):** Diversification fails to protect capital. The portfolio experiences the maximum possible drawdown.
*   **R-MDL-01 (Model Risk):** The quantitative models underpinning our entire risk management framework are invalidated.
*   **R-REP-01 (Reputational Risk):** Investors and stakeholders question the competence of the risk management function.

---

### Scenario MKT-004: Flight to Quality Freezes Corporate Debt Market

**Description:** A wave of negative economic news triggers a massive "flight to quality," where investors dump corporate bonds of all grades and flock to government-backed securities. The bid-ask spread on our company's bonds widens to unprecedented levels, making it impossible to issue new debt or roll over existing debt.

**Potential Primary Impact Nodes:**
*   **R-FIN-03 (Liquidity Risk):** A planned bond issuance to fund a major acquisition fails, putting the deal and the company's reputation at risk.
*   **R-FIN-04 (Credit Risk):** Our own credit rating comes under negative review by rating agencies due to the frozen funding markets.
*   **R-STR-02 (Market Position Risk):** The inability to fund strategic initiatives allows more liquid competitors to gain an advantage.
