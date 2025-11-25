# Crisis Simulation Library: Asset Bubble Burst Scenarios

This library provides a set of user-ready scenarios focused on the **Bursting of Asset Bubbles**. These can be used as the `{{USER_SCENARIO_INPUT}}` in the main `crisis_simulation.md` prompt.

---

### Scenario ABB-001: Dot-com Style Tech Stock Crash

**Description:** The high-flying technology sector, which has seen valuations detached from fundamental earnings for several years, experiences a sudden and brutal crash. The NASDAQ index falls 70% from its peak in a matter of weeks. Our company has significant direct investment in the sector and our pension fund is heavily exposed.

**Potential Primary Impact Nodes:**
*   **R-FIN-02 (Market Risk):** The value of our corporate investments and pension assets is decimated.
*   **R-FIN-01 (Financial Reporting Risk):** Massive impairment charges must be taken on the devalued tech stocks, erasing corporate earnings.
*   **R-EMP-02 (Talent Risk):** Employee morale plummets as the value of their stock options and 401(k)s is wiped out.

---

### Scenario ABB-002: Real Estate Market Collapse

**Description:** A nationwide housing bubble, fueled by cheap credit and speculative buying, bursts. Home prices fall by 40%, leading to a wave of mortgage defaults and foreclosures. The market for Mortgage-Backed Securities (MBS) freezes, and banks with heavy exposure to real estate loans are at risk of failure.

**Potential Primary Impact Nodes:**
*   **R-FIN-04 (Credit Risk):** If the company holds MBS or has loans collateralized by real estate, the value of that collateral evaporates.
*   **R-FIN-05 (Counterparty Risk):** Banks that the company relies on for lending and other services may become insolvent.
*   **R-STR-02 (Market Position Risk):** Consumer demand collapses as household wealth is destroyed, leading to a deep recession.

---

### Scenario ABB-003: Private Equity / Venture Capital "Unicorn" Bubble Deflates

**Description:** The private equity market, particularly for late-stage "unicorn" startups, experiences a severe correction. High-profile IPOs fail, and mega-funds are forced to write down the value of their portfolios by over 50%. Our company is a significant Limited Partner (LP) in several large PE/VC funds.

**Potential Primary Impact Nodes:**
*   **R-FIN-02 (Market Risk):** The carrying value of our private equity investments must be written down, leading to large reported losses.
*   **R-FIN-03 (Liquidity Risk):** We are subject to capital calls from the PE funds to shore up their struggling portfolio companies, creating an unexpected and significant cash drain.
*   **R-REP-01 (Reputational Risk):** The board and shareholders question the wisdom of the company's aggressive alternative investment strategy.

---

### Scenario ABB-004: Commodity Supercycle Reversal

**Description:** A decade-long "supercycle" in a key commodity (e.g., oil, copper) abruptly ends due to slowing global demand and new extraction technologies. The price of the commodity drops 80% from its peak. Our company has major operations that are vertically integrated with this commodity.

**Potential Primary Impact Nodes:**
*   **R-OPS-02 (Physical Asset Risk):** The value of our physical inventory (e.g., oil reserves, copper mines) plummets. Exploration and extraction assets may become economically unviable and need to be written off.
*   **R-FIN-02 (Market Risk):** Commodity hedging instruments result in massive losses as the price moves in the opposite direction of the hedge.
*   **R-SCM-01 (Supply Chain Risk):** The economic collapse of regions dependent on the commodity can cause widespread supplier failure and instability.
