# Crisis Simulation Library: Supply Chain Disruption Scenarios

This library provides a set of user-ready scenarios focused on **Supply Chain Disruptions**. These can be used as the `{{USER_SCENARIO_INPUT}}` in the main `crisis_simulation.md` prompt.

---

### Scenario SCD-001: Sole-Source Supplier Bankruptcy

**Description:** Our sole-source supplier for a critical, custom-designed component (e.g., a specific microchip or chemical formula) declares immediate bankruptcy and ceases all operations. There is no qualified second-source supplier, and qualifying a new one is estimated to take at least six months.

**Potential Primary Impact Nodes:**
*   **R-SCM-02 (Supplier Failure Risk):** Production of our flagship product, which requires the component, will halt completely once the current inventory is exhausted (estimated at 7 days).
*   **R-STR-02 (Market Position Risk):** Competitors will likely seize market share while our product is unavailable.
*   **R-FIN-01 (Financial Reporting Risk):** Massive revenue writedowns are imminent.

---

### Scenario SCD-002: Major Port Shutdown

**Description:** The primary seaport we use for all inbound raw materials and outbound finished goods is shut down indefinitely due to a combination of a major labor strike and a cybersecurity attack on the port's logistics systems. Rerouting to the next closest port will add 2-3 weeks of lead time and a 30% increase in shipping costs.

**Potential Primary Impact Nodes:**
*   **R-SCM-01 (Supply Chain Risk):** The entire logistics network is in chaos. Inventory holding costs will increase, and delivery schedules will be missed.
*   **R-OPS-01 (Operational Risk):** Production schedules must be constantly revised based on the uncertain arrival of materials.
*   **R-REP-01 (Reputational Risk):** Failure to meet delivery commitments damages customer trust.

---

### Scenario SCD-003: Counterfeit Components Detected

**Description:** A whistleblower reveals that a batch of counterfeit, low-quality components from an unauthorized subcontractor has entered our supply chain and has been used in products already shipped to customers. The counterfeit parts have a high failure rate and pose a significant safety risk.

**Potential Primary Impact Nodes:**
*   **R-LGL-03 (Product Liability Risk):** We are exposed to lawsuits and regulatory action due to the safety hazard. A full product recall is likely required.
*   **R-REP-01 (Reputational Risk):** Brand image is severely damaged. The "quality and safety" promise is broken.
*   **R-FIN-04 (Credit Risk):** The cost of the recall (logistics, replacement units, legal fees) will be a massive, unplanned financial hit.

---

### Scenario SCD-004: Natural Disaster Hits Key Supplier Region

**Description:** A massive earthquake followed by a tsunami strikes a region that is a central hub for three of our critical Tier-2 suppliers (i.e., suppliers to our direct suppliers). Power, water, and transportation infrastructure in the region are completely destroyed. Our direct suppliers declare `force majeure` as they cannot get the materials they need.

**Potential Primary Impact Nodes:**
*   **R-SCM-02 (Supplier Failure Risk):** Lack of visibility into Tier-2 and Tier-3 suppliers has led to a sudden, unexpected and complete cut-off of a key material.
*   **R-STR-01 (Strategic Risk):** Over-concentration of the supply base in a single geographic region is exposed as a critical strategic failure.
*   **R-OPS-01 (Operational Risk):** Production must be halted or dramatically altered, requiring expensive and time-consuming re-tooling for alternative components.
