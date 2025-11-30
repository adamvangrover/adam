# LIB-PRO-005: Industry Risk Report Generator

*   **ID:** `LIB-PRO-005`
*   **Version:** `1.0`
*   **Author:** Jules
*   **Objective:** To generate a concise, structured, and insightful risk report for a specific industry, drawing on well-established analytical frameworks like Porter's Five Forces.
*   **When to Use:** When starting analysis on a new company, to quickly get up to speed on the systemic risks and opportunities of the industry in which it operates. Also useful for portfolio-level risk management.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Industry_Name]`: The specific industry to be analyzed (e.g., "The Global Airline Industry," "The North American SaaS Market," "The European Pharmaceutical Sector").
    *   `[Key_Public_Companies]`: A list of 3-5 major public companies in the industry to serve as examples.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `IndustryAnalystAgent` or `ResearchAgent`.
    *   **External Data Integration:** This prompt is most powerful when the agent has access to external tools for web searches and financial data retrieval to enrich its analysis.
    *   **Portfolio Monitoring:** This prompt can be run periodically for all major industries represented in a credit portfolio to identify emerging trends and risks.

---

### **Example Usage**

```
[Industry_Name]: "The Global Container Shipping Industry"
[Key_Public_Companies]: "Maersk, Hapag-Lloyd, COSCO, Evergreen"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Senior Industry Analyst

# CONTEXT:
You are a senior industry analyst with deep expertise in competitive strategy and risk assessment. Your task is to create a comprehensive risk and opportunity report for a specific industry using established analytical frameworks.

# INPUTS:
*   **Industry:** `[Industry_Name]`
*   **Key Public Companies:** `[Key_Public_Companies]`

# TASK:
Generate a structured industry analysis report. The report should be objective, insightful, and forward-looking.

---
## **Industry Risk & Opportunity Report: [Industry_Name]**

### **1. Executive Summary**
*(A brief, high-level overview of the industry's key characteristics, its current state, and the most significant risks and opportunities.)*

### **2. Core Industry Characteristics**
*   **Market Size & Growth:** (e.g., "Approximately $X billion, with a projected annual growth rate of Y%...")
*   **Cyclicality:** (e.g., "Highly cyclical and tied to global GDP growth...")
*   **Key Success Factors:** (e.g., "Success in this industry is driven by operational efficiency, economies of scale, and logistics network strength.")

### **3. Competitive Landscape (Porter's Five Forces Analysis)**
*   **Threat of New Entrants:** (Low / Medium / High)
    *   **Rationale:** (e.g., "High, due to significant capital investment in vessels and infrastructure, and strong existing players' network effects.")
*   **Bargaining Power of Buyers:** (Low / Medium / High)
    *   **Rationale:** (e.g., "High, as shipping is largely a commoditized service and large customers can negotiate favorable rates.")
*   **Bargaining Power of Suppliers:** (Low / Medium / High)
    *   **Rationale:** (e.g., "Medium, key suppliers include shipbuilders and fuel providers. Fuel prices are volatile and can significantly impact costs.")
*   **Threat of Substitute Products or Services:** (Low / Medium / High)
    *   **Rationale:** (e.g., "Low, for global trade, there are few viable substitutes for container shipping. Air freight is much more expensive.")
*   **Intensity of Rivalry:** (Low / Medium / High)
    *   **Rationale:** (e.g., "High, the industry is fragmented with several large players competing aggressively on price.")

### **4. Top 3-5 Strategic Risks**
*(A bulleted list of the most significant risks facing the industry.)*
*   **Risk 1: [e.g., Geopolitical & Trade Risks]**
    *   **Description:** "The industry is highly sensitive to trade tariffs, sanctions, and geopolitical conflicts that can disrupt trade routes and volumes."
*   **Risk 2: [e.g., ESG & Regulatory Risk]**
    *   **Description:** "Increasing pressure to decarbonize and new environmental regulations (e.g., carbon taxes) will require significant capital investment in new vessels and fuels."
*   **Risk 3: [e.g., Economic Downturn]**
    *   **Description:** "A global recession would lead to a sharp decline in shipping volumes and freight rates, severely impacting profitability."

### **5. Top 3 Strategic Opportunities**
*(A bulleted list of the most significant opportunities.)*
*   **Opportunity 1: [e.g., Digitalization & Automation]**
    *   **Description:** "Opportunities exist to improve efficiency and reduce costs through better logistics software, automated port operations, and data analytics."
*   **Opportunity 2: [e.g., Consolidation]**
    *   **Description:** "Further M&A could lead to a more consolidated industry with greater pricing power."

### **6. Industry Outlook**
*(Provide a final, forward-looking statement on the industry.)*
> **Outlook:** (Stable / Positive / Negative)
> **Rationale:** "While the long-term demand drivers remain intact, the industry faces significant near-term headwinds from geopolitical uncertainty and the costs of decarbonization. Therefore, the outlook is Stable to Negative."

---
```
