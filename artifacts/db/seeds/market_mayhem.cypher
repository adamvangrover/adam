// Create the "Ouroboros" Loop
CREATE (msft:Company {legalName: "Microsoft Corp", ticker: "MSFT", sector: "Tech"})
CREATE (openai:Company {legalName: "OpenAI", type: "AI_Lab"})
CREATE (msft)-->(openai)
CREATE (openai)-->(msft)

// Create the Distressed Healthcare Rollup
CREATE (rollup:Company {legalName: "Regional Dental Partners", ticker: "PVT_DENT", type: "PE_Rollup"})
CREATE (rollup)-->(:LeveragedLoan {principal: 500000000, rate_type: "Floating"})
CREATE (rollup)-->(:RiskFactor {type: "LaborInflation", severity: "High"})
