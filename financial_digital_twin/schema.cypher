// Schema for the Financial Knowledge Graph
// Version 1.0 - Broadly Syndicated Loan Market

// ### NODES ###
// This section defines constraints for the primary entities (nodes) in the graph.
// Creating a constraint on a property for a given label will also create an index.

// 1. Company Node
// Represents borrowers, guarantors, investors, counterparties.
CREATE CONSTRAINT company_taxId IF NOT EXISTS FOR (c:Company) REQUIRE c.taxId IS UNIQUE;
CREATE CONSTRAINT company_legalEntityIdentifier IF NOT EXISTS FOR (c:Company) REQUIRE c.legalEntityIdentifier IS UNIQUE;
CREATE INDEX company_legalName IF NOT EXISTS FOR (c:Company) ON (c.legalName);

// 2. Loan Node
// Represents specific credit facilities (Term Loans, Revolvers).
CREATE CONSTRAINT loan_loanId IF NOT EXISTS FOR (l:Loan) REQUIRE l.loanId IS UNIQUE;

// 3. Security Node
// Represents tradable assets (Bonds, Syndicated Loan shares).
CREATE CONSTRAINT security_cusip IF NOT EXISTS FOR (s:Security) REQUIRE s.cusip IS UNIQUE;
CREATE CONSTRAINT security_isin IF NOT EXISTS FOR (s:Security) REQUIRE s.isin IS UNIQUE;

// 4. Collateral Node
// Represents assets securing loans.
// Assuming no single unique identifier for all collateral types initially.
// An internal ID might be added later, e.g. collateralId.

// 5. Individual Node
// Represents key executives, board members, guarantors.
// An internal ID might be needed, e.g. individualId
CREATE CONSTRAINT individual_id IF NOT EXISTS FOR (i:Individual) REQUIRE i.individualId IS UNIQUE;


// 6. Covenant Node
// Represents financial or operational performance requirements.
// An internal ID might be needed, e.g. covenantId
CREATE CONSTRAINT covenant_id IF NOT EXISTS FOR (c:Covenant) REQUIRE c.covenantId IS UNIQUE;


// 7. Financials Node
// Represents a specific financial statement (10-K, 10-Q).
// A combination of company and period end date should be unique.
CREATE CONSTRAINT financials_id IF NOT EXISTS FOR (f:Financials) REQUIRE (f.companyTaxId, f.periodEndDate) IS UNIQUE;


// ### RELATIONSHIPS (EDGES) ###
// Relationships are defined dynamically when data is loaded.
// No schema definition is required for edges in the same way as nodes,
// but we list them here for clarity.
//
// (Company)-[:IS_BORROWER_OF]->(Loan)
// (Company)-[:LENDS_TO]->(Loan)
// (Company)-[:IS_AGENT_FOR]->(Loan)
// (Loan)-[:SECURED_BY]->(Collateral)
// (Company)-[:ISSUED]->(Security)
// (Company)-[:HOLDS_POSITION_IN]->(Security) // Company as an Investor
// (Company)-[:HAS_PARENT]->(Company)
// (Individual)-[:IS_OFFICER_OF]->(Company)
// (Loan)-[:SUBJECT_TO]->(Covenant)
// (Company)-[:HAS_FINANCIALS]->(Financials)

// End of Schema
