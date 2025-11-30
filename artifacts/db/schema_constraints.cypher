// 1. Integrity Constraints (Ensuring Comprehensive Profiles)
// Every company must have a legal name and a unique ticker/ID
CREATE CONSTRAINT unique_company_id IF NOT EXISTS
FOR (c:Company) REQUIRE c.id IS UNIQUE;

// Every loan must have a principal amount defined
CREATE CONSTRAINT loan_amount_exists IF NOT EXISTS
FOR (l:Loan) REQUIRE l.principalAmount IS NOT NULL;

// 2. Performance Indexes
// Indexing Tickers for fast lookup
CREATE INDEX company_ticker_idx IF NOT EXISTS FOR (c:Company) ON (c.ticker);
// Indexing Risk Ratings for aggregate reporting
CREATE INDEX risk_rating_idx IF NOT EXISTS FOR (c:Company) ON (c.riskRating);

// 3. Vector Index for Agentic AI (Semantic Search)
// This enables the AI agents to perform "Hybrid Search" - combining graph traversal
// with vector similarity to find companies with similar risk descriptions.
CREATE VECTOR INDEX company_description_vector IF NOT EXISTS
FOR (c:Company) ON (c.descriptionEmbedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}};
