import json
import os

output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "neo4j_tool_use.jsonl")

# Sample Data. This will be replaced by the expanded dataset
# in 'neo4j_tool_use_expanded.jsonl' (see Section 2.2)
samples = [
    {
        "question": "Find all companies with a P/E ratio below 15 in the Tech sector.",
        "query": "MATCH (c:Company)-->(s:Sector {name:'Technology'}) WHERE c.pe_ratio < 15 RETURN c.name, c.ticker, c.pe_ratio"
    },
    {
        "question": "Identify distressed assets with a debt-to-equity ratio over 2.0 in the Energy sector.",
        "query": "MATCH (c:Company)-->(s:Sector {name:'Energy'}) WHERE c.debt_to_equity > 2.0 RETURN c.name, c.ticker, c.current_debt"
    },
    {
        "question": "What is the average debt-to-equity ratio for all companies in the 'Energy' sector?",
        "query": "MATCH (c:Company)-->(s:Sector {name:'Energy'}) RETURN s.name, avg(c.debt_to_equity) AS avg_d2e"
    },
    {
        "question": "Find all board members of 'Tesla' who also sit on the board of a company in the 'Aerospace' sector.",
        "query": "MATCH (p:Person)-->(:Company {name:'Tesla'}) MATCH (p)-->(c2:Company)-->(:Sector {name:'Aerospace'}) RETURN p.name, c2.name"
    },
    {
        "question": "Which company in the 'Technology' sector has the highest number of 'Patent' nodes filed after January 1, 2023?",
        "query": "MATCH (c:Company)-->(:Sector {name:'Technology'}) MATCH (c)-->(p:Patent) WHERE p.filing_date > date('2023-01-01') RETURN c.name, count(p) AS patent_count ORDER BY patent_count DESC LIMIT 1"
    },
    {
        "question": "List all 'Fund' nodes that have a holding in 'Microsoft' greater than 5% of their portfolio and are managed by 'BlackRock'.",
        "query": "MATCH (f:Fund)-[h:HOLDS]->(c:Company {name:'Microsoft'}) WHERE h.portfolio_percentage > 0.05 MATCH (mgr:Manager {name:'BlackRock'})-->(f) RETURN f.name, h.portfolio_percentage"
    },
    {
        "question": "Who is the most central person in the 'Pharmaceutical' industry, measured by the number of board seats they hold?",
        "query": "MATCH (p:Person)-->(c:Company)-->(s:Sector {name:'Pharmaceutical'}) RETURN p.name, count(DISTINCT c) AS board_seats ORDER BY board_seats DESC LIMIT 1"
    }
]


print(f"Generating {len(samples)} Stage 1 examples to {output_file}...")
with open(output_file, 'w') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\\n')
print("Done.")
