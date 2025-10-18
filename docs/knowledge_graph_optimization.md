# Knowledge Graph Optimization

## Caching

To improve the performance of the Knowledge Graph, a caching layer has been implemented using Redis. SPARQL queries and their results are cached to reduce the latency of repeated queries.

### Caching Policy

*   **TTL:** The Time-To-Live for cache entries is set to 1 hour by default.
*   **Invalidation:** The cache can be manually invalidated if necessary.

## Indexing

Proper indexing is crucial for the performance of the graph database.

### Indexing Strategy

*   Index key predicates, such as `acps:hasDirectCause`, to speed up queries that use these predicates.

## Query Optimization

*   Write efficient SPARQL queries to minimize the number of triple patterns and avoid complex joins.
*   Use `LIMIT` and `OFFSET` to paginate results and avoid fetching large amounts of data at once.
