# Data Provenance

## Provenance Model

Adam v22.0 uses the W3C PROV-O ontology to model data provenance. This allows us to track the origin and transformation of all data in the Knowledge Graph.

## Provenance Ontology

The following classes and properties are used to model data provenance:

*   `prov:Activity`: Represents a process that generates new data.
*   `prov:wasGeneratedBy`: Links a piece of data to the activity that created it.
*   `prov:used`: Links an activity to the data it used as input.
*   `prov:wasAttributedTo`: Links data to an agent (e.g., a specific Adam agent or a human expert).

## Example

Here is an example of how data provenance is modeled in Adam:

1.  The `MarketSentimentAgent` retrieves a news article about a stock.
2.  The agent analyzes the sentiment of the article and generates a sentiment score.
3.  The agent adds the sentiment score to the Knowledge Graph.
4.  The agent also adds provenance triples to the Knowledge Graph, indicating that the `MarketSentimentAgent` generated the sentiment score, and that it used the news article as input.

## Tracing Provenance

You can use SPARQL queries to trace the provenance of a specific piece of data. For example, the following query retrieves the provenance of a sentiment score:

```sparql
SELECT ?activity ?agent ?used_data
WHERE {
  <sentiment_score_uri> prov:wasGeneratedBy ?activity ;
                       prov:wasAttributedTo ?agent ;
                       prov:used ?used_data .
}
```
