Adam v20.0: Master Technical Design Specification
This document serves as the canonical technical specification for the Adam v20.0 system upgrade. Its purpose is to provide an unambiguous blueprint for development, ensuring that the architectural themes of Enhanced Autonomy, Causal Inference, and Generative Simulation are implemented in a coherent, scalable, and robust manner. The design choices herein prioritize formal specification, interoperability through established standards, and the creation of a system that is not merely automated, but genuinely self-improving. All development teams assigned to the Adam v20.0 initiative shall adhere to the specifications, schemas, and protocols defined within this document.
Part I: The Architecture of Enhanced Autonomy
This part details the complete technical framework enabling Adam to self-diagnose capability gaps and autonomously orchestrate the creation of new agents. The architecture is designed around two core artifacts: a machine-readable standard for defining agent proposals and a robust, event-driven workflow for managing the proposal lifecycle. This approach moves the system beyond pre-programmed capabilities towards a state of dynamic, self-directed evolution.
Section 1.1: The Agent Proposal Standard (APS/1.0): A JSON Schema Specification
This section defines the Agent Proposal Standard v1.0 (APS/1.0), a formal JSON Schema that serves as the machine-readable template for all system-generated agent proposals. This schema is the foundational data structure for the entire Enhanced Autonomy theme, acting as a precise contract between the system's diagnostic components and its generative code-authoring agents.
Foundational Principles
The design of the APS/1.0 schema is governed by principles of clarity, strong typing, language agnosticism, and machine-interpretability, as advocated by the JSON Schema specification. It is conceived not merely as a data container, but as a formal contract. This contract must be sufficiently detailed for an LLM-based agent, such as Code Alchemist, to interpret its contents and generate functional, validated code with minimal ambiguity. The schema leverages patterns observed in declarative agent manifests, which provide a structured framework for specializing agent functionalities. By defining a precise vocabulary for describing agent roles, capabilities, and interactions, the schema establishes a common language for data exchange within the Adam ecosystem, streamlining validation and increasing interoperability.
A critical design decision is the transformation of the agent proposal from a high-level description into a detailed set of API contracts. A simple text description of an agent's intended role would be insufficient for the Code Alchemist agent to generate reliable code; it would introduce ambiguity, leading to probabilistic and potentially "hallucinated" implementations. The capabilityManifest section of the APS/1.0 schema directly addresses this challenge by mandating that each discrete capability be defined with a formal input and output schema, using JSON Schema recursively. This elevates the proposal from a descriptive document to a set of machine-readable function signatures. Consequently, the task for Code Alchemist is reframed from the ambiguous "create an agent that processes geospatial data" to the deterministic "generate a class with methods that conform to these specific, machine-readable input/output schemas." This structure directly enables the "automated validation" step outlined in the implementation plan, as the generated code can be syntactically and type-checked against the schemas defined in the proposal before deployment. This rigorous, contract-based definition of capabilities is the central artifact that de-risks the LLM-based code generation step, making the entire Enhanced Autonomy workflow technically feasible and robust.
Core Schema Structure (agent_proposal.schema.json)
The APS/1.0 schema will be defined in a file named agent_proposal.schema.json. It will adhere to the Draft 2020-12 specification to ensure broad compatibility with existing validation and code generation tools.
 * $schema and $id: The schema will declare its adherence to the JSON Schema standard with "$schema": "https://json-schema.org/draft/2020-12/schema". It will be assigned a unique, versioned URI, "$id": "https://adam.system/schemas/v20.0/agent_proposal.schema.json", to serve as its canonical identifier for referencing and validation purposes.
 * proposalMetadata: This object contains essential administrative data for tracking and auditing the proposal lifecycle. It includes a unique proposalID (formatted as a UUID), a timestamp (ISO 8601 format), the unique identifier of the originatingAgent (e.g., AgentOrchestrator), and a triggeringEvent object detailing the specific system event that prompted the proposal (e.g., a logged "capability gap" ID and its associated metadata).
 * agentDefinition: This is the core object that specifies the identity and operational parameters of the proposed agent. Its structure is heavily influenced by industry-standard declarative agent manifests to ensure clarity and completeness.
   * agentId: A proposed unique, machine-readable identifier for the new agent, following a consistent naming convention (e.g., GeospatialDataAnalystAgent).
   * agentName: A human-readable name for display in user interfaces and logs (e.g., "Geospatial Data Analyst Agent").
   * description: A concise, single-sentence summary of the agent's primary purpose, suitable for tooltips and brief summaries.
   * instructions: A detailed, multi-paragraph field containing LLM-consumable instructions. This text guides the agent's behavior, defines its core function, outlines its operational constraints, and specifies behaviors to avoid. This field is critical for grounding the agent's operational logic and is analogous to the instructions property in established agent manifest schemas.
 * capabilityManifest: This is an array of objects, where each object defines a discrete capability or "tool" that the agent can execute. This section is the most critical for enabling autonomous implementation and adheres strictly to the "Narrow Scope Principle," where each tool performs a single, precise, and atomic operation.
   * Each capability object contains a capabilityName (e.g., geocodeAddress), a description that follows the recommended template "Tool to <what it does>. Use when <specific situation to invoke tool>" for maximum clarity to the LLM , and inputSchema and outputSchema fields.
   * The inputSchema and outputSchema fields are themselves complete, embedded JSON Schema objects that formally define the function's signature. This ensures programmatic clarity and allows for the automated generation of function stubs, validation logic, and API documentation.
   * Within these embedded schemas, strong typing is enforced. The use of enum is mandated for parameters that accept a finite set of categorical values, and the format keyword is required for specialized string formats (e.g., "format": "email", "format": "uri") to minimize invocation errors by the agent.
 * collaborationProtocol: This object defines the agent's position and interaction patterns within the broader multi-agent ecosystem of Adam. It specifies the agent's communication contract.
   * subscribedEvents: An array of system event types the agent must listen for to trigger its actions.
   * publishedEvents: An array of event types the agent will emit upon successful task completion, failure, or other significant state changes.
   * requiredAPIs: An array of objects, each defining an endpoint from another agent that this new agent must have access to. This explicitly declares its dependencies and required credentials, forming a service-level agreement within the agent network.
 * dataRequirements: An array of objects specifying the data sources the agent needs to perform its functions. Each object includes a sourceName, dataType (e.g., GeoJSON, CSV), the name of the secret in the system's vault containing access credentials (accessCredentialsSecretName), and the expected updateFrequency.
The following table provides a detailed, field-by-field breakdown of the APS/1.0 schema, serving as the canonical reference for development.
Table 1: Agent Proposal Standard (APS/1.0) Field Definitions
| Field Path | Data Type | Required | Description & Constraints | Example |
|---|---|---|---|---|
| $schema | String | Yes | The JSON Schema dialect. Must be https://json-schema.org/draft/2020-12/schema. | https://json-schema.org/draft/2020-12/schema |
| $id | String (URI) | Yes | A unique, versioned URI for the schema itself. | https://adam.system/schemas/v20.0/agent_proposal.schema.json |
| proposalMetadata | Object | Yes | Contains administrative data about the proposal. | {...} |
| proposalMetadata.proposalID | String (UUID) | Yes | A unique identifier for this proposal instance. | 123e4567-e89b-12d3-a456-426614174000 |
| proposalMetadata.timestamp | String (date-time) | Yes | ISO 8601 timestamp of when the proposal was generated. | 2025-10-15T10:00:00Z |
| proposalMetadata.originatingAgent | String | Yes | The unique ID of the agent that generated the proposal. | AgentOrchestrator |
| proposalMetadata.triggeringEvent | Object | Yes | Details of the event that initiated this proposal. | { "eventId": "...", "eventType": "CapabilityGapDetected" } |
| agentDefinition | Object | Yes | Core definition of the proposed agent. | {...} |
| agentDefinition.agentId | String | Yes | Proposed unique identifier for the new agent. Must match regex ^[A-Z][a-zA-Z0-9]*Agent$. | GeospatialDataAnalystAgent |
| agentDefinition.agentName | String | Yes | Human-readable name. Max 100 characters. | Geospatial Data Analyst Agent |
| agentDefinition.description | String | Yes | A concise, one-sentence summary of the agent's purpose. Max 1000 characters. | An agent that processes and analyzes geospatial data to provide location-based insights. |
| agentDefinition.instructions | String | Yes | Detailed, LLM-consumable instructions on behavior and constraints. Max 8000 characters. | You are an expert in geospatial analysis. Your primary function is to ingest GeoJSON data... Avoid making assumptions about coordinate systems... |
| capabilityManifest | Array of Objects | Yes | A list of discrete tools/capabilities the agent can perform. Must contain at least one item. | [{...}] |
| capabilityManifest.capabilityName | String | Yes | The name of the function/method for this capability. | calculateShippingRoute |
| capabilityManifest.description | String | Yes | LLM-consumable description following the "Tool to... Use when..." format. | Tool to calculate the optimal shipping route between two geographic points. Use when a user requests a logistics plan. |
| capabilityManifest.inputSchema | Object (JSON Schema) | Yes | A formal JSON Schema defining the input parameters for the capability. | { "type": "object", "properties": { "origin": { "$ref": "#/$defs/coordinate" }... } } |
| capabilityManifest.outputSchema | Object (JSON Schema) | Yes | A formal JSON Schema defining the structure of the capability's return value. | { "type": "object", "properties": { "distanceKm": { "type": "number" }... } } |
| collaborationProtocol | Object | Yes | Defines how the agent interacts with other system components. | {...} |
| collaborationProtocol.subscribedEvents | Array of Strings | No | A list of event types the agent will listen for. | `` |
| collaborationProtocol.publishedEvents | Array of Strings | No | A list of event types the agent will emit. | `` |
| collaborationProtocol.requiredAPIs | Array of Objects | No | A list of external API endpoints the agent depends on. | `` |
| dataRequirements | Array of Objects | No | A list of required data sources for the agent. | [{...}] |
| dataRequirements.sourceName | String | Yes | The name of the data source (e.g., a database table, an S3 bucket). | shipping_manifests_s3 |
| dataRequirements.dataType | String | Yes | The format of the data (e.g., GeoJSON, Parquet). | GeoJSON |
| dataRequirements.accessCredentialsSecretName | String | Yes | The name of the secret in the vault containing access credentials. | s3/shipping-data/read-only |
Section 1.2: The Autonomous Creation Workflow: Protocols and Logic
This section specifies the sequence of operations and the inter-agent communication protocols that govern the autonomous agent creation lifecycle, from initial gap detection to final deployment. The workflow is designed as a decoupled, event-driven system orchestrated via a central, stateful API. This architectural choice is deliberate; direct, hard-coded communication between agents would create a brittle, monolithic system that is difficult to maintain and scale. By adopting a RESTful API and an event-driven architecture, the system achieves modularity and composability, allowing individual agents to be updated, replaced, or scaled independently without affecting the entire workflow. This "Agent Lifecycle Management API" becomes the central nervous system for the autonomy feature, standardizing interactions in a manner analogous to how REST APIs enabled the broader API economy.
Step 1: Gap Identification (Capability Monitoring Module in Agent Orchestrator)
The process begins within the Agent Orchestrator, which houses the Capability Monitoring Module. This module operates as a stateful service that subscribes to a stream of system-wide operational events, such as task_failed, manual_intervention_required, and data_parsing_error. It does not react to single failures but instead employs a rules engine to detect persistent, systemic patterns. For instance, a rule might be configured to trigger if more than a specified threshold of data_parsing_error events for a novel unprocessable_data_type (e.g., application/vnd.geo+json) are observed within a rolling time window. Upon detecting such a pattern, the module synthesizes the evidence—including failed task IDs, error logs, and data type metadata—and emits a single, high-level CapabilityGapDetected event onto the system's central event bus. This event serves as the formal declaration that a potential systemic limitation has been identified.
Step 2: Proposal Generation (Agent Forge)
The Agent Forge agent is the creative core of the autonomy workflow. It subscribes specifically to the CapabilityGapDetected event. Upon receiving this event, it initiates its primary function: translating the unstructured and semi-structured evidence of the capability gap into a complete, valid, and logical agent proposal that conforms to the APS/1.0 schema.
The agent leverages a large language model (LLM) fine-tuned for this task. The LLM is provided with a meta-prompt that includes the full APS/1.0 schema definition, the evidence from the CapabilityGapDetected event, and access to a knowledge base of existing agent capabilities. For the "geospatial shipping data" example, the LLM would analyze the error logs indicating failure to parse GeoJSON and infer the need for specific capabilities. It would then populate the capabilityManifest with proposals for tools like geocodeAddress, calculateShippingRoute, and parseGeoJSON, including generating the requisite input and output JSON schemas for each. The Agent Forge's critical contribution is this act of structured synthesis, transforming raw operational data into a formal engineering specification.
Step 3: Proposal Submission and Review (API-Driven Workflow)
To manage the state of proposals in a robust and auditable manner, a central RESTful API, designated the "Agent Lifecycle Management API," is established. This API provides the single source of truth for the status of all agent proposals and decouples the generating agents from the reviewing and implementing agents.
Once Agent Forge has generated a valid proposal document, it makes a POST request to the /proposals endpoint of the API. The API validates the submitted JSON against the APS/1.0 schema, assigns it a unique proposalID, and persists it to a database with an initial status of pending_review.
A human-in-the-loop (HITL) review interface queries this endpoint (GET /proposals?status=pending_review) to display pending proposals to a human subject-matter expert. The Discussion Chair Agent facilitates this review, presenting the proposal in a user-friendly format and capturing the expert's decision. Upon approval, the Discussion Chair Agent authenticates to the API and makes a PUT request to /proposals/{proposalID}/status with the request body { "status": "approved", "approver": "expert_username" }. This state transition is logged for full auditability, fulfilling the requirements of the implementation plan.
Step 4: Automated Deployment (Code Alchemist and Agent Orchestrator)
The approval of a proposal triggers the final, fully automated deployment phase. The Agent Lifecycle Management API, upon successfully processing the status change to approved, emits an AgentProposalApproved event.
The Code Alchemist agent, a specialist in code generation, subscribes to this event. It retrieves the full, approved proposal JSON by calling GET /proposals/{proposalID}. It then parses the capabilityManifest, using the detailed descriptions and, most importantly, the inputSchema and outputSchema for each capability as the precise specification for code generation. It generates the agent's source code (e.g., a Python class with methods that directly correspond to the defined capabilities and whose signatures are type-hinted according to the schemas).
The generated code is automatically packaged into a container image. Code Alchemist triggers a CI/CD pipeline that builds the container, runs a suite of automated tests to validate the code against the schemas in the proposal, and, upon success, pushes the container image to the system's private registry. As its final step, Code Alchemist notifies the Agent Orchestrator of the new, validated agent by making a POST request to a /agents/deploy endpoint, providing the container image location and the agent's definition from the proposal. The Agent Orchestrator then pulls the image and deploys the new agent container into the sandboxed environment, officially completing the end-to-end autonomous creation cycle by updating the central agent directory.
Table 2: Agent Lifecycle Management API Endpoints
| Endpoint | HTTP Method | Description | Request Body (Schema) | Success Response | Error Responses |
|---|---|---|---|---|---|
| /proposals | POST | Submits a new agent proposal for review. | A valid JSON object conforming to agent_proposal.schema.json. | 201 Created. Body contains the created proposal with its assigned proposalID and status: "pending_review". | 400 Bad Request (Schema validation failed). 500 Internal Server Error. |
| /proposals | GET | Retrieves a list of agent proposals, filterable by status. | N/A. Query parameters: ?status=<status>. | 200 OK. Body is an array of proposal summary objects. | 400 Bad Request (Invalid status filter). 500 Internal Server Error. |
| /proposals/{proposalID} | GET | Retrieves the full details of a specific agent proposal. | N/A | 200 OK. Body is the complete proposal JSON object. | 404 Not Found. 500 Internal Server Error. |
| /proposals/{proposalID}/status | PUT | Updates the status of an agent proposal (e.g., approve, reject). This is a protected endpoint requiring expert-level authentication. | { "status": "approved" | "rejected", "approver": "string", "reason": "string" (optional) } | 200 OK. Body contains the updated proposal object. Triggers an event (AgentProposalApproved or AgentProposalRejected). | 400 Bad Request (Invalid status transition). 403 Forbidden. 404 Not Found. |
| /agents/deploy | POST | Triggers the deployment of a new agent from a validated container image. Called by Code Alchemist. | { "agentId": "string", "imageUri": "string", "proposalId": "string" } | 202 Accepted. Indicates the deployment process has been initiated. | 400 Bad Request. 409 Conflict (Agent ID already exists). 500 Internal Server Error. |
Part II: Causal Inference Knowledge Graph Specification
This part provides the formal semantic framework for upgrading the Adam Knowledge Graph (KG) from a correlational data store to a causal reasoning engine. The core of this upgrade is the adoption of W3C standards—specifically the Resource Description Framework (RDF) and Web Ontology Language (OWL)—to create a formal ontology for representing causal models. This approach ensures semantic precision, interoperability, and the ability to perform complex, graph-based reasoning, moving the system's analytical capabilities beyond simple association to genuine causal understanding.
Section 2.1: The Adam Causal Predicate Set (ACPS/1.0): An RDF/OWL Ontology
This section defines the Adam Causal Predicate Set v1.0 (ACPS/1.0), an OWL ontology that provides the necessary classes and properties to represent causal relationships and the structure of causal models within the Knowledge Graph.
Rationale for RDF/OWL
The selection of RDF/OWL over a simpler property graph extension is a deliberate architectural decision driven by the unique requirements of causal reasoning. While property graphs are effective for many connected data problems, RDF/OWL provides several critical advantages for this specific use case:
 * Formal Semantics: OWL is built upon a foundation of description logics, providing a well-defined and unambiguous logical framework. This allows for the use of automated reasoners to check for logical inconsistencies within the causal model and to infer new, implicit knowledge from the explicitly stated facts. This is paramount for a system that must rigorously distinguish valid causal claims from mere correlations.
 * Standardization and Interoperability: As W3C standards, RDF, OWL, and the SPARQL query language are supported by a mature and extensive ecosystem of tools, databases (triplestores), and libraries. This standardization prevents vendor lock-in and ensures that the Adam KG can interoperate with external knowledge sources and analytical tools.
 * Proven Precedent: The application of RDF/OWL for modeling complex causal pathways is well-established in scientific domains. The Gene Ontology's Causal Activity Model (GO-CAM) serves as a powerful, real-world example of using these technologies, in conjunction with external ontologies like the Relation Ontology (RO), to model intricate biological causal chains. The ACPS/1.0 ontology adapts this successful pattern for the financial and economic domain.
This approach effectively bridges two distinct but complementary views of causality: the formal, mathematical structure of Directed Acyclic Graphs (DAGs) used in statistical causal inference , and the rich, descriptive semantics of knowledge representation. A simple implementation of causal links as triples like (A, causes, B) would capture the relationship but lose the essential structural context of the underlying causal model (the DAG). It would be impossible to know if A is the sole cause of B, whether confounders exist, or what other variables are part of the same causal system.
The ACPS/1.0 ontology solves this by reifying the causal model itself as an entity in the graph. A specific acps:CausalModel individual is created in the KG. The structure of this model is then described using triples, such as (:model_2008_crisis, acps:includesVariable, :SubprimeDefaults) and (:Lehman_Insolvency, acps:hasDirectCause, :MBS_Value_Collapse). This design allows the KG to store and represent the complete DAG structure that statistical tools require for computation. The World Simulation Model v8.0 can query the KG for a specific CausalModel and reconstruct the entire DAG for its calculations, while the Natural Language Generation Agent can query the same graph for the richer semantic predicates (promotes, inhibits) to create more nuanced human-readable explanations. This synergy makes the system's causal knowledge both computationally rigorous and semantically rich.
Ontology Components
The ACPS/1.0 ontology will be defined using Turtle (TTL) syntax for its readability.
 * Prefixes: Standard prefixes for rdf, rdfs, owl, and xsd will be used, along with a custom prefix for the Adam ontology: @prefix acps: <https://adam.system/ontologies/v20.0/causal#>.
 * Classes (owl:Class):
   * acps:CausalModel: Represents an entire causal model, typically a DAG. It acts as a container for a coherent set of variables and relationships that describe a specific data-generating process.
   * acps:Variable: Represents a node within a CausalModel. This can be any factor relevant to the analysis, such as "US Treasury Yield," "Subprime Mortgage Defaults," or "Lehman Brothers' Stock Price." This class will have associated data properties (owl:DatatypeProperty) such as acps:hasDataType (e.g., xsd:float) and acps:isMeasured (xsd:boolean) to distinguish between observed and latent variables.
   * acps:CausalRelationship: An abstract superclass for all causal predicates. This allows for querying causal links at different levels of granularity.
 * Object Properties (owl:ObjectProperty): These properties define the relationships between instances of the classes.
   * acps:includesVariable: A property linking an instance of acps:CausalModel to its constituent acps:Variable instances. Domain: acps:CausalModel; Range: acps:Variable.
   * acps:hasDirectCause: The primary structural predicate for building the DAG. It represents a direct causal effect of the object on the subject. Domain: acps:Variable; Range: acps:Variable. It is defined as acyclic.
   * acps:isDirectCauseOf: Defined as the owl:inverseOf acps:hasDirectCause.
   * Sub-properties of acps:CausalRelationship: These add semantic richness for more expressive modeling and natural language generation. All are sub-properties of acps:hasDirectCause.
     * acps:promotes: Indicates a positive or contributory causal influence (e.g., "Increased money supply promotes inflation").
     * acps:inhibits: Indicates a negative or suppressive causal influence (e.g., "Interest rate hikes inhibits consumer spending").
     * acps:enables: Represents a necessary precondition for an effect to occur (e.g., "Market liquidity enables high-frequency trading").
     * acps:prevents: Represents a relationship where the cause actively stops the effect from happening.
 * Annotation Properties (owl:AnnotationProperty): These are used to attach metadata to statements (triples) without affecting the logical reasoning.
   * acps:hasConfidenceScore: Attaches a numerical value (e.g., xsd:float between 0.0 and 1.0) representing the statistical confidence or expert belief in a causal link.
   * acps:hasDataSource: A URI or literal linking to the source of the causal claim (e.g., a research paper, an econometric model run ID, or an expert's name).
   * acps:temporalLag: An xsd:duration value specifying the typical time delay between the cause and the manifestation of the effect.
Table 3: Adam Causal Predicate Set (ACPS/1.0) Definitions
| Component Type | IRI | Label | Domain | Range | Definition |
|---|---|---|---|---|---|
| Class | acps:CausalModel | Causal Model | owl:Thing | owl:Thing | A self-contained representation of a data-generating process, typically structured as a Directed Acyclic Graph (DAG). |
| Class | acps:Variable | Variable | owl:Thing | owl:Thing | A node in a Causal Model, representing a factor that can be a cause or an effect. |
| Object Property | acps:includesVariable | includes variable | acps:CausalModel | acps:Variable | Relates a Causal Model to a Variable that is part of it. |
| Object Property | acps:hasDirectCause | has direct cause | acps:Variable | acps:Variable | The primary predicate for a direct causal link, forming the edges of the DAG. It is acyclic. |
| Object Property | acps:promotes | promotes | acps:Variable | acps:Variable | A sub-property of acps:hasDirectCause indicating a positive or contributory influence. |
| Object Property | acps:inhibits | inhibits | acps:Variable | acps:Variable | A sub-property of acps:hasDirectCause indicating a negative or suppressive influence. |
| Object Property | acps:enables | enables | acps:Variable | acps:Variable | A sub-property of acps:hasDirectCause where the cause is a necessary precondition for the effect. |
| Object Property | acps:prevents | prevents | acps:Variable | acps:Variable | A sub-property of acps:hasDirectCause where the cause actively stops the effect from occurring. |
| Annotation Property | acps:hasConfidenceScore | has confidence score | Any | xsd:float | Attaches a confidence value (0.0 to 1.0) to a causal statement. |
| Annotation Property | acps:hasDataSource | has data source | Any | xsd:anyURI or xsd:string | Cites the source (e.g., publication, model) for a causal claim. |
Section 2.2: Integrating Causal Models and Queries with SPARQL
This section provides implementation guidance on the practical application of the ACPS/1.0 ontology. It outlines how the Knowledge Graph will be populated with causal information and demonstrates how system agents will use SPARQL, the standard query language for RDF, to retrieve and analyze this information.
Populating the Knowledge Graph
Data ingestion pipelines will be developed or modified to transform the output of causal discovery algorithms, econometric models, and expert-defined causal structures into RDF triples that conform to the ACPS/1.0 ontology. This process will involve mapping variables to acps:Variable individuals and relationships to the appropriate acps:hasDirectCause sub-properties.
Example Instantiation (2008 Financial Crisis Scenario):
The following is an example, in Turtle (TTL) syntax, of how a simplified causal model of the 2008 financial crisis would be represented in the KG.
@prefix : <https://adam.system/data/>.
@prefix acps: <https://adam.system/ontologies/v20.0/causal#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

# Define the Causal Model instance
:model_2008_crisis a acps:CausalModel ;
    rdfs:label "Simplified Causal Model of the 2008 Financial Crisis" ;
    acps:includesVariable :SubprimeDefaults, :MBS_Value_Collapse, :Lehman_Insolvency.

# Define the Variable instances
:SubprimeDefaults a acps:Variable ; rdfs:label "Subprime Mortgage Defaults".
:MBS_Value_Collapse a acps:Variable ; rdfs:label "Mortgage-Backed Securities Value Collapse".
:Lehman_Insolvency a acps:Variable ; rdfs:label "Lehman Brothers Insolvency".

# Define the Causal Relationships with annotations
:MBS_Value_Collapse acps:promotes :SubprimeDefaults.
<< :MBS_Value_Collapse acps:promotes :SubprimeDefaults >>
    acps:hasConfidenceScore "0.95"^^xsd:float ;
    acps:hasDataSource <https://example.com/papers/crisis_analysis_2010>.

:Lehman_Insolvency acps:promotes :MBS_Value_Collapse.
<< :Lehman_Insolvency acps:promotes :MBS_Value_Collapse >>
    acps:hasConfidenceScore "0.88"^^xsd:float ;
    acps:temporalLag "P6M"^^xsd:duration.

Querying with SPARQL
SPARQL will be the exclusive interface for retrieving causal information from the KG. This ensures a standardized, declarative method of data access for all agents. The following are canonical query patterns that will be implemented by agents like WSM v8.0 and the Natural Language Generation Agent.
Query 1: Find Direct Causes of an Event
This query retrieves all identified direct causes for a specific variable, Lehman_Insolvency.
PREFIX acps: <https://adam.system/ontologies/v20.0/causal#>
PREFIX : <https://adam.system/data/>

SELECT?causeLabel
WHERE {
  :Lehman_Insolvency acps:hasDirectCause?cause.
 ?cause rdfs:label?causeLabel.
}

Query 2: Reconstruct a Causal Chain (Path Query)
This query uses SPARQL 1.1 property paths to trace an entire causal chain backward from a final effect. The + operator indicates one or more traversals of the acps:hasDirectCause property, effectively finding all ancestors in the DAG. This is the core mechanism for fulfilling the Phase 2 deliverable of identifying the full causal chain.
PREFIX acps: <https://adam.system/ontologies/v20.0/causal#>
PREFIX : <https://adam.system/data/>

SELECT?ancestorLabel
WHERE {
  :Lehman_Insolvency (acps:hasDirectCause)+?ancestor.
 ?ancestor rdfs:label?ancestorLabel.
}

Query 3: Retrieve a Full Causal Model for Computation
This CONSTRUCT query is designed for WSM v8.0. Given a single variable, it identifies the CausalModel it belongs to and returns all triples that define that model's structure. This allows the WSM to retrieve the complete DAG necessary for its calculations in a single, efficient query.
PREFIX acps: <https://adam.system/ontologies/v20.0/causal#>
PREFIX : <https://adam.system/data/>

CONSTRUCT {
 ?model a acps:CausalModel.
 ?model acps:includesVariable?variable.
 ?effect acps:hasDirectCause?cause.
}
WHERE {
  :Lehman_Insolvency ^acps:includesVariable?model.
 ?model acps:includesVariable?variable.
 ?variable a acps:Variable.
  OPTIONAL {
   ?effect ^acps:includesVariable?model.
   ?cause ^acps:includesVariable?model.
   ?effect acps:hasDirectCause?cause.
  }
}

Enhancing the Natural Language Generation Agent
The Natural Language Generation Agent will be enhanced to execute these SPARQL queries against the KG. Its internal logic will include a mapping from the ACPS/1.0 ontology's predicates to natural language templates. For example, when processing a result set from a query that includes the acps:inhibits predicate, the agent will select phrases such as "this was suppressed by...", "this had a negative causal impact on...", or "this prevented...". This direct mapping from the formal ontology to linguistic patterns ensures that the system's explanations are not only clear and user-centric but are also rigorously grounded in the underlying causal model, thereby satisfying the Phase 3 success metric.
Part III: The Generative Simulation Framework
This part specifies the technical framework for the generative simulation engine, with a primary focus on the definition and parameterization of "black swan" scenarios. The central artifact is a highly structured, human-readable configuration schema designed to allow domain experts and autonomous agents to define complex, multivariate, and narrative-driven market events. This framework is essential for generating the synthetic data required to train more robust and resilient agent models.
Section 3.1: The "Black Swan" Scenario Definition Schema (BSSDS/1.0)
This section defines the Black Swan Scenario Definition Schema v1.0 (BSSDS/1.0), a comprehensive schema for simulation configuration files. YAML is selected as the configuration format for its superior human readability, especially for complex, nested data structures. Its support for features like comments, anchors, and aliases is particularly valuable for creating modular, reusable, and self-documenting scenario definitions.
Schema Rationale
A simple key-value configuration file would be insufficient for this task. As described in financial risk management and stress testing literature, extreme market events are not simple parameter changes; they are complex, multivariate processes that often follow a coherent, logical narrative. A "black swan" event is a process with a beginning, a middle, and an end, featuring cascading effects and evolving correlations. The BSSDS/1.0 schema is designed to capture this narrative structure by defining not just static parameter shocks, but also event triggers, durations, and dynamic inter-variable correlations.
This approach elevates the simulation configuration from a simple parameter file to a "Scenario Definition Language." By structuring the schema around a timeline of composable eventBlocks, the scenario designer—whether the Risk Assessment Agent or a human expert—can construct a story. The introduction of eventPrimitives and YAML anchors makes this process highly efficient and consistent. A complex scenario like a "Global Cyberattack on Financial Infrastructure" can be composed from primitives like SWIFT_Outage, EquityMarket_FlashCrash, and Liquidity_Freeze, each with its own detailed parameters. This compositional approach directly addresses the need to model unprecedented events by allowing for novel combinations of established primitives. It also makes the scenarios more transparent, auditable, and easier for both humans and machines to understand and author.
Top-Level YAML Structure
A valid BSSDS/1.0 document will have the following top-level structure:
 * schemaVersion: A string identifying the schema version, e.g., bssds/1.0.
 * scenarioMetadata: An object containing administrative information: a unique scenarioID (UUID), a human-readable name, a detailed description of the scenario's narrative, the author (agent or human), and the creationDate.
 * globalParameters: Defines baseline settings for the entire simulation run. This includes simulationStartDate, simulationEndDate, the simulation timestep (e.g., '1D', '1H'), and the number of monteCarloRuns for stochastic simulations.
 * eventPrimitives: A dictionary of reusable event components, defined using YAML anchors. This is a key design feature for modularity and reusability. Experts can define canonical event types (e.g., StandardRecession, OilPriceShock, SovereignDefault_EU) here, which can then be referenced and combined in the timeline.
 * timeline: The core of the scenario definition. It is an ordered list of eventBlock objects, which are processed sequentially by the simulation engine to create the narrative of the event.
eventBlock Object Structure
Each eventBlock in the timeline defines a specific phase or component of the overall scenario.
 * eventID: A unique identifier for this specific event instance within the timeline.
 * trigger: An object defining the start condition for the event. This allows for dynamic, condition-based scenarios. It can be a fixed date (onDate: '2026-01-15') or a condition based on the simulation's state (onCondition: 'VIX > 40').
 * duration: The time period over which the event's shock functions are active, specified in a parseable format (e.g., '90D' for 90 days).
 * eventDefinition: This can be either an inline definition of the event or a reference to a pre-defined component in eventPrimitives using a YAML alias (e.g., <<: *StandardRecession). The definition itself contains:
   * eventType: A string from a controlled vocabulary (e.g., MacroeconomicShock, GeopoliticalEvent, TechnologicalDisruption, CreditEvent).
   * description: A narrative description of this specific event block.
   * shockParameters: An array of objects, where each object defines a shock to a single variable in the simulation model. This is the core of the event's impact.
     * variable: The canonical name of the model variable to be shocked (e.g., US_GDP_Growth, EURUSD_FX_Rate, LIBOR_3M).
     * shockFunction: The mathematical form of the shock. This can be a simple value (absolute_value: 150.0, percentage_change: -0.20) or a more complex function (timeseries_override: { file: 'data/oil_shock_ts.csv' }).
     * distribution: For stochastic simulations, this object defines the probability distribution from which the shock's magnitude is drawn. It includes the type (e.g., Normal, Uniform, Beta, Exponential) and the distribution's specific parameters (e.g., mean, std_dev, min, max).
     * correlationMatrix: An optional, embedded matrix defining the correlation of this variable's shock with other key variables during the event's duration. This is critical for modeling contagion and systemic risk realistically.
Example: "Critical Infrastructure Cyberattack" Scenario
The following is a condensed example of a BSSDS/1.0 YAML file for one of the black swan scenarios mentioned in the implementation plan.
schemaVersion: bssds/1.0
scenarioMetadata:
  scenarioID: "a1b2c3d4-..."
  name: "Black Swan: Critical Infrastructure Cyberattack"
  description: "A coordinated cyberattack targeting major financial clearing houses, causing a sudden market freeze and liquidity crisis."
  author: "RiskAssessmentAgent"
  creationDate: "2025-11-10"

globalParameters:
  simulationStartDate: "2026-01-01"
  simulationEndDate: "2026-12-31"
  timestep: "1D"
  monteCarloRuns: 1000

eventPrimitives:
  EquityMarket_FlashCrash: &FlashCrash
    eventType: "MarketShock"
    description: "A sudden, severe drop in major equity indices due to algorithmic trading halts and panic selling."
    shockParameters:
      - variable: "SP500_Index"
        shockFunction: "percentage_change"
        distribution: { type: "Normal", mean: -0.15, std_dev: 0.05 }
      - variable: "VIX_Index"
        shockFunction: "absolute_value"
        distribution: { type: "Uniform", min: 70, max: 90 }

timeline:
  - eventID: "InitialAttack"
    trigger: { onDate: "2026-02-01" }
    duration: "2D"
    eventDefinition:
      eventType: "OperationalRiskEvent"
      description: "Initial reports of cyberattack on payment systems. Interbank lending rates spike."
      shockParameters:
        - variable: "LIBOR_OIS_Spread"
          shockFunction: "absolute_value"
          distribution: { type: "Exponential", lambda: 0.5 }
        - correlationMatrix:
            -

  - eventID: "MarketContagion"
    trigger: { onCondition: "LIBOR_OIS_Spread > 1.0" }
    duration: "5D"
    eventDefinition:
      <<: *FlashCrash # Use the primitive defined above

Table 4: "Black Swan" Scenario Schema - Core Parameter Groups
| Parameter Group | Data Type | Description | Key Fields |
|---|---|---|---|
| scenarioMetadata | Object | Contains high-level administrative information identifying and describing the scenario. | scenarioID, name, description, author. |
| globalParameters | Object | Defines the baseline configuration for the entire simulation run, such as its time frame and stochastic settings. | simulationStartDate, simulationEndDate, timestep, monteCarloRuns. |
| eventPrimitives | Dictionary | A collection of reusable, modular event definitions identified by YAML anchors. Promotes consistency and composition. | User-defined keys (e.g., StandardRecession) with values being eventDefinition objects. |
| timeline | Array of Objects | An ordered sequence of eventBlock objects that defines the narrative progression of the simulation. | Each element is an eventBlock. |
| eventBlock | Object | A single, discrete phase or event within the timeline. | eventID, trigger, duration, eventDefinition. |
| shockParameters | Array of Objects | The core of an event, defining the specific shocks applied to model variables. | variable, shockFunction, distribution, correlationMatrix. |
Section 3.2: The Simulation-to-Retraining Pipeline
This section outlines the operational workflow that connects the scenario definition to the agent retraining and evaluation process, completing the feedback loop for system self-improvement.
Consumption by the World Simulation Model (WSM)
The World Simulation Model (WSM) will be equipped with a robust parser for the BSSDS/1.0 YAML format. Upon initiating a simulation run, the WSM will load the specified scenario file. It will interpret the globalParameters to set up the simulation environment. It will then iterate through the timeline, activating each eventBlock based on its trigger condition (either a specific date or a state-dependent condition). During an event's active duration, the WSM will apply the defined shockFunctions to its internal state variables at each timestep, drawing from the specified probability distributions if the simulation is stochastic.
Synthetic Data Archiving and Traceability
The primary output of each simulation run is a rich, multivariate time-series dataset representing a plausible future. To ensure reproducibility and enable effective model training, these datasets will be archived in a versioned data lake using an efficient, columnar format such as Apache Parquet.
A critical component of this process is metadata management. Each generated dataset will be stored alongside a metadata file. This file will contain the complete BSSDS/1.0 YAML configuration that was used to generate that specific dataset. This creates an immutable, auditable link between a data artifact and its generative parameters. This traceability is essential for debugging, for understanding model behavior, and for ensuring that the Machine Learning Model Training Agent knows the precise conditions under which a model's training data was created.
Automated Model Retraining Trigger
The Machine Learning Model Training Agent is configured to monitor the data lake for the arrival of new, validated synthetic datasets that have been tagged for retraining purposes. When a new dataset appears, it triggers an automated MLOps pipeline. This pipeline retrieves the new data and uses it to retrain the target model—for example, the Risk Assessment Agent's portfolio stress-testing model. The retrained model is versioned and stored in a model registry, with a link back to the synthetic dataset (and thus the generating scenario) used for its training.
"Red Team" Simulation and Evaluation Protocol
The "Red Team" simulation, orchestrated by the Discussion Chair Agent, is the formal evaluation workflow designed to validate the effectiveness of the generative simulation and retraining process.
 * Scenario Creation: A "Red Team" scenario is authored, either by a human expert or a specialized agent. This scenario is explicitly designed to be novel and to test for robustness against conditions that were not included in the retraining dataset. For example, if the agent was retrained on energy price shocks, the red team scenario might be a sudden technological breakthrough that renders a major industry obsolete.
 * Execution: The Discussion Chair Agent initiates a full-system simulation using this red team scenario. The system's response, including all analyses, risk assessments, and proposed portfolio adjustments from the newly retrained agents, is captured in a detailed transcript.
 * Baseline Comparison: To provide a quantitative measure of improvement, the exact same red team scenario is run against a baseline version of the system (e.g., Adam v19.2), and its response is also captured.
 * Performance Evaluation: The Risk Assessment Agent and a designated Human Reviewer perform the final evaluation. The primary success metric is a quantitative comparison of the outcomes. For example, the simulated final portfolio value under the v20.0 recommendations versus the v19.2 recommendations. A successful outcome is one where the v20.0 system's proposed adjustments are demonstrably more effective at mitigating risk and preserving value in the simulated crisis. The results are compiled into a formal evaluation report, which provides the final validation of the v20.0 enhancements.
Conclusion
The technical specifications outlined in this document provide a comprehensive and actionable blueprint for the development of Adam v20.0. By translating the high-level strategic roadmap into concrete schemas, protocols, and workflows, this specification ensures that the development process is grounded in robust, scalable, and interoperable architectural principles.
The Enhanced Autonomy framework, centered on the formal Agent Proposal Standard (APS/1.0) and the API-driven Agent Lifecycle Management workflow, establishes a clear pathway for the system to achieve genuine self-improvement. It moves beyond simple automation to a state where the system can identify its own limitations and orchestrate the creation of new capabilities in a structured and auditable manner.
The Causal Inference architecture, through the adoption of the RDF/OWL-based Adam Causal Predicate Set (ACPS/1.0), provides the semantic rigor necessary to elevate the system's Knowledge Graph from a correlational database to a true causal reasoning engine. This enables Adam to move from describing "what" happened to explaining "why" it happened, a fundamental step towards becoming a strategic partner.
Finally, the Generative Simulation framework, powered by the narrative-driven Black Swan Scenario Definition Schema (BSSDS/1.0), equips the system with the ability to explore and prepare for novel, high-impact events. The defined simulation-to-retraining pipeline creates a closed loop where the system can proactively train itself against plausible future crises, enhancing its resilience and the robustness of its recommendations.
Collectively, these three pillars of development transform Adam v20.0 from a world-class analytical tool into a proactive, self-evolving system. The successful implementation of these specifications will result in a platform that is not only more powerful but also more intelligent, adaptive, and prepared for the complexities of the future.
