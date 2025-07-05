# RAG Agent System Overview

This document provides an overview of the RAG (Retrieval Augmented Generation) Agent system, its components, and how to use it.

## Core Components

The RAG Agent system is built upon several key abstractions and a central `Agent` class:

1.  **`Agent` (`core.agents.agent_base.Agent`)**:
    *   Orchestrates the RAG pipeline.
    *   Handles document ingestion (chunking, embedding, storing).
    *   Processes user queries (embedding query, retrieving relevant chunks, generating response with LLM).
    *   Can optionally integrate with Semantic Kernel for advanced skill/tool use.

2.  **`BaseLLMEngine` (`core.llm.base_llm_engine.BaseLLMEngine`)**:
    *   Abstract base class for language model interactions.
    *   Requires implementation of `generate_response()`.
    *   Optionally `generate_embedding()` if the LLM provider bundles it.
    *   Examples:
        *   `core.llm.engines.dummy_llm_engine.DummyLLMEngine`: For testing, echoes input.
        *   `core.llm.engines.openai_llm_engine.OpenAILLMEngine`: Conceptual connector for OpenAI models.

3.  **`BaseEmbeddingModel` (`core.embeddings.base_embedding_model.BaseEmbeddingModel`)**:
    *   Abstract base class for generating text embeddings.
    *   Requires implementation of `generate_embedding()`.
    *   Examples:
        *   `core.embeddings.models.dummy_embedding_model.DummyEmbeddingModel`: For testing, generates non-semantic embeddings.
        *   `core.embeddings.models.openai_embedding_model.OpenAIEmbeddingModel`: Conceptual connector for OpenAI embedding models.

4.  **`BaseVectorStore` (`core.vectorstore.base_vector_store.BaseVectorStore`)**:
    *   Abstract base class for vector database interactions.
    *   Requires implementation of `add_documents()` and `search()`.
    *   *(A concrete in-memory implementation will be provided in the example script for ease of use).*

5.  **Document Handling (`core.rag.document_handling`)**:
    *   `Document`: Class representing a document with content, ID, and metadata.
    *   `chunk_text()`: Utility to split text into manageable chunks for processing.

6.  **Tools (`core.tools`)**:
    *   `BaseTool`: Abstract base class for tools an agent can use.
    *   `WebSearchTool`: Example tool to fetch web content (uses sandbox `view_text_website`). Tools can be integrated via Semantic Kernel.

## Basic Usage

The `Agent` requires an LLM engine, an embedding model, and a vector store to function.

### Initialization

```python
# Conceptual imports (actual paths might vary based on project structure)
from core.agents.agent_base import Agent
from core.llm.engines.dummy_llm_engine import DummyLLMEngine
from core.embeddings.models.dummy_embedding_model import DummyEmbeddingModel
# An in-memory vector store would also be needed here (see example script)
# from core.vectorstore.stores.in_memory_vector_store import InMemoryVectorStore # Example path

# 1. Initialize components
llm_engine = DummyLLMEngine()
embedding_model = DummyEmbeddingModel(embedding_dim=128) # Ensure dim matches dummy engine if it also embeds
vector_store = InMemoryVectorStore(embedding_dim=128) # Example

# 2. Initialize Agent
rag_agent = Agent(
    llm_engine=llm_engine,
    embedding_model=embedding_model,
    vector_store=vector_store,
    agent_id="my_rag_agent"
)
```

### Ingesting Documents

Documents can be ingested as raw text or `Document` objects.

```python
from core.rag.document_handling import Document

# Ingest raw text
await rag_agent.ingest_document("This is the first document about apples.")

# Ingest a Document object
doc_content = "The second document discusses bananas and their nutritional value."
doc_metadata = {"source": "fruit_facts_vol1.txt"}
my_document = Document(content=doc_content, id="doc_banana_001", metadata=doc_metadata)
await rag_agent.ingest_document(my_document, chunk_size=100, chunk_overlap=10)
```

### Processing Queries

```python
query = "What are apples?"
response = await rag_agent.process_query(query)
print(f"Query: {query}\nResponse: {response}")

query_banana = "Tell me about bananas."
response_banana = await rag_agent.process_query(query_banana)
print(f"Query: {query_banana}\nResponse: {response_banana}")
```

## Advanced Usage

### Semantic Kernel Integration

If you initialize the `Agent` with a Semantic Kernel `Kernel` instance, it can leverage SK for more complex tasks, like using SK skills or planners.

```python
from semantic_kernel import Kernel

# Initialize SK Kernel (example, requires service configuration)
# kernel = Kernel()
# Add your LLM service connector to the kernel, e.g., OpenAI, AzureOpenAI
# from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
# kernel.add_service(OpenAIChatCompletion(service_id="chat-gpt", api_key="...", org_id="..."))


# rag_agent_with_sk = Agent(
#     llm_engine=...,
#     embedding_model=...,
#     vector_store=...,
#     kernel=kernel # Pass the configured kernel
# )

# Example: Enhancing a query using an SK skill (QueryEnhancerSkill)
# enhanced_query = await rag_agent_with_sk.enhance_query_with_sk("tell me about apples")
# response = await rag_agent_with_sk.process_query(enhanced_query)
# print(f"Enhanced Query Response: {response}")
```
The `QueryEnhancerSkill` is located in `core/agents/skills/rag_skills/`.

### Tool Integration

Tools (implementing `BaseTool` and often decorated for SK) can be registered with the agent's kernel and invoked.

```python
# Assuming rag_agent_with_sk from above and kernel is configured
# from core.tools.web_search_tool import WebSearchTool

# web_tool = WebSearchTool()
# rag_agent_with_sk.register_tool(web_tool, plugin_name="WebUtils") # Registers to SK kernel

# Using the tool (e.g., if a query requires web search)
# web_content = await rag_agent_with_sk.search_web_if_needed(
#    query="search for current apple prices",
#    direct_url="https://example.com/apple_prices" # Or provide a URL
# )
# if web_content:
#    await rag_agent_with_sk.ingest_document(f"Web context: {web_content}")
#    response = await rag_agent_with_sk.process_query("What are current apple prices based on web context?")
#    print(response)
```

## Switching LLM/Embedding Models

To use a different LLM or embedding model (e.g., a conceptual OpenAI one instead of Dummy):

```python
# from core.llm.engines.openai_llm_engine import OpenAILLMEngine
# from core.embeddings.models.openai_embedding_model import OpenAIEmbeddingModel

# openai_llm = OpenAILLMEngine(api_key="YOUR_OPENAI_KEY")
# openai_embedder = OpenAIEmbeddingModel(api_key="YOUR_OPENAI_KEY")
# Ensure vector_store embedding_dim matches the chosen embedding model, e.g., 1536 for ada-002

# vector_store_for_openai = InMemoryVectorStore(embedding_dim=1536) # Example

# openai_rag_agent = Agent(
#     llm_engine=openai_llm,
#     embedding_model=openai_embedder,
#     vector_store=vector_store_for_openai
# )

# Now use openai_rag_agent for ingestion and querying.
```

This modular design allows for flexibility in choosing and combining different backend services.
---

*Next: An example script demonstrating these concepts.*
```
