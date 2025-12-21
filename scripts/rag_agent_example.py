import asyncio
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Core components
from core.agents.agent_base import Agent
from core.embeddings.models.dummy_embedding_model import DummyEmbeddingModel
from core.llm.engines.dummy_llm_engine import DummyLLMEngine
from core.rag.document_handling import Document
from core.vectorstore.stores.in_memory_vector_store import InMemoryVectorStore

# Semantic Kernel (optional, for advanced use)
try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion  # Example connector
    # Note: Using SK requires actual LLM service configuration for most features.
    # The DummyLLMEngine is not directly usable by SK's standard connectors.
    # For this example, SK part will be mostly conceptual unless a real service is configured.
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    logger.info("Semantic Kernel library not found. Skipping SK-related examples.")

# Tools (optional)
from core.tools.web_search_tool import WebSearchTool


async def basic_rag_example():
    logger.info("\n--- Running Basic RAG Example ---")

    # 1. Initialize components
    # For DummyLLMEngine and DummyEmbeddingModel, ensure embedding_dim matches.
    # DummyLLMEngine's generate_embedding produces 128-dim if not overridden.
    # DummyEmbeddingModel defaults to 128-dim.
    llm_engine = DummyLLMEngine(model_name="dummy-chat-v1")
    embedding_model = DummyEmbeddingModel(model_name="dummy-embed-v1", embedding_dim=128)
    vector_store = InMemoryVectorStore(embedding_dim=128)

    # 2. Initialize Agent
    rag_agent = Agent(
        llm_engine=llm_engine,
        embedding_model=embedding_model,
        vector_store=vector_store,
        agent_id="example_rag_agent"
    )
    logger.info(f"RAG Agent '{rag_agent.agent_id}' initialized.")

    # 3. Ingest Documents
    logger.info("\n--- Ingesting Documents ---")
    doc1_content = "The first document talks about space exploration and Mars missions. NASA is planning new rover missions."
    doc2_content = "The second document is about culinary arts, focusing on Italian pasta dishes like Carbonara and Bolognese. These dishes are popular worldwide."
    doc_nasa_details = "Further details on NASA's Artemis program aim to return humans to the Moon by 2028, serving as a stepping stone for Mars."

    await rag_agent.ingest_document(doc1_content)
    await rag_agent.ingest_document(Document(content=doc2_content, id="doc_cuisine_002", metadata={"category": "food"}))
    await rag_agent.ingest_document(doc_nasa_details, chunk_size=50, chunk_overlap=5) # Smaller chunks for this specific text

    # 4. Process Queries
    logger.info("\n--- Processing Queries ---")
    queries = [
        "What is NASA planning regarding Mars?",
        "Tell me about Italian food.",
        "What is the Artemis program?"
    ]

    for query in queries:
        response = await rag_agent.process_query(query)
        logger.info(f"Query: {query}\nResponse: {response}\n")

    logger.info("Basic RAG Example Finished.")


async def advanced_rag_with_sk_and_tools_example():
    if not SK_AVAILABLE:
        logger.warning("Skipping Advanced RAG example as Semantic Kernel is not available.")
        return

    logger.info("\n--- Running Advanced RAG Example with Semantic Kernel and Tools ---")

    # 1. Initialize Core Components (as above)
    # For SK to use a real LLM, you'd use OpenAILLMEngine or similar, not DummyLLMEngine for SK parts.
    # However, our RAG Agent's LLM can still be Dummy for its direct calls.
    # SK Kernel needs its own LLM service.

    llm_engine = DummyLLMEngine(model_name="dummy-chat-for-rag") # RAG Agent's internal LLM
    embedding_model = DummyEmbeddingModel(embedding_dim=128)
    vector_store = InMemoryVectorStore(embedding_dim=128)

    # 2. Initialize Semantic Kernel
    kernel = Kernel()

    # --- IMPORTANT ---
    # To run SK parts that call an LLM (like QueryEnhancerSkill or planning),
    # you MUST add a real LLM service to the kernel.
    # Example (requires OPENAI_API_KEY and org_id in env or passed):
    # try:
    #     kernel.add_service(OpenAIChatCompletion(service_id="default", env_file_path=".env")) # Loads from .env
    #     logger.info("OpenAI Chat Completion service added to SK Kernel.")
    #     SK_LLM_CONFIGURED = True
    # except Exception as e:
    #     logger.warning(f"Could not configure OpenAI service for SK Kernel: {e}. SK LLM features will be limited.")
    #     SK_LLM_CONFIGURED = False
    # For this script, we'll assume SK_LLM_CONFIGURED = False if not explicitly set up with real keys.
    SK_LLM_CONFIGURED = False
    logger.warning("Semantic Kernel is initialized, but NO LLM SERVICE IS CONFIGURED for it in this example.")
    logger.warning("SK skills requiring LLM calls (like QueryEnhancerSkill) will likely use a fallback or fail if not configured.")


    # 3. Initialize Agent with SK Kernel
    rag_agent_sk = Agent(
        llm_engine=llm_engine,
        embedding_model=embedding_model,
        vector_store=vector_store,
        agent_id="advanced_rag_agent",
        kernel=kernel
    )
    logger.info(f"RAG Agent '{rag_agent_sk.agent_id}' initialized with Semantic Kernel.")

    # 4. Register a Tool
    web_tool = WebSearchTool() # Uses sandbox's view_text_website
    rag_agent_sk.register_tool(web_tool) # Registers with plugin_name="web_search" by default
    logger.info("WebSearchTool registered with the agent's kernel.")

    # 5. Ingest some initial documents (same as basic example for context)
    await rag_agent_sk.ingest_document("Mars is a planet. NASA sends rovers there.")
    await rag_agent_sk.ingest_document("The Moon is Earth's natural satellite. Artemis program targets it.")

    # 6. Example: Using a SK skill (QueryEnhancerSkill)
    # This skill uses an LLM, so it needs SK_LLM_CONFIGURED = True with a real LLM.
    if SK_LLM_CONFIGURED:
        logger.info("\n--- Using SK QueryEnhancerSkill ---")
        original_query = "tell me about moon missions"
        enhanced_query = await rag_agent_sk.enhance_query_with_sk(original_query)
        logger.info(f"Original Query: {original_query} -> Enhanced Query (SK): {enhanced_query}")
        response = await rag_agent_sk.process_query(enhanced_query or original_query) # Use enhanced if available
        logger.info(f"Response to (enhanced) query: {response}\n")
    else:
        logger.info("\n--- SK QueryEnhancerSkill (Skipped due to no LLM in SK Kernel) ---")
        logger.info("To run this, configure an LLM service in the SK Kernel above.")


    # 7. Example: Using the WebSearchTool via agent's helper method
    logger.info("\n--- Using WebSearchTool ---")
    # This tool uses the sandbox's view_text_website, does not need external LLM for its own execution.
    # We need a real URL for view_text_website to work.
    # For this example, we'll simulate the outcome.

    # Conceptual: User asks something that implies needing web search
    query_needing_web = "search for recent news on Artemis program"
    # The sandbox's view_text_website tool would need a specific URL.
    # Let's assume we've identified a URL through some means (or a more advanced tool did).
    # For this example, we'll use a placeholder URL.
    # To make this runnable, we'd need a URL the sandbox can access or mock view_text_website.
    # Since we can't guarantee external access, the WebSearchTool itself has simulation logic for query-only.

    # Scenario 1: Tool decides it needs a URL
    web_content_from_query = await rag_agent_sk.search_web_if_needed(query=query_needing_web)
    logger.info(f"Web search result (from query only): {web_content_from_query}") # Likely an info message from the tool

    # Scenario 2: Providing a direct URL (more likely to yield content with current WebSearchTool)
    # This URL is for demonstration; view_text_website may or may not access it.
    sample_url = "https://www.nasa.gov/specials/artemis/"
    logger.info(f"Attempting to fetch content from URL: {sample_url} (This may fail in sandbox if URL is blocked or tool is restricted)")
    web_content_from_url = await rag_agent_sk.search_web_if_needed(query="artemis info", direct_url=sample_url)

    if web_content_from_url and not web_content_from_url.startswith("Error:") and not web_content_from_url.startswith("Info:"):
        logger.info(f"Content fetched from URL (first 200 chars): {web_content_from_url[:200]}...")
        await rag_agent_sk.ingest_document(f"Web context about Artemis: {web_content_from_url}", chunk_size=150, chunk_overlap=15)
        response_after_web = await rag_agent_sk.process_query("What are the latest updates on Artemis based on web context?")
        logger.info(f"Response (after web search): {response_after_web}\n")
    else:
        logger.warning(f"Could not fetch or use web content from URL. Result: {web_content_from_url}")

    logger.info("Advanced RAG Example Finished.")


async def main():
    await basic_rag_example()
    # Uncomment to run the advanced example.
    # Note: For full functionality of advanced example, SK Kernel needs an LLM service configured
    # and the WebSearchTool needs URLs that the sandbox environment can access.
    await advanced_rag_with_sk_and_tools_example()

if __name__ == "__main__":
    asyncio.run(main())
