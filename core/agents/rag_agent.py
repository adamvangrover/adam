import logging
import asyncio
from typing import Any, Dict, Optional, List, Tuple

from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from core.embeddings.base_embedding_model import BaseEmbeddingModel
from core.vectorstore.base_vector_store import BaseVectorStore

try:
    from semantic_kernel import Kernel
except ImportError:
    Kernel = Any
    logging.warning("semantic_kernel module not found.")

logger = logging.getLogger(__name__)


class RAGAgent(AgentBase):
    """
    An agent that implements a Retrieval-Augmented Generation (RAG) pipeline.
    It can ingest documents and answer queries based on the ingested content.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        constitution: Optional[Dict[str, Any]] = None,
        kernel: Optional[Kernel] = None,
        llm_engine: Optional[BaseLLMEngine] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        **kwargs
    ):
        """
        Initializes the RAGAgent.
        """
        super().__init__(config, constitution, kernel)

        self.llm_engine = llm_engine
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.tools = {}
        self.state = "idle"

        # Log initialization details
        log_message = f"RAG Agent {type(self).__name__} initialized"
        if self.kernel:
            log_message += " with Semantic Kernel instance."
        else:
            log_message += "."
        logging.info(log_message)

    async def execute(self, skill_name: str, **kwargs) -> Optional[Any]:
        """
        Executes a RAG skill based on the skill_name.
        """
        if skill_name == 'process_rag_query':
            query = kwargs.get('query')
            if not query:
                return {"error": "Query not provided for process_rag_query skill."}
            return await self.process_query(query)

        elif skill_name == 'ingest_document':
            doc_input = kwargs.get('document')
            if not doc_input:
                return {"error": "Document not provided for ingest_document skill."}
            await self.ingest_document(doc_input)
            return {"status": "ingestion_complete"}

        else:
            logging.warning(f"Unknown skill '{skill_name}' requested from RAGAgent.")
            return None

    async def process_query(self, query: str) -> str:
        """
        Processes a user query using the RAG pipeline.
        """
        if not self.llm_engine or not self.embedding_model or not self.vector_store:
            return "RAG components (LLM, Embedding Model, Vector Store) are not fully initialized."

        self.state = "active"
        logging.info(f"RAG Agent {self.config.get('agent_id', 'unknown')} received query: {query}")

        # 1. Generate query embedding
        query_embedding = await self.embedding_model.generate_embedding(query)
        logging.debug(f"RAG Agent generated query embedding.")

        # 2. Search for relevant documents in the vector store
        retrieved_docs = await self.vector_store.search(query_embedding, top_k=3)
        logging.debug(f"RAG Agent retrieved {len(retrieved_docs)} documents.")

        # 3. Format context from retrieved documents
        # Assuming retrieved_docs is a list of tuples (content, score) or similar
        context = "\n".join([doc[0] for doc in retrieved_docs]) if retrieved_docs else ""

        if not context:
            logging.warning(f"RAG Agent found no relevant context for query: {query}")

        logging.info(f"RAG Agent prepared context (length: {len(context)}).")

        # 4. Generate response using LLM with query and context
        response = await self.llm_engine.generate_response(prompt=query, context=context)
        logging.info(f"RAG Agent generated response.")

        self.state = "idle"
        return response

    async def ingest_document(self, doc_input: Any, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Ingests a document or text into the RAG system.
        """
        if not self.embedding_model or not self.vector_store:
            logging.error("RAG Agent: Cannot ingest, components missing.")
            return

        # Local import to avoid circular dependency if any
        try:
            from core.rag.document_handling import Document, chunk_text
        except ImportError:
            # Fallback if module missing
            logging.warning("core.rag.document_handling not found, using simple string handling.")

            class Document:
                def __init__(self, content, metadata=None):
                    self.content = content
                    self.metadata = metadata or {}
                    self.id = "unknown"

            def chunk_text(text, chunk_size, chunk_overlap):
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-chunk_overlap)]

        if isinstance(doc_input, str):
            doc = Document(content=doc_input)
        elif isinstance(doc_input, Document):
            doc = doc_input
        else:
            logging.error(f"RAG Agent: Invalid document input type: {type(doc_input)}")
            return

        self.state = "ingesting"
        logging.info(f"RAG Agent ingesting document (ID: {doc.id}, Source: {doc.metadata.get('source', 'N/A')})...")

        try:
            text_chunks = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not text_chunks:
                logging.warning(f"RAG Agent: No chunks generated for document {doc.id}.")
                self.state = "idle"
                return

            logging.info(f"RAG Agent: Document {doc.id} split into {len(text_chunks)} chunks.")

            documents_to_add = []
            for chunk_text_content in text_chunks:
                chunk_embedding = await self.embedding_model.generate_embedding(chunk_text_content)
                documents_to_add.append((chunk_text_content, chunk_embedding))

            if documents_to_add:
                await self.vector_store.add_documents(documents_to_add)
                logging.info(
                    f"RAG Agent: Finished embedding and adding {len(documents_to_add)} chunks for document {doc.id}.")
            else:
                logging.info(f"RAG Agent: No document chunks were added to vector store for doc {doc.id}.")

        except Exception as e:
            logging.error(f"RAG Agent: Failed to ingest document (ID: {doc.id}): {e}")
        finally:
            self.state = "idle"

    async def enhance_query_with_sk(self, query: str) -> str:
        """
        Enhances a query using a Semantic Kernel skill.
        """
        if not self.kernel:
            logging.warning(f"RAG Agent: Semantic Kernel not available. Returning original query.")
            return query

        skill_name = "QueryEnhancerSkill"
        function_name = "enhance"

        try:
            # Check if plugin exists, if not try to load (mock path logic preserved from original)
            if hasattr(self.kernel, 'plugins') and skill_name not in self.kernel.plugins:
                # Logic to load would go here, omitting specific path assumptions for now
                pass

            if hasattr(self.kernel, 'plugins') and skill_name in self.kernel.plugins:
                enhancer_function = self.kernel.plugins[skill_name][function_name]
                # Assuming invoke takes arguments
                result = await self.kernel.invoke(enhancer_function, query=query)
                return str(result)

            return query
        except Exception as e:
            logging.error(f"RAG Agent: Error running QueryEnhancerSkill: {e}")
            return query

    def register_tool(self, tool_instance: Any, plugin_name: Optional[str] = None):
        """
        Registers a tool with the agent's Semantic Kernel instance.
        """
        if not self.kernel:
            logging.warning(f"RAG Agent: Cannot register tool, Semantic Kernel not available.")
            return

        if not hasattr(tool_instance, "name"):
            logging.error(f"RAG Agent: Tool instance does not have a 'name' attribute.")
            return

        tool_name = plugin_name or tool_instance.name

        try:
            # SK v1.x
            if hasattr(self.kernel, 'add_plugin'):
                self.kernel.add_plugin(plugin_instance=tool_instance, plugin_name=tool_name)
            self.tools[tool_name] = tool_instance
            logging.info(f"RAG Agent: Registered tool '{tool_name}' with Semantic Kernel.")
        except Exception as e:
            logging.error(f"RAG Agent: Failed to register tool '{tool_name}': {e}")

    async def invoke_tool(self, plugin_name: str, function_name: str, **kwargs) -> Optional[str]:
        """
        Invokes a registered tool's function via Semantic Kernel.
        """
        if not self.kernel:
            logging.warning(f"RAG Agent: Cannot invoke tool, Semantic Kernel not available.")
            return None

        try:
            if hasattr(self.kernel, 'plugins') and plugin_name in self.kernel.plugins:
                target_function = self.kernel.plugins[plugin_name][function_name]
                result = await self.kernel.invoke(target_function, **kwargs)
                return str(result)
        except Exception as e:
            logging.error(f"RAG Agent: Error invoking tool '{plugin_name}.{function_name}': {e}")
            return None

    async def search_web_if_needed(self, query: str, direct_url: Optional[str] = None) -> Optional[str]:
        """
        Example of how an agent might decide to use the web search tool.
        """
        if "search for" in query.lower() or "find on the web" in query.lower() or direct_url:
            logging.info(f"RAG Agent: Web search triggered for query: '{query}'")
            plugin_name = "web_search"
            function_name = "fetch_web_content"
            tool_args = {}
            if direct_url:
                tool_args["url"] = direct_url
            if query and not direct_url:
                tool_args["query"] = query

            return await self.invoke_tool(plugin_name, function_name, **tool_args)
        return None

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills (MCP).
        """
        return {
            "name": "RAGAgent",
            "description": "Handles document ingestion and retrieval-augmented generation queries.",
            "skills": [
                {
                    "name": "process_rag_query",
                    "description": "Answers a query based on documents in the vector store.",
                    "parameters": [
                        {"name": "query", "type": "string", "description": "The user's query."}
                    ]
                },
                {
                    "name": "ingest_document",
                    "description": "Ingests a document into the RAG system's vector store.",
                    "parameters": [
                        {"name": "document", "type": "any", "description": "The document content or object to ingest."}
                    ]
                }
            ]
        }
