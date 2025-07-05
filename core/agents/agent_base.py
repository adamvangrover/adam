# core/agents/agent_base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional # Optional was already here, ensure Dict and Any are used consistently
import logging
import json
import asyncio

# Import Kernel for type hinting
from semantic_kernel import Kernel


# Configure logging (you could also have a central logging config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AgentBase(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and behavior expected of all agents.
    This version incorporates MCP, A2A, and Semantic Kernel.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        """
        Initializes the AgentBase. Subclasses should call super().__init__(config, kernel)
        to ensure proper initialization. The config dictionary provides agent-specific
        configuration parameters, and kernel is an optional Semantic Kernel instance.
        """
        self.config = config
        self.kernel = kernel # Store the Semantic Kernel instance
        self.context: Dict[str, Any] = {}
        self.peer_agents: Dict[str, AgentBase] = {}  # For A2A
        # Updated log message to reflect potential kernel presence
        log_message = f"Agent {type(self).__name__} initialized with config: {config}"
        if kernel:
            log_message += " and Semantic Kernel instance."
        else:
            log_message += "."
        logging.info(log_message)


    def set_context(self, context: Dict[str, Any]):
        """
        Sets the MCP context for the agent. This context contains
        information needed to perform the agent's task.
        """
        self.context = context
        logging.debug(f"Agent {type(self).__name__} context set: {context}")

    def get_context(self) -> Dict[str, Any]:
        """
        Returns the current MCP context.
        """
        return self.context

    def add_peer_agent(self, agent: 'AgentBase'):
        """
        Adds a peer agent for A2A communication.
        """
        self.peer_agents[agent.name] = agent
        logging.info(f"Agent {self.name} added peer agent: {agent.name}")

    async def send_message(self, target_agent: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends an A2A message to another agent and waits for the response.
        """
        if target_agent not in self.peer_agents:
            raise ValueError(f"Agent '{target_agent}' is not a known peer.")

        logging.info(f"Agent {self.name} sending message to {target_agent}: {message}")
        response = await self.peer_agents[target_agent].receive_message(self.name, message)
        logging.info(f"Agent {self.name} received response from {target_agent}: {response}")
        return response

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        This is the main entry point for agent execution.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills (MCP). This should be overridden
        by subclasses to describe their specific capabilities.
        """
        return {
            "name": type(self).__name__,
            "description": self.config.get("description", "No description provided"),
            "skills": []
        }

    async def receive_message(self, sender_agent: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handles incoming A2A messages. Subclasses should override
        this to define how they respond to messages.
        """
        logging.info(f"Agent {self.name} received message from {sender_agent}: {message}")
        return None  # Default: No response


    async def run_semantic_kernel_skill(self, skill_collection_name: str, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Executes a Semantic Kernel skill from a specific collection.
        This assumes the agent has access to a Semantic Kernel instance (self.kernel)
        and that skills have been imported into collections.
        """
        if not hasattr(self, 'kernel') or not self.kernel:
            raise AttributeError("Agent does not have access to a Semantic Kernel instance (self.kernel).")
        if not hasattr(self.kernel, 'skills') or not hasattr(self.kernel.skills, 'get_function'):
             raise AttributeError("Semantic Kernel instance does not have 'skills.get_function' method. SK version might be different than expected.")


        # Get the Semantic Kernel function from the specified collection and skill name.
        # For SK v1.x Python, this is typically kernel.plugins[plugin_name][function_name]
        # or kernel.skills.get_function(skill_collection_name, skill_name) if skills are registered that way.
        # The problem description suggests kernel.skills.get_function(skill_collection_name, skill_name).
        try:
            sk_function = self.kernel.skills.get_function(skill_collection_name, skill_name)
        except Exception as e: # Broad exception to catch issues if .skills or .get_function doesn't exist as expected
            logging.error(f"Error accessing SK function '{skill_name}' in collection '{skill_collection_name}': {e}. This might be due to an unexpected SK version or structure.")
            raise ValueError(f"Could not retrieve Semantic Kernel skill '{skill_name}' from collection '{skill_collection_name}'. Error: {e}")


        if not sk_function:
            raise ValueError(f"Semantic Kernel skill '{skill_name}' not found in collection '{skill_collection_name}'.")

        # Create the Semantic Kernel context.
        # For SK v1.x, context is often handled via KernelArguments or directly passed to invoke.
        # The method `kernel.create_new_context()` is from older versions (e.g., v0.9.x)
        # If using SK v1.x, `kernel.run_async(sk_function, input_vars=input_vars)` might not be the right way.
        # It would be more like: `await self.kernel.invoke(sk_function, **input_vars)`
        # or `await sk_function.invoke(variables=input_vars)`
        # Given the existing `self.kernel.run`, I will assume it's for an older SK version compatible with create_new_context.
        # However, the instruction for SK v1.x style get_function is a bit conflicting.
        # Let's stick to the existing run pattern but log a warning if create_new_context is missing.
        
        sk_context = None
        if hasattr(self.kernel, 'create_new_context') and callable(self.kernel.create_new_context):
            sk_context = self.kernel.create_new_context()
            # Set input variables for the Semantic Kernel function.
            for var_name, var_value in input_vars.items():
                sk_context[var_name] = var_value
        else:
            # If create_new_context is not available (e.g. SK v1.x), input_vars are usually passed directly to run/invoke.
            # The existing self.kernel.run call takes input_vars, so this might be fine.
            logging.debug("kernel.create_new_context() not found, assuming input_vars passed directly to kernel.run().")


        # Execute the Semantic Kernel function.
        # The existing code is: result = await self.kernel.run(sk_function, input_vars=input_vars)
        # For SK v1.x, it would be more like: result = await self.kernel.invoke(sk_function, **input_vars)
        # or await sk_function.invoke(input_vars)
        # Given the instruction to keep `kernel.run`, I'll use it.
        # If sk_context was created, some SK versions expect it in run: await self.kernel.run(sk_function, context=sk_context)
        # If not, input_vars directly: await self.kernel.run(sk_function, input_vars=input_vars)
        # The original code did not pass sk_context to run.
        
        # Using input_vars directly as per the original structure of kernel.run call
        result = await self.kernel.run_async(sk_function, input_vars=input_vars) # kernel.run is often run_async

        # Return the result as a string.

        # Set input variables for the Semantic Kernel function.
        for var_name, var_value in input_vars.items():
            sk_context[var_name] = var_value

        # Execute the Semantic Kernel function.
        result = await self.kernel.run(sk_function, input_vars=input_vars)

        # Return the result as a string.
        return str(result)


# New Agent class for RAG pipeline
from core.llm.base_llm_engine import BaseLLMEngine
from core.embeddings.base_embedding_model import BaseEmbeddingModel
from core.vectorstore.base_vector_store import BaseVectorStore

class Agent:
    def __init__(
        self,
        llm_engine: BaseLLMEngine,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        agent_id: str = "rag_agent",
        kernel: Optional[Kernel] = None, # Added kernel parameter
    ):
        self.agent_id = agent_id
        self.llm_engine = llm_engine
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.kernel = kernel # Store the kernel instance
        self.tools = {} # To store instantiated tools
        self.state = "idle"
        log_message = f"RAG Agent {self.agent_id} initialized"
        if self.kernel:
            log_message += " with Semantic Kernel instance."
        else:
            log_message += "."
        logging.info(log_message)

    async def process_query(self, query: str) -> str:
        """
        Processes a user query using the RAG pipeline.
        """
        self.state = "active"
        logging.info(f"RAG Agent {self.agent_id} received query: {query}")

        # 1. Generate query embedding
        query_embedding = await self.embedding_model.generate_embedding(query)
        logging.debug(f"RAG Agent {self.agent_id} generated query embedding.")

        # 2. Search for relevant documents in the vector store
        retrieved_docs = await self.vector_store.search(query_embedding, top_k=3)
        logging.debug(f"RAG Agent {self.agent_id} retrieved {len(retrieved_docs)} documents.")

        # 3. Format context from retrieved documents
        context = "\n".join([doc[0] for doc in retrieved_docs])
        if not context:
            logging.warning(f"RAG Agent {self.agent_id} found no relevant context for query: {query}")
            # Fallback or specific handling for no context can be added here
            # For now, proceed with an empty context string.

        logging.info(f"RAG Agent {self.agent_id} prepared context (length: {len(context)}).")


        # 4. Generate response using LLM with query and context
        response = await self.llm_engine.generate_response(prompt=query, context=context)
        logging.info(f"RAG Agent {self.agent_id} generated response.")

        self.state = "idle"
        return response

    async def ingest_document(self, doc_input: Any, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Ingests a document or text into the RAG system.
        - If doc_input is a string, it's treated as raw content.
        - If doc_input is a Document object, its content is used.
        - Chunks the document content.
        - Generates embeddings for each chunk.
        - Adds each chunk and its embedding to the vector store.
        """
        from core.rag.document_handling import Document, chunk_text # Local import

        if isinstance(doc_input, str):
            doc = Document(content=doc_input)
        elif isinstance(doc_input, Document):
            doc = doc_input
        else:
            logging.error(f"RAG Agent {self.agent_id}: Invalid document input type: {type(doc_input)}")
            return

        self.state = "ingesting"
        logging.info(f"RAG Agent {self.agent_id} ingesting document (ID: {doc.id}, Source: {doc.metadata.get('source', 'N/A')})...")

        try:
            text_chunks = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not text_chunks:
                logging.warning(f"RAG Agent {self.agent_id}: No chunks generated for document {doc.id}. Content might be empty or too short.")
                self.state = "idle"
                return

            logging.info(f"RAG Agent {self.agent_id}: Document {doc.id} split into {len(text_chunks)} chunks.")

            documents_to_add = []
            for i, chunk_text_content in enumerate(text_chunks):
                # We could add chunk-specific metadata here, e.g., chunk_index, document_id
                # For now, the vector store interface takes (text, embedding).
                # If BaseVectorStore.add_documents can handle metadata, this would be the place to pass it.
                chunk_embedding = await self.embedding_model.generate_embedding(chunk_text_content)
                # Storing the chunk text along with its embedding.
                # The vector store would ideally also store/link metadata like doc.id, doc.metadata, chunk_id.
                # Current BaseVectorStore.add_documents takes List[Tuple[str, List[float]]]
                documents_to_add.append((chunk_text_content, chunk_embedding))

            if documents_to_add:
                await self.vector_store.add_documents(documents_to_add)
                logging.info(f"RAG Agent {self.agent_id}: Finished embedding and adding {len(documents_to_add)} chunks for document {doc.id}.")
            else:
                logging.info(f"RAG Agent {self.agent_id}: No document chunks were added to vector store for doc {doc.id}.")

        except Exception as e:
            logging.error(f"RAG Agent {self.agent_id}: Failed to ingest document (ID: {doc.id}): {e}")
            # Optionally re-raise or handle more gracefully
            # raise # Uncomment if the caller should handle this exception
        finally:
            self.state = "idle"

    def get_status(self) -> Dict[str, str]:
        return {"agent_id": self.agent_id, "state": self.state}

    async def enhance_query_with_sk(self, query: str) -> str:
        """
        Enhances a query using a Semantic Kernel skill.
        This is a demonstration of how the agent can use SK.
        """
        if not self.kernel:
            logging.warning(f"RAG Agent {self.agent_id}: Semantic Kernel not available. Returning original query.")
            return query

        # Import KernelArguments if using SK v1.x style
        try:
            from semantic_kernel.functions import KernelArguments
        except ImportError:
            # Fallback or error if SK version is not as expected
            logging.error("Failed to import KernelArguments from semantic_kernel.functions. Ensure Semantic Kernel v1.x is installed.")
            return query

        skill_name = "QueryEnhancerSkill"
        function_name = "enhance" # Assuming the prompt file name (skprompt.txt) translates to 'enhance' function

        if skill_name not in self.kernel.plugins:
            try:
                # Path to the directory containing the 'QueryEnhancerSkill' directory
                skills_directory_path = "core/agents/skills/rag_skills"
                # The import_plugin_from_prompt_directory expects the path to the skill group, then the specific skill dir name
                self.kernel.import_plugin_from_prompt_directory(skills_directory_path, skill_name)
                logging.info(f"RAG Agent {self.agent_id}: Loaded {skill_name} into Semantic Kernel from {skills_directory_path}/{skill_name}.")
            except Exception as e:
                logging.error(f"RAG Agent {self.agent_id}: Failed to load {skill_name}: {e}")
                return query

        try:
            if skill_name not in self.kernel.plugins or function_name not in self.kernel.plugins[skill_name]:
                logging.error(f"RAG Agent {self.agent_id}: {skill_name} or function {function_name} not found in kernel plugins after attempting load.")
                return query

            enhancer_function = self.kernel.plugins[skill_name][function_name]

            kernel_args = KernelArguments(query=query)
            result = await self.kernel.invoke(enhancer_function, kernel_args)

            enhanced_query = str(result)
            logging.info(f"RAG Agent {self.agent_id}: Enhanced query from '{query}' to '{enhanced_query}' using SK.")
            return enhanced_query
        except Exception as e:
            logging.error(f"RAG Agent {self.agent_id}: Error running QueryEnhancerSkill: {e}")
            return query

    def register_tool(self, tool_instance: Any, plugin_name: Optional[str] = None):
        """
        Registers a tool with the agent's Semantic Kernel instance.
        The tool_instance should have methods decorated with @kernel_function.
        """
        if not self.kernel:
            logging.warning(f"RAG Agent {self.agent_id}: Cannot register tool, Semantic Kernel not available.")
            return

        if not hasattr(tool_instance, "name"):
            logging.error(f"RAG Agent {self.agent_id}: Tool instance does not have a 'name' attribute.")
            return

        tool_name = plugin_name or tool_instance.name

        try:
            # SK v1.x: plugins are typically Python objects whose methods are decorated.
            # kernel.add_plugin(plugin_instance=tool_instance, plugin_name=tool_name)
            self.kernel.add_plugin(plugin_instance=tool_instance, plugin_name=tool_name)
            self.tools[tool_name] = tool_instance # Keep a reference if needed
            logging.info(f"RAG Agent {self.agent_id}: Registered tool '{tool_name}' with Semantic Kernel.")
        except Exception as e:
            logging.error(f"RAG Agent {self.agent_id}: Failed to register tool '{tool_name}': {e}")

    async def invoke_tool(self, plugin_name: str, function_name: str, **kwargs) -> Optional[str]:
        """
        Invokes a registered tool's function via Semantic Kernel.
        """
        if not self.kernel:
            logging.warning(f"RAG Agent {self.agent_id}: Cannot invoke tool, Semantic Kernel not available.")
            return None

        if plugin_name not in self.kernel.plugins or function_name not in self.kernel.plugins[plugin_name]:
            logging.error(f"RAG Agent {self.agent_id}: Tool/function '{plugin_name}.{function_name}' not found in Semantic Kernel.")
            return None

        try:
            from semantic_kernel.functions import KernelArguments
            kernel_args = KernelArguments(**kwargs)
            target_function = self.kernel.plugins[plugin_name][function_name]

            result = await self.kernel.invoke(target_function, kernel_args)
            return str(result)
        except Exception as e:
            logging.error(f"RAG Agent {self.agent_id}: Error invoking tool '{plugin_name}.{function_name}': {e}")
            return None

    async def search_web_if_needed(self, query: str, direct_url: Optional[str] = None) -> Optional[str]:
        """
        Example of how an agent might decide to use the web search tool.
        """
        # Simple logic: if the query asks to "search" or "find on the web", use the tool.
        # A more sophisticated agent would use an LLM prompt or a planner to decide this.
        if "search for" in query.lower() or "find on the web" in query.lower() or "what is" in query.lower() or direct_url:
            logging.info(f"RAG Agent {self.agent_id}: Web search triggered for query: '{query}' or URL: '{direct_url}'")

            # Name of the plugin and function should match how WebSearchTool was registered
            # and its @kernel_function decorated method.
            plugin_name = "web_search" # Default name from WebSearchTool.name
            function_name = "fetch_web_content" # From @kernel_function name in WebSearchTool

            tool_args = {}
            if direct_url:
                tool_args["url"] = direct_url
            if query and not direct_url: # Only pass query if no direct URL, or pass both if desired by tool
                tool_args["query"] = query

            if not tool_args:
                logging.warning(f"RAG Agent {self.agent_id}: Web search tool invoked without query or URL.")
                return "No query or URL provided for web search."

            return await self.invoke_tool(plugin_name, function_name, **tool_args)
        return None
