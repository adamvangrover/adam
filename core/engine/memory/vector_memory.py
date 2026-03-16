import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any

class VectorMemoryManager:
    """
    Manages semantic RAG storage and retrieval using ChromaDB.
    Replaces static flat JSON context injections.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logging.info(f"Initializing VectorMemoryManager at {self.persist_directory}")
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # We load a local embedding model so we don't rely on outbound API calls for basic Memory
        logging.info("Loading SentenceTransformer embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create the main intelligence collection
        self.collection = self.client.get_or_create_collection(name="system_intelligence")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Generate dense vector embeddings for the text chunks
        return self.embedding_model.encode(texts).tolist()

    def ingest_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Embeds a document and stores it in the vector database.
        """
        if not text.strip():
            return
            
        logging.info(f"Ingesting document: {document_id}")
        
        # Simple chunking strategy for demonstration (split by paragraphs)
        chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 50]
        
        if not chunks:
            return
            
        embeddings = self._generate_embeddings(chunks)
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {} for _ in chunks]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        logging.info(f"Successfully vectorized and stored {len(chunks)} chunks for {document_id}")

    def semantic_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the vector database for structurally similar intelligence.
        """
        logging.info(f"Executing Semantic Search for: '{query}'")
        query_embedding = self._generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        formatted_results = []
        # ChromaDB returns parallel arrays. We zip them together.
        if results and results.get('documents') and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
                
        return formatted_results
