from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
import os


class VectorStoreManager:
    def __init__(
        self, 
        persist_directory: str = "vector_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store manager
        
        Args:
            persist_directory: Directory to persist vector store
            model_name: HuggingFace model name for embeddings
        """
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self._vectorstore = None
    
    def create_vectorstore(self, documents: List[Document], force_recreate: bool = False) -> Chroma:
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to store
            force_recreate: If True, delete existing vector store
            
        Returns:
            Chroma vector store instance
        """
        # Delete existing collection if requested
        if force_recreate and os.path.exists(self.persist_directory):
            if self._vectorstore is None:
                self._vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            self._vectorstore.delete_collection()
            self._vectorstore = None
            
        # Create new vector store
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        return self._vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store
        
        Returns:
            Chroma vector store instance or None if not exists
        """
        if not os.path.exists(self.persist_directory):
            return None
            
        self._vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        return self._vectorstore
        
    def get_collection_data(self) -> Dict[str, Any]:
        """
        Get collection data including embeddings, documents and metadata
        
        Returns:
            Dictionary with embeddings, documents and metadata
        """
        if self._vectorstore is None:
            self.load_vectorstore()
            
        if self._vectorstore is None:
            return {"embeddings": [], "documents": [], "metadatas": []}
            
        return self._vectorstore._collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        
    @property
    def vectorstore(self) -> Optional[Chroma]:
        """Get current vector store"""
        if self._vectorstore is None:
            return self.load_vectorstore()
        return self._vectorstore
