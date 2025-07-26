from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory  
from langchain.chains import ConversationalRetrievalChain
from langchain_core.vectorstores import VectorStore
from typing import Dict, Any, Optional, List
import time


class RAGChainManager:
    def __init__(
        self,
        vectorstore: VectorStore,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        search_k: int = 30,
        memory_k: int = 3
    ):
        """
        Initialize RAG chain manager
        
        Args:
            vectorstore: Vector store to retrieve documents from
            model_name: Name of the Gemini model to use
            temperature: Temperature for text generation
            search_k: Number of documents to retrieve per query
            memory_k: Number of conversation turns to keep in memory
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.temperature = temperature
        self.search_k = search_k
        self.memory_k = memory_k
        
        # Initialize components
        self.llm = None
        self.memory = None
        self.retriever = None
        self.conversation_chain = None
        
        # Create the chain
        self._create_chain()
        
    def _create_chain(self) -> None:
        """Create the RAG conversation chain"""
        # Create LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature
        )
        
        # Create memory
        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history', 
            return_messages=True,
            k=self.memory_k
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.search_k}
        )
        
        # Create conversation chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory
        )
    
    def reset_memory(self) -> None:
        """Reset conversation memory"""
        self.memory.clear()
    
    def query(self, question: str, return_source_docs: bool = False) -> Dict[str, Any]:
        """
        Process a query through the RAG chain
        
        Args:
            question: User's question to answer
            return_source_docs: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        start_time = time.time()
        result = self.conversation_chain.invoke({"question": question})
        end_time = time.time()
        
        response = {
            "answer": result["answer"],
            "processing_time": end_time - start_time
        }
        
        if return_source_docs and "source_documents" in result:
            response["source_documents"] = result["source_documents"]
            
        return response
