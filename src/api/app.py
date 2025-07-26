from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional

from src.data_processing.document_loader import DocumentLoader
from src.data_processing.document_splitter import DocumentSplitter
from src.vectorstore.vector_store import VectorStoreManager
from src.rag.rag_chain import RAGChainManager
from src.visualization.embeddings_viz import create_embeddings_visualization, save_visualization

# Constants
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge-base")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")
VISUALIZATIONS_DIR = os.getenv("VISUALIZATIONS_DIR", "./visualizations")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Models for API
class QueryRequest(BaseModel):
    question: str
    return_source_docs: bool = False
    
class QueryResponse(BaseModel):
    answer: str
    processing_time: float
    source_documents: Optional[List[Dict[str, Any]]] = None
    
class StatusResponse(BaseModel):
    status: str
    documents_count: int = 0
    chunks_count: int = 0
    is_initialized: bool = False

# Initialize FastAPI app
app = FastAPI(
    title="RAG API with LangChain",
    description="API for Retrieval-Augmented Generation with LangChain and Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vector_store_manager = VectorStoreManager(
    persist_directory=VECTOR_DB_DIR,
    model_name=EMBEDDING_MODEL
)
rag_chain_manager = None

# State variables
initialized = False
documents_count = 0
chunks_count = 0


# Dependency to ensure system is initialized
def get_rag_manager():
    if not initialized or rag_chain_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized. Run /initialize first.")
    return rag_chain_manager


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic HTML welcome page"""
    return """
    <html>
        <head>
            <title>RAG API with LangChain</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                }
                .endpoint {
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>RAG API with LangChain</h1>
            <p>Welcome to the Retrieval-Augmented Generation API!</p>
            <div class="endpoint">
                <h3>GET /status</h3>
                <p>Check the system status</p>
            </div>
            <div class="endpoint">
                <h3>POST /initialize</h3>
                <p>Initialize or reinitialize the system</p>
            </div>
            <div class="endpoint">
                <h3>POST /query</h3>
                <p>Query the RAG system</p>
            </div>
            <div class="endpoint">
                <h3>GET /visualize</h3>
                <p>Generate embeddings visualization</p>
            </div>
            <div class="endpoint">
                <h3>GET /reset</h3>
                <p>Reset conversation memory</p>
            </div>
        </body>
    </html>
    """


@app.get("/status", response_model=StatusResponse)
async def status():
    """Get system status"""
    global initialized, documents_count, chunks_count
    
    return StatusResponse(
        status="initialized" if initialized else "not_initialized",
        documents_count=documents_count,
        chunks_count=chunks_count,
        is_initialized=initialized
    )


@app.post("/initialize", response_model=StatusResponse)
async def initialize(background_tasks: BackgroundTasks, force_recreate: bool = False):
    """Initialize or reinitialize the system"""
    global initialized, rag_chain_manager, documents_count, chunks_count
    
    def _initialize():
        global initialized, rag_chain_manager, documents_count, chunks_count
        
        # Load documents
        document_loader = DocumentLoader(knowledge_base_dir=KNOWLEDGE_BASE_DIR)
        documents = document_loader.load_documents()
        documents_count = len(documents)
        
        # Split documents
        document_splitter = DocumentSplitter(chunk_size=1024, chunk_overlap=256)
        chunks = document_splitter.split_documents(documents)
        chunks_count = len(chunks)
        
        # Create or load vector store
        vector_store = vector_store_manager.create_vectorstore(chunks, force_recreate=force_recreate)
        
        # Create RAG chain
        rag_chain_manager = RAGChainManager(
            vectorstore=vector_store,
            model_name=MODEL_NAME,
            temperature=0.7,
            search_k=30
        )
        
        initialized = True
    
    # Try to load existing vectorstore if not force recreating
    if not force_recreate and not initialized:
        vector_store = vector_store_manager.load_vectorstore()
        if vector_store is not None:
            rag_chain_manager = RAGChainManager(
                vectorstore=vector_store,
                model_name=MODEL_NAME,
                temperature=0.7,
                search_k=30
            )
            initialized = True
            
            # Get collection info
            collection_data = vector_store_manager.get_collection_data()
            chunks_count = len(collection_data.get("documents", []))
            documents_count = chunks_count  # Approximate
            return StatusResponse(
                status="initialized",
                documents_count=documents_count,
                chunks_count=chunks_count,
                is_initialized=True
            )
    
    # Otherwise initialize in background
    background_tasks.add_task(_initialize)
    
    return StatusResponse(
        status="initializing",
        is_initialized=False
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag_manager: RAGChainManager = Depends(get_rag_manager)):
    """Query the RAG system"""
    try:
        result = rag_manager.query(request.question, return_source_docs=request.return_source_docs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/visualize", response_class=FileResponse)
async def visualize():
    """Generate embeddings visualization"""
    global initialized
    
    if not initialized:
        raise HTTPException(status_code=503, detail="System not initialized. Run /initialize first.")
        
    try:
        # Get collection data
        collection_data = vector_store_manager.get_collection_data()
        
        # Extract data
        vectors = np.array(collection_data["embeddings"])
        documents = collection_data["documents"]
        doc_types = [metadata.get("doc_type", "unknown") for metadata in collection_data["metadatas"]]
        
        # Create visualization
        fig = create_embeddings_visualization(vectors, doc_types, documents)
        
        # Save visualization
        output_path = save_visualization(fig, VISUALIZATIONS_DIR)
        
        # Return HTML file
        return output_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating visualization: {str(e)}")


@app.get("/reset")
async def reset_memory(rag_manager: RAGChainManager = Depends(get_rag_manager)):
    """Reset conversation memory"""
    rag_manager.reset_memory()
    return {"status": "success", "message": "Conversation memory reset"}
