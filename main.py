import os
import uvicorn
import argparse
from dotenv import load_dotenv

from src.utils.helpers import load_environment_variables


def main():
    # Load environment variables
    load_dotenv()
    config = load_environment_variables()
    
    # Create required directories
    os.makedirs("./visualizations", exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG API with LangChain")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Display configuration
    print("Starting RAG API with the following configuration:")
    print(f"- Knowledge Base: {config['knowledge_base_dir']}")
    print(f"- Vector DB: {config['vector_db_dir']}")
    print(f"- LLM Model: {config['model_name']}")
    print(f"- Embedding Model: {config['embedding_model']}")
    
    # Start Uvicorn server
    uvicorn.run(
        "src.api.app:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
