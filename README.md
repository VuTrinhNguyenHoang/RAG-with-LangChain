# RAG with LangChain Project

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and HuggingFace embeddings, with a FastAPI backend.

## Project Structure

```
RAG-with-LangChain/
├── knowledge-base/                     # Directory with document source files
│   ├── company/                        # Company information documents
│   ├── employees/                      # Employee information documents
│   ├── schools/                        # School information documents
│   └── visas/                          # Visa information documents
├── notebooks/                          # Jupyter notebooks for experimentation
│   └── test.ipynb                      # Original RAG implementation notebook
├── src/                                # Source code directory
│   ├── api/                            # FastAPI application
│   │   └── app.py                      # Main FastAPI application
│   ├── data_processing/                # Document loading and processing
│   │   ├── document_loader.py          # Document loading utilities
│   │   └── document_splitter.py        # Document splitting utilities
│   ├── rag/                            # RAG implementation
│   │   └── rag_chain.py                # RAG chain management
│   ├── utils/                          # Utility functions
│   │   └── helpers.py                  # Helper functions
│   ├── vectorstore/                    # Vector database management
│   │   └── vector_store.py             # Chroma vector store management
│   └── visualization/                  # Data visualization
│       └── embeddings_viz.py           # TSNE visualization of embeddings
├── vector_db/                          # Vector database storage (created by the application)
├── visualizations/                     # Output directory for visualizations (created by the application)
├── main.py                             # Application entry point
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key_here
KNOWLEDGE_BASE_DIR=./knowledge-base
VECTOR_DB_DIR=./vector_db
MODEL_NAME=gemini-2.0-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your environment variables

## Running the Application

To start the FastAPI application:

```bash
python main.py
```

This will start the server on `http://0.0.0.0:8000` by default.

### Command-line Options

- `--host`: Host to bind the server to (default: "0.0.0.0")
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development

Example:

```bash
python main.py --host 127.0.0.1 --port 9000 --reload
```

## API Endpoints

- `GET /`: HTML welcome page
- `GET /status`: Check system status
- `POST /initialize`: Initialize or reinitialize the system
- `POST /query`: Query the RAG system
- `GET /visualize`: Generate embeddings visualization
- `GET /reset`: Reset conversation memory

## API Usage Examples

### Initialize the System

```bash
curl -X POST http://localhost:8000/initialize
```

### Query the RAG System

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you briefly describe the Korea Study Center?"}'
```

### Reset Conversation Memory

```bash
curl -X GET http://localhost:8000/reset
```

## Development

The original notebook implementation is preserved in `notebooks/test.ipynb` for reference and experimentation.
