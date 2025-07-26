import os
import time
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv


def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Check if GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY is not set. RAG functionality will not work.")
        
    # Return configuration as dictionary
    return {
        "knowledge_base_dir": os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge-base"),
        "vector_db_dir": os.getenv("VECTOR_DB_DIR", "./vector_db"),
        "model_name": os.getenv("MODEL_NAME", "gemini-2.0-flash"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    }


def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper
    

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """
    Save data as JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False
        
        
def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Loaded dictionary or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None
