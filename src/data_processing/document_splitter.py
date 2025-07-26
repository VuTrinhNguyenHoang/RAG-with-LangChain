from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


class DocumentSplitter:
    def __init__(
        self, 
        chunk_size: int = 1024, 
        chunk_overlap: int = 256,
        separators: List[str] = None
    ):
        """
        Initialize document splitter with configuration
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting text
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        return self.text_splitter.split_documents(documents)
