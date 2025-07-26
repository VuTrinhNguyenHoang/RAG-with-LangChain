from langchain.document_loaders import DirectoryLoader, TextLoader
import os
import glob
from typing import List, Dict, Any
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, knowledge_base_dir: str):
        """
        Initialize document loader with path to knowledge base directory
        
        Args:
            knowledge_base_dir: Path to knowledge base directory containing document folders
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.text_loader_kwargs = {"autodetect_encoding": True}
        
    def load_documents(self) -> List[Document]:
        """
        Load all documents from knowledge base directory and add metadata
        
        Returns:
            List of Document objects with appropriate metadata
        """
        # Get list of directories in knowledge-base
        folders = glob.glob(os.path.join(self.knowledge_base_dir, "*"))
        
        # Initialize list to hold documents
        documents = []
        
        # Loop through each folder to load documents
        for folder in folders:
            # Use folder name as document type
            doc_type = os.path.basename(folder)
            
            # Create loader for all .md files in the folder
            loader = DirectoryLoader(
                folder, 
                glob="**/*.md", 
                loader_cls=TextLoader, 
                loader_kwargs=self.text_loader_kwargs
            )
            
            # Load documents from the folder
            folder_docs = loader.load()
            
            # Add metadata and append to main list
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
                
        return documents
