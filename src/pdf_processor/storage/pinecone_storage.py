"""Module for storing embeddings in Pinecone using LangChain."""
import os
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain.schema.document import Document
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PineconeStorage:
    """Handles storage and retrieval of document embeddings in Pinecone."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super(PineconeStorage, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone connection with LangChain integration."""
        if not hasattr(self, 'initialized'):
            try:
                # Initialize Pinecone
                self.pc = PineconeClient(
                    api_key=api_key,
                    environment=environment
                )
                
                # Create index if it doesn't exist
                dimension = 384  # Default for all-MiniLM-L6-v2
                if index_name not in self.pc.list_indexes().names():
                    self.pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric='cosine'
                    )
                    logger.info(f"Created new Pinecone index: {index_name}")
                
                # Initialize embeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                # Initialize LangChain's Pinecone integration
                self.vectorstore = Pinecone.from_existing_index(
                    index_name=index_name,
                    embedding=self.embeddings,
                    text_key='text'
                )
                
                self.initialized = True
                logger.info("Pinecone storage initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {str(e)}")
                raise
                
    def upload_documents(self, texts: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Upload chunks to Pinecone using LangChain's vectorstore."""
        try:
            if not texts:
                logger.warning("No texts provided for upload")
                return False
                
            # Prepare documents with metadata
            documents = [
                {
                    'text': text,
                    'metadata': metadata or {}
                }
                for text in texts
            ]
            
            # Upload to Pinecone
            self.vectorstore.add_texts(
                texts=[doc['text'] for doc in documents],
                metadatas=[doc['metadata'] for doc in documents]
            )
            
            logger.info(f"Successfully uploaded {len(documents)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload documents to Pinecone: {str(e)}")
            return False
        
    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert our chunk format to LangChain documents."""
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata=chunk['metadata']
            )
            documents.append(doc)
        return documents
        
    def upload_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Upload chunks to Pinecone using LangChain's vectorstore."""
        try:
            logger.info(f"Converting {len(chunks)} chunks to LangChain documents")
            documents = self._chunks_to_documents(chunks)
            
            # Process in batches
            for i in tqdm(range(0, len(documents), batch_size)):
                batch = documents[i:i + batch_size]
                
                try:
                    self.vectorstore.add_documents(batch)
                    logger.debug(f"Successfully uploaded batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
                    
            logger.info(f"Successfully uploaded {len(documents)} documents to Pinecone")
            
        except Exception as e:
            logger.error(f"Error in upload_batch: {str(e)}")
            
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search using LangChain's vectorstore."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
            