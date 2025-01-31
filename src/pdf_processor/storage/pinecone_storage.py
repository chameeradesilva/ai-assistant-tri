"""Module for storing embeddings in Pinecone."""
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time

class PineconeStorage:
    _instance = None
    
    def __new__(cls, api_key: str, environment: str, index_name: str):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(PineconeStorage, cls).__new__(cls)
            cls._instance._initialize(api_key, environment, index_name)
        return cls._instance
        
    def _initialize(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone connection."""
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=384,  # dimension for all-MiniLM-L6-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment
                )
            )
            time.sleep(1)  # Rate limit after index creation
        
        self.index = self.pc.Index(index_name)

    def upload_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Upload chunks to Pinecone in batches."""
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            vectors = [(
                chunk['id'],
                chunk['embedding'],
                chunk['metadata']
            ) for chunk in batch]
            
            self.index.upsert(vectors=vectors)
            time.sleep(1)  # Rate limit between batch uploads 
            