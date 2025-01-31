"""Module for generating embeddings from text chunks."""
from typing import List, Dict, Any
import logging
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    _instance = None
    
    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2'):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding generator."""
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
            self.initialized = True

    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks with batching.
        
        Args:
            chunks: List of dictionaries containing text chunks
            batch_size: Number of texts to process in each batch (default 32)
            
        Returns:
            List of chunks with embeddings added
        """
        texts = [chunk['text'] for chunk in chunks]
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")
            batch_embeddings = self._process_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress
            processed = min(i + batch_size, len(texts))
            logger.info(f"Progress: {processed}/{len(texts)} texts processed")
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding
            # Add metadata for timestamp
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['timestamp'] = datetime.now().isoformat()
        
        return chunks 