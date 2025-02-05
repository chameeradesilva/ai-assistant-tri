"""PDF Processor Factory for creating appropriate extractors based on configuration."""
import os
import yaml
import logging
from typing import Dict, Any
from src.pdf_processor.extractors.basic_extractor import BasicExtractor
from src.pdf_processor.extractors.advanced_extractor import AdvancedExtractor
from src.pdf_processor.extractors.google_vision_extractor import GoogleVisionExtractor
from .text_processor import TextProcessor
from .embedding.embedding_generator import EmbeddingGenerator
from .storage.pinecone_storage import PineconeStorage
from .nlp.text_cleaner import TextCleaner
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class PDFProcessorFactory:
    """Factory class for creating PDF processors based on configuration."""
    
    def __init__(self, config_path: str = 'config/processing.yaml'):
        """Initialize the factory with configuration."""
        self.config = self._load_config(config_path)
        self.processing_config = self.config.get('pdf_processing', {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def create_extractor(self) -> Any:
        """Create and return appropriate extractor based on configuration."""
        path_config = self.processing_config.get('processing_path', {})
        active_path = path_config.get('active_path', 'basic')
        
        logger.info(f"Creating extractor for processing path: {active_path}")
        
        try:
            # Always use basic extractor for now as it's our focus
            return BasicExtractor(self.processing_config)
                
        except Exception as e:
            logger.error(f"Error creating extractor: {str(e)}")
            raise
            
    def get_active_path_info(self) -> Dict[str, Any]:
        """Get information about the currently active processing path."""
        path_config = self.processing_config.get('processing_path', {})
        active_path = path_config.get('active_path', 'basic')
        paths = path_config.get('paths', {})
        
        return {
            'active_path': active_path,
            'path_info': paths.get(active_path, {}),
            'components': paths.get(active_path, {}).get('components', [])
        }
        
    def create_text_processor(self) -> TextProcessor:
        """Create a text processor."""
        return TextProcessor(
            chunk_size=self.config['chunking']['chunk_size'],
            overlap=self.config['chunking']['chunk_overlap']
        )
        
    def create_text_cleaner(self) -> TextCleaner:
        """Create a text cleaner for NLP preprocessing."""
        return TextCleaner(self.config)
        
    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """Create an embedding model."""
        return HuggingFaceEmbeddings(
            model_name=self.config['embedding']['model_name'],
            model_kwargs={'device': 'cpu'}
            # TODO: change to 'cuda' if using GPU
        )
        
    def create_storage(self, api_key: str, environment: str, index_name: str) -> PineconeStorage:
        """Create a Pinecone storage instance."""
        return PineconeStorage(api_key, environment, index_name) 