"""Factory module for creating PDF processing components."""
from typing import Optional
from .extractors.pymupdf_extractor import PyMuPDFExtractor
from .text_processor import TextProcessor
from .embedding.embedding_generator import EmbeddingGenerator
from .storage.pinecone_storage import PineconeStorage

class PDFProcessorFactory:
    @staticmethod
    def create_extractor(tesseract_path: Optional[str] = None) -> PyMuPDFExtractor:
        """Create a PDF text extractor."""
        return PyMuPDFExtractor(tesseract_path)

    @staticmethod
    def create_text_processor(chunk_size: int = 1000, overlap: int = 200) -> TextProcessor:
        """Create a text processor."""
        return TextProcessor(chunk_size, overlap)

    @staticmethod
    def create_embedding_generator(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingGenerator:
        """Create an embedding generator (singleton)."""
        return EmbeddingGenerator(model_name)

    @staticmethod
    def create_storage(api_key: str, environment: str, index_name: str) -> PineconeStorage:
        """Create a Pinecone storage instance (singleton)."""
        return PineconeStorage(api_key, environment, index_name) 