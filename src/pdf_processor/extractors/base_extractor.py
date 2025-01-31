"""Base class for PDF text extraction."""
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        pass 