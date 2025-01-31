"""Text processing module for cleaning and chunking text."""
import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from langdetect import detect
from .ocr.ocr_processor import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50, tesseract_path: Optional[str] = None):
        """Initialize text processor with chunk size and overlap."""
        self.chunk_size = max(100, min(chunk_size, 1000))  # Ensure reasonable chunk size
        self.overlap = min(overlap, self.chunk_size // 4)  # Ensure overlap is not too large
        self.ocr_processor = OCRProcessor(tesseract_path)
        logger.info(f"Initialized TextProcessor with chunk_size={self.chunk_size}, overlap={self.overlap}")

    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            # Only use first 1000 characters for language detection
            sample_text = text[:1000]
            return detect(sample_text)
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}, defaulting to 'en'")
            return 'en'

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        try:
            logger.info("Starting text preprocessing")
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            processed_text = text.strip()
            logger.info(f"Text preprocessing complete. Original length: {len(text)}, Processed length: {len(processed_text)}")
            return processed_text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text

    def _find_chunk_boundary(self, text: str, end_pos: int) -> int:
        """Find natural boundary for text chunk based on layout and content."""
        try:
            # Ensure end_pos is within text bounds
            end_pos = min(end_pos, len(text))
            search_window = min(self.chunk_size // 2, end_pos)
            
            # Priority for boundaries: paragraph > sentence > word
            text_window = text[max(0, end_pos - search_window):end_pos]
            
            # Look for paragraph break
            if '\n\n' in text_window:
                relative_pos = text_window.rindex('\n\n')
                return end_pos - search_window + relative_pos
                
            # Look for sentence break
            if '.' in text_window:
                relative_pos = text_window.rindex('.')
                return end_pos - search_window + relative_pos + 1
                
            # Look for word break
            if ' ' in text_window:
                relative_pos = text_window.rindex(' ')
                return end_pos - search_window + relative_pos + 1
                
            return end_pos
            
        except Exception as e:
            logger.error(f"Error finding chunk boundary: {str(e)}")
            return end_pos

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        try:
            logger.info("Starting text chunking process")
            
            if not text:
                logger.warning("Empty text provided for chunking")
                return chunks

            text = self.preprocess_text(text)
            language = self.detect_language(text)
            total_length = len(text)
            
            # Update metadata
            metadata.update({
                'language': language,
                'total_length': total_length,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap
            })
            
            # Extract circular_no metadata if present
            if 'circular_no' in metadata:
                try:
                    metadata['doc_id'] = metadata['circular_no'].split()[0]
                except Exception as e:
                    logger.warning(f"Failed to extract doc_id from circular_no: {str(e)}")
            
            start = 0
            chunk_count = 0
            last_end = 0  # Track the last ending position
            
            while start < total_length:
                # Ensure we're making progress
                if start <= last_end and chunk_count > 0:
                    logger.error(f"Chunking stuck at position {start}, breaking to avoid infinite loop")
                    break
                    
                end = min(start + self.chunk_size, total_length)
                
                if end < total_length:
                    end = self._find_chunk_boundary(text, end)
                
                # Ensure we're making progress
                if end <= start:
                    end = min(start + self.chunk_size, total_length)
                
                chunk_text = text[start:end]
                chunk_id = str(uuid.uuid4())
                
                # Create chunk-specific metadata
                chunk_metadata = {
                    **metadata,
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': end,
                    'chunk_length': len(chunk_text),
                    'is_first_chunk': start == 0,
                    'is_last_chunk': end >= total_length
                }
                
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                chunk_count += 1
                logger.info(f"Created chunk {chunk_count}: {start}-{end} ({len(chunk_text)} chars)")
                
                last_end = end
                start = end - self.overlap
                
                # Safety check to prevent infinite loops
                if chunk_count > total_length // 10:  # Assuming average chunk size of 10 chars (very conservative)
                    logger.error("Too many chunks created, possible infinite loop. Breaking.")
                    break

            logger.info(f"Chunking complete. Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in text chunking: {str(e)}")
            # Return a single chunk with the original text if chunking fails
            return [{
                'id': str(uuid.uuid4()),
                'text': text,
                'metadata': {**metadata, 'chunking_error': str(e)}
            }] 