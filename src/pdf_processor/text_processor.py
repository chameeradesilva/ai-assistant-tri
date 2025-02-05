"""Text processing module for cleaning and chunking text."""
import re
import uuid
import logging
import psutil
import time
from typing import List, Dict, Any, Optional, Tuple
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from .ocr.ocr_processor import OCRProcessor

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, config: dict):
        """Initialize text processor with configuration."""
        chunking_config = config['chunking']
        self.chunk_size = chunking_config.get('chunk_size', 1000)
        self.overlap = chunking_config.get('chunk_overlap', 200)
        self.max_chunks = chunking_config.get('max_chunks_per_doc', 1000)
        self.min_chunk_size = chunking_config.get('min_chunk_size', 50)
        self.max_iterations = chunking_config.get('max_iterations', 10000)
        self.memory_limit_mb = chunking_config.get('memory_limit_mb', 1000)
        
        self.supported_languages = config['nlp'].get('languages', ['english'])
        self.date_formats = config['metadata'].get('date_formats', ['%Y-%m-%d'])
        
        self.ocr_processor = OCRProcessor(config['ocr'].get('tesseract_path'))
        
        logger.info(f"Initialized TextProcessor with config: {chunking_config}")

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.memory_limit_mb:
                logger.warning(f"Memory usage ({memory_mb:.2f}MB) exceeds limit ({self.memory_limit_mb}MB)")
                return False
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {str(e)}")
            return True  # Continue on error

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of input text with confidence score."""
        try:
            # Only use first 1000 characters for language detection
            sample_text = text[:1000].strip()
            if not sample_text:
                logger.warning("Empty text provided for language detection")
                return ('en', 0.0)
                
            logger.debug(f"Detecting language from sample: {sample_text[:100]}...")
            
            # Perform multiple detections for confidence
            detections = []
            for _ in range(3):
                try:
                    lang = detect(sample_text)
                    detections.append(lang)
                except LangDetectException as e:
                    logger.warning(f"Detection attempt failed: {str(e)}")
                    
            if not detections:
                logger.warning("All language detection attempts failed")
                return ('en', 0.0)
                
            # Calculate confidence
            primary_lang = max(set(detections), key=detections.count)
            confidence = detections.count(primary_lang) / len(detections)
            
            logger.info(f"Detected language: {primary_lang} with confidence: {confidence:.2f}")
            return (primary_lang, confidence)
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return ('en', 0.0)

    def preprocess_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """Clean and normalize text with metadata tracking."""
        try:
            start_time = time.time()
            logger.info("Starting text preprocessing")
            
            original_length = len(text)
            logger.debug(f"Original text length: {original_length}")
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            processed_text = text.strip()
            processed_length = len(processed_text)
            
            # Update metadata with preprocessing info
            metadata['preprocessing'] = {
                'original_length': original_length,
                'processed_length': processed_length,
                'processing_time': time.time() - start_time
            }
            
            logger.info(f"Text preprocessing complete. Reduction: {(original_length - processed_length) / original_length:.2%}")
            return processed_text
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text

    def _find_chunk_boundary(self, text: str, end_pos: int) -> int:
        """Find natural boundary for text chunk with validation."""
        try:
            # Ensure end_pos is within text bounds
            end_pos = min(end_pos, len(text))
            if end_pos <= 0:
                return 0
                
            search_window = min(self.chunk_size // 2, end_pos)
            text_window = text[max(0, end_pos - search_window):end_pos]
            
            # Priority for boundaries: paragraph > sentence > word
            boundaries = [
                ('\n\n', 0),  # Paragraph break
                ('।', 0),     # Sinhala/Tamil sentence end
                ('.', 1),     # English sentence end
                (' ', 1)      # Word break
            ]
            
            for separator, offset in boundaries:
                if separator in text_window:
                    relative_pos = text_window.rindex(separator)
                    return end_pos - search_window + relative_pos + offset
                    
            return end_pos
            
        except Exception as e:
            logger.error(f"Error finding chunk boundary: {str(e)}")
            return end_pos

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with enhanced safety mechanisms."""
        chunks = []
        try:
            start_time = time.time()
            logger.info("Starting text chunking process")
            
            if not text or not text.strip():
                logger.warning("Empty text provided for chunking")
                return chunks
                
            # Preprocess text
            text = self.preprocess_text(text, metadata)
            if not text.strip():
                logger.warning("Text is empty after preprocessing")
                return chunks
                
            # Detect language
            language, confidence = self.detect_language(text)
            total_length = len(text)
            
            # Initialize chunking metadata
            chunk_metadata = {
                'language': language,
                'language_confidence': confidence,
                'total_length': total_length,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
                'start_time': start_time
            }
            
            # Update document metadata
            metadata.update(chunk_metadata)
            
            start = 0
            chunk_count = 0
            iteration_count = 0
            
            while start < total_length:
                # Safety checks
                iteration_count += 1
                if iteration_count > self.max_iterations:
                    logger.error("Maximum iteration count exceeded")
                    break
                    
                if chunk_count >= self.max_chunks:
                    logger.error("Maximum chunk count exceeded")
                    break
                    
                if not self._check_memory_usage():
                    logger.error("Memory usage limit exceeded")
                    break
                    
                # Find chunk boundary
                end = min(start + self.chunk_size, total_length)
                end = self._find_chunk_boundary(text, end)
                
                chunk_text = text[start:end].strip()
                
                # Validate chunk
                if not chunk_text or len(chunk_text) < self.min_chunk_size:
                    logger.warning(f"Skipping small chunk at position {start}-{end}")
                    start = end
                    continue
                    
                chunk_id = str(uuid.uuid4())
                
                # Create chunk-specific metadata
                chunk_metadata = {
                    **metadata,
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': end,
                    'chunk_length': len(chunk_text),
                    'is_first_chunk': start == 0,
                    'is_last_chunk': end >= total_length,
                    'iteration': iteration_count,
                    'processing_time': time.time() - start_time
                }
                
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                chunk_count += 1
                logger.info(f"Created chunk {chunk_count}: {start}-{end} ({len(chunk_text)} chars)")
                
                # Update start position with overlap
                start = end - self.overlap
                if start >= total_length:
                    break
                    
            # Final metadata update
            metadata['chunking_results'] = {
                'total_chunks': chunk_count,
                'total_iterations': iteration_count,
                'processing_time': time.time() - start_time,
                'average_chunk_size': sum(len(c['text']) for c in chunks) / len(chunks) if chunks else 0
            }
            
            logger.info(f"Chunking complete. Created {len(chunks)} chunks in {iteration_count} iterations")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in text chunking: {str(e)}", exc_info=True)
            return [] 