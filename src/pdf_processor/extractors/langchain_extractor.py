"""LangChain-based PDF extractor with hybrid processing capabilities."""
import os
from typing import List, Dict, Any, Optional
import logging
import shutil
from pathlib import Path
import fitz
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LangChainExtractor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LangChain extractor with configuration."""
        self.config = config
        self.temp_dir = Path(config['ocr']['temp_dir'])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def _is_image_based_pdf(self, file_path: str) -> bool:
        """Determine if PDF is primarily image-based."""
        try:
            doc = fitz.open(file_path)
            text_content = 0
            total_pages = len(doc)
            
            for page in doc:
                if page.get_text().strip():
                    text_content += 1
                    
            doc.close()
            text_ratio = text_content / total_pages
            return text_ratio < self.config['loader']['image_based_threshold']
            
        except Exception as e:
            logger.error(f"Error checking PDF type: {str(e)}")
            return True  # Default to image-based processing on error
            
    def _process_with_langchain(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process PDF using LangChain's UnstructuredPDFLoader."""
        try:
            loader = UnstructuredPDFLoader(
                file_path,
                mode="elements",
                strategy=self.config['ocr']['strategy'],
                languages=[self.config['ocr']['language']],
            )
            
            documents = loader.load()
            
            # Configure text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunking']['chunk_size'],
                chunk_overlap=self.config['chunking']['chunk_overlap'],
                length_function=len,
                separators=self.config['chunking']['separators']
            )
            
            # Split documents
            splits = text_splitter.split_documents(documents)
            
            # Process splits into our format
            results = []
            for i, split in enumerate(splits):
                chunk_metadata = {
                    **metadata,
                    'chunk_id': f"{metadata.get('id', 'doc')}_{i}",
                    'page_numbers': split.metadata.get('page_numbers', []),
                    'source': file_path,
                    'chunk_index': i,
                    'total_chunks': len(splits)
                }
                
                results.append({
                    'text': split.page_content,
                    'metadata': chunk_metadata
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in LangChain processing: {str(e)}")
            return []
            
    def _process_with_pymupdf(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process PDF using PyMuPDF for text-based documents."""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                text += doc[page_num].get_text() + "\n\n"
                
            doc.close()
            
            # Use LangChain's text splitter for consistency
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunking']['chunk_size'],
                chunk_overlap=self.config['chunking']['chunk_overlap'],
                length_function=len,
                separators=self.config['chunking']['separators']
            )
            
            splits = text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            
            results = []
            for i, split in enumerate(splits):
                chunk_metadata = {
                    **metadata,
                    'chunk_id': f"{metadata.get('id', 'doc')}_{i}",
                    'source': file_path,
                    'chunk_index': i,
                    'total_chunks': len(splits)
                }
                
                results.append({
                    'text': split.page_content,
                    'metadata': chunk_metadata
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in PyMuPDF processing: {str(e)}")
            return []
            
    def extract_text(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text from PDF using appropriate method based on content."""
        try:
            mode = self.config['loader']['mode']
            
            if mode == 'langchain_only':
                return self._process_with_langchain(file_path, metadata)
            elif mode == 'pymupdf_only':
                return self._process_with_pymupdf(file_path, metadata)
            else:  # hybrid mode
                is_image_based = self._is_image_based_pdf(file_path)
                logger.info(f"PDF type detected: {'image-based' if is_image_based else 'text-based'}")
                
                if is_image_based:
                    return self._process_with_langchain(file_path, metadata)
                else:
                    return self._process_with_pymupdf(file_path, metadata)
                    
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            return []
            
        finally:
            # Cleanup temporary files if configured
            if self.config['ocr']['cleanup_temp']:
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info("Temporary OCR files cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary files: {str(e)}") 