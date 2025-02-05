"""Basic PDF extractor using PyMuPDF and Tesseract OCR."""
import os
import fitz
import pytesseract
from PIL import Image
import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
import tempfile
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BasicExtractor:
    """Basic PDF extractor using PyMuPDF and Tesseract OCR."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the basic extractor with configuration."""
        self.config = config
        self.ocr_config = config.get('ocr', {})
        self.table_config = config.get('table', {})
        self.image_config = config.get('image', {})
        
        # Create temp directory if it doesn't exist
        self.temp_dir = self.ocr_config.get('temp_dir', 'temp/ocr')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Configure Tesseract
        self._configure_tesseract()
        
    def _configure_tesseract(self):
        """Configure Tesseract OCR settings."""
        # Get Tesseract path from environment variable
        tesseract_path = os.getenv('TESSERACT_PATH')
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Using Tesseract from environment variable: {tesseract_path}")
        else:
            logger.warning("TESSERACT_PATH environment variable not set or path not found. Make sure Tesseract is installed and TESSERACT_PATH is set correctly in .env file.")
            
        # Configure OCR settings
        self.tesseract_config = f"""--psm {self.ocr_config.get('psm', 3)} --oem {self.ocr_config.get('oem', 3)}"""
        
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and tables from PDF using PyMuPDF and Tesseract."""
        start_time = datetime.now()
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Process each page
            pages = []
            for page_num in range(total_pages):
                page_dict = self._process_page(doc[page_num], page_num + 1)
                pages.append(page_dict)
                
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'metadata': {
                    'filename': os.path.basename(pdf_path),
                    'total_pages': total_pages,
                    'processing_time': processing_time,
                    'extraction_method': 'basic',
                    'components_used': ['pymupdf', 'tesseract']
                },
                'pages': pages
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {str(e)}")
            raise
            
    def _process_page(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Process a single PDF page."""
        try:
            # Extract text using PyMuPDF
            text = page.get_text()
            
            # Get page dimensions
            page_rect = page.rect
            width, height = page_rect.width, page_rect.height
            
            # Extract tables only if enabled in config
            tables = []
            if self.table_config.get('enabled', True):
                tables = self._extract_tables_from_text(text)
            
            return {
                'page_number': page_num,
                'width': width,
                'height': height,
                'text': text,
                'tables': tables
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            return {
                'page_number': page_num,
                'error': str(e)
            }
            
    def _extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """Extract tables from text content."""
        try:
            # Split text into lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            if not lines:
                return []
                
            # Find potential table sections
            tables = []
            current_table = []
            in_table = False
            min_columns = 2
            
            for line in lines:
                # Skip very long lines as they're likely paragraphs
                if len(line) > 100:
                    if current_table:
                        df = self._create_dataframe(current_table)
                        if not df.empty:
                            tables.append(df)
                        current_table = []
                    in_table = False
                    continue
                    
                # Split line into parts
                parts = [p.strip() for p in line.split('  ') if p.strip()]
                
                # If line has multiple columns, it might be part of a table
                if len(parts) >= min_columns:
                    current_table.append(parts)
                    in_table = True
                elif in_table:
                    # If we were in a table but this line doesn't match,
                    # try to create a table from what we have
                    if current_table:
                        df = self._create_dataframe(current_table)
                        if not df.empty:
                            tables.append(df)
                        current_table = []
                    in_table = False
                    
            # Process any remaining table data
            if current_table:
                df = self._create_dataframe(current_table)
                if not df.empty:
                    tables.append(df)
                    
            return tables
            
        except Exception as e:
            logger.warning(f"Failed to extract tables from text: {str(e)}")
            return []
            
    def _create_dataframe(self, rows: List[List[str]]) -> pd.DataFrame:
        """Create a DataFrame from rows of data."""
        try:
            if not rows:
                return pd.DataFrame()
                
            # Find the maximum number of columns
            max_cols = max(len(row) for row in rows)
            
            # Pad rows to have equal length
            padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            
            # Create DataFrame
            if len(padded_rows) > 1:
                # Use first row as header if it doesn't contain numbers
                first_row = padded_rows[0]
                if not any(part.replace('.', '').isdigit() for part in first_row):
                    df = pd.DataFrame(padded_rows[1:], columns=first_row)
                else:
                    df = pd.DataFrame(padded_rows, columns=[f'Column_{i+1}' for i in range(max_cols)])
            else:
                df = pd.DataFrame(padded_rows, columns=[f'Column_{i+1}' for i in range(max_cols)])
                
            # Clean up the DataFrame
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df = df.replace('', pd.NA).dropna(how='all')
            
            # Only return if we have at least 2 columns and some data
            if len(df.columns) >= 2 and not df.empty:
                logger.info(f"Created table with {len(df)} rows and {len(df.columns)} columns")
                return df
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to create DataFrame: {str(e)}")
            return pd.DataFrame() 