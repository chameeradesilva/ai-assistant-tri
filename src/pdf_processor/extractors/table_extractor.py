"""Table extraction module using Camelot, Tabula, and OpenCV for visual table detection."""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import camelot
import tabula
import pandas as pd
import fitz
from PIL import Image
import io
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class TableExtractor:
    """Handles table extraction from PDFs using multiple engines and visual detection."""
    
    def __init__(self, config: dict):
        """Initialize table extractor with configuration."""
        self.config = config['table']
        self.engine = self.config.get('engine', 'camelot')
        
    def _detect_table_structure(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect table structure using OpenCV."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )

            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

            # Combine lines
            table_mask = cv2.add(horizontal_lines, vertical_lines)

            # Find contours of table cells
            contours, _ = cv2.findContours(
                table_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter and sort contours
            min_area = 100  # Minimum area to be considered a table cell
            table_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if region has grid-like structure
                    roi = table_mask[y:y+h, x:x+w]
                    if self._verify_table_structure(roi):
                        table_regions.append({
                            'bbox': (x, y, x+w, y+h),
                            'area': area,
                            'aspect_ratio': w/h if h > 0 else 0
                        })

            return table_regions

        except Exception as e:
            logger.error(f"Visual table detection failed: {str(e)}")
            return []

    def _verify_table_structure(self, roi: np.ndarray) -> bool:
        """Verify if a region contains a valid table structure."""
        try:
            # Count line intersections
            horizontal_lines = cv2.reduce(roi, 0, cv2.REDUCE_MAX)
            vertical_lines = cv2.reduce(roi, 1, cv2.REDUCE_MAX)
            
            h_count = np.count_nonzero(horizontal_lines)
            v_count = np.count_nonzero(vertical_lines)
            
            # Require minimum number of lines in both directions
            min_lines = 2
            if h_count >= min_lines and v_count >= min_lines:
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Table structure verification failed: {str(e)}")
            return False

    def _extract_with_camelot(self, pdf_path: str, page_num: int) -> List[pd.DataFrame]:
        """Extract tables using Camelot."""
        try:
            # Configure Camelot options based on flavor
            flavor = self.config.get('flavor', 'lattice')  # Changed default to lattice
            
            logger.info(f"Attempting Camelot extraction on page {page_num} with flavor={flavor}")
            
            # Configure options based on flavor
            if flavor == 'lattice':
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor=flavor,
                    line_scale=40,
                    edge_tol=500
                )
            else:  # stream flavor
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor=flavor,
                    edge_tol=500,
                    row_tol=10
                )
            
            # Lower accuracy threshold and log findings
            extracted_tables = []
            for table in tables:
                logger.info(f"Found table with accuracy: {table.accuracy}%")
                if table.accuracy > 50:  # Lower threshold from 80% to 50%
                    extracted_tables.append(table.df)
                
            logger.info(f"Extracted {len(extracted_tables)} tables from page {page_num}")
            return extracted_tables
            
        except Exception as e:
            logger.warning(f"Camelot extraction failed for page {page_num}: {str(e)}")
            return []
            
    def _extract_with_tabula(self, pdf_path: str, page_num: int) -> List[pd.DataFrame]:
        """Extract tables using Tabula."""
        try:
            # Read tables
            tables = tabula.read_pdf(
                pdf_path,
                pages=page_num + 1,  # Tabula uses 1-based page numbers
                multiple_tables=True,
                guess=True,
                lattice=True
            )
            
            return [table for table in tables if not table.empty]
            
        except Exception as e:
            logger.warning(f"Tabula extraction failed for page {page_num}: {str(e)}")
            return []
            
    def _is_table_region(self, page: fitz.Page, bbox: fitz.Rect) -> bool:
        """Determine if a region likely contains a table."""
        try:
            # Get text blocks in the region
            blocks = page.get_text("blocks", clip=bbox)
            if not blocks:
                return False
                
            # Check for grid-like structure - more lenient
            lines = page.get_drawings(clip=bbox)
            if len(lines) > 3:  # Reduced from 5 to 3 lines
                logger.info(f"Found table region with {len(lines)} lines")
                return True
                
            # Check text alignment and spacing
            x_coords = []
            for block in blocks:
                x_coords.append(block[0])  # x0 coordinate
                x_coords.append(block[2])  # x1 coordinate
                
            # Look for consistent column alignment - more lenient
            x_coords.sort()
            if len(x_coords) < 4:  # Too few coordinates to be a table
                return False
                
            diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            if len(set(round(d, 1) for d in diffs)) < len(diffs) * 0.5:  # Reduced from 0.7 to 0.5
                logger.info("Found table region based on column alignment")
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Table region detection failed: {str(e)}")
            return False
            
    def extract_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from a PDF page."""
        tables = []
        try:
            # Try primary engine first
            if self.engine == 'camelot':
                dataframes = self._extract_with_camelot(pdf_path, page_num)
                if not dataframes:  # Fallback to Tabula
                    dataframes = self._extract_with_tabula(pdf_path, page_num)
            else:
                dataframes = self._extract_with_tabula(pdf_path, page_num)
                if not dataframes:  # Fallback to Camelot
                    dataframes = self._extract_with_camelot(pdf_path, page_num)
                    
            # Process extracted tables
            for idx, df in enumerate(dataframes):
                table_data = {
                    'page': page_num,
                    'table_index': idx,
                    'data': df.to_dict('records'),
                    'headers': df.columns.tolist(),
                    'num_rows': len(df),
                    'num_cols': len(df.columns)
                }
                tables.append(table_data)
                
            return tables
            
        except Exception as e:
            logger.error(f"Table extraction failed for page {page_num}: {str(e)}")
            return []
            
    def get_table_regions(self, page: fitz.Page) -> List[fitz.Rect]:
        """Identify regions that likely contain tables."""
        try:
            regions = []
            # Get text blocks
            blocks = page.get_text("blocks")
            
            # Group adjacent blocks that might form tables
            current_region = None
            for block in blocks:
                bbox = fitz.Rect(block[:4])
                
                if self._is_table_region(page, bbox):
                    if current_region is None:
                        current_region = bbox
                    else:
                        # Merge with previous region if close
                        if bbox.y0 - current_region.y1 < 20:  # Arbitrary threshold
                            current_region = current_region | bbox
                        else:
                            regions.append(current_region)
                            current_region = bbox
                elif current_region is not None:
                    regions.append(current_region)
                    current_region = None
                    
            if current_region is not None:
                regions.append(current_region)
                
            return regions
            
        except Exception as e:
            logger.error(f"Table region detection failed: {str(e)}")
            return []

    def _parse_table_text(self, text: str) -> pd.DataFrame:
        """Parse OCR text into a table structure."""
        try:
            # Split into lines and clean
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if not lines:
                return pd.DataFrame()
                
            # Try to detect table structure
            data = []
            max_cols = 0
            
            # First pass: split lines and find max columns
            split_lines = []
            for line in lines:
                # Split on multiple spaces and remove quotes
                parts = [p.strip().strip('"\'') for p in line.split('  ') if p.strip()]
                if parts:  # Only add non-empty lines
                    split_lines.append(parts)
                    max_cols = max(max_cols, len(parts))
                    
            # Second pass: normalize data
            for parts in split_lines:
                # Pad with empty strings if needed
                row = parts + [''] * (max_cols - len(parts))
                data.append(row)
                
            # Create DataFrame
            if data and max_cols >= 2:  # Require at least 2 columns
                # Use first row as headers if it looks like headers
                if len(data) > 1:
                    headers = [str(h).strip() for h in data[0]]
                    df = pd.DataFrame(data[1:], columns=headers)
                else:
                    # Generate column names if no headers
                    df = pd.DataFrame(data, columns=[f'Column_{i+1}' for i in range(max_cols)])
                    
                # Clean up the DataFrame
                df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
                df = df.replace('', pd.NA).dropna(how='all')  # Remove empty rows
                
                return df
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to parse table text: {str(e)}")
            return pd.DataFrame()
            
    def extract_tables_from_image(self, image: Image.Image) -> List[pd.DataFrame]:
        """Extract tables from an image using visual detection."""
        try:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect tables visually
            table_regions = self._detect_table_structure(cv_image)
            logger.info(f"Detected {len(table_regions)} potential table regions")
            
            tables = []
            for idx, region in enumerate(table_regions):
                try:
                    # Extract region from original image
                    x1, y1, x2, y2 = region['bbox']
                    
                    # Add padding around region
                    padding = 10
                    y1 = max(0, y1 - padding)
                    y2 = min(cv_image.shape[0], y2 + padding)
                    x1 = max(0, x1 - padding)
                    x2 = min(cv_image.shape[1], x2 + padding)
                    
                    table_img = cv_image[y1:y2, x1:x2]
                    
                    # Preprocess image for better OCR
                    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    
                    # Additional preprocessing
                    kernel = np.ones((1, 1), np.uint8)
                    thresh = cv2.dilate(thresh, kernel, iterations=1)
                    thresh = cv2.erode(thresh, kernel, iterations=1)
                    
                    # Convert to PIL for OCR
                    table_pil = Image.fromarray(thresh)
                    
                    # Try OCR with table-optimized settings
                    try:
                        import pytesseract
                        # Use tesseract with table configuration
                        config = """--psm 6 
                                  --oem 3 
                                  -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-() "
                                  -c preserve_interword_spaces=1
                                  -c tessedit_do_invert=0"""
                        text = pytesseract.image_to_string(table_pil, config=config)
                        
                        # Parse text into table structure
                        df = self._parse_table_text(text)
                        if not df.empty and len(df.columns) >= 2:  # Require at least 2 columns
                            tables.append(df)
                            logger.info(f"Successfully extracted table {idx} using OCR with {df.shape[0]} rows and {df.shape[1]} columns")
                        
                    except Exception as ocr_error:
                        logger.warning(f"OCR extraction failed for table {idx}: {str(ocr_error)}")
                            
                except Exception as e:
                    logger.warning(f"Failed to process table region {idx}: {str(e)}")
                    continue
            
            return tables
            
        except Exception as e:
            logger.error(f"Image-based table extraction failed: {str(e)}")
            return [] 