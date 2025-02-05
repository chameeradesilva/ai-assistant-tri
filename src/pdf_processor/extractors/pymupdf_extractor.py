"""PyMuPDF implementation for text extraction with enhanced table and OCR support."""
import os
import fitz
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pytesseract
from PIL import Image
import io
import numpy as np
import re
from src.pdf_processor.ocr.image_processor import ImagePreprocessor
from src.pdf_processor.nlp.text_cleaner import TextCleaner
from src.pdf_processor.extractors.table_extractor import TableExtractor
from langdetect import detect

logger = logging.getLogger(__name__)

class PyMuPDFExtractor:
    """Enhanced PDF text extractor using PyMuPDF with table detection and OCR."""
    
    def __init__(self, config: dict):
        """Initialize the PyMuPDF extractor with components."""
        self.config = config
        self.image_processor = ImagePreprocessor(config)
        self.table_extractor = TableExtractor(config)
        self.text_cleaner = TextCleaner(config)
        
        # Configure Tesseract
        if 'TESSERACT_PATH' in os.environ:
            pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_PATH']
            
        # Initialize metadata tracking
        self.metadata_config = config['metadata']
        self.date_formats = self.metadata_config.get('date_formats', ['%Y-%m-%d'])
        self.required_fields = set(self.metadata_config.get('required_fields', []))
        
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract and standardize document metadata."""
        try:
            metadata = {}
            
            # Extract PDF metadata
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata.update({
                    'title': pdf_metadata.get('title', ''),
                    'author': pdf_metadata.get('author', ''),
                    'subject': pdf_metadata.get('subject', ''),
                    'keywords': [k.strip() for k in pdf_metadata.get('keywords', '').split(',') if k.strip()],
                    'creator': pdf_metadata.get('creator', ''),
                    'producer': pdf_metadata.get('producer', '')
                })
                
            # Try to extract document date
            date_str = pdf_metadata.get('creationDate', '') if pdf_metadata else ''
            if date_str:
                for date_format in self.date_formats:
                    try:
                        date = datetime.strptime(date_str, date_format)
                        metadata['issued_date'] = date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
                        
            # Extract document properties with safe fallbacks
            try:
                metadata.update({
                    'page_count': len(doc),
                    'file_size': os.path.getsize(doc.name) if doc.name else 0,
                    'is_encrypted': doc.is_encrypted if hasattr(doc, 'is_encrypted') else False,
                    'pdf_version': doc.version if hasattr(doc, 'version') else None,
                    'needs_pass': doc.needs_pass if hasattr(doc, 'needs_pass') else False,
                    'has_links': doc.has_links() if hasattr(doc, 'has_links') else False,
                    'has_annots': any(page.annots() for page in doc) if hasattr(doc[0], 'annots') else False
                })
            except Exception as e:
                logger.warning(f"Error extracting document properties: {str(e)}")
                
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return {}
            
    def _extract_document_type(self, text: str) -> str:
        """Determine document type based on content patterns."""
        try:
            patterns = self.config['nlp']['content_tags']
            for category in patterns:
                if category['category'] == 'document_type':
                    for pattern in category['patterns']:
                        if re.search(pattern['pattern'], text, re.IGNORECASE):
                            return pattern['tag']
            return 'unknown'
        except Exception as e:
            logger.error(f"Document type detection failed: {str(e)}")
            return 'unknown'
            
    def _process_image(self, image: Image.Image, dpi: int = 300) -> Tuple[str, float]:
        """Process image with OCR using enhanced preprocessing."""
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.image_processor.preprocess(image)
            
            # Configure OCR settings - only use English for now
            config = f"""--psm {self.config['ocr'].get('psm', 6)} 
                        --oem {self.config['ocr'].get('oem', 3)}
                        -l eng"""  # Only use English for OCR
                        
            # Perform OCR with confidence
            ocr_data = pytesseract.image_to_data(
                processed_image,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf > -1:  # Valid confidence score
                    text_parts.append(ocr_data['text'][i])
                    confidences.append(conf)
                    
            text = ' '.join(text_parts).strip()
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            logger.info(f"OCR completed in {processing_time:.2f}s with confidence {avg_confidence:.2f}%")
            
            return text, avg_confidence / 100
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return "", 0.0

    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect document language and determine if it should be processed."""
        try:
            # Only use first 1000 characters for language detection
            sample_text = text[:1000].strip()
            if not sample_text:
                return ('en', 0.0)
                
            # Perform multiple detections for confidence
            detections = []
            for _ in range(3):
                try:
                    lang = detect(sample_text)
                    detections.append(lang)
                except Exception:
                    continue
                    
            if not detections:
                return ('en', 0.0)
                
            # Calculate confidence
            primary_lang = max(set(detections), key=detections.count)
            confidence = detections.count(primary_lang) / len(detections)
            
            logger.info(f"Detected language: {primary_lang} with confidence: {confidence:.2f}")
            return (primary_lang, confidence)
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return ('en', 0.0)

    def _extract_page_images(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Extract images from page with enhanced metadata."""
        images = []
        try:
            # First try to get the page as an image
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Process full page image
            text, confidence = self._process_image(image)
            if text or confidence > 0:
                images.append({
                    'image': image,
                    'bbox': list(page.rect),
                    'type': 'full_page',
                    'dpi': 300,
                    'size': image.size,
                    'ocr_text': text,
                    'ocr_confidence': confidence
                })
                
            # Then try to get embedded images
            try:
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = page.parent.extract_image(xref)
                        if base_image:
                            image_data = base_image["image"]
                            image = Image.open(io.BytesIO(image_data))
                            bbox = page.get_image_bbox(img)
                            
                            if bbox:
                                # Process embedded image
                                text, confidence = self._process_image(image)
                                images.append({
                                    'image': image,
                                    'bbox': list(bbox),
                                    'type': 'embedded',
                                    'xref': xref,
                                    'size': image.size,
                                    'colorspace': base_image.get('colorspace', ''),
                                    'ocr_text': text,
                                    'ocr_confidence': confidence
                                })
                                
                    except Exception as e:
                        logger.warning(f"Failed to extract embedded image {img_index}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed to get embedded images: {str(e)}")
                
            return images
            
        except Exception as e:
            logger.error(f"Image extraction failed: {str(e)}")
            return []
            
    def _process_page(self, page: fitz.Page, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single page with enhanced metadata tracking."""
        try:
            start_time = time.time()
            page_content = {
                'text': "",
                'tables': [],
                'images': [],
                'metadata': {
                    'page_number': page.number + 1,
                    'rotation': page.rotation,
                    'mediabox': list(page.mediabox),
                    'processing_start': start_time
                }
            }
            
            logger.info(f"Processing page {page.number + 1}")
            
            # Get page as image for visual processing
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            page_image = Image.open(io.BytesIO(img_data))
            
            # Try visual table detection first
            logger.info("Attempting visual table detection")
            visual_tables = self.table_extractor.extract_tables_from_image(page_image)
            if visual_tables:
                logger.info(f"Found {len(visual_tables)} tables using visual detection")
                page_content['tables'].extend(visual_tables)
            
            # If no tables found visually, try traditional methods
            if not page_content['tables'] and self.config['table'].get('enabled', True):
                table_regions = self.table_extractor.get_table_regions(page)
                logger.info(f"Found {len(table_regions)} table regions using traditional method")
                
                for region in table_regions:
                    tables = self.table_extractor.extract_tables(
                        page.parent.name,
                        page.number
                    )
                    if tables:
                        # Clean table text
                        cleaned_tables = [
                            self.text_cleaner.clean_table_text(table)
                            for table in tables
                        ]
                        page_content['tables'].extend(cleaned_tables)
                        logger.info(f"Extracted {len(cleaned_tables)} tables using traditional method")
            
            # Try normal text extraction
            text = page.get_text()
            if text.strip():
                logger.info(f"Extracted {len(text)} characters of text directly")
                page_content['text'] = self.text_cleaner.clean_text(text)
            else:
                logger.info("No direct text found, trying OCR")
                # If no text found, process images
                if self.config['ocr'].get('enabled', True):
                    images = self._extract_page_images(page)
                    logger.info(f"Found {len(images)} images/regions to process")
                    
                    for img_data in images:
                        if img_data['ocr_text']:
                            # Clean OCR text
                            cleaned_text = self.text_cleaner.clean_text(img_data['ocr_text'])
                            if img_data['type'] == 'full_page':
                                page_content['text'] = cleaned_text
                                
                                # Try table extraction again with OCR text if no tables found
                                if not page_content['tables']:
                                    logger.info("Attempting table extraction from OCR text")
                                    tables = self.table_extractor.extract_tables(
                                        page.parent.name,
                                        page.number
                                    )
                                    if tables:
                                        cleaned_tables = [
                                            self.text_cleaner.clean_table_text(table)
                                            for table in tables
                                        ]
                                        page_content['tables'].extend(cleaned_tables)
                                        logger.info(f"Extracted {len(cleaned_tables)} tables from OCR text")
                            else:
                                page_content['images'].append({
                                    'bbox': img_data['bbox'],
                                    'text': cleaned_text,
                                    'confidence': img_data['ocr_confidence'],
                                    'metadata': {
                                        'type': img_data['type'],
                                        'size': img_data['size']
                                    }
                                })
                                
            # Update page metadata
            processing_time = time.time() - start_time
            page_content['metadata'].update({
                'processing_time': processing_time,
                'text_length': len(page_content['text']),
                'table_count': len(page_content['tables']),
                'image_count': len(page_content['images'])
            })
            
            return page_content
            
        except Exception as e:
            logger.error(f"Page processing failed: {str(e)}", exc_info=True)
            return {
                'text': "",
                'tables': [],
                'images': [],
                'metadata': {'page_number': page.number + 1, 'error': str(e)}
            }
            
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text and tables from PDF with enhanced metadata."""
        try:
            start_time = time.time()
            doc = fitz.open(file_path)
            
            # Extract document metadata
            metadata = self._extract_metadata(doc)
            metadata['processing_start'] = start_time
            
            # Get initial text sample for language detection
            sample_text = ""
            for page in doc[:min(3, len(doc))]:  # Check first 3 pages
                sample_text += page.get_text()
                if len(sample_text) > 1000:
                    break
                    
            # Detect language
            lang, confidence = self._detect_language(sample_text)
            metadata['detected_language'] = lang
            metadata['language_confidence'] = confidence
            
            # Skip non-English documents
            if lang not in ['en', 'eng', 'english']:
                logger.info(f"Skipping non-English document (detected: {lang})")
                doc.close()
                return {
                    'metadata': {
                        **metadata,
                        'skipped_reason': 'non_english_document',
                        'processing_status': 'skipped'
                    },
                    'pages': []
                }
            
            # Process pages for English documents
            pages = []
            total_text = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_content = self._process_page(page, metadata)
                    pages.append(page_content)
                    total_text.append(page_content['text'])
                    
                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
                    pages.append({
                        'text': "",
                        'tables': [],
                        'images': [],
                        'metadata': {'page_number': page_num + 1, 'error': str(e)}
                    })
                    
            # Combine all text for document-level analysis
            full_text = ' '.join(total_text)
            
            # Detect document type
            doc_type = self._extract_document_type(full_text)
            
            # Update final metadata
            processing_time = time.time() - start_time
            metadata.update({
                'document_type': doc_type,
                'processing_time': processing_time,
                'total_pages': len(pages),
                'total_text_length': len(full_text),
                'total_tables': sum(len(p['tables']) for p in pages),
                'total_images': sum(len(p['images']) for p in pages),
                'processing_end': time.time(),
                'processing_status': 'completed'
            })
            
            # Validate required metadata fields
            missing_fields = self.required_fields - set(metadata.keys())
            if missing_fields:
                logger.warning(f"Missing required metadata fields: {missing_fields}")
                
            doc.close()
            return {
                'metadata': metadata,
                'pages': pages
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {str(e)}")
            return {
                'metadata': {
                    'error': str(e),
                    'processing_status': 'failed'
                },
                'pages': []
            } 