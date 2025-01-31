"""OCR processor module with language detection and multi-language support."""
from typing import Dict, Any, Optional, Tuple
import pytesseract
from langdetect import detect
import cv2
import numpy as np
from PIL import Image

class OCRProcessor:
    """Handles OCR processing with language detection and confidence scoring."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize OCR processor with optional Tesseract path."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Language mapping for Tesseract
        self.language_map = {
            'si': 'sin',  # Sinhala
            'ta': 'tam',  # Tamil
            'en': 'eng',  # English
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails
    
    def process_image(self, image: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Process image with OCR and return text with metadata."""
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # First try English OCR
        text = pytesseract.image_to_string(image, lang='eng')
        language = self.detect_language(text)
        
        # If not English and we have support for detected language, reprocess
        if language != 'en' and language in self.language_map:
            text = pytesseract.image_to_string(
                image, 
                lang=self.language_map[language]
            )
            
        # Get confidence scores
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidence_scores = [float(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        metadata = {
            'language': language,
            'ocr_confidence': avg_confidence,
            'ocr_engine': 'tesseract',
            'word_count': len(data['text']),
        }
        
        return text, metadata 