"""PyMuPDF implementation for text extraction."""
import fitz
from typing import Optional
import pytesseract
from PIL import Image
import io
from .base_extractor import BaseExtractor

class PyMuPDFExtractor(BaseExtractor):
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize the PyMuPDF extractor."""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF with OCR fallback."""
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Try normal text extraction first
                page_text = page.get_text()
                
                # If no text found, try OCR
                if not page_text.strip():
                    print(f"Using OCR for page {page_num + 1}")
                    page_text = self._ocr_page(page)
                
                text += page_text + "\n"
            
            doc.close()
            return text
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return ""
            
    def _ocr_page(self, page: fitz.Page) -> str:
        """Perform OCR on a page using Tesseract."""
        try:
            # Get page as image
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            return text
            
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return "" 