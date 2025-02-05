"""Image preprocessing module for OCR optimization."""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for OCR optimization."""
    
    def __init__(self, config: dict):
        """Initialize image preprocessor with configuration."""
        self.config = config['image']
        
    def _convert_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    def _convert_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to the image."""
        try:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        except Exception as e:
            logger.warning(f"Denoising failed: {str(e)}")
            return image
            
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew."""
        try:
            # Convert to grayscale and get edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            if lines is None:
                return image
                
            # Calculate skew angle
            angles = []
            for rho, theta in lines[0]:
                angle = theta * 180 / np.pi
                if angle < 45 or angle > 135:
                    angles.append(angle)
                    
            if not angles:
                return image
                
            skew_angle = 90 - np.mean(angles)
            if abs(skew_angle) < 1:  # Skip minor corrections
                return image
                
            # Rotate image
            height, width = image.shape[:2]
            center = (width//2, height//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                                
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image
            
    def _remove_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove horizontal and vertical lines."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold the image
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine masks and remove lines
            mask = cv2.add(remove_horizontal, remove_vertical)
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            
            return result
            
        except Exception as e:
            logger.warning(f"Line removal failed: {str(e)}")
            return image
            
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            block_size = self.config.get('block_size', 11)
            c_value = self.config.get('c_value', 2)
            
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c_value
            )
            
        except Exception as e:
            logger.warning(f"Adaptive thresholding failed: {str(e)}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Apply all preprocessing steps to the image."""
        if not self.config.get('enable_preprocessing', True):
            return image
            
        try:
            # Convert to OpenCV format
            cv2_image = self._convert_to_cv2(image)
            
            # Apply preprocessing steps based on configuration
            if self.config.get('denoise', True):
                cv2_image = self._denoise(cv2_image)
                
            if self.config.get('deskew', True):
                cv2_image = self._deskew(cv2_image)
                
            if self.config.get('remove_lines', True):
                cv2_image = self._remove_lines(cv2_image)
                
            if self.config.get('adaptive_threshold', True):
                cv2_image = self._adaptive_threshold(cv2_image)
                
            # Convert back to PIL Image
            return self._convert_to_pil(cv2_image)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return image  # Return original image on error 