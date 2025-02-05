"""NLP text preprocessing module with enhanced multilingual support."""
import re
import logging
from typing import List, Dict, Any, Set, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Download required NLTK resources
REQUIRED_NLTK_RESOURCES = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'punkt_tab'
]

def download_nltk_resources():
    """Download required NLTK resources."""
    for resource in REQUIRED_NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

# Download resources
download_nltk_resources()

logger = logging.getLogger(__name__)

class TextCleaner:
    """Handles NLP preprocessing of extracted text with multilingual support."""
    
    def __init__(self, config: dict):
        """Initialize text cleaner with configuration."""
        self.config = config['nlp']
        self.languages = self.config.get('languages', ['english'])
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize stopwords for all supported languages
        self.stop_words = {}
        for lang in self.languages:
            try:
                self.stop_words[lang] = set(stopwords.words(lang))
            except Exception as e:
                logger.warning(f"Failed to load stopwords for {lang}: {str(e)}")
                self.stop_words[lang] = set()
                
        self.custom_patterns = self.config.get('custom_patterns', [])
        self.content_tags = self.config.get('content_tags', [])
        
    def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            return detect(text[:1000])
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return 'english'
            
    def _remove_special_chars(self, text: str, language: str) -> str:
        """Remove special characters while preserving language-specific characters."""
        if language in ['sinhala', 'tamil']:
            # Preserve Sinhala/Tamil unicode ranges
            text = re.sub(r'[^\u0D80-\u0DFF\u0B80-\u0BFF\s.,!?-]', ' ', text)
        else:
            # Standard Latin cleanup
            text = re.sub(r'[^\w\s.,!?-]', ' ', text)
            
        return ' '.join(text.split())
        
    def _apply_custom_patterns(self, text: str) -> str:
        """Apply custom regex patterns for text cleaning."""
        for pattern in self.custom_patterns:
            text = re.sub(pattern['pattern'], pattern.get('replace', ''), text)
        return text
        
    def _lemmatize_text(self, tokens: List[str], language: str) -> List[str]:
        """Lemmatize tokens based on language."""
        if language == 'english':
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens  # Skip lemmatization for non-English text
        
    def _remove_stopwords(self, tokens: List[str], language: str) -> List[str]:
        """Remove stop words based on language."""
        stop_words = self.stop_words.get(language, set())
        return [token for token in tokens if token.lower() not in stop_words]
        
    def _extract_content_tags(self, text: str) -> Dict[str, List[str]]:
        """Extract content tags based on configured patterns."""
        tags = {}
        try:
            for category in self.content_tags:
                category_name = category['category']
                category_tags = set()
                
                for pattern in category['patterns']:
                    matches = re.finditer(pattern['pattern'], text, re.IGNORECASE)
                    for match in matches:
                        category_tags.add(pattern['tag'])
                        
                if category_tags:
                    tags[category_name] = list(category_tags)
                    
        except Exception as e:
            logger.error(f"Content tagging failed: {str(e)}")
            
        return tags
        
    def clean_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Apply all text cleaning steps with metadata tracking."""
        if not self.config.get('enable_preprocessing', True):
            return text, {}
            
        try:
            metadata = {
                'original_length': len(text),
                'processing_steps': []
            }
            
            # Detect language
            language = self._detect_language(text)
            metadata['detected_language'] = language
            
            # Apply custom patterns first
            text = self._apply_custom_patterns(text)
            metadata['processing_steps'].append('custom_patterns')
            
            # Convert to lowercase if configured
            if self.config.get('lowercase', True):
                text = text.lower()
                metadata['processing_steps'].append('lowercase')
                
            # Remove special characters
            if self.config.get('remove_special_chars', True):
                text = self._remove_special_chars(text, language)
                metadata['processing_steps'].append('special_chars')
                
            # Tokenize
            tokens = word_tokenize(text)
            metadata['token_count'] = len(tokens)
            
            # Remove stopwords if configured
            if self.config.get('remove_stopwords', True):
                tokens = self._remove_stopwords(tokens, language)
                metadata['processing_steps'].append('stopwords')
                
            # Lemmatize if configured
            if self.config.get('lemmatize', True):
                tokens = self._lemmatize_text(tokens, language)
                metadata['processing_steps'].append('lemmatization')
                
            # Join tokens back into text
            cleaned_text = ' '.join(tokens)
            
            # Extract content tags
            content_tags = self._extract_content_tags(cleaned_text)
            if content_tags:
                metadata['content_tags'] = content_tags
                
            metadata['final_length'] = len(cleaned_text)
            metadata['reduction_ratio'] = 1 - (len(cleaned_text) / metadata['original_length'])
            
            return cleaned_text, metadata
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return text, {'error': str(e)}
            
    def clean_table_text(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean text in table cells while preserving structure."""
        try:
            cleaned_data = []
            metadata = {
                'rows_processed': 0,
                'cells_processed': 0,
                'cells_cleaned': 0
            }
            
            for row in table_data['data']:
                cleaned_row = {}
                for key, value in row.items():
                    metadata['cells_processed'] += 1
                    if isinstance(value, str):
                        cleaned_text, cell_metadata = self.clean_text(value)
                        cleaned_row[key] = cleaned_text
                        metadata['cells_cleaned'] += 1
                    else:
                        cleaned_row[key] = value
                cleaned_data.append(cleaned_row)
                metadata['rows_processed'] += 1
                
            table_data['data'] = cleaned_data
            table_data['cleaning_metadata'] = metadata
            return table_data
            
        except Exception as e:
            logger.error(f"Table text cleaning failed: {str(e)}")
            return table_data 