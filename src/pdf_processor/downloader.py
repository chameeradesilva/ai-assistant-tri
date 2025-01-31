"""Module for downloading and managing PDFs from URLs."""
import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PDFDownloader:
    """Handles downloading PDFs and managing metadata from JSON sources."""
    
    def __init__(self, download_dir: str = "downloads"):
        """Initialize the downloader with download directory."""
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
    def load_metadata_from_json(self, json_path: str) -> List[Dict]:
        """Load metadata from a JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {json_path}: {str(e)}")
            return []
            
    def download_pdf(self, url: str, metadata: Dict) -> Optional[str]:
        """
        Download a PDF file and return its local path.
        
        Args:
            url: URL of the PDF
            metadata: Associated metadata dictionary
            
        Returns:
            Optional[str]: Local path to downloaded PDF or None if download failed
        """
        try:
            # Create filename from circular number or title
            if metadata.get('circular_no'):
                filename = f"{metadata['circular_no'].replace(' ', '_')}.pdf"
            else:
                # Use sanitized title if no circular number
                title = metadata['title'].replace(' ', '_')
                filename = f"{title[:50]}.pdf"
            
            local_path = self.download_dir / filename
            
            # Don't redownload if exists
            if local_path.exists():
                logger.info(f"PDF already exists: {filename}")
                return str(local_path)
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            logger.info(f"Successfully downloaded: {filename}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None
            
    def process_json_metadata(self, json_paths: List[str]) -> List[Dict]:
        """
        Process multiple JSON metadata files and download PDFs.
        
        Args:
            json_paths: List of paths to JSON metadata files
            
        Returns:
            List[Dict]: List of processed metadata with local PDF paths
        """
        all_metadata = []
        
        for json_path in json_paths:
            metadata_list = self.load_metadata_from_json(json_path)
            
            for metadata in metadata_list:
                pdf_url = metadata.get('pdf_url')
                if not pdf_url:
                    continue
                    
                # Download PDF
                local_path = self.download_pdf(pdf_url, metadata)
                if local_path:
                    # Add additional metadata
                    enriched_metadata = {
                        **metadata,
                        'local_pdf_path': local_path,
                        'download_timestamp': datetime.now().isoformat(),
                        'json_source': os.path.basename(json_path)
                    }
                    all_metadata.append(enriched_metadata)
                    
        return all_metadata 