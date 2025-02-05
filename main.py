"""Main script for PDF processing with PyMuPDF and Tesseract OCR."""
import os
import glob
import argparse
import shutil
from datetime import datetime
from time import time
import logging
import logging.config
import yaml
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv
from src.pdf_processor.factory import PDFProcessorFactory

# Load environment variables
load_dotenv()

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    
    try:
        if os.path.exists(default_path):
            with open(default_path, 'rt') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
            logging.info(f"Logging configuration loaded from {default_path}")
        else:
            logging.basicConfig(level=default_level)
            logging.warning(f"Logging configuration file not found at {default_path}. Using basic configuration.")
    except Exception as e:
        logging.basicConfig(level=default_level)
        logger.error(f"Error loading logging configuration: {str(e)}")

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def get_pdf_files(test_mode: bool = False) -> List[str]:
    """Get list of PDF files to process."""
    # First check data/pdfs directory
    target_dir = "data/pdfs"
    os.makedirs(target_dir, exist_ok=True)
    
    # Get PDFs from data/pdfs
    pdf_files = glob.glob(os.path.join(target_dir, "*.pdf"))
    
    # If no PDFs found in data/pdfs, try copying from downloads
    if not pdf_files:
        downloads_dir = os.path.expanduser("~/Downloads")
        download_pdfs = glob.glob(os.path.join(downloads_dir, "*.pdf"))
        
        if download_pdfs:
            for pdf_file in download_pdfs:
                filename = os.path.basename(pdf_file)
                target_path = os.path.join(target_dir, filename)
                try:
                    shutil.copy2(pdf_file, target_path)
                    pdf_files.append(target_path)
                    logger.info(f"Copied {filename} to {target_dir}")
                except Exception as e:
                    logger.error(f"Failed to copy {filename}: {str(e)}")
    
    if not pdf_files:
        logger.warning("No PDF files found in data/pdfs or downloads folder")
        return []
        
    # In test mode, return only the first PDF
    if test_mode and pdf_files:
        selected_pdf = pdf_files[0]
        logger.info(f"Test mode: Selected {os.path.basename(selected_pdf)} for processing")
        return [selected_pdf]
        
    return pdf_files

def upload_to_pinecone(content: Dict[str, Any], metadata: Dict[str, Any], factory: PDFProcessorFactory) -> bool:
    """Upload processed content to Pinecone."""
    try:
        # Initialize Pinecone storage
        pinecone_storage = factory.create_storage(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT'),
            index_name=os.getenv('PINECONE_INDEX_NAME')
        )
        
        # Combine all text from pages
        all_text = "\n".join(
            page.get('text', '') 
            for page in content.get('pages', [])
            if isinstance(page, dict)
        )
        
        # Create document chunks
        text_processor = factory.create_text_processor()
        chunks = text_processor.create_chunks(all_text)
        
        # Create document metadata
        doc_metadata = {
            'source': metadata['filename'],
            'page_count': metadata['total_pages'],
            'extraction_method': metadata['extraction_method'],
            'processing_time': metadata['processing_time']
        }
        
        # Upload to Pinecone
        logger.info("Uploading chunks to Pinecone...")
        pinecone_storage.upload_documents(chunks, doc_metadata)
        logger.info("Successfully uploaded to Pinecone")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload to Pinecone: {str(e)}")
        return False

def process_pdf(file_path: str, extractor: Any) -> Dict[str, Any]:
    """Process a single PDF file."""
    try:
        logger.info(f"\nProcessing: {os.path.basename(file_path)}")
        logger.info("=" * 50)
        
        # Extract content
        content = extractor.extract_text(file_path)
        
        # Prepare metadata
        metadata = {
            'filename': os.path.basename(file_path),
            'processing_time': content.get('metadata', {}).get('processing_time', 0),
            'total_pages': content.get('metadata', {}).get('total_pages', 0),
            'processing_status': 'completed',
            'extraction_method': 'basic_pymupdf_tesseract'
        }
        
        # Process tables
        all_tables = []
        for page in content.get('pages', []):
            if isinstance(page, dict) and 'tables' in page:
                all_tables.extend(page['tables'])
        
        # Save results
        if all_tables:
            output_dir = os.path.join('output', 'tables')
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, table in enumerate(all_tables, 1):
                if isinstance(table, pd.DataFrame):
                    output_path = os.path.join(
                        output_dir,
                        f"{os.path.splitext(metadata['filename'])[0]}_table_{idx}.csv"
                    )
                    table.to_csv(output_path, index=False)
                    logger.info(f"Saved table {idx} to {output_path}")
        
        # Save full extraction results
        output_dir = os.path.join('output', 'extractions')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(metadata['filename'])[0]}_extraction.json"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(content, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved extraction results to {output_path}")
        return {'content': content, 'metadata': metadata}
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return {
            'content': None,
            'metadata': {
                'filename': os.path.basename(file_path),
                'processing_status': 'failed',
                'error': str(e)
            }
        }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Process PDFs with PyMuPDF and Tesseract')
    parser.add_argument('--test', action='store_true', help='Run in test mode with single file')
    parser.add_argument('--config', default='config/processing.yaml', help='Path to processing configuration')
    parser.add_argument('--upload', action='store_true', help='Upload processed content to Pinecone')
    args = parser.parse_args()
    
    logger.info("\n=== PDF PROCESSING PIPELINE STARTED ===\n")
    if args.test:
        logger.info("Running in TEST MODE - Will process only one PDF file")
    start_time = time()
    
    try:
        # Initialize components
        logger.info("1. Initializing PDF processor")
        factory = PDFProcessorFactory(config_path=args.config)
        extractor = factory.create_extractor()
        logger.info("PDF processor initialized")
        
        # Get PDF files to process
        logger.info("\n2. Collecting PDF files")
        pdf_files = get_pdf_files(test_mode=args.test)
        if not pdf_files:
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        logger.info("\n3. Starting PDF processing")
        results = []
        
        for pdf_file in pdf_files:
            result = process_pdf(pdf_file, extractor)
            
            # Upload to Pinecone if requested
            if args.upload and result['content'] is not None:
                upload_success = upload_to_pinecone(
                    result['content'],
                    result['metadata'],
                    factory
                )
                result['metadata']['pinecone_upload'] = 'success' if upload_success else 'failed'
                
            results.append(result['metadata'])
            
        # Print summary
        success_count = sum(1 for r in results if r.get('processing_status') == 'completed')
        failed_count = len(results) - success_count
        total_time = time() - start_time
        
        logger.info("\n=== PROCESSING SUMMARY ===")
        logger.info(f"Total files processed: {len(results)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info("=========================\n")
        
    except Exception as e:
        logger.error("\nError in main process")
        logger.exception(str(e))
        
if __name__ == "__main__":
    main()