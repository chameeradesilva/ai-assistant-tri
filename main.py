"""Main script for PDF processing and Pinecone upload."""
import os
import glob
import logging
import logging.config
import yaml
import argparse
from datetime import datetime
from time import time
from typing import Dict, Optional, List
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessorFactory
from src.pdf_processor.downloader import PDFDownloader

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """
    Setup logging configuration from the YAML file.
    
    Args:
        default_path (str): Path to the logging configuration YAML file
        default_level (int): Default logging level if config file is not found
        
    This function:
    1. Creates the logs directory if it doesn't exist
    2. Loads the YAML configuration file
    3. Configures all loggers based on the YAML settings
    """
    # Create logs directory if it doesn't exist
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
        logging.error(f"Error loading logging configuration: {str(e)}")

# Initialize logging first, before any other operations
setup_logging()

# Get the logger for this module
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def process_pdf(file_path: str, metadata: Dict, factory: PDFProcessorFactory, pinecone_config: Optional[Dict] = None) -> bool:
    """
    Process a single PDF file.
    
    Args:
        file_path: Path to the PDF file
        metadata: Metadata dictionary for the PDF
        factory: PDFProcessorFactory instance
        pinecone_config: Optional Pinecone configuration
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    start_time = time()
    filename = os.path.basename(file_path)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"STARTING NEW FILE: {filename}")
        logger.info(f"{'='*50}")
        
        # Create components using factory
        logger.info(f"\n1. Creating PDF processing components for {filename}")
        extractor = factory.create_extractor(tesseract_path=os.getenv('TESSERACT_PATH'))
        text_processor = factory.create_text_processor()
        embedding_generator = factory.create_embedding_generator()
        
        # Extract text
        logger.info(f"\n2. Starting text extraction from {filename}")
        extraction_start = time()
        text = extractor.extract_text(file_path)
        extraction_time = time() - extraction_start
        logger.info(f"   ✓ Text extraction completed in {extraction_time:.2f} seconds")
        
        if not text:
            logger.error(f"   ✗ ERROR: No text extracted from {filename}")
            return False
            
        # Enrich metadata
        logger.info(f"\n3. Enriching metadata for {filename}")
        metadata.update({
            'source': filename,
            'file_type': 'pdf',
            'processed_date': str(datetime.now().isoformat())
        })
        logger.info("   ✓ Metadata enriched")
        logger.debug(f"Updated metadata: {metadata}")  # Added detailed debug logging
        
        # Process text and generate embeddings
        logger.info(f"\n4. Chunking text for {filename}")
        chunk_start = time()
        chunks = text_processor.chunk_text(text, metadata)
        chunk_time = time() - chunk_start
        logger.info(f"   ✓ Text chunking completed in {chunk_time:.2f} seconds")
        logger.info(f"   ✓ Generated {len(chunks)} chunks")
        logger.debug(f"Chunk sizes: {[len(chunk['text']) for chunk in chunks]}")  # Added debug info
        
        logger.info(f"\n5. Generating embeddings for {filename}")
        embedding_start = time()
        chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
        embedding_time = time() - embedding_start
        logger.info(f"   ✓ Embedding generation completed in {embedding_time:.2f} seconds")
        
        # Save to output
        logger.info(f"\n6. Preparing output for {filename}")
        output_file = f"{os.path.splitext(filename)[0]}_processed.json"
        output_path = os.path.join('output', output_file)
        os.makedirs('output', exist_ok=True)
        logger.info(f"   ✓ Output directory prepared")
        
        # Upload to Pinecone if configured
        if pinecone_config:
            logger.info(f"\n7. Uploading to Pinecone for {filename}")
            upload_start = time()
            storage = factory.create_storage(
                api_key=pinecone_config['api_key'],
                environment=pinecone_config['environment'],
                index_name=pinecone_config['index_name']
            )
            storage.upload_batch(chunks_with_embeddings)
            upload_time = time() - upload_start
            logger.info(f"   ✓ Pinecone upload completed in {upload_time:.2f} seconds")
        
        processing_time = time() - start_time
        logger.info(f"\n{'='*50}")
        logger.info(f"COMPLETED {filename} in {processing_time:.2f} seconds")
        logger.info(f"{'='*50}\n")
        return True
        
    except Exception as e:
        logger.error(f"\n{'='*50}")
        logger.error(f"ERROR PROCESSING {filename}")
        logger.exception(f"Error details: {str(e)}")  # This will log the full stack trace
        logger.error(f"{'='*50}\n")
        return False

def get_json_metadata_files(test_mode: bool = False) -> List[str]:
    """Get list of JSON metadata files to process."""
    logger = logging.getLogger(__name__)
    logger.info("\nScanning for JSON metadata files...")
    pattern = "output/json/test_*.json" if test_mode else "output/json/tri_publications_*.json"
    json_files = glob.glob(pattern)
    logger.info(f"Found {len(json_files)} JSON metadata files")
    return json_files

def main():
    logger = logging.getLogger(__name__)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF files and upload to Pinecone')
    parser.add_argument('--test', action='store_true', help='Run in test mode with single file')
    args = parser.parse_args()

    logger.info("\n=== PDF PROCESSING PIPELINE STARTED ===\n")
    if args.test:
        logger.info("Running in TEST MODE - Will process only one document")
    
    start_time = time()
    success_count = 0
    failed_count = 0
    
    try:
        # Initialize factory and downloader
        logger.info("1. Initializing components")
        factory = PDFProcessorFactory()
        downloader = PDFDownloader()
        logger.info("   ✓ Components initialized")
        
        # Configure Pinecone
        logger.info("\n2. Setting up Pinecone configuration")
        pinecone_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENVIRONMENT'),
            'index_name': os.getenv('PINECONE_INDEX_NAME')
        }
        logger.info("   ✓ Pinecone configuration loaded")
        
        # Get JSON metadata files
        json_files = get_json_metadata_files(test_mode=args.test)
        if not json_files:
            logger.warning("No JSON metadata files found")
            return
            
        # Process metadata and download PDFs
        logger.info("\n3. Processing metadata and downloading PDFs")
        all_metadata = downloader.process_json_metadata(json_files)
        
        if args.test:
            all_metadata = all_metadata[:1]  # Take only the first document in test mode
            
        total_files = len(all_metadata)
        
        if total_files == 0:
            logger.warning("No valid PDFs found in metadata")
            return
            
        # Process each PDF file
        logger.info("\n4. Starting PDF processing pipeline")
        for i, metadata in enumerate(all_metadata, 1):
            logger.info(f"\nProcessing file {i} of {total_files}")
            file_path = metadata['local_pdf_path']
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"File: {metadata.get('title')} (Size: {file_size:.2f} MB)")
            
            if process_pdf(file_path, metadata, factory, pinecone_config):
                success_count += 1
            else:
                failed_count += 1
                
        # Print summary
        total_time = time() - start_time
        logger.info("\n=== PROCESSING SUMMARY ===")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info("=========================\n")
        
    except Exception as e:
        logger.error("\nError in main process")
        logger.exception(str(e))  # This will log the full stack trace
        
if __name__ == "__main__":
    main()