"""Main script for PDF processing and Pinecone upload."""
import os
import glob
import logging
import argparse
from datetime import datetime
from time import time
from typing import Dict, Optional, List
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessorFactory
from src.pdf_processor.downloader import PDFDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
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
    
    try:
        print(f"\n{'='*50}")
        print(f"STARTING NEW FILE: {filename}")
        print(f"{'='*50}")
        
        # Create components using factory
        print(f"\n1. Creating PDF processing components for {filename}")
        extractor = factory.create_extractor(tesseract_path=os.getenv('TESSERACT_PATH'))
        text_processor = factory.create_text_processor()
        embedding_generator = factory.create_embedding_generator()
        
        # Extract text
        print(f"\n2. Starting text extraction from {filename}")
        extraction_start = time()
        text = extractor.extract_text(file_path)
        extraction_time = time() - extraction_start
        print(f"   ✓ Text extraction completed in {extraction_time:.2f} seconds")
        
        if not text:
            print(f"   ✗ ERROR: No text extracted from {filename}")
            return False
            
        # Enrich metadata
        print(f"\n3. Enriching metadata for {filename}")
        metadata.update({
            'source': filename,
            'file_type': 'pdf',
            'processed_date': str(datetime.now().isoformat())
        })
        print("   ✓ Metadata enriched")
        
        # Process text and generate embeddings
        print(f"\n4. Chunking text for {filename}")
        chunk_start = time()
        chunks = text_processor.chunk_text(text, metadata)
        chunk_time = time() - chunk_start
        print(f"   ✓ Text chunking completed in {chunk_time:.2f} seconds")
        print(f"   ✓ Generated {len(chunks)} chunks")
        
        print(f"\n5. Generating embeddings for {filename}")
        embedding_start = time()
        chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
        embedding_time = time() - embedding_start
        print(f"   ✓ Embedding generation completed in {embedding_time:.2f} seconds")
        
        # Save to output
        print(f"\n6. Preparing output for {filename}")
        output_file = f"{os.path.splitext(filename)[0]}_processed.json"
        output_path = os.path.join('output', output_file)
        os.makedirs('output', exist_ok=True)
        print(f"   ✓ Output directory prepared")
        
        # Upload to Pinecone if configured
        if pinecone_config:
            print(f"\n7. Uploading to Pinecone for {filename}")
            upload_start = time()
            storage = factory.create_storage(
                api_key=pinecone_config['api_key'],
                environment=pinecone_config['environment'],
                index_name=pinecone_config['index_name']
            )
            storage.upload_batch(chunks_with_embeddings)
            upload_time = time() - upload_start
            print(f"   ✓ Pinecone upload completed in {upload_time:.2f} seconds")
        
        processing_time = time() - start_time
        print(f"\n{'='*50}")
        print(f"COMPLETED {filename} in {processing_time:.2f} seconds")
        print(f"{'='*50}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR PROCESSING {filename}")
        print(f"Error: {str(e)}")
        print(f"{'='*50}\n")
        return False

def get_json_metadata_files(test_mode: bool = False) -> List[str]:
    """Get list of JSON metadata files to process."""
    print("\nScanning for JSON metadata files...")
    pattern = "output/json/test_*.json" if test_mode else "output/json/tri_publications_*.json"
    json_files = glob.glob(pattern)
    print(f"Found {len(json_files)} JSON metadata files")
    return json_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF files and upload to Pinecone')
    parser.add_argument('--test', action='store_true', help='Run in test mode with single file')
    args = parser.parse_args()

    print("\n=== PDF PROCESSING PIPELINE STARTED ===\n")
    if args.test:
        print("Running in TEST MODE - Will process only one document")
    
    start_time = time()
    success_count = 0
    failed_count = 0
    
    try:
        # Initialize factory and downloader
        print("1. Initializing components")
        factory = PDFProcessorFactory()
        downloader = PDFDownloader()
        print("   ✓ Components initialized")
        
        # Configure Pinecone
        print("\n2. Setting up Pinecone configuration")
        pinecone_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENVIRONMENT'),
            'index_name': os.getenv('PINECONE_INDEX_NAME')
        }
        print("   ✓ Pinecone configuration loaded")
        
        # Get JSON metadata files
        json_files = get_json_metadata_files(test_mode=args.test)
        if not json_files:
            print("No JSON metadata files found")
            return
            
        # Process metadata and download PDFs
        print("\n3. Processing metadata and downloading PDFs")
        all_metadata = downloader.process_json_metadata(json_files)
        
        if args.test:
            all_metadata = all_metadata[:1]  # Take only the first document in test mode
            
        total_files = len(all_metadata)
        
        if total_files == 0:
            print("No valid PDFs found in metadata")
            return
            
        # Process each PDF file
        print("\n4. Starting PDF processing pipeline")
        for i, metadata in enumerate(all_metadata, 1):
            print(f"\nProcessing file {i} of {total_files}")
            file_path = metadata['local_pdf_path']
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"File: {metadata.get('title')} (Size: {file_size:.2f} MB)")
            
            if process_pdf(file_path, metadata, factory, pinecone_config):
                success_count += 1
            else:
                failed_count += 1
                
        # Print summary
        total_time = time() - start_time
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total files processed: {total_files}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Total time: {total_time:.2f} seconds")
        print("=========================\n")
        
    except Exception as e:
        print(f"\nError in main process: {str(e)}")
        logger.exception("Error in main process")  # Added detailed error logging
        
if __name__ == "__main__":
    main()