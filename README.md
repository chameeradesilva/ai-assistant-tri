# Sri Lankan Tea Industry AI Assistant

This project implements an AI-powered assistant specialized for the Sri Lankan tea industry. It processes PDF documents and creates a knowledge base for intelligent querying and information retrieval.

## Features

- PDF text extraction with support for scanned documents
- Multi-language support with OCR capabilities
- Efficient text chunking and processing
- High-quality embeddings generation using Sentence Transformers
- Vector storage and retrieval using Pinecone
- Optimized for tea industry domain knowledge
- Layout-aware dynamic chunking

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR (required for multi-language support):
   - Windows: Download installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - Add language data packs for Sinhala and Tamil support

5. Create a `.env` file with your Pinecone credentials:
```
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment  # e.g., "gcp-starter"
PINECONE_INDEX=your_index_name
TESSERACT_PATH=/path/to/tesseract  # Optional, if not in system PATH
```

## Multi-Language Support

The system supports multiple languages with a focus on:
- English (primary)
- Sinhala (with Tesseract OCR)
- Tamil (with Tesseract OCR)

Language detection is automatic, and the system will:
1. Detect document language
2. Apply appropriate OCR if needed
3. Process text with language-specific considerations
4. Store language metadata for better retrieval

## Workflow

1. **Document Processing**: 
   - PDF documents are processed using PyMuPDF
   - Automatic language detection
   - OCR fallback for non-English text
2. **Text Chunking**: 
   - Layout-aware dynamic chunking (500 tokens with 50 token overlap)
   - Preserves document structure and context
3. **Embedding Generation**: Text chunks are converted to embeddings using Sentence Transformers
4. **Vector Storage**: Embeddings are stored in Pinecone with enhanced metadata:
   - Document source and ID
   - Language information
   - OCR confidence scores (when applicable)
   - Page number and position
   - Chunk context (is_first_chunk, is_last_chunk)
   - Circular number metadata
   - Custom tea industry specifics

## Usage

```python
from src.pdf_processor import PDFProcessor

# Initialize the processor
processor = PDFProcessor()

# Process a PDF file
processor.process_pdf("path/to/your/tea_document.pdf")
```

## Best Practices

- Keep PDF documents focused on tea industry content
- Ensure proper metadata tagging for better organization
- Regular index maintenance for optimal performance
- Monitor and manage vector storage capacity

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Specify your license here]

## Metadata Fields

New metadata fields include:
- `language`: Detected document language
- `ocr_confidence`: Confidence score for OCR processing
- `doc_id`: Extracted from circular number
- `chunk_length`: Length of each text chunk
- `is_first_chunk`/`is_last_chunk`: Chunk position indicators
- `total_length`: Original document length
- `chunk_size`/`overlap`: Chunking parameters
