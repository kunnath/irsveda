#!/usr/bin/env python3
"""
Extract content from PDFs and store in Qdrant with improved chunking.
This script uses the advanced PDF extractor to process PDFs and store them
in the Qdrant vector database with improved chunking and metadata.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("process_pdfs")

# Import our modules
from advanced_pdf_extractor import extract_advanced_chunks
from enhanced_iris_qdrant import EnhancedIrisQdrantClient


def process_pdf(pdf_path: str, qdrant_client, store_in_qdrant=True):
    """
    Process a PDF file and optionally store chunks in Qdrant.
    
    Args:
        pdf_path: Path to the PDF file
        qdrant_client: EnhancedIrisQdrantClient instance
        store_in_qdrant: Whether to store chunks in Qdrant
        
    Returns:
        List of extracted chunks
    """
    # Simple progress callback
    def report_progress(current, total):
        logger.info(f"Processing: {current}/{total} ({current/total*100:.1f}%)")
    
    # Start timer
    start_time = time.time()
    
    # Extract chunks with advanced extractor
    logger.info(f"Extracting content from: {pdf_path}")
    chunks = extract_advanced_chunks(pdf_path, report_progress)
    
    # End timer
    elapsed_time = time.time() - start_time
    
    # Print statistics
    logger.info(f"Extraction completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total chunks extracted: {len(chunks)}")
    
    if chunks:
        # Show sample of the first chunk
        logger.info(f"Sample of first extracted chunk:")
        logger.info(f"Text (first 100 chars): {chunks[0]['text'][:100]}...")
        logger.info(f"Page: {chunks[0]['page']}")
        logger.info(f"Keywords: {', '.join(chunks[0].get('keywords', [])[:5])}")
        
        # Count dosha-related chunks
        dosha_chunks = [c for c in chunks if c.get('is_dosha_related', False)]
        logger.info(f"Dosha-related chunks: {len(dosha_chunks)}/{len(chunks)}")
        
        # Store in Qdrant if requested
        if store_in_qdrant:
            try:
                logger.info(f"Storing {len(chunks)} chunks in Qdrant...")
                point_ids = qdrant_client.store_chunks(chunks)
                logger.info(f"Successfully stored {len(point_ids)} chunks in Qdrant")
            except Exception as e:
                logger.error(f"Error storing chunks in Qdrant: {e}")
    else:
        logger.warning("No chunks were extracted. The PDF may not contain relevant content.")
    
    return chunks


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDFs and store in Qdrant')
    parser.add_argument('pdf_files', nargs='+', help='PDF files to process')
    parser.add_argument('--qdrant-host', default='localhost', help='Qdrant host (default: localhost)')
    parser.add_argument('--qdrant-port', type=int, default=6333, help='Qdrant port (default: 6333)')
    parser.add_argument('--collection', default='enhanced_iris_chunks', help='Qdrant collection name')
    parser.add_argument('--skip-qdrant', action='store_true', help='Skip storing in Qdrant')
    parser.add_argument('--output-dir', help='Directory to save JSON output files')
    args = parser.parse_args()
    
    # Initialize Qdrant client
    if not args.skip_qdrant:
        try:
            logger.info(f"Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}...")
            qdrant_client = EnhancedIrisQdrantClient(
                host=args.qdrant_host,
                port=args.qdrant_port,
                collection_name=args.collection
            )
            
            # Create collection if it doesn't exist
            qdrant_client.create_collection()
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return 1
    else:
        qdrant_client = None
        logger.info("Skipping Qdrant storage (--skip-qdrant specified)")
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output files will be saved to: {args.output_dir}")
    
    # Process each PDF file
    total_chunks = 0
    for pdf_file in args.pdf_files:
        if not os.path.isfile(pdf_file):
            logger.error(f"PDF file not found: {pdf_file}")
            continue
        
        logger.info(f"Processing PDF: {pdf_file}")
        
        # Extract and store chunks
        chunks = process_pdf(pdf_file, qdrant_client, not args.skip_qdrant)
        total_chunks += len(chunks)
        
        # Save chunks to JSON file if output directory is specified
        if args.output_dir and chunks:
            output_file = os.path.join(args.output_dir, f"{os.path.basename(pdf_file)}.json")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    # Create output structure
                    output = {
                        "source": pdf_file,
                        "chunks_count": len(chunks),
                        "chunks": chunks
                    }
                    json.dump(output, f, indent=2)
                logger.info(f"Chunks written to: {output_file}")
            except Exception as e:
                logger.error(f"Error writing output file: {e}")
    
    logger.info(f"Processing complete. Total chunks extracted: {total_chunks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
