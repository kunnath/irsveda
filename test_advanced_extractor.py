#!/usr/bin/env python3
"""
Test script for the advanced PDF extractor.
This script demonstrates how to use the new advanced PDF extractor to extract
content from PDFs with better chunking and more comprehensive extraction.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
import json
from advanced_pdf_extractor import extract_advanced_chunks


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract content from PDF files using advanced chunking')
    parser.add_argument('pdf_file', help='PDF file to process')
    parser.add_argument('-o', '--output', help='Output JSON file (default: output.json)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')
    args = parser.parse_args()
    
    # Verify the PDF file exists
    if not os.path.isfile(args.pdf_file):
        print(f"Error: PDF file not found: {args.pdf_file}")
        return 1
    
    # Set output file
    output_file = args.output if args.output else "output.json"
    
    print(f"Processing: {args.pdf_file}")
    print(f"Output will be written to: {output_file}")
    
    # Simple progress callback
    def report_progress(current, total):
        if args.verbose or current % 5 == 0:  # Show progress every 5 steps or if verbose
            print(f"Processing: {current}/{total} ({current/total*100:.1f}%)")
    
    # Start timer
    start_time = time.time()
    
    # Extract chunks
    chunks = extract_advanced_chunks(args.pdf_file, report_progress)
    
    # End timer
    elapsed_time = time.time() - start_time
    
    # Print statistics
    print(f"\nExtraction completed in {elapsed_time:.2f} seconds")
    print(f"Total chunks extracted: {len(chunks)}")
    
    if chunks:
        # Show sample of the first chunk
        print("\nSample of first extracted chunk:")
        print(f"Text (first 150 chars): {chunks[0]['text'][:150]}...")
        print(f"Page: {chunks[0]['page']}")
        print(f"Relevance: {chunks[0]['relevance_score']:.2f}")
        print(f"Keywords: {', '.join(chunks[0].get('keywords', [])[:10])}")
        
        # Count dosha-related chunks
        dosha_chunks = [c for c in chunks if c.get('is_dosha_related', False)]
        print(f"\nDosha-related chunks: {len(dosha_chunks)}/{len(chunks)}")
        
        # Count chunks by extraction method
        methods = {}
        for chunk in chunks:
            method = chunk.get('extraction_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        print("\nChunks by extraction method:")
        for method, count in methods.items():
            print(f"  - {method}: {count}")
        
        # Write chunks to output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Create output structure
                output = {
                    "source": args.pdf_file,
                    "extraction_time": elapsed_time,
                    "chunks_count": len(chunks),
                    "chunks": chunks
                }
                json.dump(output, f, indent=2)
            print(f"\nChunks written to: {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            return 1
    else:
        print("No chunks were extracted. The PDF may not contain relevant content or may be image-based without OCR support.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
