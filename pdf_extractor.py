import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Union
import io
from PIL import Image
import os
import pytesseract


def extract_iris_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract iris-related chunks from PDF that contain how/why/when questions.
    Uses OCR if regular text extraction doesn't yield results.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of iris-related text chunks with metadata
    """
    doc = fitz.open(pdf_path)
    iris_chunks = []
    
    # First try standard text extraction
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if "iris" in text.lower():
            # Split text into paragraphs for better context
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if 'iris' in para.lower() and re.search(r'\b(how|why|when)\b', para.lower()):
                    # Add page number for reference
                    iris_chunks.append({
                        "text": para.strip(),
                        "page": page_num + 1,  # 1-based page numbering
                        "source": pdf_path
                    })
    
    # If no iris content found and the document appears to be scanned, try OCR
    if not iris_chunks:
        print(f"No iris content found via standard extraction. Attempting OCR...")
        
        # Process a subset of pages to save time
        pages_to_process = min(10, len(doc))
        
        try:
            # Verify Tesseract is available
            pytesseract.get_tesseract_version()
            
            for page_num in range(pages_to_process):
                print(f"OCR processing page {page_num+1}/{pages_to_process}...")
                page = doc[page_num]
                
                # Get the page as an image at higher resolution for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # Use OCR to extract text
                ocr_text = pytesseract.image_to_string(img)
                
                # Check for iris-related content
                if "iris" in ocr_text.lower() or "iridology" in ocr_text.lower():
                    # Split into paragraphs
                    paragraphs = ocr_text.split('\n\n')
                    for para in paragraphs:
                        if ('iris' in para.lower() or 'iridology' in para.lower()) and re.search(r'\b(how|why|when)\b', para.lower()):
                            iris_chunks.append({
                                "text": para.strip(),
                                "page": page_num + 1,  # 1-based page numbering
                                "source": pdf_path,
                                "extraction_method": "ocr"
                            })
        except Exception as e:
            print(f"OCR processing error: {str(e)}")
    
    print(f"Extracted {len(iris_chunks)} iris-related chunks from {pdf_path}")
    return iris_chunks
