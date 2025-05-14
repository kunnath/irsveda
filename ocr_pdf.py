import fitz  # PyMuPDF
import sys
import io
from PIL import Image
import os
import pytesseract
import re

def ocr_pdf_for_iris_content(pdf_path):
    """
    Performs OCR on a PDF file to extract iris-related content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries with extracted iris-related content
    """
    print(f"Performing OCR on PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return []

    # Check Tesseract installation
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {tesseract_version}")
    except Exception as e:
        print(f"Error checking Tesseract: {str(e)}")
        print("Make sure Tesseract OCR is installed and in your PATH")
        return []

    try:
        doc = fitz.open(pdf_path)
        print(f"Successfully opened PDF with {len(doc)} pages")
        iris_chunks = []
        
        # Process a subset of pages to save time
        # In a real application, you might want to process all pages
        pages_to_process = min(10, len(doc))
        
        for page_num in range(pages_to_process):
            print(f"Processing page {page_num+1}/{pages_to_process}...")
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
                    if "iris" in para.lower() or "iridology" in para.lower():
                        # Check for how/why/when questions
                        if re.search(r'\b(how|why|when)\b', para.lower()):
                            iris_chunks.append({
                                "page": page_num + 1,
                                "text": para.strip(),
                                "source": pdf_path
                            })
        
        print(f"\nExtracted {len(iris_chunks)} iris-related chunks from {pages_to_process} pages")
        
        # Display sample chunks
        if iris_chunks:
            print("\nSample extracted content:")
            for i, chunk in enumerate(iris_chunks[:3]):  # Show up to 3 samples
                print(f"\n--- Sample {i+1} (Page {chunk['page']}) ---")
                print(chunk['text'])
        
        return iris_chunks
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return []

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ocr_pdf_for_iris_content(sys.argv[1])
    else:
        ocr_pdf_for_iris_content("Iridology_Simplified.pdf")
