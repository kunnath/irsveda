import fitz  # PyMuPDF
import sys
import io
from PIL import Image
import numpy as np
import os

def analyze_pdf(pdf_path):
    """
    Analyze a PDF file to check for iris-related content.
    
    Args:
        pdf_path: Path to the PDF file
    """
    print(f"Analyzing PDF: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        
        # Basic information
        print(f"Number of pages: {len(doc)}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
        
        # Check for iris-related content
        iris_pages = []
        iris_mentions = 0
        
        for page_num, page in enumerate(doc):
            text = page.get_text().lower()
            iris_count = text.count("iris")
            eye_count = text.count("eye")
            iridology_count = text.count("iridology")
            ayurved_count = text.count("ayurved")
            
            total_mentions = iris_count + iridology_count
            
            if total_mentions > 0:
                iris_pages.append(page_num + 1)
                iris_mentions += total_mentions
                print(f"Page {page_num + 1}: {total_mentions} iris-related mentions")
                
                # Print a sample of text around iris mentions
                if iris_count > 0:
                    text_chunks = text.split("iris")
                    for i in range(1, len(text_chunks)):
                        before = text_chunks[i-1][-50:] if len(text_chunks[i-1]) > 50 else text_chunks[i-1]
                        after = text_chunks[i][:50] if len(text_chunks[i]) > 50 else text_chunks[i]
                        print(f"  Context: ...{before}iris{after}...")
        
        print("-" * 50)
        print(f"Total iris-related mentions: {iris_mentions}")
        print(f"Iris content found on pages: {iris_pages if iris_pages else 'None'}")
        
        # Check if the PDF is likely a scanned document
        if iris_mentions == 0:
            # Check image content
            print("\nChecking if document is scanned or contains images...")
            has_images = False
            image_count = 0
            sample_page = min(5, len(doc)-1)  # Use the first few pages as sample
            
            for page_num in range(sample_page + 1):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                if image_list:
                    has_images = True
                    image_count += len(image_list)
                    
                    # Extract a sample image
                    if page_num == 0 and image_list:
                        xref = image_list[0][0]
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        extension = base_image["ext"]
                        
                        # Save image for reference
                        sample_path = f"sample_page_image.{extension}"
                        with open(sample_path, "wb") as f:
                            f.write(image_data)
                        print(f"Saved sample image from page 1 to {sample_path}")
                        
                        # TODO: If needed, run OCR on this image to extract text
            
            print(f"Document has {image_count} images in the first {sample_page + 1} pages.")
            if has_images and iris_mentions == 0:
                print("This appears to be a scanned document or image-based PDF.")
                print("OCR processing might be needed for text extraction.")
        
        # If no iris content found, check for any useful health-related content
        if not iris_pages:
            health_terms = ["health", "medicine", "diagnosis", "healing", "medical"]
            health_pages = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text().lower()
                for term in health_terms:
                    if term in text:
                        health_pages.append(page_num + 1)
                        break
            
            if health_pages:
                print(f"No iris content, but health-related content found on pages: {health_pages}")
            else:
                print("No health-related content found.")
        
    except Exception as e:
        print(f"Error analyzing PDF: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_pdf(sys.argv[1])
    else:
        analyze_pdf("Orange White Modern Pitch Deck Presentation.pdf")
