from PIL import Image
import pytesseract
import sys

def test_tesseract():
    """Simple test to verify Tesseract OCR is working"""
    try:
        # First, check tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        
        # Test OCR on the sample image
        image_path = "sample_page_image.png"
        print(f"Attempting OCR on: {image_path}")
        
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        
        print(f"Extracted {len(text)} characters of text")
        print("Sample of extracted text:")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
        
        # Look for iris-related content
        if "iris" in text.lower() or "iridology" in text.lower():
            print("Found iris-related content!")
        else:
            print("No iris-related content found in this image.")
            
    except Exception as e:
        print(f"Error testing Tesseract: {str(e)}")

if __name__ == "__main__":
    test_tesseract()
