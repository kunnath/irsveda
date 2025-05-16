# Test script to identify string concatenation error in report generation
import os
import sys
from iris_zone_analyzer import IrisZoneAnalyzer
from iris_report_generator import IrisReportGenerator
import traceback

# Create mock zone_results to use for testing
def create_mock_zone_results():
    # Create a simple PIL image for testing
    import numpy as np
    from PIL import Image
    
    # Create a simple color image (100x100 pixels)
    mock_img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    
    # Minimal mock data with required fields
    return {
        "original_image": mock_img,  # mock PIL Image
        "zone_map": mock_img,        # mock PIL Image
        "boundary_image": mock_img,  # mock PIL Image
        "health_summary": {
            "overall_health": "balanced",
            "dosha_balance": {
                "vata": 0.4,
                "pitta": 0.3,
                "kapha": 0.3
            }
        },
        "zones_analysis": {
            "zone1": {
                "name": "Brain Region",
                "health_indication": {
                    "condition": "normal",
                    "confidence": 0.85,
                    "suggestion": "Maintain your balanced lifestyle."
                },
                "ayurvedic_mapping": {
                    "systems": ["nervous", "brain"],
                    "description": "Related to nervous system and brain function.",
                    "dominant_dosha": "vata",
                    "dosha_qualities": ["mobile", "dry", "light"]
                }
            }
        }
    }

def main():
    try:
        # First, get a real iris image analysis to ensure valid data
        analyzer = IrisZoneAnalyzer()
        
        # Use a sample image provided in the repo if possible
        sample_paths = [
            os.path.join("sample_images", img) for img in os.listdir("sample_images") 
            if img.endswith('.jpg') or img.endswith('.png')
        ] if os.path.exists("sample_images") else []
        
        # Use the first sample image or a default one if no samples are available
        image_path = sample_paths[0] if sample_paths else "sample_page_image.png"
        
        print(f"Using image: {image_path}")
        
        # Process the iris image to get complete data
        zone_results = analyzer.process_iris_image(image_path)
        
        if "error" in zone_results:
            print(f"Error in iris analysis: {zone_results['error']}")
            # Fall back to mock data
            print("Using mock data instead")
            zone_results = create_mock_zone_results()
        
        # Now generate report
        report_generator = IrisReportGenerator()
        
        print("Generating HTML report...")
        html_report = report_generator.generate_html_report(zone_results)
        print("HTML report generated successfully.")
        
        print("Generating PDF report...")
        # Use a try-except block inside our main try-except to get detailed error info
        try:
            pdf_report = report_generator.generate_report(zone_results)
            print("PDF report generated successfully.")
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            print("Traceback:")
            traceback.print_exc()
            # Let's make this more verbose to find the exact issue
            if hasattr(e, "__traceback__"):
                # Get the traceback as a string
                tb_str = ''.join(traceback.format_tb(e.__traceback__))
                print(f"Detailed traceback: {tb_str}")

    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
