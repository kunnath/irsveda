# Test specifically if the fix for the Detailed Zone Analysis works
import os
from iris_report_generator import IrisReportGenerator
from PIL import Image
import numpy as np

# Create a simple test dataset that resembles the real data
def create_test_data():
    # Create a test image
    test_img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    
    # Create test data structure
    return {
        "original_image": test_img,
        "zone_map": test_img,
        "boundary_image": test_img,
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
                "name": "Brain Zone",
                "health_indication": {
                    "condition": "normal",
                    "confidence": 0.85,  # <-- This is where the error could occur
                    "suggestion": "Maintain your balanced lifestyle."
                },
                "ayurvedic_mapping": {
                    "systems": ["nervous", "brain"],
                    "description": "Related to nervous system and brain function.",
                    "dominant_dosha": "vata",
                    "dosha_qualities": ["mobile", "dry", "light"]
                }
            },
            "zone2": {
                "name": "Digestive Zone",
                "health_indication": {
                    "condition": "stressed",
                    "confidence": 0.75,  # <-- This is where the error could occur
                    "suggestion": "Consider digestive herbs."
                },
                "ayurvedic_mapping": {
                    "systems": ["digestive", "liver"],
                    "description": "Related to digestion and liver function.",
                    "dominant_dosha": "pitta",
                    "dosha_qualities": ["hot", "sharp", "light"]
                }
            }
        }
    }

def main():
    print("Testing Detailed Zone Analysis in PDF report generation...")
    
    # Create test data
    test_data = create_test_data()
    
    # Create report generator
    report_gen = IrisReportGenerator()
    
    try:
        # Generate PDF report
        pdf_report = report_gen.generate_report(test_data)
        
        # If we get here, there was no exception
        if pdf_report:
            print("SUCCESS: PDF report generated successfully!")
            # Save the PDF to a file for inspection
            pdf_path = "test_detailed_zone_report.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_report)
            print(f"PDF saved to {pdf_path} for inspection")
        else:
            print("ERROR: PDF report generation returned None")
    except Exception as e:
        print(f"ERROR: Exception during PDF generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
