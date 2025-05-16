from iris_zone_analyzer import IrisZoneAnalyzer
from iris_report_generator import IrisReportGenerator
import traceback

def test_report_generation():
    """Test the report generation with PIL Images from the iris zone analyzer."""
    try:
        print("Initializing analyzers...")
        analyzer = IrisZoneAnalyzer()
        report_generator = IrisReportGenerator()
        
        # Process an iris image
        print("Processing iris image...")
        zone_results = analyzer.process_iris_image('./irs.png')
        
        if "error" in zone_results:
            print(f"Error in analysis: {zone_results['error']}")
            return
        
        # Check the types of images in zone_results
        from PIL import Image
        for key in ["original_image", "boundary_image", "zone_map"]:
            if isinstance(zone_results[key], Image.Image):
                print(f"✓ {key} is a PIL Image as expected")
            else:
                print(f"× {key} is not a PIL Image, it's a {type(zone_results[key])}")
        
        # Try generating reports
        print("\nGenerating HTML report...")
        user_info = {
            "name": "Test User",
            "age": 30,
            "gender": "Female",
            "email": "test@example.com",
            "concerns": "General health check"
        }
        
        try:
            html_report = report_generator.generate_html_report(zone_results, user_info)
            print("✓ HTML report generated successfully")
        except Exception as e:
            print(f"× Error generating HTML report: {e}")
            traceback.print_exc()
            return
        
        print("\nGenerating PDF report...")
        try:
            pdf_report = report_generator.generate_report(zone_results, user_info)
            print("✓ PDF report generated successfully")
        except Exception as e:
            print(f"× Error generating PDF report: {e}")
            traceback.print_exc()
            return
        
        print("\nAll tests passed! Report generation is working correctly.")
        
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_report_generation()
