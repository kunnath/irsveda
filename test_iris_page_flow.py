import os
import traceback
from iris_zone_analyzer import IrisZoneAnalyzer

def test_iris_page_flow():
    """
    Test the iris analysis feature flow as it would be used in the Iris Zone Analysis page.
    """
    print("Testing Iris Zone Analysis page flow...")
    
    try:
        # Initialize the analyzer
        analyzer = IrisZoneAnalyzer()
        
        # Step 1: Process the iris image (this is what would happen when a user uploads an image)
        image_path = './irs.png'
        if not os.path.exists(image_path):
            raise ValueError(f"Test image {image_path} not found")
        
        result = analyzer.process_iris_image(image_path)
        
        # Step 2: Check for errors in the result
        if "error" in result:
            print(f"Analysis error: {result['error']}")
            return False
        
        # Step 3: Extract the components that would be displayed on the page
        original_image = result["original_image"]
        boundary_image = result["boundary_image"]
        zone_map = result["zone_map"]
        zones_analysis = result["zones_analysis"]
        health_summary = result["health_summary"]
        
        # Step 4: Test manipulations that might happen on the page
        # For example, accessing zone details
        print("Testing zone data access:")
        for zone_name, zone_data in zones_analysis.items():
            print(f"- {zone_name}: {zone_data['name']}")
            # Access ayurvedic mapping
            systems = zone_data["ayurvedic_mapping"]["systems"]
            dosha = zone_data["ayurvedic_mapping"]["dominant_dosha"]
            print(f"  - Systems: {', '.join(systems)}")
            print(f"  - Dominant dosha: {dosha}")
            
        # Step 5: Test health summary display
        print("\nHealth Summary:")
        print(f"- Overall health: {health_summary['overall_health']}")
        print(f"- Dosha balance: {health_summary['dosha_balance']}")
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error during test: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_iris_page_flow()
