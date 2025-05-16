import os
from iris_zone_analyzer import IrisZoneAnalyzer

def test_process_iris_image():
    print("Testing process_iris_image function...")
    analyzer = IrisZoneAnalyzer()
    
    # Test with the iris image
    image_path = './irs.png'
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        return
    
    try:
        result = analyzer.process_iris_image(image_path)
        if "error" in result:
            print(f"Error in analysis: {result['error']}")
            return False
        
        # Check if all expected components are in the result
        expected_keys = ["original_image", "boundary_image", "zone_map", "zones_analysis", "health_summary"]
        for key in expected_keys:
            if key not in result:
                print(f"Error: Missing {key} in the result")
                return False
                
        # Check the health summary
        health_summary = result["health_summary"]
        if "overall_health" not in health_summary or "dosha_balance" not in health_summary:
            print("Error: Incomplete health summary")
            return False
            
        print("Success! All components in the result are valid.")
        print(f"Overall health: {health_summary['overall_health']}")
        print(f"Dosha balance: {health_summary['dosha_balance']}")
        return True
        
    except Exception as e:
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_process_iris_image()
