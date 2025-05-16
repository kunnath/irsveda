import os
import sys
import traceback
from iris_zone_analyzer import IrisZoneAnalyzer

def diagnose_tuple_error():
    """
    Try to diagnose the 'tuple' object does not support item assignment error
    by stepping through each part of the process and checking for tuple operations.
    """
    try:
        # Initialize analyzer
        analyzer = IrisZoneAnalyzer()
        
        # Step 1: Test preprocess_image
        print("Testing preprocess_image...")
        image_path = './irs.png'
        if not os.path.exists(image_path):
            print(f"Error: Test image {image_path} not found.")
            return
        
        preprocessed = analyzer.preprocess_image(image_path)
        print("✓ preprocess_image works")
        
        # Step 2: Test detect_iris_and_pupil
        print("\nTesting detect_iris_and_pupil...")
        boundaries, boundary_image = analyzer.detect_iris_and_pupil(preprocessed)
        print("✓ detect_iris_and_pupil works")
        print(f"  - Pupil radius: {boundaries['pupil']['radius']}")
        print(f"  - Iris radius: {boundaries['iris']['radius']}")
        
        # Step 3: Test generate_zone_map
        print("\nTesting generate_zone_map...")
        # Wrap in try-except to catch tuple errors specifically
        try:
            zone_map = analyzer.generate_zone_map(preprocessed, boundaries)
            print("✓ generate_zone_map works")
        except TypeError as e:
            if "'tuple' object does not support item assignment" in str(e):
                print(f"× Error in generate_zone_map: {e}")
                print("  - This is likely where the tuple error is occurring")
                traceback.print_exc()
                return
            else:
                raise
        
        # Step 4: Test analyze_iris_zones
        print("\nTesting analyze_iris_zones...")
        try:
            zones_analysis = analyzer.analyze_iris_zones(preprocessed, boundaries)
            print("✓ analyze_iris_zones works")
            
            # Check if there are any tuple values in zones_analysis that might cause problems
            # when being manipulated later
            print("  - Checking for tuple values in analysis results...")
            for zone_name, zone_data in zones_analysis.items():
                # Check RGB means
                rgb_means = None
                if "color_features" in zone_data:
                    color_features = zone_data["color_features"]
                    # Check types
                    for key, value in color_features.items():
                        if isinstance(value, tuple):
                            print(f"    Warning: {zone_name}.color_features.{key} is a tuple: {value}")
                
                # Check other potential tuple values
                if "ayurvedic_mapping" in zone_data:
                    mapping = zone_data["ayurvedic_mapping"]
                    for key, value in mapping.items():
                        if isinstance(value, tuple):
                            print(f"    Warning: {zone_name}.ayurvedic_mapping.{key} is a tuple: {value}")
        except TypeError as e:
            if "'tuple' object does not support item assignment" in str(e):
                print(f"× Error in analyze_iris_zones: {e}")
                print("  - This is likely where the tuple error is occurring")
                traceback.print_exc()
                return
            else:
                raise
        
        # Step 5: Test the full process_iris_image method
        print("\nTesting full process_iris_image method...")
        try:
            result = analyzer.process_iris_image(image_path)
            if "error" in result:
                print(f"× Error in process_iris_image: {result['error']}")
            else:
                print("✓ process_iris_image works")
        except Exception as e:
            print(f"× Error in process_iris_image: {e}")
            traceback.print_exc()
            return
        
        print("\nAll tests passed! No tuple assignment errors detected.")
        
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_tuple_error()
