import traceback
import os
from iris_zone_analyzer import IrisZoneAnalyzer

def test_iris_analysis():
    print("Starting comprehensive iris analyzer test...")
    analyzer = IrisZoneAnalyzer()
    
    # Test with different images if available
    images_to_test = ['./irs.png']
    sample_dir = './sample_images'
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images_to_test.append(os.path.join(sample_dir, file))
    
    # Test each image
    for image_path in images_to_test:
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping...")
            continue
            
        print(f"Testing with image: {image_path}")
        
        try:
            # First try directly accessing the internal methods to pinpoint the issue
            print("Step 1: Preprocessing image")
            preprocessed = analyzer.preprocess_image(image_path)
            
            print("Step 2: Detecting iris and pupil")
            boundaries, boundary_image = analyzer.detect_iris_and_pupil(preprocessed)
            
            print("Step 3: Generating zone map")
            zone_map = analyzer.generate_zone_map(preprocessed, boundaries)
            
            print("Step 4: Analyzing iris zones")
            zones_analysis = analyzer.analyze_iris_zones(preprocessed, boundaries)
            
            print("All steps completed successfully!")
            
            # Now run the full process
            result = analyzer.process_iris_image(image_path)
            if "error" in result:
                print(f"Error in analysis: {result['error']}")
            else:
                print(f"Successfully analyzed {image_path}")
                # Check if key components exist
                for key in ["original_image", "boundary_image", "zone_map", "zones_analysis", "health_summary"]:
                    if key not in result:
                        print(f"Warning: {key} missing from results")
        except Exception as e:
            print(f"Exception during analysis of {image_path}: {str(e)}")
            traceback.print_exc()
    
    print("Test completed!")

if __name__ == "__main__":
    test_iris_analysis()
