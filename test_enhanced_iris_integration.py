#!/usr/bin/env python3
"""
Test script for Enhanced Iris Analysis Service
Verifies that the integration works correctly
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
import cv2

def create_sample_iris_image(filename="test_iris.jpg", size=(400, 400)):
    """Create a simple sample iris image for testing."""
    # Create a simple iris-like image
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Create outer circle (iris boundary)
    center = (size[0]//2, size[1]//2)
    outer_radius = min(size)//2 - 10
    inner_radius = outer_radius//3
    
    # Fill iris area with brown color
    cv2.circle(img, center, outer_radius, (139, 69, 19), -1)  # Brown
    
    # Add pupil (black circle)
    cv2.circle(img, center, inner_radius, (0, 0, 0), -1)  # Black
    
    # Add some texture (spots) to make it interesting
    for i in range(10):
        spot_x = center[0] + np.random.randint(-outer_radius//2, outer_radius//2)
        spot_y = center[1] + np.random.randint(-outer_radius//2, outer_radius//2)
        spot_radius = np.random.randint(3, 10)
        spot_color = (np.random.randint(50, 200), np.random.randint(30, 150), np.random.randint(10, 100))
        cv2.circle(img, (spot_x, spot_y), spot_radius, spot_color, -1)
    
    # Save the image
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return filename

def test_enhanced_iris_analysis():
    """Test the enhanced iris analysis service."""
    print("Testing Enhanced Iris Analysis Service...")
    
    try:
        # Import the enhanced analysis service
        from enhanced_iris_analysis_service import EnhancedIrisAnalysisService
        print("✅ Successfully imported EnhancedIrisAnalysisService")
        
        # Create test image
        test_image_path = create_sample_iris_image()
        print(f"✅ Created test iris image: {test_image_path}")
        
        # Initialize the service
        analyzer = EnhancedIrisAnalysisService(output_dir="test_output")
        print("✅ Initialized EnhancedIrisAnalysisService")
        
        # Test sensitivity preset
        params = analyzer.set_sensitivity_preset("medium")
        print(f"✅ Set sensitivity preset: {len(params)} parameters")
        
        # Test parameter updates
        updated_params = analyzer.update_detection_params(min_area=30, max_area=1500)
        print(f"✅ Updated detection parameters: min_area={updated_params['min_area']}, max_area={updated_params['max_area']}")
        
        # Test comprehensive analysis
        print("🔍 Running comprehensive analysis...")
        results = analyzer.analyze_iris_comprehensive(
            test_image_path,
            include_zones=True,
            include_doshas=True,
            include_segmentation=True
        )
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return False
        
        print("✅ Comprehensive analysis completed successfully!")
        
        # Check results structure
        expected_keys = ['timestamp', 'image_path', 'analysis_params']
        for key in expected_keys:
            if key in results:
                print(f"✅ Found expected key: {key}")
            else:
                print(f"⚠️ Missing expected key: {key}")
        
        # Test segmentation results
        if 'segmentation_analysis' in results:
            seg_results = results['segmentation_analysis']
            segments_count = seg_results.get('total_segments', 0)
            print(f"✅ Segmentation analysis: {segments_count} segments detected")
            
            if 'detection_methods_used' in seg_results:
                methods = seg_results['detection_methods_used']
                print(f"✅ Detection methods used: {', '.join(methods)}")
        
        # Test data export
        csv_path = analyzer.export_segments_csv()
        if csv_path and os.path.exists(csv_path):
            print(f"✅ Successfully exported segments to: {csv_path}")
        else:
            print("⚠️ No segments data to export")
        
        # Test analysis summary
        summary = analyzer.get_analysis_summary()
        print(f"✅ Analysis summary: {summary.get('segments_detected', 0)} segments detected")
        
        # Cleanup
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)
            print("✅ Cleaned up test image")
        
        print("\n🎉 All tests passed! Enhanced Iris Analysis Service is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available.")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_dependencies():
    """Test that basic dependencies are available."""
    print("Testing basic dependencies...")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not available")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow available")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib available")
    except ImportError:
        print("❌ Matplotlib not available")
        return False
    
    print("✅ All basic dependencies are available!")
    return True

if __name__ == "__main__":
    print("Enhanced Iris Analysis Service - Integration Test")
    print("=" * 50)
    
    # Test dependencies first
    if not test_basic_dependencies():
        print("❌ Basic dependencies test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Test the enhanced analysis service
    if test_enhanced_iris_analysis():
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
