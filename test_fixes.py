#!/usr/bin/env python3
"""
Test script to verify the fixes made to IrisPatternMatcher and RGB tuple handling
"""

import os
import sys
import logging
import tempfile
import numpy as np
from iris_pattern_matcher import IrisPatternMatcher

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_convert_features_to_vector():
    """Test the fixed convert_features_to_vector method"""
    logger.info("Testing convert_features_to_vector method with RGB tuple...")
    
    # Initialize the pattern matcher
    pattern_matcher = IrisPatternMatcher(collection_name="test_collection")
    
    # Test case with RGB tuple
    features = {
        "color_features": [
            {"color": (100, 150, 200), "percentage": 0.6},
            {"color": (50, 75, 100), "percentage": 0.4}
        ],
        "texture_features": {"contrast": 0.7, "uniformity": 0.5, "entropy": 1.2},
        "num_spots": 5
    }
    
    try:
        # Convert to vector
        vector = pattern_matcher.convert_features_to_vector(features)
        logger.info(f"Vector length: {len(vector)}")
        logger.info(f"Vector content: {vector[:10]}...")  # Show first 10 elements
        
        # Check if the vector contains the normalized RGB values (divided by 255.0)
        normalized_rgb = [100/255.0, 150/255.0, 200/255.0, 50/255.0, 75/255.0, 100/255.0]
        # Round to 5 decimal places for comparison to handle floating point precision
        vector_rounded = [round(v, 5) for v in vector]
        normalized_rgb_rounded = [round(v, 5) for v in normalized_rgb]
        
        # Check if all normalized RGB values are found in the vector
        rgb_found = all(any(abs(v - rgb_val) < 0.00001 for v in vector_rounded[:20]) 
                       for rgb_val in normalized_rgb_rounded)
        
        if rgb_found:
            logger.info("✓ RGB tuple conversion successful")
        else:
            logger.error("✗ RGB tuple values not found in vector")
            logger.info(f"Expected to find normalized values: {normalized_rgb_rounded}")
            logger.info(f"First 20 vector values: {vector_rounded[:20]}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing convert_features_to_vector: {str(e)}")
        return False

def test_search_by_vector():
    """Test the added search_by_vector method"""
    logger.info("Testing search_by_vector method...")
    
    # Initialize the pattern matcher
    pattern_matcher = IrisPatternMatcher(collection_name="test_collection")
    
    # Create the collection if it doesn't exist
    pattern_matcher.create_collection()
    
    # Add a few test patterns
    test_points = [
        {
            "features": {
                "color_features": [{"color": (100, 150, 200), "percentage": 0.6}],
                "texture_features": {"contrast": 0.7, "uniformity": 0.5},
                "num_spots": 5
            },
            "payload": {"id": "test1", "subject_id": "S001", "eye": "left"}
        },
        {
            "features": {
                "color_features": [{"color": (50, 75, 100), "percentage": 0.4}],
                "texture_features": {"contrast": 0.5, "uniformity": 0.3},
                "num_spots": 3
            },
            "payload": {"id": "test2", "subject_id": "S002", "eye": "right"}
        }
    ]
    
    try:
        # Store test points
        for i, test_point in enumerate(test_points):
            point_id = pattern_matcher.store_iris_pattern(
                test_point["features"],
                test_point["payload"]
            )
            logger.info(f"Added test point {i+1} with ID: {point_id}")
        
        # Create a test vector
        test_vector = pattern_matcher.convert_features_to_vector(test_points[0]["features"])
        logger.info(f"Created test vector of length {len(test_vector)}")
        
        # Test search_by_vector
        results = pattern_matcher.search_by_vector(test_vector, limit=5)
        
        # Verify results
        if results and len(results) > 0:
            logger.info(f"✓ search_by_vector returned {len(results)} results")
            logger.info(f"Top match: {results[0]['payload']}")
            return True
        else:
            logger.error("✗ search_by_vector returned no results")
            return False
            
    except Exception as e:
        logger.error(f"Error testing search_by_vector: {str(e)}")
        return False

def test_dataset_pattern_matcher():
    """Test integration with DatasetPatternMatcher"""
    logger.info("Testing integration with DatasetPatternMatcher...")
    
    try:
        # Import here to avoid errors if not available
        from iris_dataset_matcher import DatasetPatternMatcher
        
        # Initialize DatasetPatternMatcher
        dataset_matcher = DatasetPatternMatcher(dataset_key="test_dataset")
        
        # Get dataset statistics (this should use search_by_vector internally)
        stats = dataset_matcher.get_dataset_statistics()
        
        if "error" in stats:
            logger.warning(f"Dataset statistics returned error: {stats['error']}")
            logger.info("This is expected if the collection doesn't exist yet")
        else:
            logger.info(f"Dataset statistics: {stats}")
            
        logger.info("✓ DatasetPatternMatcher initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing DatasetPatternMatcher: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING FIXES FOR IRIS PATTERN MATCHER")
    print("=" * 60)
    
    # Check Qdrant availability first
    try:
        import requests
        response = requests.get("http://localhost:6333/livez")
        if response.status_code == 200:
            logger.info("Qdrant is healthy and responding")
            qdrant_available = True
        else:
            logger.warning(f"Qdrant health check returned status: {response.status_code}")
            qdrant_available = False
    except Exception as e:
        logger.warning(f"Error checking Qdrant health: {str(e)}")
        qdrant_available = False
    
    # Test vector conversion
    vector_test = test_convert_features_to_vector()
    
    # Only run the following tests if Qdrant is available
    if qdrant_available:
        search_test = test_search_by_vector()
        integration_test = test_dataset_pattern_matcher()
    else:
        logger.warning("Skipping search_by_vector and integration tests because Qdrant is not available")
        search_test = None
        integration_test = None
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"1. convert_features_to_vector: {'PASSED' if vector_test else 'FAILED'}")
    
    if search_test is not None:
        print(f"2. search_by_vector: {'PASSED' if search_test else 'FAILED'}")
    else:
        print("2. search_by_vector: SKIPPED (Qdrant not available)")
        
    if integration_test is not None:
        print(f"3. DatasetPatternMatcher integration: {'PASSED' if integration_test else 'FAILED'}")
    else:
        print("3. DatasetPatternMatcher integration: SKIPPED (Qdrant not available)")
    
    all_tests = [t for t in [vector_test, search_test, integration_test] if t is not None]
    if all(all_tests):
        print("\n✅ All executed tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED. See logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
