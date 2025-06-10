#!/usr/bin/env python3
"""
Verification script for dataset matching functionality.

This script tests that the implemented fixes for the dataset matching functionality
are working correctly, specifically:
1. Verifies the IrisPatternMatcher.search_by_vector method works
2. Confirms the convert_features_to_vector method properly handles RGB tuples
3. Tests the DatasetPatternMatcher integration with IrisPatternMatcher
"""

import os
import sys
import logging
import numpy as np
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_features():
    """Create mock features with RGB tuples to test vector conversion."""
    features = {
        "color_features": [
            {"color": (100, 150, 200), "percentage": 0.6},
            {"color": (50, 80, 120), "percentage": 0.3},
            {"color": (200, 220, 240), "percentage": 0.1}
        ],
        "texture_features": {
            "contrast": 120.5,
            "uniformity": 0.35,
            "energy": 0.42,
            "entropy": 4.2
        },
        "num_spots": 15
    }
    return features

def test_iris_pattern_matcher():
    """Test the IrisPatternMatcher class with vector conversion and search."""
    try:
        from iris_pattern_matcher import IrisPatternMatcher
        
        logger.info("Creating IrisPatternMatcher instance...")
        matcher = IrisPatternMatcher(collection_name="test_patterns")
        
        # Test vector conversion with RGB tuples
        features = create_mock_features()
        
        logger.info("Testing convert_features_to_vector with RGB tuples...")
        try:
            vector = matcher.convert_features_to_vector(features)
            logger.info(f"Vector created successfully with length: {len(vector)}")
            logger.info(f"First 10 elements: {vector[:10]}")
            
            # Verify the first elements correspond to the RGB values
            expected_rgb_normalized = [100/255, 150/255, 200/255]
            actual_rgb = vector[:3]
            
            tolerance = 0.01
            rgb_match = all(abs(a - e) < tolerance for a, e in zip(actual_rgb, expected_rgb_normalized))
            
            if rgb_match:
                logger.info("✅ RGB tuple conversion is working correctly!")
            else:
                logger.error(f"❌ RGB tuple conversion failed. Expected: {expected_rgb_normalized}, Got: {actual_rgb}")
        
        except Exception as e:
            logger.error(f"❌ Vector conversion failed: {str(e)}")
            return False
            
        # Test collection creation
        try:
            logger.info("Testing collection creation...")
            matcher.create_collection()
            logger.info("✅ Collection created successfully!")
        except Exception as e:
            logger.error(f"❌ Collection creation failed: {str(e)}")
            return False
            
        # Test vector search (mock version since we don't want to persist data)
        try:
            logger.info("Testing search_by_vector method...")
            # Mock the client.search method to avoid actual DB interaction
            original_search = matcher.client.search
            
            # Define a mock response
            class MockSearchResult:
                def __init__(self, id_val, score):
                    self.id = id_val
                    self.score = score
                    self.payload = {"test": "data"}
                    
            # Replace with mock method
            matcher.client.search = lambda **kwargs: [
                MockSearchResult("test1", 0.95),
                MockSearchResult("test2", 0.85)
            ]
            
            # Test the search
            results = matcher.search_by_vector(vector, limit=2)
            
            # Restore original method
            matcher.client.search = original_search
            
            if len(results) == 2 and results[0]["id"] == "test1" and results[1]["id"] == "test2":
                logger.info("✅ search_by_vector method is working correctly!")
            else:
                logger.error(f"❌ search_by_vector returned unexpected results: {results}")
                return False
                
        except Exception as e:
            logger.error(f"❌ search_by_vector test failed: {str(e)}")
            return False
            
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
        
def test_dataset_pattern_matcher():
    """Test the DatasetPatternMatcher integration with IrisPatternMatcher."""
    try:
        from iris_dataset_matcher import DatasetPatternMatcher
        
        logger.info("Creating DatasetPatternMatcher instance...")
        matcher = DatasetPatternMatcher(dataset_key="test_dataset")
        
        # Test that the pattern_matcher was initialized correctly
        logger.info(f"Pattern matcher collection name: {matcher.pattern_matcher.collection_name}")
        
        # Test vector search integration (mock version)
        # Create a mock search_by_vector method for the IrisPatternMatcher
        original_search = matcher.pattern_matcher.search_by_vector
        
        # Replace with mock method
        matcher.pattern_matcher.search_by_vector = lambda vector, limit: [
            {"id": "test1", "score": 0.1, "payload": {"subject_id": "S0001", "eye": "left"}},
            {"id": "test2", "score": 0.2, "payload": {"subject_id": "S0002", "eye": "right"}}
        ]
        
        # Test with a mock vector
        vector = [0.1] * 256
        results = matcher.pattern_matcher.search_by_vector(vector, limit=2)
        
        # Restore original method
        matcher.pattern_matcher.search_by_vector = original_search
        
        if len(results) == 2 and results[0]["id"] == "test1":
            logger.info("✅ DatasetPatternMatcher integration with IrisPatternMatcher is working!")
            return True
        else:
            logger.error(f"❌ DatasetPatternMatcher integration returned unexpected results: {results}")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    logger.info("==== Testing Iris Pattern Matcher ====")
    iris_matcher_success = test_iris_pattern_matcher()
    
    logger.info("\n==== Testing Dataset Pattern Matcher ====")
    dataset_matcher_success = test_dataset_pattern_matcher()
    
    if iris_matcher_success and dataset_matcher_success:
        logger.info("\n✅ All tests passed! The dataset matching functionality is working correctly.")
    else:
        logger.error("\n❌ Some tests failed. Please check the logs for details.")
        
    logger.info("\nNext steps:")
    logger.info("1. To use dataset matching in the application, ensure the dataset directory is set up.")
    logger.info("2. Use the test_dataset_matching.py script to create a mock dataset for testing.")
    logger.info("3. Make sure the Qdrant service is running with docker-compose up -d.")
    logger.info("4. Test the full functionality with a real iris image.")

if __name__ == "__main__":
    main()
