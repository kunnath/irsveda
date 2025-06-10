#!/usr/bin/env python3
"""
Dataset Matching Test Script

This script helps test and debug the iris dataset matching functionality by:
1. Creating a mock dataset if it doesn't exist
2. Testing the dataset pattern matching with a sample image
3. Diagnosing and fixing common issues
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_dataset(dataset_key, num_samples=5, force=False):
    """
    Create a mock dataset for testing purposes.
    
    Args:
        dataset_key: The dataset key (e.g., "casia_thousand")
        num_samples: Number of sample images to create
        force: Whether to force recreation of the dataset if it exists
    """
    # Define the base directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    dataset_dir = os.path.join(base_dir, dataset_key)
    
    # Check if dataset already exists
    if os.path.exists(dataset_dir) and os.listdir(dataset_dir) and not force:
        logger.info(f"Dataset {dataset_key} already exists. Use --force to recreate it.")
        return dataset_dir
    
    logger.info(f"Creating mock dataset in {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample images
    for i in range(num_samples):
        # Create a subject directory
        subject_id = f"S{i+1:04d}"
        subject_dir = os.path.join(dataset_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Create separate directories for left and right eyes
        for eye in ["L", "R"]:
            eye_dir = os.path.join(subject_dir, f"{eye}1")
            os.makedirs(eye_dir, exist_ok=True)
            
            # Create mock iris image (circle on white background)
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            
            # Draw iris
            cv2.circle(img, (200, 200), 150, (100, 150, 180), -1)
            
            # Draw pupil
            cv2.circle(img, (200, 200), 50, (30, 30, 30), -1)
            
            # Add some random texture and patterns
            for _ in range(20):
                x = np.random.randint(50, 350)
                y = np.random.randint(50, 350)
                r = np.random.randint(5, 20)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(img, (x, y), r, color, -1)
            
            # Save the image
            img_path = os.path.join(eye_dir, f"{subject_id}_{eye}1_001.jpg")
            cv2.imwrite(img_path, img)
            
            # Also save to subject dir for compatibility
            cv2.imwrite(os.path.join(subject_dir, f"{subject_id}_{eye}.jpg"), img)
            
            logger.info(f"Created mock image: {img_path}")
    
    # Create empty metadata.csv file
    metadata_path = os.path.join(base_dir, "dataset_metadata.csv")
    if not os.path.exists(metadata_path) or force:
        with open(metadata_path, "w") as f:
            f.write("image_id,dataset,subject_id,eye,session,filename,path,processed,segmented\n")
        logger.info(f"Created metadata file: {metadata_path}")
    
    logger.info(f"Mock dataset creation complete. Created {num_samples} subjects with L/R eyes.")
    return dataset_dir

def create_test_image():
    """Create a test iris image and save it to a temporary file."""
    # Create mock iris image (circle on white background)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw iris
    cv2.circle(img, (200, 200), 150, (100, 150, 180), -1)
    
    # Draw pupil
    cv2.circle(img, (200, 200), 50, (30, 30, 30), -1)
    
    # Add some random texture and patterns
    for _ in range(20):
        x = np.random.randint(50, 350)
        y = np.random.randint(50, 350)
        r = np.random.randint(5, 20)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img, (x, y), r, color, -1)
    
    # Save to temporary file
    fd, path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    cv2.imwrite(path, img)
    logger.info(f"Created test image: {path}")
    
    return path

def test_dataset_matcher(dataset_key, test_image_path=None):
    """Test the DatasetPatternMatcher with a test image."""
    try:
        # Import the DatasetPatternMatcher
        from iris_dataset_matcher import DatasetPatternMatcher
        
        # Create a test image if not provided
        if test_image_path is None:
            test_image_path = create_test_image()
        
        logger.info(f"Testing pattern matching with image: {test_image_path}")
        
        # Create the matcher and test it
        matcher = DatasetPatternMatcher(dataset_key=dataset_key)
        
        # Get dataset statistics
        try:
            stats = matcher.get_dataset_statistics()
            logger.info(f"Dataset statistics: {stats}")
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
        
        # Try matching with the test image
        try:
            results = matcher.match_iris_with_dataset(test_image_path)
            if "error" in results:
                logger.error(f"Matching error: {results['error']}")
            else:
                logger.info(f"Matching results: {results}")
        except Exception as e:
            logger.error(f"Error during matching: {str(e)}")
        
        # Clean up the test image if we created it
        if test_image_path is None:
            os.unlink(test_image_path)
            
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.info("Make sure the iris_dataset_matcher.py file exists and is properly implemented.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

def check_docker_service():
    """Check if Docker and Qdrant are running."""
    try:
        import subprocess
        
        # Check Docker
        logger.info("Checking Docker status...")
        docker_result = subprocess.run(["docker", "info"], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        if docker_result.returncode != 0:
            logger.warning("Docker doesn't appear to be running. Qdrant may not be available.")
            logger.info("Start Docker and then run: docker-compose up -d")
        else:
            logger.info("Docker is running.")
            
            # Check Qdrant container
            qdrant_result = subprocess.run(["docker", "ps", "-q", "--filter", "name=qdrant"],
                                          stdout=subprocess.PIPE,
                                          text=True)
            
            if not qdrant_result.stdout.strip():
                logger.warning("Qdrant container doesn't appear to be running.")
                logger.info("Start it with: docker-compose up -d")
            else:
                logger.info(f"Qdrant container seems to be running (ID: {qdrant_result.stdout.strip()})")
                
                # Check Qdrant health
                try:
                    import requests
                    response = requests.get("http://localhost:6333/livez")
                    if response.status_code == 200:
                        logger.info("Qdrant is healthy and responding")
                    else:
                        logger.warning(f"Qdrant health check returned status: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error checking Qdrant health: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error checking Docker services: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test and debug iris dataset matching")
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"],
                        default="casia_thousand", help="Dataset to test")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to create")
    parser.add_argument("--force", action="store_true", help="Force recreation of dataset")
    parser.add_argument("--test-image", help="Path to test image (creates one if not specified)")
    parser.add_argument("--check-services", action="store_true", help="Check Docker and Qdrant services")
    
    args = parser.parse_args()
    
    # First check services if requested
    if args.check_services:
        check_docker_service()
    
    # Create dataset
    create_mock_dataset(args.dataset, args.samples, args.force)
    
    # Test the matcher
    test_dataset_matcher(args.dataset, args.test_image)
    
    print("\nTo test the dataset matching in your application:")
    print("1. Make sure your application can access the dataset directory:")
    print(f"   ls -la datasets/{args.dataset}")
    print("\n2. If you're having issues, check that:")
    print("   - The dataset directory exists and contains subject folders with images")
    print("   - Docker and Qdrant are running (run with --check-services)")
    print("   - The metadata.csv file exists in the datasets directory")
    print("   - Your dataset_matcher.py properly handles missing or invalid datasets")
    print("\n3. Run the test with a specific image:")
    print(f"   python {sys.argv[0]} --test-image path/to/your/iris/image.jpg")

if __name__ == "__main__":
    main()
