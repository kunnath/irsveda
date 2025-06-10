#!/usr/bin/env python3
"""
Dataset Setup Verification and Fix Script

This script helps verify and fix the iris dataset setup by:
1. Checking if the datasets directory and metadata.csv exist
2. Ensuring mock dataset is properly created
3. Populating metadata.csv with existing dataset information
4. Verifying Qdrant connection
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_dataset_directory(dataset_key="casia_thousand", create=False):
    """
    Verify that the dataset directory exists and has expected structure.
    
    Args:
        dataset_key: The dataset key (e.g., "casia_thousand")
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Tuple of (exists, path)
    """
    # Define the base directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    dataset_dir = os.path.join(base_dir, dataset_key)
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        logger.warning(f"Datasets directory doesn't exist: {base_dir}")
        if create:
            logger.info(f"Creating datasets directory: {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        logger.warning(f"Dataset directory doesn't exist: {dataset_dir}")
        if create:
            logger.info(f"Creating dataset directory: {dataset_dir}")
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create info file
            info_file = os.path.join(dataset_dir, "dataset_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"Dataset: CASIA-Iris-Thousand (Mock)\n")
                f.write(f"Description: This is a mock dataset for testing purposes.\n")
                f.write(f"Created by: verify_dataset_setup.py\n")
                
            return False, dataset_dir
        else:
            return False, dataset_dir
    
    # Check if it has content
    contents = os.listdir(dataset_dir)
    if not contents:
        logger.warning(f"Dataset directory is empty: {dataset_dir}")
        return False, dataset_dir
    
    logger.info(f"Dataset directory exists and contains {len(contents)} items: {dataset_dir}")
    return True, dataset_dir

def verify_metadata_csv(create=False):
    """
    Verify that the metadata.csv file exists and has expected structure.
    
    Args:
        create: Whether to create the file if it doesn't exist
        
    Returns:
        Tuple of (exists, path)
    """
    # Define the path
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    metadata_path = os.path.join(base_dir, "dataset_metadata.csv")
    
    # Check if file exists
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file doesn't exist: {metadata_path}")
        if create:
            logger.info(f"Creating metadata file: {metadata_path}")
            # Create with header
            with open(metadata_path, "w") as f:
                f.write("image_id,dataset,subject_id,eye,session,filename,path,processed,segmented,"
                        "iris_center_x,iris_center_y,iris_radius,"
                        "pupil_center_x,pupil_center_y,pupil_radius\n")
            return False, metadata_path
        else:
            return False, metadata_path
    
    # Check if it's valid
    try:
        df = pd.read_csv(metadata_path)
        logger.info(f"Metadata file exists with {len(df)} entries and {len(df.columns)} columns.")
        # Check for required columns
        required_columns = ["image_id", "dataset", "subject_id", "eye", "path"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Metadata file is missing required columns: {', '.join(missing)}")
            return False, metadata_path
        return True, metadata_path
    except Exception as e:
        logger.error(f"Error reading metadata file: {str(e)}")
        return False, metadata_path

def verify_qdrant_connection():
    """
    Verify that Qdrant is running and accessible.
    
    Returns:
        Boolean indicating if Qdrant is accessible
    """
    try:
        # Try to access Qdrant API
        response = requests.get("http://localhost:6333/livez", timeout=5)
        if response.status_code == 200:
            logger.info("Qdrant is healthy and responding")
            return True
        else:
            logger.warning(f"Qdrant health check returned status: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Error connecting to Qdrant: {str(e)}")
        logger.info("Make sure Docker is running and Qdrant container is started:")
        logger.info("docker-compose up -d")
        return False

def populate_metadata(dataset_key="casia_thousand"):
    """
    Populate metadata.csv with information from existing dataset files.
    
    Args:
        dataset_key: The dataset key (e.g., "casia_thousand")
        
    Returns:
        Boolean indicating success
    """
    try:
        # Import IrisDatasetManager and process the dataset
        from iris_dataset_manager import IrisDatasetManager
        
        logger.info(f"Processing {dataset_key} dataset to populate metadata...")
        manager = IrisDatasetManager()
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", dataset_key)
        manager.process_casia_thousand(dataset_dir)
        
        logger.info(f"Metadata now contains {len(manager.metadata)} entries.")
        return True
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify and fix iris dataset setup")
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"],
                        default="casia_thousand", help="Dataset to verify")
    parser.add_argument("--fix", action="store_true", help="Fix issues automatically")
    parser.add_argument("--create-mock", action="store_true", help="Create a mock dataset if missing")
    
    args = parser.parse_args()
    
    # Check dataset directory
    dataset_exists, dataset_path = verify_dataset_directory(args.dataset, create=args.fix)
    
    # Check metadata.csv
    metadata_exists, metadata_path = verify_metadata_csv(create=args.fix)
    
    # Check Qdrant connection
    qdrant_ok = verify_qdrant_connection()
    
    if args.create_mock and not dataset_exists:
        logger.info("Creating mock dataset...")
        from test_dataset_matching import create_mock_dataset
        create_mock_dataset(args.dataset, num_samples=5, force=True)
        dataset_exists = True
        
    # Populate metadata if needed
    if dataset_exists and (args.fix or (not metadata_exists)):
        logger.info("Populating metadata...")
        populate_metadata(args.dataset)
    
    # Print summary
    print("\n=== Dataset Setup Summary ===")
    print(f"Dataset directory: {'✅ OK' if dataset_exists else '❌ Missing'} - {dataset_path}")
    print(f"Metadata file:     {'✅ OK' if metadata_exists else '❌ Missing'} - {metadata_path}")
    print(f"Qdrant connection: {'✅ OK' if qdrant_ok else '❌ Not Connected'} - http://localhost:6333")
    
    if not dataset_exists or not metadata_exists or not qdrant_ok:
        print("\n=== Fix Instructions ===")
        if not dataset_exists:
            print("1. Create a mock dataset for testing:")
            print(f"   python test_dataset_matching.py --dataset {args.dataset} --samples 5 --force")
        if not metadata_exists:
            print("2. Create and populate the metadata:")
            print(f"   python verify_dataset_setup.py --dataset {args.dataset} --fix")
        if not qdrant_ok:
            print("3. Start Qdrant using Docker Compose:")
            print("   docker-compose up -d")
        
    print("\n=== Next Steps ===")
    print("1. Test the dataset matching functionality:")
    print(f"   python test_dataset_matching.py --dataset {args.dataset}")
    print("\n2. Verify that the iris pattern matcher works in your application.")

if __name__ == "__main__":
    main()
