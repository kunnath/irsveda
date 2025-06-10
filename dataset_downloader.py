
"""
Dataset Downloader for Iris Research Datasets

This script enables downloading and preparing iris research datasets for pattern matching.
"""

import os
import argparse
import requests
from tqdm import tqdm
import zipfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset download URLs - Replace with actual URLs when available
DATASET_URLS = {
    "casia_thousand": "https://example.com/datasets/casia-thousand.zip",
    "nd_iris_0405": "https://example.com/datasets/nd-iris-0405.zip",
    "ubiris_v2": "https://example.com/datasets/ubiris-v2.zip"
}

def download_file(url, destination):
    """
    Download a file with a progress bar.
    
    Args:
        url: The URL to download from
        destination: Where to save the file
    """
    if os.path.exists(destination):
        logger.info(f"File already exists at {destination}")
        return
    
    logger.info(f"Downloading from {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    
    logger.info(f"Download complete: {destination}")

def extract_archive(archive_path, extract_to):
    """
    Extract a zip archive.
    
    Args:
        archive_path: Path to the zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_to}")
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract the archive
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, extract_to)
    
    logger.info(f"Extraction complete: {extract_to}")

def setup_dataset(dataset_key, base_dir=None):
    """
    Set up a dataset for use.
    
    Args:
        dataset_key: Key of the dataset to setup
        base_dir: Base directory for datasets
    """
    from iris_dataset_manager import IrisDatasetManager
    
    # Initialize the dataset manager
    manager = IrisDatasetManager(base_dir)
    
    # Get dataset path
    dataset_dir = os.path.join(manager.base_dir, dataset_key)
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        logger.error(f"Dataset {dataset_key} not found or empty.")
        logger.info(f"Please obtain the dataset from official sources and place it in {dataset_dir}")
        logger.info("For more information, see the README.md in the datasets directory.")
        return False
    
    try:
        # Process the dataset with the setup_dataset function
        logger.info(f"Processing dataset {dataset_key}...")
        # Use the module-level setup_dataset function
        from iris_dataset_manager import setup_dataset as setup_iris_dataset
        setup_iris_dataset(dataset_key=dataset_key, limit=100)  # Limit to 100 samples for faster processing
        logger.info(f"Dataset {dataset_key} setup complete.")
        return True
    except Exception as e:
        logger.error(f"Error setting up dataset {dataset_key}: {str(e)}")
        return False

def simulate_dataset(dataset_key, base_dir=None, num_samples=10):
    """
    Simulate a dataset for testing (creates placeholder files).
    
    Args:
        dataset_key: Key of the dataset to simulate
        base_dir: Base directory for datasets
        num_samples: Number of sample images to create
    """
    from iris_dataset_manager import IrisDatasetManager
    import numpy as np
    import cv2
    
    # Initialize the dataset manager
    manager = IrisDatasetManager(base_dir)
    
    # Get dataset path
    dataset_dir = os.path.join(manager.base_dir, dataset_key)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample subjects and eyes
    logger.info(f"Creating simulated dataset {dataset_key} with {num_samples} samples")
    
    for i in range(num_samples):
        # Create random subject ID
        subject_id = f"S{i+1:04d}"
        subject_dir = os.path.join(dataset_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Create left and right eye images
        for eye in ["L", "R"]:
            # Create a simulated iris image (black circle on white background)
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
            img_path = os.path.join(subject_dir, f"{subject_id}_{eye}.jpg")
            cv2.imwrite(img_path, img)
    
    logger.info(f"Created {num_samples} simulated iris samples in {dataset_dir}")
    
    # Update metadata
    try:
        # Use the module-level setup_dataset function
        from iris_dataset_manager import setup_dataset as setup_iris_dataset
        setup_iris_dataset(dataset_key=dataset_key, limit=num_samples)
        logger.info(f"Simulated dataset {dataset_key} setup complete")
        return True
    except Exception as e:
        logger.error(f"Error setting up simulated dataset: {str(e)}")
        return False

def main():
    """Main function for command-line use."""
    parser = argparse.ArgumentParser(description="Download and prepare iris research datasets")
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2", "all"],
                        default="casia_thousand", help="Dataset to download and prepare")
    parser.add_argument("--data-dir", help="Override the default data directory")
    parser.add_argument("--simulate", action="store_true", 
                        help="Create a simulated dataset for testing")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples to generate for simulation")
    
    args = parser.parse_args()
    
    datasets_to_process = ["casia_thousand", "nd_iris_0405", "ubiris_v2"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets_to_process:
        if args.simulate:
            simulate_dataset(dataset, args.data_dir, args.samples)
        else:
            setup_dataset(dataset, args.data_dir)

if __name__ == "__main__":
    main()