#!/bin/bash
# Setup script for iris dataset integration

echo "IridoVeda - Iris Dataset Integration Setup"
echo "=========================================="

# Check requirements
echo "Checking requirements..."

# Check Python version
python_version=$(python3 --version | cut -d ' ' -f 2)
echo "Python version: $python_version"

# Check if key packages are installed
echo "Checking required packages..."
pip3 install -q numpy scikit-learn opencv-python qdrant-client tqdm

# Create dataset directory
echo "Creating dataset directory structure..."
mkdir -p datasets/casia_thousand
mkdir -p datasets/nd_iris_0405
mkdir -p datasets/ubiris_v2

# Create dataset info files
echo "Creating dataset information files..."

cat > datasets/casia_thousand/README.md << EOL
# CASIA-Iris-Thousand Dataset

## Overview
The CASIA-Iris-Thousand dataset contains 20,000 iris images from 1,000 subjects (both left and right eyes).

## How to Obtain
This dataset needs to be obtained from the official source:
http://biometrics.idealtest.org/

## Citation
CASIA Iris Image Database, http://biometrics.idealtest.org/
EOL

cat > datasets/nd_iris_0405/README.md << EOL
# ND-IRIS-0405 Dataset

## Overview
The ND-IRIS-0405 dataset contains 64,980 iris images from 356 subjects collected over multiple sessions.

## How to Obtain
This dataset needs to be obtained from the official source:
https://cvrl.nd.edu/projects/data/

## Citation
Computer Vision Research Lab, University of Notre Dame
EOL

cat > datasets/ubiris_v2/README.md << EOL
# UBIRIS.v2 Dataset

## Overview
The UBIRIS.v2 dataset contains 11,102 images from 261 subjects in non-ideal conditions, captured at visible wavelengths.

## How to Obtain
This dataset needs to be obtained from the official source:
http://iris.di.ubi.pt

## Citation
UBIRIS: A noisy iris image database, http://iris.di.ubi.pt
EOL

# Create sample data for testing
echo "Creating sample data for testing..."

# Run the dataset setup script in test mode
echo "Running dataset manager test..."
python3 -c "from iris_dataset_manager import setup_dataset; setup_dataset('casia_thousand', limit=5)"

# Create mock training script
cat > train_iris_dataset.py << EOL
#!/usr/bin/env python3

import os
import argparse
import logging
from iris_dataset_manager import IrisDatasetManager
from iris_dataset_matcher import DatasetPatternMatcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Iris Pattern Matcher Training Script")
    
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"], 
                       default="casia_thousand", help="Dataset to use")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to process (None for all)")
    parser.add_argument("--download", action="store_true", help="Download/prepare the dataset")
    parser.add_argument("--process", action="store_true", help="Process the dataset")
    parser.add_argument("--features", action="store_true", help="Extract features from the dataset")
    parser.add_argument("--import_to_qdrant", action="store_true", help="Import features to Qdrant")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    # Initialize the dataset manager
    dataset_manager = IrisDatasetManager()
    
    # Initialize the matcher
    matcher = DatasetPatternMatcher(dataset_key=args.dataset)
    
    # Run requested steps
    if args.all or args.download:
        logger.info(f"Downloading/preparing {args.dataset} dataset")
        dataset_dir = dataset_manager.download_dataset(args.dataset)
        logger.info(f"Dataset prepared in {dataset_dir}")
    
    if args.all or args.process:
        logger.info(f"Processing {args.dataset} dataset")
        if args.dataset == "casia_thousand":
            dataset_manager.process_casia_thousand(os.path.join(dataset_manager.base_dir, args.dataset))
        # Add processing for other datasets as needed
        logger.info(f"Dataset processed")
    
    if args.all or args.process:
        logger.info(f"Segmenting {args.dataset} dataset (limit: {args.limit})")
        dataset_manager.segment_dataset(args.dataset, limit=args.limit)
        logger.info(f"Dataset segmentation complete")
    
    if args.all or args.features:
        logger.info(f"Extracting features from {args.dataset} dataset (limit: {args.limit})")
        features = dataset_manager.extract_features(args.dataset, limit=args.limit)
        logger.info(f"Feature extraction complete. {len(features)} images processed")
    
        if args.all or args.import_to_qdrant:
            logger.info(f"Importing features to Qdrant")
            dataset_manager.import_to_qdrant(features, collection_name=f"iris_{args.dataset}")
            logger.info(f"Import to Qdrant complete")
    
    if args.all or args.stats:
        logger.info(f"Getting statistics for {args.dataset}")
        stats = matcher.get_dataset_statistics()
        
        if "error" in stats:
            logger.error(f"Statistics error: {stats['error']}")
        else:
            logger.info(f"Dataset: {stats.get('dataset_name', args.dataset)}")
            logger.info(f"Patterns in Qdrant: {stats.get('total_patterns', 0)}")
            logger.info(f"Images in metadata: {stats.get('total_images', 0)}")
            logger.info(f"Subjects: {stats.get('subjects', 0)}")
            logger.info(f"Left eyes: {stats.get('left_eyes', 0)}")
            logger.info(f"Right eyes: {stats.get('right_eyes', 0)}")
            logger.info(f"Segmented: {stats.get('segmented_images', 0)}")

if __name__ == "__main__":
    main()
EOL

chmod +x train_iris_dataset.py

echo "Setup complete!"
echo
echo "To integrate a dataset:"
echo "1. Obtain the dataset from the official source"
echo "2. Place it in the appropriate directory under datasets/"
echo "3. Run ./train_iris_dataset.py --dataset casia_thousand --all"
echo 
echo "For more options, run ./train_iris_dataset.py --help"
