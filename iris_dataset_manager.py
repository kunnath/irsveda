"""
Iris Dataset Manager for IridoVeda.

This module provides functionality for managing and processing iris datasets
for pattern matching and model training.
"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
import requests
from tqdm import tqdm
import zipfile
import tarfile
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisDatasetManager:
    """
    Class for managing iris datasets, including downloading, preprocessing,
    and organizing for training and pattern matching.
    """
    
    DATASET_INFO = {
        "casia_thousand": {
            "name": "CASIA-Iris-Thousand",
            "description": "A dataset containing 20,000 iris images from 1,000 subjects",
            "url": "https://download-link-placeholder/casia-thousand.zip",
            "citation": "CASIA Iris Image Database, http://biometrics.idealtest.org/"
        },
        "nd_iris_0405": {
            "name": "ND-IRIS-0405",
            "description": "A dataset with 64,980 iris images from 356 subjects",
            "url": "https://download-link-placeholder/nd-iris-0405.zip",
            "citation": "Computer Vision Research Lab, University of Notre Dame"
        },
        "ubiris_v2": {
            "name": "UBIRIS.v2",
            "description": "A dataset with 11,102 images from 261 subjects in non-ideal conditions",
            "url": "https://download-link-placeholder/ubiris-v2.zip",
            "citation": "UBIRIS: A noisy iris image database, http://iris.di.ubi.pt"
        }
    }
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the dataset manager.
        
        Args:
            base_dir: Base directory for storing datasets
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create a dataframe to track metadata
        self.metadata_file = os.path.join(self.base_dir, "dataset_metadata.csv")
        self.load_or_create_metadata()
    
    def load_or_create_metadata(self):
        """Load existing metadata or create a new dataframe."""
        if os.path.exists(self.metadata_file):
            self.metadata = pd.read_csv(self.metadata_file)
        else:
            self.metadata = pd.DataFrame(columns=[
                "image_id", "dataset", "subject_id", "eye", "session", 
                "filename", "path", "processed", "segmented",
                "iris_center_x", "iris_center_y", "iris_radius",
                "pupil_center_x", "pupil_center_y", "pupil_radius"
            ])
            self.save_metadata()
    
    def save_metadata(self):
        """Save metadata to CSV."""
        self.metadata.to_csv(self.metadata_file, index=False)
    
    def download_dataset(self, dataset_key: str):
        """
        Download a dataset if not already present.
        
        Args:
            dataset_key: Key of the dataset to download ("casia_thousand", "nd_iris_0405", "ubiris_v2")
            
        Returns:
            Path to the dataset
        """
        if dataset_key not in self.DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        dataset_info = self.DATASET_INFO[dataset_key]
        dataset_dir = os.path.join(self.base_dir, dataset_key)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # For actual implementation, you would download the dataset here
        # This would require proper authorization and authentication for these datasets
        logger.info(f"Note: To use {dataset_info['name']}, you need to obtain it from the official source.")
        logger.info(f"Citation: {dataset_info['citation']}")
        
        # Create a placeholder file to indicate dataset information
        info_file = os.path.join(dataset_dir, "dataset_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Dataset: {dataset_info['name']}\n")
            f.write(f"Description: {dataset_info['description']}\n")
            f.write(f"Citation: {dataset_info['citation']}\n")
            f.write("\nNote: This is a placeholder. The actual dataset needs to be obtained from the official source.\n")
            
        return dataset_dir
    
    def process_casia_thousand(self, dataset_dir: str):
        """
        Process the CASIA-Iris-Thousand dataset.
        
        Args:
            dataset_dir: Path to the extracted dataset
        """
        logger.info("Processing CASIA-Iris-Thousand dataset...")
        
        # Check if the directory exists and has content
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory {dataset_dir} does not exist.")
            return
            
        # Get list of subject directories (S0001, S0002, etc.)
        subject_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('S') and os.path.isdir(os.path.join(dataset_dir, d))]
        
        if not subject_dirs:
            logger.warning(f"No subject directories found in {dataset_dir}")
            
            # Still create a sample entry in the metadata for testing
            sample_entry = {
                "image_id": "S1001_L1_001",
                "dataset": "casia_thousand",
                "subject_id": "1001",
                "eye": "left",
                "session": 1,
                "filename": "S1001_L1_001.jpg",
                "path": os.path.join(dataset_dir, "S1001/L1/S1001_L1_001.jpg"),
                "processed": False,
                "segmented": False,
                "iris_center_x": None,
                "iris_center_y": None,
                "iris_radius": None,
                "pupil_center_x": None,
                "pupil_center_y": None,
                "pupil_radius": None
            }
            self.metadata = pd.concat([self.metadata, pd.DataFrame([sample_entry])], ignore_index=True)
            self.save_metadata()
            return
            
        logger.info(f"Found {len(subject_dirs)} subject directories.")
        
        # Process each subject directory
        new_entries = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir[1:]  # Remove 'S' prefix
            subject_path = os.path.join(dataset_dir, subject_dir)
            
            # Check for eye directories (L1, R1, etc.)
            eye_dirs = [d for d in os.listdir(subject_path) if (d.startswith('L') or d.startswith('R')) and os.path.isdir(os.path.join(subject_path, d))]
            
            for eye_dir in eye_dirs:
                eye = 'left' if eye_dir.startswith('L') else 'right'
                eye_path = os.path.join(subject_path, eye_dir)
                
                # Process each image in the eye directory
                img_files = [f for f in os.listdir(eye_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in img_files:
                    img_path = os.path.join(eye_path, img_file)
                    image_id = img_file.split('.')[0]
                    
                    # Check if this image is already in metadata
                    if len(self.metadata[self.metadata["image_id"] == image_id]) == 0:
                        # Add to metadata
                        new_entries.append({
                            "image_id": image_id,
                            "dataset": "casia_thousand",
                            "subject_id": subject_id,
                            "eye": eye,
                            "session": 1,  # Default for CASIA
                            "filename": img_file,
                            "path": img_path,
                            "processed": False,
                            "segmented": False,
                            "iris_center_x": None,
                            "iris_center_y": None,
                            "iris_radius": None,
                            "pupil_center_x": None,
                            "pupil_center_y": None,
                            "pupil_radius": None
                        })
        
        # Add all new entries to metadata
        if new_entries:
            self.metadata = pd.concat([self.metadata, pd.DataFrame(new_entries)], ignore_index=True)
            logger.info(f"Added {len(new_entries)} new entries to metadata.")
            self.save_metadata()
        else:
            logger.info("No new entries to add to metadata.")
        
    def segment_dataset(self, dataset_key: str, limit: int = None):
        """
        Segment all images in a dataset using our iris segmentation.
        
        Args:
            dataset_key: Key of the dataset to segment
            limit: Optional limit on number of images to process
        """
        from iris_advanced_segmentation import preprocess_image, segment_iris
        
        # Filter metadata for this dataset and unsegmented images
        dataset_images = self.metadata[(self.metadata["dataset"] == dataset_key) & 
                                      (~self.metadata["segmented"])]
        
        if limit:
            dataset_images = dataset_images.head(limit)
        
        logger.info(f"Segmenting {len(dataset_images)} images from {dataset_key}...")
        
        for idx, row in tqdm(dataset_images.iterrows(), total=len(dataset_images)):
            try:
                # Skip if path doesn't exist (as it will for our placeholder)
                if not os.path.exists(row["path"]):
                    logger.warning(f"File not found: {row['path']} (Using placeholder data)")
                    
                    # For placeholder, update with dummy data
                    self.metadata.loc[idx, "segmented"] = True
                    self.metadata.loc[idx, "iris_center_x"] = 320
                    self.metadata.loc[idx, "iris_center_y"] = 240
                    self.metadata.loc[idx, "iris_radius"] = 120
                    self.metadata.loc[idx, "pupil_center_x"] = 320
                    self.metadata.loc[idx, "pupil_center_y"] = 240
                    self.metadata.loc[idx, "pupil_radius"] = 40
                    continue
                
                # Real processing would happen here
                image = cv2.imread(row["path"])
                preprocessed, _, _ = preprocess_image(image)
                segmentation_data = segment_iris(preprocessed)
                
                if segmentation_data:
                    iris_center = segmentation_data["iris_center"]
                    pupil_center = segmentation_data["pupil_center"]
                    
                    # Update metadata
                    self.metadata.loc[idx, "segmented"] = True
                    self.metadata.loc[idx, "iris_center_x"] = iris_center[0]
                    self.metadata.loc[idx, "iris_center_y"] = iris_center[1]
                    self.metadata.loc[idx, "iris_radius"] = segmentation_data["iris_radius"]
                    
                    if pupil_center:
                        self.metadata.loc[idx, "pupil_center_x"] = pupil_center[0]
                        self.metadata.loc[idx, "pupil_center_y"] = pupil_center[1]
                        self.metadata.loc[idx, "pupil_radius"] = segmentation_data["pupil_radius"]
            
            except Exception as e:
                logger.error(f"Error segmenting {row['path']}: {str(e)}")
        
        self.save_metadata()
        logger.info(f"Segmentation complete. {len(dataset_images)} images processed.")
    
    def extract_features(self, dataset_key: str, limit: int = None):
        """
        Extract features from segmented images.
        
        Args:
            dataset_key: Key of the dataset to process
            limit: Optional limit on number of images to process
            
        Returns:
            Dictionary of features by image_id
        """
        from iris_feature_extractor import extract_all_features
        
        # Filter metadata for this dataset and segmented images
        dataset_images = self.metadata[(self.metadata["dataset"] == dataset_key) & 
                                      (self.metadata["segmented"])]
        
        if limit:
            dataset_images = dataset_images.head(limit)
        
        logger.info(f"Extracting features from {len(dataset_images)} images...")
        
        features_by_id = {}
        
        for _, row in tqdm(dataset_images.iterrows(), total=len(dataset_images)):
            try:
                # Skip if path doesn't exist (as it will for our placeholder)
                if not os.path.exists(row["path"]):
                    logger.warning(f"File not found: {row['path']} (Using placeholder data)")
                    
                    # Generate dummy features for development purposes
                    features_by_id[row["image_id"]] = {
                        "color_features": [{"color": [100, 120, 140], "percentage": 0.6}],
                        "texture_features": {"contrast": 0.7, "uniformity": 0.5, "entropy": 1.2},
                        "num_spots": 5
                    }
                    continue
                
                # Real processing would happen here
                image = cv2.imread(row["path"])
                
                # Create segmentation data from stored metadata
                segmentation_data = {
                    "iris_center": (row["iris_center_x"], row["iris_center_y"]),
                    "iris_radius": row["iris_radius"],
                    "pupil_center": (row["pupil_center_x"], row["pupil_center_y"]) if pd.notna(row["pupil_center_x"]) else None,
                    "pupil_radius": row["pupil_radius"] if pd.notna(row["pupil_radius"]) else None,
                    
                    # Create masks (these would be generated properly in real processing)
                    "iris_mask": np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
                    "pupil_mask": np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                }
                
                # Draw the masks
                cv2.circle(segmentation_data["iris_mask"], 
                           (int(segmentation_data["iris_center"][0]), int(segmentation_data["iris_center"][1])), 
                           int(segmentation_data["iris_radius"]), 255, -1)
                
                if segmentation_data["pupil_center"] and segmentation_data["pupil_radius"]:
                    cv2.circle(segmentation_data["pupil_mask"],
                              (int(segmentation_data["pupil_center"][0]), int(segmentation_data["pupil_center"][1])),
                              int(segmentation_data["pupil_radius"]), 255, -1)
                
                # Extract features
                features = extract_all_features(image, segmentation_data)
                features_by_id[row["image_id"]] = features
                
            except Exception as e:
                logger.error(f"Error extracting features from {row['path']}: {str(e)}")
        
        logger.info(f"Feature extraction complete. {len(features_by_id)} images processed.")
        return features_by_id
        
    def import_to_qdrant(self, features_by_id: Dict[str, Dict], collection_name: str = "iris_dataset"):
        """
        Import dataset features to Qdrant for pattern matching.
        
        Args:
            features_by_id: Dictionary of features by image_id
            collection_name: Name of the Qdrant collection
        """
        from iris_pattern_matcher import IrisPatternMatcher
        
        logger.info(f"Importing {len(features_by_id)} feature sets to Qdrant...")
        
        # Initialize pattern matcher
        pattern_matcher = IrisPatternMatcher(collection_name=collection_name)
        pattern_matcher.create_collection()
        
        # Track success/failure
        success_count = 0
        failure_count = 0
        
        # Import each feature set
        for image_id, features in tqdm(features_by_id.items()):
            try:
                # Get metadata for this image
                image_metadata = self.metadata[self.metadata["image_id"] == image_id]
                
                if len(image_metadata) == 0:
                    logger.warning(f"No metadata found for {image_id}")
                    continue
                
                metadata_row = image_metadata.iloc[0]
                
                # Create payload
                payload = {
                    "image_id": image_id,
                    "dataset": metadata_row["dataset"],
                    "subject_id": metadata_row["subject_id"],
                    "eye": metadata_row["eye"],
                    "session": int(metadata_row["session"]),
                    "filename": metadata_row["filename"],
                }
                
                # Store in Qdrant
                point_id = pattern_matcher.store_iris_pattern(features, payload)
                
                if point_id:
                    success_count += 1
                else:
                    failure_count += 1
                    logger.warning(f"Failed to store {image_id} in Qdrant")
                    
            except Exception as e:
                logger.error(f"Error importing {image_id} to Qdrant: {str(e)}")
                failure_count += 1
        
        logger.info(f"Import complete. Success: {success_count}, Failures: {failure_count}")

# Utility function for the entire process
def setup_dataset(dataset_key: str = "casia_thousand", limit: int = 10):
    """
    Set up a dataset for use with IridoVeda.
    
    Args:
        dataset_key: Key of the dataset to set up
        limit: Maximum number of images to process
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize manager
        manager = IrisDatasetManager()
        
        # Download/prepare dataset
        dataset_dir = manager.download_dataset(dataset_key)
        
        # Process specific dataset
        if dataset_key == "casia_thousand":
            manager.process_casia_thousand(dataset_dir)
        # Add other datasets as needed
        
        # Segment the dataset
        manager.segment_dataset(dataset_key, limit=limit)
        
        # Extract features
        features = manager.extract_features(dataset_key, limit=limit)
        
        # Import to Qdrant
        manager.import_to_qdrant(features, collection_name=f"iris_{dataset_key}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up dataset: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    setup_dataset(limit=5)  # Start with a small subset
