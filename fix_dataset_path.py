#!/usr/bin/env python3
"""
Fix for the missing dataset path issue and Qdrant serialization problems.

This script creates a simple mock dataset for testing purposes and properly
formats data for Qdrant to fix serialization issues with numpy types.
"""

import os
import argparse
import logging
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Any, Dict, Union, List
import uuid

# Try to import Qdrant client - this is optional for the script to work
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_values(obj):
    """
    Convert numpy values to Python native types for JSON serialization.
    
    Args:
        obj: The object containing numpy values
        
    Returns:
        Object with numpy values converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_values(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def create_mock_dataset(dataset_key, num_samples=5):
    """
    Create a mock dataset for testing purposes.
    
    Args:
        dataset_key: The dataset key (e.g., "casia_thousand")
        num_samples: Number of sample images to create
    """
    # Define the base directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    dataset_dir = os.path.join(base_dir, dataset_key)
    
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
            
            logger.info(f"Created mock image: {img_path}")
    
    # Create empty metadata.csv file
    metadata_path = os.path.join(base_dir, "dataset_metadata.csv")
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w") as f:
            f.write("image_id,dataset,subject_id,eye,session,filename,path,processed,segmented\n")
        logger.info(f"Created metadata file: {metadata_path}")
    
    logger.info(f"Mock dataset creation complete. Created {num_samples} subjects with L/R eyes.")
    return dataset_dir

def create_qdrant_collection(dataset_key):
    """
    Create a Qdrant collection for the dataset.
    
    Args:
        dataset_key: The dataset key (e.g., "casia_thousand")
    
    Returns:
        True if successful, False otherwise
    """
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant client is not available. Install with: pip install qdrant-client")
        return False
    
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        collection_name = f"iris_{dataset_key}"
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
            return True
            
        # Create collection
        vector_size = 256
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {str(e)}")
        return False

def add_to_qdrant(dataset_key, features, metadata):
    """
    Add features to Qdrant collection with proper handling of numpy types.
    
    Args:
        dataset_key: The dataset key
        features: Feature dictionary
        metadata: Metadata dictionary
    
    Returns:
        Point ID if successful, None otherwise
    """
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant client is not available. Cannot add to collection.")
        return None
        
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        collection_name = f"iris_{dataset_key}"
        
        # Get a unique ID
        point_id = str(uuid.uuid4())
        
        # Convert numpy values to native Python types
        features = convert_numpy_values(features)
        metadata = convert_numpy_values(metadata)
        
        # Create a vector from features
        vector = []
        
        # Add color features
        for color_data in features.get("color_features", [])[:3]:
            rgb = color_data.get("color", [0, 0, 0])
            vector.extend([c / 255.0 for c in rgb])
            vector.append(color_data.get("percentage", 0.0))
        
        # Add texture features
        texture = features.get("texture_features", {})
        vector.append(texture.get("contrast", 0.0))
        vector.append(texture.get("uniformity", 0.0))
        vector.append(texture.get("energy", 0.0))
        vector.append(texture.get("entropy", 0.0))
        
        # Add spot count
        vector.append(float(features.get("num_spots", 0)) / 20.0)  # Normalize
        
        # Fill to expected size
        vector_size = 256
        if len(vector) < vector_size:
            vector.extend([0.0] * (vector_size - len(vector)))
            
        # Create payload
        payload = {
            "metadata": metadata,
            "image_id": metadata.get("image_id", ""),
            "dataset": dataset_key,
            "subject_id": metadata.get("subject_id", ""),
            "eye": metadata.get("eye", ""),
            "feature_summary": {
                "num_spots": features.get("num_spots", 0),
                "contrast": texture.get("contrast", 0.0)
            }
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        # Add to collection
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        logger.info(f"Added point {point_id} to collection {collection_name}")
        return point_id
        
    except Exception as e:
        logger.error(f"Error adding to Qdrant: {str(e)}")
        return None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix dataset paths and create mock datasets")
    parser.add_argument("--dataset", default="casia_thousand", 
                        choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"],
                        help="Dataset to create")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to create")
    parser.add_argument("--populate-qdrant", action="store_true", 
                        help="Populate Qdrant with mock data")
    
    args = parser.parse_args()
    
    # Create the mock dataset
    create_mock_dataset(args.dataset, args.samples)
    
    if args.populate_qdrant:
        logger.info(f"Setting up Qdrant collection for {args.dataset}...")
        
        # Create the collection
        if create_qdrant_collection(args.dataset):
            # Add mock data to the collection
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
            dataset_dir = os.path.join(base_dir, args.dataset)
            
            # Check if metadata file exists
            metadata_file = os.path.join(base_dir, "dataset_metadata.csv")
            if not os.path.exists(metadata_file):
                with open(metadata_file, 'w') as f:
                    f.write("image_id,dataset,subject_id,eye,session,filename,path\n")
            
            # Add mock data
            added = 0
            for i in range(args.samples):
                subject_id = f"S{i+1:04d}"
                
                for eye in ["left", "right"]:
                    # Create a mock feature set
                    features = {
                        "color_features": [
                            {"color": [100, 150, 180], "percentage": 0.6},
                            {"color": [30, 30, 30], "percentage": 0.2},
                            {"color": [255, 255, 255], "percentage": 0.2}
                        ],
                        "texture_features": {
                            "contrast": float(np.random.rand()),
                            "uniformity": float(np.random.rand()),
                            "energy": float(np.random.rand()),
                            "entropy": float(np.random.rand())
                        },
                        "num_spots": int(np.random.randint(2, 15))
                    }
                    
                    # Create metadata
                    image_id = f"{subject_id}_{eye[0].upper()}1_001"
                    metadata = {
                        "image_id": image_id,
                        "dataset": args.dataset,
                        "subject_id": subject_id,
                        "eye": eye,
                        "session": 1,
                        "filename": f"{image_id}.jpg"
                    }
                    
                    # Add to Qdrant
                    if add_to_qdrant(args.dataset, features, metadata):
                        added += 1
                    
                    # Add to metadata file
                    with open(metadata_file, 'a') as f:
                        path = os.path.join(dataset_dir, subject_id, f"{eye[0].upper()}1", f"{image_id}.jpg")
                        f.write(f"{image_id},{args.dataset},{subject_id},{eye},1,{image_id}.jpg,{path}\n")
            
            logger.info(f"Added {added} mock patterns to Qdrant")
        
    logger.info("Fix completed successfully")

if __name__ == "__main__":
    main()
