"""
Dataset pattern matching integration for IridoVeda.

This script enhances the IrisPatternMatcher by integrating public datasets.
"""

import os
import argparse
import logging
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm

from iris_dataset_manager import IrisDatasetManager
from iris_feature_extractor import extract_all_features
from iris_pattern_matcher import IrisPatternMatcher
from iris_advanced_segmentation import preprocess_image, segment_iris

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPatternMatcher:
    """
    Class to integrate public datasets with pattern matching functionality.
    """
    
    def __init__(
        self,
        dataset_key: str = "casia_thousand",
        collection_name: str = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        """
        Initialize the dataset pattern matcher.
        
        Args:
            dataset_key: Key identifying the dataset ("casia_thousand", "nd_iris_0405", "ubiris_v2")
            collection_name: Name for the Qdrant collection (defaults to iris_{dataset_key})
            qdrant_host: Host address of the Qdrant server
            qdrant_port: Port number for the Qdrant server
        """
        self.dataset_key = dataset_key
        self.collection_name = collection_name or f"iris_{dataset_key}"
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Initialize dataset manager and pattern matcher
        self.dataset_manager = IrisDatasetManager()
        self.pattern_matcher = IrisPatternMatcher(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=self.collection_name
        )
        
        # Ensure the collection exists
        self.pattern_matcher.create_collection()
    
    def setup_dataset(self, limit: int = None):
        """
        Set up the dataset for pattern matching.
        
        Args:
            limit: Maximum number of images to process (None for all)
            
        Returns:
            True if successful
        """
        try:
            # Download/prepare dataset
            dataset_dir = self.dataset_manager.download_dataset(self.dataset_key)
            
            # Process based on dataset type
            if self.dataset_key == "casia_thousand":
                self.dataset_manager.process_casia_thousand(dataset_dir)
            # Add more datasets as needed
            
            # Segment the dataset
            self.dataset_manager.segment_dataset(self.dataset_key, limit=limit)
            
            # Extract features
            features = self.dataset_manager.extract_features(self.dataset_key, limit=limit)
            
            # Import to Qdrant
            self.dataset_manager.import_to_qdrant(features, collection_name=self.collection_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up dataset: {str(e)}")
            return False
    
    def match_iris_with_dataset(self, iris_image_path: str, limit: int = 5) -> Dict[str, Any]:
        """
        Match an iris image with the dataset patterns.
        
        Args:
            iris_image_path: Path to the iris image
            limit: Maximum number of matches to return
            
        Returns:
            Dictionary with match results
        """
        try:
            from advanced_iris_analyzer import AdvancedIrisAnalyzer
            
            # Process the iris image
            analyzer = AdvancedIrisAnalyzer()
            analysis_result = analyzer.analyze_iris(iris_image_path)
            
            if "error" in analysis_result:
                return {"error": analysis_result["error"]}
            
            # Extract features from analysis
            features = {
                "color_features": [
                    {
                        "color": eval(c["color"].replace("rgb", "")),
                        "percentage": c["percentage"]
                    }
                    for c in analysis_result["features"]["color_summary"]
                ],
                "texture_features": analysis_result["features"]["texture_stats"],
                "num_spots": analysis_result["features"]["spot_count"]
            }
            
            # Search for similar patterns in the dataset
            vector = self.pattern_matcher.convert_features_to_vector(features)
            similar_patterns = self.pattern_matcher.search_by_vector(vector, limit=limit)
            
            # Enhance results with dataset metadata
            enhanced_results = []
            for pattern in similar_patterns:
                pattern_id = pattern["id"]
                score = pattern["score"]
                payload = pattern["payload"]
                
                # Get metadata for this pattern
                metadata = self.dataset_manager.metadata[
                    self.dataset_manager.metadata["image_id"] == payload.get("image_id", "")
                ]
                
                if len(metadata) > 0:
                    metadata_row = metadata.iloc[0].to_dict()
                    
                    enhanced_results.append({
                        "id": pattern_id,
                        "score": score,
                        "dataset": payload.get("dataset", self.dataset_key),
                        "subject_id": payload.get("subject_id", "unknown"),
                        "eye": payload.get("eye", "unknown"),
                        "session": payload.get("session", 1),
                        "filename": payload.get("filename", ""),
                        "features": payload.get("feature_summary", {})
                    })
                else:
                    # Use payload directly if metadata not found
                    enhanced_results.append({
                        "id": pattern_id,
                        "score": score,
                        "dataset": payload.get("dataset", self.dataset_key),
                        "subject_id": payload.get("subject_id", "unknown"),
                        "eye": payload.get("eye", "unknown"),
                        "session": payload.get("session", 1),
                        "filename": payload.get("filename", ""),
                        "features": payload.get("feature_summary", {})
                    })
            
            return {
                "matches": enhanced_results,
                "analysis": analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error matching iris with dataset: {str(e)}")
            return {"error": f"Match failed: {str(e)}"}
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        try:
            # Get collection info from Qdrant
            collection_info = self.pattern_matcher.client.get_collection(self.collection_name)
            
            # Get dataset metadata
            dataset_metadata = self.dataset_manager.metadata[
                self.dataset_manager.metadata["dataset"] == self.dataset_key
            ]
            
            # Calculate statistics
            stats = {
                "dataset_name": self.dataset_manager.DATASET_INFO[self.dataset_key]["name"],
                "total_patterns": collection_info.vectors_count,
                "total_images": len(dataset_metadata),
                "subjects": dataset_metadata["subject_id"].nunique() if not dataset_metadata.empty else 0,
                "left_eyes": len(dataset_metadata[dataset_metadata["eye"] == "left"]) if not dataset_metadata.empty else 0,
                "right_eyes": len(dataset_metadata[dataset_metadata["eye"] == "right"]) if not dataset_metadata.empty else 0,
                "segmented_images": len(dataset_metadata[dataset_metadata["segmented"] == True]) if not dataset_metadata.empty else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            return {"error": f"Failed to get statistics: {str(e)}"}

def main():
    """Run the dataset pattern matcher as a standalone script."""
    parser = argparse.ArgumentParser(description="Iris Dataset Pattern Matcher")
    
    parser.add_argument("--setup", action="store_true", help="Set up the dataset")
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"], 
                        default="casia_thousand", help="Dataset to use")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of images to process")
    parser.add_argument("--match", type=str, help="Path to an iris image to match against the dataset")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    
    args = parser.parse_args()
    
    matcher = DatasetPatternMatcher(dataset_key=args.dataset)
    
    if args.setup:
        logger.info(f"Setting up {args.dataset} dataset (limit: {args.limit})...")
        success = matcher.setup_dataset(limit=args.limit)
        if success:
            logger.info("Dataset setup complete")
        else:
            logger.error("Dataset setup failed")
    
    if args.match:
        logger.info(f"Matching iris image {args.match} against {args.dataset}...")
        results = matcher.match_iris_with_dataset(args.match)
        if "error" in results:
            logger.error(f"Match error: {results['error']}")
        else:
            logger.info(f"Found {len(results['matches'])} matches")
            for i, match in enumerate(results["matches"]):
                logger.info(f"Match {i+1}: Subject {match['subject_id']}, "
                           f"{match['eye']} eye, Score: {match['score']:.4f}")
    
    if args.stats:
        logger.info(f"Getting statistics for {args.dataset}...")
        stats = matcher.get_dataset_statistics()
        if "error" in stats:
            logger.error(f"Statistics error: {stats['error']}")
        else:
            logger.info(f"Dataset: {stats['dataset_name']}")
            logger.info(f"Patterns: {stats['total_patterns']}")
            logger.info(f"Images: {stats['total_images']}")
            logger.info(f"Subjects: {stats['subjects']}")
            logger.info(f"Left eyes: {stats['left_eyes']}")
            logger.info(f"Right eyes: {stats['right_eyes']}")
            logger.info(f"Segmented: {stats['segmented_images']}")

if __name__ == "__main__":
    main()
