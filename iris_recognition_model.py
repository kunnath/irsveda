#!/usr/bin/env python3
"""
Iris Recognition Model Builder

This script builds an iris recognition model using the CASIA dataset and
provides functionality to evaluate new iris images against the model.
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import pickle
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IrisModel:
    """Class for building and using iris recognition models"""
    
    def __init__(self, dataset_key: str = "casia_thousand", model_dir: str = None):
        """
        Initialize the iris model
        
        Args:
            dataset_key: Key identifying the dataset to use
            model_dir: Directory to store model files
        """
        self.dataset_key = dataset_key
        
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "iris_models"
            )
        
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize dependencies
        from iris_advanced_segmentation import preprocess_image, segment_iris
        from iris_feature_extractor import extract_all_features
        from iris_dataset_matcher import DatasetPatternMatcher
        
        self.dataset_matcher = DatasetPatternMatcher(dataset_key=dataset_key)
        
        # Model paths
        self.model_paths = {
            "features": os.path.join(self.model_dir, f"{dataset_key}_features.pkl"),
            "metadata": os.path.join(self.model_dir, f"{dataset_key}_metadata.json"),
            "model": os.path.join(self.model_dir, f"{dataset_key}_model.pkl")
        }
    
    def build_model(self, limit: int = None, force: bool = False):
        """
        Build the iris recognition model using the dataset
        
        Args:
            limit: Maximum number of samples to use
            force: Whether to force rebuilding even if model exists
        
        Returns:
            True if successful
        """
        # Check if model already exists
        if os.path.exists(self.model_paths["model"]) and not force:
            logger.info(f"Model already exists at {self.model_paths['model']}")
            logger.info("Use --force to rebuild")
            return True
        
        try:
            logger.info(f"Building iris recognition model using {self.dataset_key} dataset")
            
            # Step 1: Collect dataset statistics
            stats = self.dataset_matcher.get_dataset_statistics()
            if "error" in stats:
                logger.error(f"Error getting dataset statistics: {stats['error']}")
                return False
            
            logger.info(f"Dataset: {stats['dataset_name']}")
            logger.info(f"Total patterns: {stats['total_patterns']}")
            logger.info(f"Subjects: {stats['subjects']}")
            
            # Step 2: Extract features from dataset
            from iris_dataset_manager import IrisDatasetManager
            dataset_manager = IrisDatasetManager()
            
            # Get list of segmented images from metadata
            dataset_images = dataset_manager.metadata[
                (dataset_manager.metadata["dataset"] == self.dataset_key) & 
                (dataset_manager.metadata["segmented"])
            ]
            
            if limit:
                dataset_images = dataset_images.head(limit)
            
            # Set up the model data structure
            model_data = {
                "features": {},
                "feature_vectors": {},
                "subject_indices": {},
                "threshold": 0.3,  # Default similarity threshold
                "vector_size": 128  # Default vector size
            }
            
            # Process each image in the dataset
            logger.info(f"Processing {len(dataset_images)} images...")
            for idx, row in tqdm(dataset_images.iterrows(), total=len(dataset_images)):
                subject_id = row["subject_id"]
                eye = row["eye"]
                image_id = row["image_id"]
                
                # Get existing feature data from Qdrant
                search_result = self.dataset_matcher.pattern_matcher.client.scroll(
                    collection_name=self.dataset_matcher.collection_name,
                    scroll_filter={"must": [{"key": "metadata.image_id", "match": {"value": image_id}}]},
                    limit=1
                )
                
                if search_result and search_result[0]:
                    vectors = search_result[0]
                    if vectors:
                        vector = vectors[0].vector
                        payload = vectors[0].payload
                        
                        # Store feature vector
                        model_data["feature_vectors"][image_id] = vector
                        
                        # Store feature summary
                        if "feature_summary" in payload:
                            model_data["features"][image_id] = payload["feature_summary"]
                        
                        # Track subject indices
                        if subject_id not in model_data["subject_indices"]:
                            model_data["subject_indices"][subject_id] = []
                        
                        model_data["subject_indices"][subject_id].append(image_id)
            
            # Calculate the model's vector size based on actual data
            if model_data["feature_vectors"]:
                first_key = next(iter(model_data["feature_vectors"]))
                model_data["vector_size"] = len(model_data["feature_vectors"][first_key])
            
            # Save the model
            with open(self.model_paths["model"], "wb") as f:
                pickle.dump(model_data, f)
            
            # Save metadata
            model_metadata = {
                "dataset": self.dataset_key,
                "created": datetime.now().isoformat(),
                "num_subjects": len(model_data["subject_indices"]),
                "num_images": len(model_data["feature_vectors"]),
                "vector_size": model_data["vector_size"],
                "threshold": model_data["threshold"]
            }
            
            with open(self.model_paths["metadata"], "w") as f:
                json.dump(model_metadata, f, indent=2)
            
            logger.info(f"Model built and saved to {self.model_paths['model']}")
            logger.info(f"Model includes {len(model_data['subject_indices'])} subjects and {len(model_data['feature_vectors'])} images")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return False
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the iris recognition model
        
        Returns:
            Loaded model data or None if not found
        """
        try:
            if not os.path.exists(self.model_paths["model"]):
                logger.error(f"Model file not found: {self.model_paths['model']}")
                return None
            
            with open(self.model_paths["model"], "rb") as f:
                model_data = pickle.load(f)
            
            logger.info(f"Loaded model with {len(model_data['subject_indices'])} subjects and {len(model_data['feature_vectors'])} images")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def evaluate_image(self, image_path: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Evaluate a new iris image against the model
        
        Args:
            image_path: Path to the iris image
            top_n: Number of top matches to return
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Load the model
            model_data = self.load_model()
            if not model_data:
                return {"error": "Model not loaded"}
            
            # Process the iris image
            from advanced_iris_analyzer import AdvancedIrisAnalyzer
            analyzer = AdvancedIrisAnalyzer()
            
            analysis_result = analyzer.analyze_iris(image_path)
            if "error" in analysis_result:
                return {"error": analysis_result["error"]}
            
            # Extract features and convert to vector
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
            
            # Generate vector from features
            vector = self.dataset_matcher.pattern_matcher.convert_features_to_vector(features)
            
            # Compare with vectors in the model
            results = []
            for image_id, model_vector in model_data["feature_vectors"].items():
                # Calculate cosine similarity
                similarity = self._cosine_similarity(vector, model_vector)
                
                # Find which subject this image belongs to
                subject_id = None
                for s_id, images in model_data["subject_indices"].items():
                    if image_id in images:
                        subject_id = s_id
                        break
                
                results.append({
                    "image_id": image_id,
                    "subject_id": subject_id,
                    "similarity": similarity,
                    "features": model_data["features"].get(image_id, {})
                })
            
            # Sort by similarity (higher is better)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Get top N results
            top_matches = results[:top_n]
            
            # Calculate confidence based on similarity scores
            if top_matches:
                # If top match is significantly higher than others, high confidence
                if len(top_matches) > 1 and top_matches[0]["similarity"] > 0.9:
                    if top_matches[0]["similarity"] - top_matches[1]["similarity"] > 0.2:
                        confidence = "high"
                    else:
                        confidence = "medium"
                elif top_matches[0]["similarity"] > 0.7:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                confidence = "unknown"
            
            # Group matches by subject
            subjects = {}
            for match in top_matches:
                subject_id = match["subject_id"]
                if subject_id not in subjects:
                    subjects[subject_id] = {
                        "subject_id": subject_id,
                        "matches": [],
                        "avg_similarity": 0
                    }
                subjects[subject_id]["matches"].append(match)
            
            # Calculate average similarity by subject
            for subject_id, data in subjects.items():
                data["avg_similarity"] = sum(m["similarity"] for m in data["matches"]) / len(data["matches"])
            
            # Sort subjects by average similarity
            sorted_subjects = sorted(
                subjects.values(), 
                key=lambda x: x["avg_similarity"], 
                reverse=True
            )
            
            # Format the result
            evaluation_result = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "analysis": analysis_result,
                "top_matches": top_matches,
                "subjects": sorted_subjects,
                "confidence": confidence
            }
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating image: {str(e)}")
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score [0-1]
        """
        # Handle different vector sizes
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def visualize_evaluation(self, evaluation_result: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate a visualization of the evaluation results
        
        Args:
            evaluation_result: Results from evaluate_image
            output_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        try:
            if "error" in evaluation_result:
                logger.error(f"Cannot visualize evaluation with error: {evaluation_result['error']}")
                return None
            
            # Create output directory if needed
            if output_path is None:
                output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 
                    "evaluation_results"
                )
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a filename based on the image name
                image_path = evaluation_result["image_path"]
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"{base_filename}_evaluation_{timestamp}.png")
            
            # Load the original image
            image_path = evaluation_result["image_path"]
            original_image = cv2.imread(image_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create the visualization
            fig = plt.figure(figsize=(14, 10))
            
            # Title
            plt.suptitle(f"Iris Recognition Evaluation - {self.dataset_key}", fontsize=16)
            
            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(original_rgb)
            plt.title("Query Image")
            plt.axis("off")
            
            # Segmentation if available
            analysis = evaluation_result.get("analysis", {})
            if "image_paths" in analysis and analysis["image_paths"].get("segmentation"):
                segmentation_image = analysis["image_paths"].get("segmentation")
                
                # If it's base64, we need to convert it back
                if isinstance(segmentation_image, str) and segmentation_image.startswith("data:image"):
                    import base64
                    # Extract the base64 part and decode
                    base64_data = segmentation_image.split(",")[1]
                    image_data = base64.b64decode(base64_data)
                    
                    # Convert to numpy array
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(image_data))
                    segmentation_image = np.array(img)
                
                plt.subplot(2, 3, 2)
                if isinstance(segmentation_image, np.ndarray):
                    plt.imshow(cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB))
                plt.title("Segmentation")
                plt.axis("off")
            
            # Display top match details
            plt.subplot(2, 3, 3)
            plt.axis("off")
            
            top_match = evaluation_result.get("top_matches", [{}])[0]
            confidence = evaluation_result.get("confidence", "unknown")
            
            if top_match:
                match_text = [
                    f"Top Match Subject: {top_match.get('subject_id', 'unknown')}",
                    f"Similarity: {top_match.get('similarity', 0):.4f}",
                    f"Confidence: {confidence}"
                ]
                plt.text(0.1, 0.5, "\n".join(match_text), fontsize=12, verticalalignment='center')
            else:
                plt.text(0.1, 0.5, "No matches found", fontsize=12, verticalalignment='center')
            
            # Display subject matches
            subjects = evaluation_result.get("subjects", [])
            plt.subplot(2, 1, 2)
            plt.axis("off")
            
            if subjects:
                subject_text = ["Subject Matches:"]
                for i, subject in enumerate(subjects[:5]):
                    subj_id = subject.get("subject_id", "unknown")
                    avg_sim = subject.get("avg_similarity", 0)
                    num_matches = len(subject.get("matches", []))
                    
                    subject_text.append(f"{i+1}. Subject {subj_id}: {avg_sim:.4f} "
                                       f"({num_matches} matches)")
                
                plt.text(0.1, 0.5, "\n".join(subject_text), fontsize=12, verticalalignment='center')
            else:
                plt.text(0.1, 0.5, "No subject matches found", fontsize=12, verticalalignment='center')
            
            # Save the visualization
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Evaluation visualization saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error visualizing evaluation: {str(e)}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Iris Recognition Model Builder")
    
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"], 
                       default="casia_thousand", help="Dataset to use")
    parser.add_argument("--build", action="store_true", help="Build the model")
    parser.add_argument("--force", action="store_true", help="Force model rebuilding")
    parser.add_argument("--evaluate", type=str, help="Path to image to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top matches to show")
    
    args = parser.parse_args()
    
    # Create the model
    model = IrisModel(dataset_key=args.dataset)
    
    # Build the model if requested
    if args.build:
        success = model.build_model(limit=args.limit, force=args.force)
        if not success:
            return 1
    
    # Evaluate image if requested
    if args.evaluate:
        if not os.path.exists(args.evaluate):
            logger.error(f"Image not found: {args.evaluate}")
            return 1
        
        evaluation = model.evaluate_image(args.evaluate, top_n=args.top_n)
        
        if "error" in evaluation:
            logger.error(f"Evaluation error: {evaluation['error']}")
            return 1
        
        # Show results
        print("\n" + "="*50)
        print(f"Iris Recognition Results for {os.path.basename(args.evaluate)}")
        print("="*50)
        
        # Top matches
        top_matches = evaluation.get("top_matches", [])
        print(f"\nFound {len(top_matches)} matches (showing top {min(args.top_n, len(top_matches))}):")
        for i, match in enumerate(top_matches[:args.top_n]):
            print(f"{i+1}. Subject {match.get('subject_id', 'unknown')}, "
                  f"Similarity: {match.get('similarity', 0):.4f}")
        
        # Confidence
        confidence = evaluation.get("confidence", "unknown")
        print(f"\nOverall confidence: {confidence}")
        
        # Subject matches
        subjects = evaluation.get("subjects", [])
        if subjects:
            print("\nTop subject matches:")
            for i, subject in enumerate(subjects[:3]):
                print(f"{i+1}. Subject {subject.get('subject_id', 'unknown')}, "
                      f"Average similarity: {subject.get('avg_similarity', 0):.4f}, "
                      f"Matches: {len(subject.get('matches', []))}")
        
        # Visualize
        viz_path = model.visualize_evaluation(evaluation)
        if viz_path:
            print(f"\nVisualization saved to: {viz_path}")
    
    # If no action specified, show help
    if not args.build and not args.evaluate:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
