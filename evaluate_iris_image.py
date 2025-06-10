#!/usr/bin/env python3
"""
Iris Image Evaluation Tool

This script processes an iris image and matches it against the CASIA dataset.
It can be used to evaluate a single image or to build a model for iris recognition.
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our project modules
from iris_advanced_segmentation import preprocess_image, segment_iris
from iris_feature_extractor import extract_all_features
from iris_dataset_matcher import DatasetPatternMatcher
from advanced_iris_analyzer import AdvancedIrisAnalyzer

# Import connection checker
try:
    from check_qdrant_connection import check_qdrant_connection
except ImportError:
    def check_qdrant_connection(host="localhost", port=6333, retries=1, wait_time=2):
        """Fallback function if check_qdrant_connection module is not available"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False

class IrisEvaluator:
    """
    Class for evaluating iris images against research datasets
    """
    
    def __init__(
        self, 
        dataset_key: str = "casia_thousand",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        check_connection: bool = True
    ):
        """
        Initialize the iris evaluator
        
        Args:
            dataset_key: Key identifying the dataset to use
            qdrant_host: Host address of the Qdrant server
            qdrant_port: Port number for the Qdrant server
            check_connection: Whether to check Qdrant connection on init
        """
        self.dataset_key = dataset_key
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Check Qdrant connection first if requested
        if check_connection:
            if not check_qdrant_connection(qdrant_host, qdrant_port, retries=2, wait_time=1):
                logger.warning(f"Cannot connect to Qdrant at {qdrant_host}:{qdrant_port}.")
                logger.warning("Some functionality may be limited. Make sure Qdrant is running.")
                logger.info("Consider running: docker-compose up -d qdrant")
                self.qdrant_available = False
            else:
                self.qdrant_available = True
        else:
            self.qdrant_available = True
        
        # Initialize the iris analyzer and dataset matcher
        try:
            self.analyzer = AdvancedIrisAnalyzer(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port
            )
            
            self.dataset_matcher = DatasetPatternMatcher(
                dataset_key=dataset_key,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port
            )
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            logger.warning("Iris evaluation will run in limited mode without dataset matching")
            self.qdrant_available = False
    
    def evaluate_image(self, image_path: str, visualize: bool = True) -> Dict[str, Any]:
        """
        Evaluate an iris image against the dataset
        
        Args:
            image_path: Path to the iris image file
            visualize: Whether to generate visualization
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Check if file exists
            if not os.path.isfile(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            logger.info(f"Processing iris image: {image_path}")
            
            # Step 1: Analyze the iris using AdvancedIrisAnalyzer
            analysis_result = self.analyzer.analyze_iris(image_path)
            
            if "error" in analysis_result:
                return {"error": analysis_result["error"]}
            
            # Step 2: Match with dataset (if Qdrant is available)
            dataset_matches = []
            if self.qdrant_available:
                try:
                    # Check connection again before attempting to match
                    if check_qdrant_connection(self.qdrant_host, self.qdrant_port):
                        dataset_match_results = self.dataset_matcher.match_iris_with_dataset(
                            image_path, 
                            limit=10
                        )
                        dataset_matches = dataset_match_results.get("matches", [])
                    else:
                        logger.error(f"Cannot connect to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
                        logger.info("Skipping dataset matching. Please ensure Qdrant is running.")
                except Exception as e:
                    logger.error(f"Dataset matching error: {str(e)}")
                    logger.info("Continuing with partial results (without dataset matching)")
            else:
                logger.info("Qdrant not available. Skipping dataset matching.")
            
            # Step 3: Generate health insights
            health_insights = self.analyzer.generate_health_insights(analysis_result)
            
            # Step 4: Combine results
            evaluation_result = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "analysis": analysis_result,
                "dataset_matches": dataset_matches,
                "health_insights": health_insights
            }
            
            # Step 5: Visualize if requested
            if visualize:
                visualization = self._visualize_results(
                    image_path, 
                    analysis_result, 
                    dataset_matches
                )
                evaluation_result["visualization"] = visualization
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating iris image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def _visualize_results(
        self, 
        image_path: str,
        analysis_result: Dict[str, Any],
        matches: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate visualizations of the evaluation results
        
        Args:
            image_path: Path to the original image
            analysis_result: Results from iris analysis
            matches: List of dataset matches
            
        Returns:
            Dictionary with paths to visualization images
        """
        try:
            # Create output directory
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "evaluation_results"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Base filename for outputs
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output = os.path.join(output_dir, f"{base_filename}_{timestamp}")
            
            # Load the original image
            original_image = cv2.imread(image_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # 1. Segmentation visualization
            if "segmentation" in analysis_result and analysis_result.get("image_paths", {}).get("segmentation"):
                segmentation_image = analysis_result.get("image_paths", {}).get("segmentation")
                segmentation_path = f"{base_output}_segmentation.png"
                
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
                
                if isinstance(segmentation_image, np.ndarray):
                    cv2.imwrite(segmentation_path, segmentation_image)
            else:
                segmentation_path = None
            
            # 2. Zone visualization
            if "image_paths" in analysis_result and analysis_result["image_paths"].get("zones"):
                zones_image = analysis_result["image_paths"].get("zones")
                zones_path = f"{base_output}_zones.png"
                
                # Similar processing as above if needed
                if isinstance(zones_image, str) and zones_image.startswith("data:image"):
                    import base64
                    base64_data = zones_image.split(",")[1]
                    image_data = base64.b64decode(base64_data)
                    
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(image_data))
                    zones_image = np.array(img)
                
                if isinstance(zones_image, np.ndarray):
                    cv2.imwrite(zones_path, zones_image)
            else:
                zones_path = None
            
            # 3. Matches visualization
            if matches:
                plt.figure(figsize=(12, 8))
                plt.suptitle(f"Matches from {self.dataset_key} Dataset", fontsize=16)
                
                # Original image
                plt.subplot(2, 3, 1)
                plt.imshow(original_rgb)
                plt.title("Query Image")
                plt.axis("off")
                
                # Display top 5 matches
                for i, match in enumerate(matches[:5]):
                    plt.subplot(2, 3, i+2)
                    plt.text(0.5, 0.5, f"Subject: {match.get('subject_id', 'unknown')}\nEye: {match.get('eye', 'unknown')}\nScore: {match.get('score', 0):.4f}", 
                            ha='center', va='center', fontsize=12)
                    plt.axis("off")
                
                matches_path = f"{base_output}_matches.png"
                plt.savefig(matches_path, bbox_inches='tight')
                plt.close()
            else:
                matches_path = None
            
            return {
                "segmentation": segmentation_path,
                "zones": zones_path,
                "matches": matches_path
            }
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return {}

def ensure_dataset_setup(dataset_key: str, num_samples: int = 10):
    """
    Ensure the dataset is set up for evaluation
    
    Args:
        dataset_key: Key of the dataset to set up
        num_samples: Number of samples to use in simulation mode
    """
    # Create a temporary dataset for testing if needed
    try:
        # Import here to avoid circular imports
        import iris_dataset_manager
        
        # Create dataset dir if it doesn't exist
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        dataset_dir = os.path.join(base_dir, dataset_key)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Check if dataset directory is empty or doesn't have expected structure
        if not os.path.exists(os.path.join(dataset_dir, "dataset_info.txt")):
            logger.info(f"Setting up simulated {dataset_key} dataset with {num_samples} samples")
            
            # Create a basic structure for the dataset
            for i in range(1, num_samples + 1):
                subject_id = f"S{i:04d}"
                subject_dir = os.path.join(dataset_dir, subject_id)
                os.makedirs(os.path.join(subject_dir, "L1"), exist_ok=True)
                os.makedirs(os.path.join(subject_dir, "R1"), exist_ok=True)
                
                # Create a placeholder image or link to a sample image
                with open(os.path.join(subject_dir, f"{subject_id}_L.jpg"), "w") as f:
                    f.write("Placeholder for left iris image")
                with open(os.path.join(subject_dir, f"{subject_id}_R.jpg"), "w") as f:
                    f.write("Placeholder for right iris image")
            
            # Create dataset info file
            with open(os.path.join(dataset_dir, "dataset_info.txt"), "w") as f:
                f.write(f"Dataset: {dataset_key} (Simulated)\n")
                f.write(f"Subjects: {num_samples}\n")
                f.write("Note: This is a simulated dataset for development purposes.\n")
        
        # Initialize and set up the dataset matcher
        from iris_dataset_matcher import DatasetPatternMatcher
        matcher = DatasetPatternMatcher(dataset_key=dataset_key)
        
        # Check if dataset is already in Qdrant
        stats = matcher.get_dataset_statistics()
        
        if stats.get("total_patterns", 0) < num_samples:
            logger.info(f"Setting up {dataset_key} in Qdrant")
            matcher.setup_dataset(limit=num_samples)
        else:
            logger.info(f"{dataset_key} dataset already set up with {stats.get('total_patterns', 0)} patterns")
            
    except Exception as e:
        logger.error(f"Error ensuring dataset setup: {str(e)}")

def main():
    """Main function to run the iris evaluator"""
    parser = argparse.ArgumentParser(description="Iris Image Evaluation Tool")
    
    parser.add_argument("image_path", type=str, help="Path to the iris image to evaluate")
    parser.add_argument("--dataset", choices=["casia_thousand", "nd_iris_0405", "ubiris_v2"], 
                       default="casia_thousand", help="Dataset to use for evaluation")
    parser.add_argument("--setup", action="store_true", help="Set up the dataset if needed")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples for dataset simulation")
    parser.add_argument("--skip-qdrant-check", action="store_true", help="Skip Qdrant connection check")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--force-offline", action="store_true", 
                       help="Force offline mode (no Qdrant dependency)")
    
    args = parser.parse_args()
    
    # Check if Qdrant is running
    if not args.skip_qdrant_check and not args.force_offline:
        logger.info("Checking if Qdrant is running...")
        if not check_qdrant_connection(args.qdrant_host, args.qdrant_port):
            logger.error("Qdrant is not available. Some features will be limited.")
            logger.info("To start Qdrant: docker-compose up -d qdrant")
            logger.info("To run without Qdrant: Use --force-offline flag")
            
            # Prompt user to continue
            if not args.force_offline:
                try:
                    response = input("Continue without dataset matching? (y/n): ").strip().lower()
                    if response != 'y':
                        logger.info("Exiting. Start Qdrant and try again.")
                        return 1
                except:
                    # In non-interactive environments, continue anyway
                    pass
    
    # Ensure the dataset is set up if requested
    if args.setup and not args.force_offline:
        ensure_dataset_setup(args.dataset, args.samples)
    
    # Create the evaluator
    evaluator = IrisEvaluator(
        dataset_key=args.dataset,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        check_connection=not args.skip_qdrant_check
    )
    
    # Evaluate the image
    results = evaluator.evaluate_image(args.image_path, visualize=not args.no_viz)
    
    # Print results summary
    if "error" in results:
        logger.error(f"Evaluation error: {results['error']}")
        return 1
    
    # Display summary
    print("\n" + "="*50)
    print(f"Iris Evaluation Results for {os.path.basename(args.image_path)}")
    print("="*50)
    
    # Analysis summary
    analysis = results.get("analysis", {})
    print(f"\nIris Analysis:")
    if "segmentation" in analysis:
        seg = analysis["segmentation"]
        print(f"- Iris: center=({seg.get('iris_center', 'unknown')}), radius={seg.get('iris_radius', 'unknown')}")
        print(f"- Pupil: center=({seg.get('pupil_center', 'unknown')}), radius={seg.get('pupil_radius', 'unknown')}")
    
    if "features" in analysis:
        features = analysis["features"]
        print("\nFeature Summary:")
        print(f"- Spot count: {features.get('spot_count', 'unknown')}")
        if "texture_stats" in features:
            texture = features["texture_stats"]
            print(f"- Texture: contrast={texture.get('contrast', 'unknown'):.2f}, "
                 f"uniformity={texture.get('uniformity', 'unknown'):.2f}")
    
    # Dataset matches
    matches = results.get("dataset_matches", [])
    print(f"\nFound {len(matches)} matches in the {args.dataset} dataset:")
    for i, match in enumerate(matches[:5]):
        print(f"{i+1}. Subject {match.get('subject_id', 'unknown')}, "
              f"{match.get('eye', 'unknown')} eye, "
              f"Score: {match.get('score', 0):.4f}")
    
    # Health insights
    insights = results.get("health_insights", {})
    print("\nHealth Insights:")
    if "error" in insights:
        print(f"- Error: {insights['error']}")
    else:
        print(f"- Overall assessment: {insights.get('overall_assessment', 'unknown')}")
        findings = insights.get("key_findings", [])
        for i, finding in enumerate(findings):
            print(f"- Finding {i+1}: {finding.get('finding', 'unknown')}")
            if "suggestion" in finding:
                print(f"  Suggestion: {finding.get('suggestion', '')}")
    
    # Visualization paths
    if "visualization" in results:
        viz = results["visualization"]
        print("\nVisualizations saved to:")
        for k, v in viz.items():
            if v:
                print(f"- {k}: {v}")
    
    print("\nEvaluation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
