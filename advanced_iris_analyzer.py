"""
Advanced iris analyzer for IridoVeda.

This module integrates advanced iris segmentation, feature extraction,
and pattern matching to provide comprehensive iris analysis.
"""

import cv2
import numpy as np
import os
import tempfile
import json
import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Import our modules
from iris_advanced_segmentation import preprocess_image, segment_iris, extract_iris_zones
from iris_feature_extractor import extract_all_features
from iris_pattern_matcher import IrisPatternMatcher
from iris_zone_analyzer import IrisZoneAnalyzer  # Import the existing zone analyzer

# Try to import the suggested queries generator (optional)
try:
    from suggested_queries import SuggestedQueriesGenerator
    QUERIES_GENERATOR_AVAILABLE = True
except ImportError:
    QUERIES_GENERATOR_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedIrisAnalyzer:
    """
    Class for advanced iris analysis, integrating segmentation, feature extraction, and pattern matching.
    """
    
    def __init__(
        self, 
        qdrant_host: str = "localhost", 
        qdrant_port: int = 6333, 
        collection_name: str = "iris_patterns"
    ):
        """
        Initialize the advanced iris analyzer.
        
        Args:
            qdrant_host: Host of the Qdrant server
            qdrant_port: Port of the Qdrant server
            collection_name: Name of the collection to store iris patterns
        """
        self.pattern_matcher = IrisPatternMatcher(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        # Initialize the existing zone analyzer for compatibility
        self.zone_analyzer = IrisZoneAnalyzer()
        
        # Initialize the query generator if available
        self.query_generator = None
        if QUERIES_GENERATOR_AVAILABLE:
            self.query_generator = SuggestedQueriesGenerator()
        
        # Ensure the collection exists
        self.pattern_matcher.create_collection()
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            # Check if file exists
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Read the image
            image = cv2.imread(image_path)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    def analyze_iris(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive iris analysis on an image.
        
        Args:
            image_path: Path to the iris image file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load the image
            original_image = self.load_image(image_path)
            if original_image is None:
                return {"error": "Failed to load image"}
            
            # Track processing time
            start_time = time.time()
            
            # Step 1: Preprocess the image
            preprocessed_image, gray_image, _ = preprocess_image(original_image)
            
            # Step 2: Segment the iris
            segmentation_data = segment_iris(preprocessed_image)
            if segmentation_data is None:
                return {"error": "Failed to segment iris - no iris detected"}
            
            # Step 3: Extract iris zones
            zone_data = extract_iris_zones(segmentation_data)
            if zone_data:
                segmentation_data.update(zone_data)
            
            # Step 4: Extract features
            features = extract_all_features(original_image, segmentation_data)
            
            # Step 5: Search for similar patterns
            similar_patterns = self.pattern_matcher.search_similar_patterns(features, limit=5)
            
            # Step 6: Also run the existing zone analyzer for compatibility
            zone_results = self.zone_analyzer.process_iris_image(image_path)
            
            # Step 7: Generate suggested health queries (if query generator is available)
            suggested_queries = []
            if self.query_generator is not None:
                suggested_queries = self.query_generator.generate_query_from_iris_features(features)
            
            # Step 8: Create a combined result
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time,
                "segmentation": {
                    "iris_center": segmentation_data["iris_center"],
                    "iris_radius": segmentation_data["iris_radius"],
                    "pupil_center": segmentation_data["pupil_center"],
                    "pupil_radius": segmentation_data["pupil_radius"]
                },
                "features": {
                    "color_summary": [
                        {
                            "color": f"rgb({int(c['color'][0])}, {int(c['color'][1])}, {int(c['color'][2])})",
                            "percentage": c["percentage"]
                        }
                        for c in features.get("color_features", [])[:3]
                    ],
                    "spot_count": features.get("num_spots", 0),
                    "texture_stats": features.get("texture_features", {})
                },
                "similar_patterns": similar_patterns,
                "suggested_queries": suggested_queries,
                "image_paths": {
                    "segmentation": self._save_visualization(segmentation_data["segmentation_image"], "segmentation"),
                    "zones": self._save_visualization(zone_data["zone_visualization"], "zones") if zone_data else None
                }
            }
            
            # Add zone analysis from the existing analyzer if available
            if zone_results and "error" not in zone_results:
                analysis_result["zone_analysis"] = {
                    "health_summary": zone_results.get("health_summary", {}),
                    "zones_analysis": zone_results.get("zones_analysis", {})
                }
                
                # Add original visualization paths
                analysis_result["image_paths"].update({
                    "original": self._convert_to_base64(original_image),
                    "zone_map": zone_results.get("zone_map", None),
                    "boundary_image": zone_results.get("boundary_image", None),
                })
            
            # Store this analysis in Qdrant for future reference
            pattern_id = self.pattern_matcher.store_iris_pattern(
                features,
                {
                    "timestamp": analysis_result["timestamp"],
                    "filename": os.path.basename(image_path)
                }
            )
            
            if pattern_id:
                analysis_result["pattern_id"] = pattern_id
            
            # Generate suggested queries if the generator is available
            if self.query_generator and "suggested_queries" not in analysis_result:
                analysis_result["suggested_queries"] = self.query_generator.generate_query_from_iris_features(analysis_result.get("features", {}))
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in iris analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _save_visualization(self, image: np.ndarray, name: str) -> Optional[str]:
        """
        Save a visualization image to a temporary file and return its path.
        
        Args:
            image: Image to save
            name: Name for the temporary file
            
        Returns:
            Base64 encoded image data
        """
        if image is None:
            return None
        
        try:
            return self._convert_to_base64(image)
            
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return None
    
    def _convert_to_base64(self, image: np.ndarray) -> str:
        """
        Convert an image to base64 for web display.
        
        Args:
            image: Image to convert
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Encode the image
            _, buffer = cv2.imencode('.png', image)
            image_bytes = buffer.tobytes()
            import base64
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            return None
    
    def generate_health_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate health insights based on iris analysis.
        
        Args:
            analysis_result: Result from analyze_iris
            
        Returns:
            Dictionary with health insights
        """
        try:
            if "error" in analysis_result:
                return {"error": analysis_result["error"]}
            
            # Use existing zone analysis if available
            if "zone_analysis" in analysis_result:
                zone_analysis = analysis_result["zone_analysis"]
                
                # Extract key insights
                insights = {
                    "overall_health": zone_analysis.get("health_summary", {}).get("overall_health", "unknown"),
                    "dosha_balance": zone_analysis.get("health_summary", {}).get("dosha_balance", {}),
                    "key_findings": []
                }
                
                # Extract findings from zones
                zones_analysis = zone_analysis.get("zones_analysis", {})
                for zone_name, zone_data in zones_analysis.items():
                    condition = zone_data.get("health_indication", {}).get("condition", "unknown")
                    confidence = zone_data.get("health_indication", {}).get("confidence", 0)
                    
                    # Only include significant findings
                    if condition != "normal" and confidence > 0.5:
                        insights["key_findings"].append({
                            "zone": zone_data.get("name", zone_name),
                            "condition": condition,
                            "confidence": confidence,
                            "suggestion": zone_data.get("health_indication", {}).get("suggestion", "")
                        })
                
                # Add feature-based insights
                features = analysis_result.get("features", {})
                spot_count = features.get("spot_count", 0)
                
                # Add insights based on spots
                if spot_count > 10:
                    insights["key_findings"].append({
                        "type": "feature",
                        "finding": f"High number of pigment spots detected ({spot_count})",
                        "suggestion": "Consider detoxification protocols and improved hydration"
                    })
                    
                return insights
                
            # If no existing zone analysis, create basic insights from features
            features = analysis_result.get("features", {})
            segmentation = analysis_result.get("segmentation", {})
            
            # Basic insights
            insights = {
                "overall_assessment": "Analysis completed",
                "key_findings": []
            }
            
            # Add pupil-iris ratio insight
            pupil_radius = segmentation.get("pupil_radius")
            iris_radius = segmentation.get("iris_radius")
            if pupil_radius is not None and iris_radius and iris_radius > 0:
                pupil_ratio = pupil_radius / iris_radius
                if pupil_ratio > 0.45:
                    insights["key_findings"].append({
                        "type": "structure",
                        "finding": "Large pupil-to-iris ratio detected",
                        "suggestion": "May indicate autonomic nervous system activity or light sensitivity"
                    })
                elif pupil_ratio < 0.2:
                    insights["key_findings"].append({
                        "type": "structure",
                        "finding": "Small pupil-to-iris ratio detected",
                        "suggestion": "May indicate constricted autonomic response"
                    })
            
            # Add spot count insight
            spot_count = features.get("spot_count", 0)
            if spot_count > 10:
                insights["key_findings"].append({
                    "type": "feature",
                    "finding": f"High number of pigment spots detected ({spot_count})",
                    "suggestion": "Consider detoxification protocols and improved hydration"
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating health insights: {str(e)}")
            return {"error": f"Failed to generate insights: {str(e)}"}
