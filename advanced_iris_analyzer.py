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
import re

# Import our modules
from iris_advanced_segmentation import preprocess_image, segment_iris, extract_iris_zones
from iris_feature_extractor import extract_all_features
from iris_pattern_matcher import IrisPatternMatcher
from iris_zone_analyzer import IrisZoneAnalyzer  # Import the existing zone analyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our new deep analysis modules (after logger is defined)
try:
    from enhanced_iris_spot_analyzer import EnhancedIrisSpotAnalyzer
    ENHANCED_SPOT_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_SPOT_ANALYZER_AVAILABLE = False
    logger.warning("Enhanced spot analyzer not available")

try:
    from iris_deep_analysis import IrisDeepAnalyzer
    DEEP_ANALYZER_AVAILABLE = True
except ImportError:
    DEEP_ANALYZER_AVAILABLE = False
    logger.warning("Deep analyzer not available")

# Try to import the suggested queries generator (optional)
try:
    from suggested_queries import SuggestedQueriesGenerator
    QUERIES_GENERATOR_AVAILABLE = True
except ImportError:
    QUERIES_GENERATOR_AVAILABLE = False

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
        
        # Initialize the enhanced spot analyzer if available
        self.enhanced_spot_analyzer = None
        if ENHANCED_SPOT_ANALYZER_AVAILABLE:
            self.enhanced_spot_analyzer = EnhancedIrisSpotAnalyzer()
        
        # Initialize the deep analyzer if available
        self.deep_analyzer = None
        if DEEP_ANALYZER_AVAILABLE:
            self.deep_analyzer = IrisDeepAnalyzer()
        
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
    
    def analyze_iris(self, image_path: str, enhanced_qdrant_client=None) -> Dict[str, Any]:
        """
        Perform comprehensive iris analysis.
        
        Args:
            image_path: Path to the iris image
            enhanced_qdrant_client: Optional enhanced Qdrant client for pattern matching
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load and preprocess image
            image = self.load_image(image_path)
            preprocessed = preprocess_image(image)
            
            # Perform iris segmentation
            segmentation_results = segment_iris(preprocessed)
            if "error" in segmentation_results:
                return {"error": f"Segmentation failed: {segmentation_results['error']}"}
                
            # Extract zones using segmentation data
            zones_data = extract_iris_zones(preprocessed, segmentation_results)
            
            # Perform zone analysis using updated analyzer
            zone_analysis = self.zone_analyzer.analyze_zones(
                preprocessed,
                {
                    'iris_center': segmentation_results['iris_center'],
                    'iris_radius': segmentation_results['iris_radius'],
                    'pupil_radius': segmentation_results['pupil_radius']
                }
            )
            
            # Extract all iris features
            features = extract_all_features(preprocessed, segmentation_results)
            
            # Perform pattern matching if client is available
            pattern_matches = None
            if enhanced_qdrant_client:
                pattern_matches = self.pattern_matcher.find_matches(
                    features,
                    enhanced_qdrant_client
                )
            
            # Perform enhanced spot analysis if available
            spot_analysis = None
            if self.enhanced_spot_analyzer:
                spot_analysis = self.enhanced_spot_analyzer.comprehensive_spot_analysis(
                    preprocessed,
                    segmentation_results
                )
            
            # Perform deep analysis if available
            deep_analysis = None
            if self.deep_analyzer:
                deep_analysis = self.deep_analyzer.analyze(
                    preprocessed,
                    segmentation_results,
                    features
                )
            
            # Generate suggested queries if available
            suggested_queries = None
            if self.query_generator:
                suggested_queries = self.query_generator.generate_queries(
                    zone_analysis,
                    spot_analysis,
                    deep_analysis
                )
            
            # Combine all results
            results = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'segmentation_results': segmentation_results,
                'zone_analysis': zone_analysis,
                'features': features,
            }
            
            if pattern_matches:
                results['pattern_matches'] = pattern_matches
            if spot_analysis:
                results['spot_analysis'] = spot_analysis
            if deep_analysis:
                results['deep_analysis'] = deep_analysis
            if suggested_queries:
                results['suggested_queries'] = suggested_queries
                
            return results
            
        except Exception as e:
            logger.error(f"Error in iris analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
    
    def fetch_knowledge_summary(self, features: Dict[str, Any], enhanced_qdrant_client = None) -> Dict[str, Any]:
        """
        Fetch relevant knowledge from the Qdrant database to enhance the analysis summary.
        
        Args:
            features: Extracted iris features
            enhanced_qdrant_client: Optional EnhancedIrisQdrantClient instance
            
        Returns:
            Dictionary with knowledge-based insights
        """
        if not enhanced_qdrant_client:
            logger.warning("No Qdrant client provided for knowledge retrieval")
            return {
                "available": False,
                "message": "Knowledge base not accessible"
            }
            
        # Build targeted queries based on extracted features
        queries = []
        
        # Add queries based on pupil/iris ratio if available
        if "pupil_iris_ratio" in features:
            ratio = features["pupil_iris_ratio"]
            if ratio > 0.4:
                queries.append("large pupil iris ratio significance")
            elif ratio < 0.25:
                queries.append("small pupil iris ratio meaning")
            else:
                queries.append("normal pupil iris ratio")
        
        # Add queries based on spot count
        if "spot_count" in features:
            spot_count = features["spot_count"]
            if spot_count > 10:
                queries.append("numerous iris spots significance")
            elif spot_count > 5:
                queries.append("moderate iris spots meaning")
            elif spot_count > 0:
                queries.append("few iris spots interpretation")
        
        # Add queries based on dominant colors
        if "color_summary" in features and features["color_summary"]:
            color_info = features["color_summary"][0]
            color_str = color_info.get("color", "")
            # Extract RGB values from the color string
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                if r > 150 and g > 150 and b < 100:  # Yellow/amber
                    queries.append("yellow amber iris meaning")
                elif r > 150 and g < 100 and b < 100:  # Reddish
                    queries.append("red or reddish iris significance")
                elif r < 100 and g < 100 and b > 150:  # Blue
                    queries.append("blue iris in iridology")
                elif r < 100 and g > 150 and b < 100:  # Green
                    queries.append("green iris meaning in iridology")
                elif r > 100 and g > 50 and b < 50:  # Brown
                    queries.append("brown iris iridology significance")
        
        # Add general interpretation queries
        queries.append("iris analysis interpretation principles")
        
        # Use the first 3 queries to avoid overloading
        knowledge_results = {}
        insights = []
        
        # Get dosha information if available in the zone analysis
        dosha_proportions = self._get_dosha_proportions_from_features(features)
        primary_dosha = max(dosha_proportions, key=dosha_proportions.get) if dosha_proportions else None
        
        # Process regular queries first
        for i, query in enumerate(queries[:3]):
            try:
                # Use multi_query_search for better results
                results = enhanced_qdrant_client.multi_query_search(query, limit=1)
                if results and len(results) > 0:
                    # Extract the most relevant information from each result
                    top_result = results[0]
                    insights.append({
                        "query": query,
                        "insight": top_result.get("text", "No information available"),
                        "source": top_result.get("source", "Unknown"),
                        "page": top_result.get("page", 0),
                        "score": top_result.get("score", 0)
                    })
            except Exception as e:
                logger.error(f"Error fetching knowledge for query '{query}': {str(e)}")
        
        # Add dosha-specific information if primary dosha is available
        dosha_insights = []
        if primary_dosha and dosha_proportions[primary_dosha] >= 0.4:
            try:
                # Create dosha-specific query based on iris features
                dosha_query = f"{primary_dosha} dosha iris characteristics"
                
                # Use the new dosha-specific search
                dosha_results = enhanced_qdrant_client.search_by_dosha(
                    dosha_type=primary_dosha,
                    query="iris characteristics", 
                    limit=2
                )
                
                if dosha_results and len(dosha_results) > 0:
                    for result in dosha_results:
                        dosha_insights.append({
                            "dosha": primary_dosha,
                            "query": dosha_query,
                            "insight": result.get("text", "No dosha information available"),
                            "source": result.get("source", "Unknown"),
                            "page": result.get("page", 0),
                            "score": result.get("score", 0),
                            "is_dosha_specific": True
                        })
            except Exception as e:
                logger.error(f"Error fetching dosha knowledge: {str(e)}")
        
        # Add secondary dosha information if available
        secondary_doshas = [d for d in dosha_proportions if d != primary_dosha and dosha_proportions[d] >= 0.25]
        for dosha in secondary_doshas[:1]:  # Only get info for the strongest secondary dosha
            try:
                # Get secondary dosha information
                sec_results = enhanced_qdrant_client.search_by_dosha(
                    dosha_type=dosha,
                    query="brief description", 
                    limit=1
                )
                
                if sec_results and len(sec_results) > 0:
                    result = sec_results[0]
                    dosha_insights.append({
                        "dosha": dosha,
                        "query": f"{dosha} dosha brief description",
                        "insight": result.get("text", "No information available"),
                        "source": result.get("source", "Unknown"),
                        "page": result.get("page", 0),
                        "score": result.get("score", 0),
                        "is_dosha_specific": True,
                        "is_secondary": True
                    })
            except Exception as e:
                logger.error(f"Error fetching secondary dosha info: {str(e)}")
        
        # Combine all insights
        all_insights = insights + dosha_insights
        
        # Create the knowledge results
        knowledge_results = {
            "available": len(all_insights) > 0,
            "insights": all_insights,
            "queries_used": queries[:3],
            "dosha_distribution": dosha_proportions,
            "primary_dosha": primary_dosha
        }
        
        return knowledge_results

    def _get_dosha_proportions_from_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract dosha proportions from iris features.
        
        Args:
            features: Extracted iris features
            
        Returns:
            Dictionary with dosha types as keys and their proportions as values
        """
        dosha_proportions = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
        
        # Try to get dosha information from iris color
        if "color_summary" in features and features["color_summary"]:
            color_info = features["color_summary"][0]
            color_str = color_info.get("color", "")
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
            
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                
                # Associate colors with doshas
                if r < 100 and g < 100 and b > 150:  # Blue - Vata
                    dosha_proportions["vata"] += 0.6
                    dosha_proportions["pitta"] += 0.2
                    dosha_proportions["kapha"] += 0.2
                elif r > 150 and g < 100 and b < 100:  # Reddish - Pitta
                    dosha_proportions["pitta"] += 0.7
                    dosha_proportions["vata"] += 0.2
                    dosha_proportions["kapha"] += 0.1
                elif r > 100 and g > 50 and b < 50:  # Brown - Kapha
                    dosha_proportions["kapha"] += 0.6
                    dosha_proportions["pitta"] += 0.3
                    dosha_proportions["vata"] += 0.1
                elif r < 100 and g > 150 and b < 100:  # Green - Pitta/Kapha
                    dosha_proportions["pitta"] += 0.5
                    dosha_proportions["kapha"] += 0.4
                    dosha_proportions["vata"] += 0.1
                elif r > 150 and g > 150 and b < 100:  # Yellow/amber - Pitta/Vata
                    dosha_proportions["pitta"] += 0.5
                    dosha_proportions["vata"] += 0.4
                    dosha_proportions["kapha"] += 0.1
        
        # Adjust based on pupil/iris ratio if available
        if "pupil_iris_ratio" in features:
            ratio = features["pupil_iris_ratio"]
            if ratio > 0.4:  # Large pupils - more Vata
                dosha_proportions["vata"] = min(1.0, dosha_proportions["vata"] + 0.2)
                dosha_proportions["pitta"] = max(0.0, dosha_proportions["pitta"] - 0.1)
                dosha_proportions["kapha"] = max(0.0, dosha_proportions["kapha"] - 0.1)
            elif ratio < 0.25:  # Small pupils - more Kapha
                dosha_proportions["kapha"] = min(1.0, dosha_proportions["kapha"] + 0.2)
                dosha_proportions["vata"] = max(0.0, dosha_proportions["vata"] - 0.1)
                dosha_proportions["pitta"] = max(0.0, dosha_proportions["pitta"] - 0.1)
        
        # Normalize to ensure they sum to 1.0
        total = sum(dosha_proportions.values())
        if total > 0:
            dosha_proportions = {k: v/total for k, v in dosha_proportions.items()}
        
        return dosha_proportions
