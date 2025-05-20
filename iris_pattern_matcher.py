"""
Iris pattern matching module for IridoVeda.

This module provides functionality to store and retrieve iris patterns using Qdrant,
enabling similarity search and pattern matching.
"""

import numpy as np
import uuid
from typing import Dict, List, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct, Distance, VectorParams
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisPatternMatcher:
    """
    Class for matching iris patterns using Qdrant vector database.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "iris_patterns"):
        """
        Initialize the iris pattern matcher.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection to store iris patterns
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = 256  # Default size, will be updated when features are processed
    
    def create_collection(self, vector_size: Optional[int] = None) -> bool:
        """
        Create a collection for iris patterns if it doesn't exist.
        
        Args:
            vector_size: Size of the feature vectors
            
        Returns:
            True if successful
        """
        try:
            # Update vector size if provided
            if vector_size:
                self.vector_size = vector_size
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False
    
    def convert_features_to_vector(self, features: Dict[str, Any]) -> list:
        """
        Convert extracted features to a single vector representation.
        
        Args:
            features: Dictionary of iris features
            
        Returns:
            A flat vector representation
        """
        try:
            vector = []
            
            # Add color features
            if "color_features" in features and features["color_features"]:
                # For each dominant color, add RGB values and percentage
                for color_data in features["color_features"][:3]:  # Use top 3 colors
                    rgb = color_data["color"]
                    # Handle different rgb formats (tuple, list, np.array)
                    if isinstance(rgb, tuple) or isinstance(rgb, list):
                        # Convert to list for consistency and manually normalize each component
                        rgb_list = list(rgb) if isinstance(rgb, tuple) else rgb
                        vector.extend([r / 255.0 for r in rgb_list])
                    else:
                        # Assume it's a numpy array that can be divided
                        vector.extend(rgb / 255.0)  # Normalize to [0, 1]
                    vector.append(color_data["percentage"])

            # Add texture features
            if "texture_features" in features:
                # Add contrast, uniformity, energy, entropy
                texture = features["texture_features"]
                if "contrast" in texture:
                    vector.append(texture["contrast"] / 255.0)  # Normalize
                if "uniformity" in texture:
                    vector.append(texture["uniformity"])
                if "energy" in texture:
                    vector.append(texture["energy"] / 255.0)  # Normalize
                if "entropy" in texture:
                    vector.append(min(texture["entropy"] / 8.0, 1.0))  # Normalize, max entropy is ~8
                
                # Add histogram summary (we can't add the full histogram as it would be too large)
                if "lbp_histogram" in texture:
                    hist = np.array(texture["lbp_histogram"])
                    if len(hist) > 0:
                        # Add statistical measures of histogram
                        vector.append(np.mean(hist))
                        vector.append(np.std(hist))
                        vector.append(np.median(hist))
                        # Add a few samples from histogram
                        samples = np.linspace(0, len(hist)-1, 10, dtype=int)
                        vector.extend(hist[samples])
            
            # Add spot statistics
            if "spots" in features:
                spots = features["spots"]
                # Add number of spots (normalized)
                vector.append(min(len(spots) / 50.0, 1.0))  # Normalize, assume max 50 spots
                
                if spots:
                    # Add statistics about spots
                    areas = [spot["area"] for spot in spots]
                    positions = [spot["relative_position"] for spot in spots]
                    intensities = [spot["intensity"] for spot in spots if "intensity" in spot]
                    
                    vector.append(np.mean(areas) / 500.0)  # Normalized average area
                    vector.append(np.std(areas) / 500.0)  # Normalized std dev of areas
                    vector.append(np.mean(positions))  # Average relative position [0-1]
                    
                    if intensities:
                        vector.append(np.mean(intensities) / 255.0)  # Normalized average intensity
            
            # Add radial features
            if "radial_features" in features and "radial_data" in features["radial_features"]:
                radial_data = features["radial_features"]["radial_data"]
                if radial_data:
                    # Sample mean intensities at key angles
                    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions
                    for angle in angles:
                        # Find closest radial data point to this angle
                        closest = min(radial_data, key=lambda x: abs(x["angle"] - angle))
                        vector.append(closest["mean_intensity"] / 255.0)
                        vector.append(closest["std_intensity"] / 128.0)  # Normalize
            
            # Fill any remaining space with zeros to ensure consistent vector size
            if len(vector) < self.vector_size:
                vector.extend([0] * (self.vector_size - len(vector)))
            
            # Truncate if too long
            if len(vector) > self.vector_size:
                vector = vector[:self.vector_size]
            
            return vector
            
        except Exception as e:
            logger.error(f"Error converting features to vector: {str(e)}")
            # Return a default vector filled with zeros
            return [0] * self.vector_size
    
    def store_iris_pattern(self, features: Dict[str, Any], metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Store iris pattern features in Qdrant for future matching.
        
        Args:
            features: Dictionary containing extracted iris features
            metadata: Optional metadata for this iris pattern
            
        Returns:
            Pattern ID if successful, None otherwise
        """
        try:
            # Convert features to vector
            vector = self.convert_features_to_vector(features)
            
            # Get a unique ID for this pattern
            pattern_id = str(uuid.uuid4())
            
            # Ensure metadata is a dictionary
            metadata = metadata or {}
            
            # Extract main color for summary
            main_color = []
            color_features = features.get("color_features", [])
            if color_features:
                color_val = color_features[0].get("color", (0, 0, 0))
                
                # Handle different types properly
                if isinstance(color_val, tuple) or isinstance(color_val, list):
                    main_color = list(color_val) if isinstance(color_val, tuple) else color_val
                elif hasattr(color_val, 'tolist'):
                    main_color = color_val.tolist()
                else:
                    main_color = color_val
            
            # Ensure all numpy values are converted to native Python types
            def convert_numpy_values(data):
                if isinstance(data, dict):
                    return {k: convert_numpy_values(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [convert_numpy_values(item) for item in data]
                elif hasattr(data, 'tolist') and callable(getattr(data, 'tolist')):
                    return data.tolist()
                elif hasattr(data, 'item') and callable(getattr(data, 'item')):
                    return data.item()
                else:
                    return data
            
            # Convert all numpy values in features
            features = convert_numpy_values(features)
                    
            feature_summary = {
                "num_spots": features.get("num_spots", 0),
                "main_color": main_color,
                "contrast": features.get("texture_features", {}).get("contrast", 0),
                "uniformity": features.get("texture_features", {}).get("uniformity", 0)
            }
            
            # Create the point
            point = PointStruct(
                id=pattern_id,
                vector=vector,
                payload={
                    "metadata": metadata,
                    "feature_summary": feature_summary,
                    # Store a subset of the original features to save space
                    "color_features": [
                        {"color": list(c["color"]) if isinstance(c["color"], tuple) else 
                                 c["color"].tolist() if hasattr(c["color"], 'tolist') else c["color"], 
                         "percentage": c["percentage"]} 
                        for c in features.get("color_features", [])[:3]
                    ],
                    "num_spots": features.get("num_spots", 0),
                    "texture_stats": {
                        k: features.get("texture_features", {}).get(k, 0)
                        for k in ["contrast", "uniformity", "energy", "entropy"]
                    }
                }
            )
            
            # Insert the point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored iris pattern with ID: {pattern_id}")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error storing iris pattern: {str(e)}")
            return None
    
    def search_similar_patterns(self, features: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar iris patterns.
        
        Args:
            features: Dictionary of iris features to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching patterns with scores and metadata
        """
        try:
            # Convert features to vector
            vector = self.convert_features_to_vector(features)
            
            # Perform the search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {}),
                    "feature_summary": result.payload.get("feature_summary", {})
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching iris patterns: {str(e)}")
            return []
    
    def search_by_vector(self, vector: list, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar patterns by vector.
        
        Args:
            vector: Vector representation of iris features
            limit: Maximum number of results to return
            
        Returns:
            List of similar patterns with score and payload
        """
        # Check if collection exists
        collections = self.client.get_collections()
        collection_exists = any(collection.name == self.collection_name for collection in collections.collections)
        
        if not collection_exists:
            logger.info(f"Collection '{self.collection_name}' not found")
            return []
        
        # Perform vector search
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit
            )
            
            # Process results
            patterns = []
            for match in search_results:
                patterns.append({
                    "id": match.id,
                    "score": match.score,
                    "payload": match.payload if match.payload else {}
                })
            
            return patterns
        except Exception as e:
            logger.error(f"Error searching by vector: {str(e)}")
            return []
    
    def get_pattern_by_id(self, pattern_id: str) -> Dict[str, Any]:
        """
        Retrieve an iris pattern by ID.
        
        Args:
            pattern_id: ID of the pattern to retrieve
            
        Returns:
            Pattern data with features and metadata
        """
        try:
            # Get the point
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[pattern_id]
            )
            
            if not points:
                return None
                
            point = points[0]
            
            return {
                "id": point.id,
                "metadata": point.payload.get("metadata", {}),
                "feature_summary": point.payload.get("feature_summary", {}),
                "color_features": point.payload.get("color_features", []),
                "texture_stats": point.payload.get("texture_stats", {})
            }
            
        except Exception as e:
            logger.error(f"Error retrieving iris pattern: {str(e)}")
            return None
