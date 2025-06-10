import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List, Any, Tuple, Optional

# Import dosha quantification model
from dosha_quantification_model import DoshaQuantificationModel

class IrisPredictor:
    """Class for analyzing iris images and generating predictions/queries."""
    
    def __init__(self):
        """Initialize the iris predictor."""
        # In a real implementation, you would load models here
        # Initialize the dosha quantification model
        self.dosha_model = DoshaQuantificationModel()
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the iris image for analysis.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # In a real implementation, you would resize, normalize, etc.
        
        return img
    
    def detect_iris(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect and extract the iris region from an eye image.
        
        Args:
            image: Input eye image
            
        Returns:
            Extracted iris image and metadata
        """
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Detect the eye
        # 2. Find the iris boundary using Hough transform
        # 3. Extract the iris region
        
        # For now, just return the original image and dummy metadata
        height, width = image.shape[:2]
        
        # Dummy iris circle (center_x, center_y, radius)
        iris_circle = (width // 2, height // 2, min(width, height) // 3)
        
        metadata = {
            "iris_detected": True,
            "iris_circle": iris_circle,
            "confidence": 0.95
        }
        
        return image, metadata
    
    def analyze_iris(self, iris_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the iris image and identify health indicators.
        
        Args:
            iris_image: Preprocessed iris image
            
        Returns:
            Dictionary containing analysis results
        """
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Segment the iris into zones
        # 2. Detect features in each zone
        # 3. Map features to health conditions
        
        # Dummy analysis results
        return {
            "zones": {
                "liver": {"condition": "normal", "confidence": 0.85},
                "kidney": {"condition": "stressed", "confidence": 0.75},
                "lungs": {"condition": "normal", "confidence": 0.90},
                "heart": {"condition": "normal", "confidence": 0.82},
                "stomach": {"condition": "inflamed", "confidence": 0.78},
            },
            "overall_health": "good",
        }
    
    def generate_queries(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate natural language queries based on iris analysis.
        
        Args:
            analysis: Results of iris analysis
            
        Returns:
            List of queries related to the analysis
        """
        queries = []
        
        # Generate queries for concerning areas
        for zone, info in analysis["zones"].items():
            if info["condition"] != "normal" and info["confidence"] > 0.7:
                queries.append(f"Why does the iris show {info['condition']} condition in {zone} zone?")
                queries.append(f"How to improve {zone} health when iris shows {info['condition']} signs?")
        
        # Add general query if no specific issues
        if not queries:
            queries.append("What does a healthy iris look like?")
            
        return queries
    
    def generate_enhanced_queries(self, analysis: Dict[str, Any], qdrant_client=None) -> List[str]:
        """
        Generate more relevant and contextual queries based on iris analysis and knowledge base.
        
        Args:
            analysis: Results of iris analysis
            qdrant_client: Optional Qdrant client for knowledge base lookup
            
        Returns:
            List of contextually relevant queries
        """
        # Start with basic queries
        basic_queries = self.generate_queries(analysis)
        
        if qdrant_client is None:
            return basic_queries
            
        # Create enriched queries by leveraging the knowledge base
        enriched_queries = []
        seed_terms = set()
        
        # Extract key terms based on analysis
        for zone, info in analysis["zones"].items():
            if info["condition"] != "normal":
                # Use zone and condition as seed terms
                seed_terms.add(zone)
                seed_terms.add(info["condition"])
                
                # Create enrichment queries to search knowledge base
                enrichment_query = f"{zone} {info['condition']} iridology"
                
                try:
                    # Search for related content in the knowledge base
                    results = qdrant_client.search(enrichment_query, limit=3)
                    
                    if results:
                        # Extract key terms from search results
                        for result in results:
                            # Generate a specific query based on this result
                            context = result["text"][:200]  # Use first 200 chars as context
                            
                            # Extract key phrases from the context
                            phrases = self._extract_key_phrases(context)
                            
                            for phrase in phrases:
                                if len(phrase) > 3 and phrase.lower() not in ("what", "when", "where", "which", "this", "that"):
                                    related_query = f"What is the relationship between {zone} and {phrase} in iridology?"
                                    if related_query not in enriched_queries and related_query not in basic_queries:
                                        enriched_queries.append(related_query)
                                        
                            # Create a treatment-focused query
                            if info["condition"] in ("stressed", "inflamed", "weak", "compromised"):
                                treatment_query = f"Ayurvedic remedies for {info['condition']} {zone} as shown in iris"
                                if treatment_query not in enriched_queries and treatment_query not in basic_queries:
                                    enriched_queries.append(treatment_query)
                except Exception as e:
                    print(f"Error enriching queries with knowledge base: {str(e)}")
        
        # Combine basic and enriched queries, prioritizing enriched ones
        combined_queries = []
        
        # Limit to reasonable number of queries
        max_queries = 8
        
        # Add most relevant enriched queries first
        for query in enriched_queries[:max_queries//2]:
            if query not in combined_queries and len(combined_queries) < max_queries:
                combined_queries.append(query)
        
        # Add basic queries if we still have space
        for query in basic_queries:
            if query not in combined_queries and len(combined_queries) < max_queries:
                combined_queries.append(query)
                
        # Fill any remaining slots with enriched queries
        for query in enriched_queries[max_queries//2:]:
            if query not in combined_queries and len(combined_queries) < max_queries:
                combined_queries.append(query)
        
        return combined_queries
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for query generation."""
        # Simple implementation using common NLP techniques
        import re
        from collections import Counter
        
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most common words as potential key terms
        word_counts = Counter(filtered_words)
        common_words = [word for word, _ in word_counts.most_common(5)]
        
        # Extract bigrams (pairs of adjacent words)
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 6:  # Avoid very short bigrams
                    bigrams.append(bigram)
        
        # Combine single words and bigrams
        key_phrases = common_words + bigrams
        
        return key_phrases[:7]  # Limit to top 7 phrases
    
    def process_iris_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an iris image and return analysis and queries.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Dictionary with analysis results and suggested queries
        """
        # Preprocess the image
        preprocessed_img = self.preprocess_image(image_path)
        
        # Detect iris
        iris_img, iris_metadata = self.detect_iris(preprocessed_img)
        
        if not iris_metadata["iris_detected"]:
            return {"error": "No iris detected in the image"}
        
        # Analyze iris
        analysis = self.analyze_iris(iris_img)
        
        # Generate queries
        queries = self.generate_queries(analysis)
        
        # Add dosha quantification analysis
        dosha_analysis = self.analyze_dosha_profile(image_path)
        
        return {
            "analysis": analysis,
            "metadata": iris_metadata,
            "queries": queries,
            "image": iris_img,
            "dosha_analysis": dosha_analysis
        }
    
    def analyze_dosha_profile(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze iris image to generate dosha profile and health indicators.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Dictionary containing dosha analysis results
        """
        try:
            # Process the iris image using the dosha quantification model
            dosha_report = self.dosha_model.process_iris_image(image_path)
            return dosha_report
        except Exception as e:
            print(f"Error in dosha analysis: {str(e)}")
            return {"error": f"Dosha analysis failed: {str(e)}"}
