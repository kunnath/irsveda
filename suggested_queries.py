"""
Suggested Queries Module for IridoVeda

This module generates relevant suggested queries based on the content in the knowledge base.
It helps users discover information by providing pre-formulated questions.
"""

import numpy as np
import nltk
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK packages are available
try:
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class SuggestedQueriesGenerator:
    """
    Generate suggested queries based on the knowledge base content.
    """
    
    def __init__(self):
        """Initialize the suggested queries generator."""
        self.stop_words = set(stopwords.words('english'))
        
        # Common prefixes for generating questions
        self.question_prefixes = [
            "What is", "How does", "Can", "Why is", "What are",
            "How can", "What causes", "How to identify", "Is there",
            "What do", "When should", "Where are"
        ]
        
        # Iridology-specific topics for generating questions
        self.iridology_topics = [
            "iris patterns", "iris colors", "iris zones", "health indicators",
            "digestive system", "nervous system", "constitutional types",
            "lymphatic system", "blood circulation", "detoxification",
            "iris diagnosis", "genetic predispositions", "constitutional weaknesses",
            "organ systems", "tissue integrity", "inflammation signs"
        ]
        
        # Predefined set of high-quality queries
        self.predefined_queries = [
            "What are the main iris constitutional types in iridology?",
            "How does iris color correlate with health conditions?",
            "What can the pupil shape tell us about health?",
            "How are the zones of the iris mapped to organs?",
            "What are radii solaris in iridology and what do they mean?",
            "How does the lymphatic system appear in the iris?",
            "What are the signs of digestive weakness in iris patterns?",
            "How does stress manifest in iris markings?",
            "What do white marks in the iris indicate?",
            "How can the iris show hormonal imbalances?",
            "What do brown spots in the iris mean?",
            "How does Ayurvedic iridology differ from Western iridology?",
            "What are the most reliable indicators in an iris analysis?",
            "How can iridology help with early disease detection?",
            "What is the significance of the collarette in iris analysis?",
            "How accurate is iridology in identifying health conditions?",
            "What do fiber structures in the iris represent?",
            "How can treatments be personalized using iris analysis?",
            "What are the main principles of iridology diagnosis?",
            "How does iris topography relate to organ systems?"
        ]
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and word.isalpha() and len(word) > 3
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Return most common words
        return [word for word, _ in word_freq.most_common(10)]
    
    def generate_queries_from_chunks(self, chunks: List[Dict[str, Any]], 
                                    num_queries: int = 5) -> List[str]:
        """
        Generate queries based on the content of text chunks.
        
        Args:
            chunks: List of text chunks from the knowledge base
            num_queries: Number of queries to generate
            
        Returns:
            List of generated queries
        """
        # Extract all text and combine
        all_text = " ".join([chunk.get("text", "") for chunk in chunks])
        
        # Extract keywords
        keywords = self.extract_keywords(all_text)
        
        # Generate queries by combining keywords with question prefixes
        generated_queries = []
        
        # Mix topic-based and keyword-based queries
        for _ in range(min(num_queries, 20)):
            if random.random() < 0.5 and keywords:
                # Keyword-based query
                prefix = random.choice(self.question_prefixes)
                keyword = random.choice(keywords)
                
                # Add some variation
                if random.random() < 0.3:
                    topic = random.choice(self.iridology_topics)
                    query = f"{prefix} {keyword} related to {topic}?"
                else:
                    query = f"{prefix} {keyword} in iridology?"
                    
                generated_queries.append(query)
            else:
                # Use a predefined query
                query = random.choice(self.predefined_queries)
                if query not in generated_queries:
                    generated_queries.append(query)
        
        # Remove duplicates and limit to requested number
        unique_queries = list(set(generated_queries))
        return unique_queries[:num_queries]
    
    def get_suggested_queries(self, num_queries: int = 5) -> List[str]:
        """
        Get a list of suggested queries without needing chunks.
        This is useful when no specific context is available.
        
        Args:
            num_queries: Number of queries to return
            
        Returns:
            List of suggested queries
        """
        # Shuffle the predefined queries and select the requested number
        queries = self.predefined_queries.copy()
        random.shuffle(queries)
        return queries[:num_queries]
    
    def generate_query_from_iris_features(self, features: Dict[str, Any]) -> List[str]:
        """
        Generate specific queries based on iris analysis features.
        
        Args:
            features: Dictionary of iris features
            
        Returns:
            List of relevant queries
        """
        queries = []
        
        # Generate queries based on detected features
        if "color_features" in features and features["color_features"]:
            dominant_color = features["color_features"][0]["color"]
            # Convert RGB tuple/list to string description
            if isinstance(dominant_color, (list, tuple)) and len(dominant_color) == 3:
                r, g, b = dominant_color
                if r > 100 and g > 100 and b < 80:  # Yellowish
                    queries.append("What does a yellowish iris indicate in iridology?")
                elif r > 100 and g < 80 and b < 80:  # Reddish
                    queries.append("What do reddish tones in the iris signify?")
                elif r < 80 and g < 80 and b > 100:  # Bluish
                    queries.append("What health traits are associated with blue iris?")
                elif r < 100 and g > 100 and b < 100:  # Greenish
                    queries.append("What does a green iris indicate about constitution?")
                elif r > 100 and g > 80 and b > 60:  # Brown
                    queries.append("What are the health indicators in brown iris patterns?")
        
        if "spot_count" in features:
            spot_count = features.get("spot_count", 0)
            if spot_count > 10:
                queries.append("What do multiple spots in the iris indicate?")
            elif 5 <= spot_count <= 10:
                queries.append("What is the significance of iris spots?")
            
        if "texture_features" in features:
            texture = features.get("texture_features", {})
            if texture.get("density", 0) > 0.7:
                queries.append("What does a dense iris fiber pattern indicate?")
            elif texture.get("density", 0) < 0.3:
                queries.append("What health conditions relate to loose iris fibers?")
                
            if texture.get("contrast", 0) > 0.7:
                queries.append("What does high contrast in iris patterns mean?")
        
        # Add some general iris analysis queries
        general_queries = [
            "How can iris analysis help with preventive health?",
            "What organ systems can be assessed through iris patterns?",
            "How accurate is iridology for health assessment?"
        ]
        
        # Combine and return unique queries
        all_queries = queries + general_queries
        random.shuffle(all_queries)
        
        # Return unique queries, up to 5
        unique_queries = []
        for query in all_queries:
            if query not in unique_queries:
                unique_queries.append(query)
                if len(unique_queries) >= 5:
                    break
                    
        return unique_queries
