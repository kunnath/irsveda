from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, util
import torch
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re

class EnhancedIrisQdrantClient:
    def __init__(self, host="localhost", port=6333, collection_name="iris_chunks"):
        """Initialize enhanced Qdrant client with better embedding model and search methods."""
        self.client = QdrantClient(host=host, port=port)
        
        # Use a more powerful model for better embeddings
        self.model = SentenceTransformer("all-mpnet-base-v2")  # More advanced than MiniLM
        self.collection_name = collection_name
        
    def create_collection(self):
        """Create the collection for storing iris chunks if it doesn't exist."""
        vector_size = self.model.get_sentence_embedding_dimension()
        
        # Check if collection exists first
        collections = self.client.get_collections()
        collection_exists = any(collection.name == self.collection_name for collection in collections.collections)
        
        if not collection_exists:
            # Only create if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                # Add payload indexing for faster filtering
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0
                )
            )
            print(f"Created new collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")
        
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Embed and store chunks in Qdrant with improved embedding strategy.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            List of IDs for the stored points
        """
        # Process in batches for better memory usage
        batch_size = 32
        all_point_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            
            # Extract texts for embedding
            texts = [self._prepare_text_for_embedding(chunk) for chunk in batch_chunks]
            
            # Generate embeddings
            print(f"Generating embeddings for batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}...")
            embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
            
            # Create points with unique IDs
            point_ids = [str(uuid.uuid4()) for _ in batch_chunks]
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=self._enhance_payload(chunk)
                )
                for point_id, vector, chunk in zip(point_ids, embeddings, batch_chunks)
            ]
            
            # Upload points to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            all_point_ids.extend(point_ids)
            print(f"Stored {len(point_ids)} chunks in batch")
        
        return all_point_ids
    
    def _prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """Prepare text for embedding with enhanced context."""
        # Start with the main text
        text = chunk["text"]
        
        # Enhance with keywords if available
        if "keywords" in chunk and chunk["keywords"]:
            text += f" Keywords: {', '.join(chunk['keywords'])}"
            
        # Add entity information if available
        if "entities" in chunk and chunk["entities"]:
            entity_texts = [e["text"] for e in chunk["entities"]]
            if entity_texts:
                text += f" Entities: {', '.join(entity_texts)}"
                
        return text
    
    def _enhance_payload(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the payload with additional metadata for improved search."""
        # Start with a copy of the chunk
        payload = chunk.copy()
        
        # Add word and sentence counts
        if "text" in payload:
            text = payload["text"]
            payload["word_count"] = len(text.split())
            
        # Add flags for filtering
        if "keywords" in payload:
            payload["has_keywords"] = len(payload["keywords"]) > 0
            
        if "entities" in payload:
            payload["has_entities"] = len(payload["entities"]) > 0
            
        return payload
    
    def hybrid_search(self, query: str, limit: int = 10, min_score: float = 0.6) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of matching chunks with scores and highlights
        """
        # Pre-process query
        processed_query, query_keywords = self._process_query(query)
        
        # Generate query embedding
        query_vector = self.model.encode(processed_query).tolist()
        
        # Prepare Qdrant search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit * 2,  # Fetch more for re-ranking
            score_threshold=min_score  # Filter by minimum score
        )
        
        # Re-rank results with hybrid scoring
        scored_results = []
        for match in search_results:
            # Start with vector similarity score
            base_score = match.score
            
            # Get text and metadata
            text = match.payload.get("text", "")
            keywords = match.payload.get("keywords", [])
            
            # Calculate keyword overlap boost
            keyword_boost = self._calculate_keyword_overlap(query_keywords, keywords, text)
            
            # Calculate final score with boosting
            final_score = min(1.0, base_score + keyword_boost)
            
            # Generate highlights
            highlights = self._generate_highlights(text, query_keywords)
            
            # Add to results
            scored_results.append({
                "text": text,
                "page": match.payload.get("page", 0),
                "source": match.payload.get("source", ""),
                "score": final_score,
                "vector_score": base_score,
                "keyword_boost": keyword_boost,
                "highlights": highlights,
                "extraction_method": match.payload.get("extraction_method", "standard"),
                "keywords": keywords
            })
        
        # Sort by final score and limit results
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
    
    def _process_query(self, query: str) -> Tuple[str, List[str]]:
        """Process query to extract keywords and improve search."""
        # Extract keywords from query
        words = query.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'with', 'by', 'about', 'is', 'are'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Enhance query to improve semantic matching
        enhanced_query = query
        
        return enhanced_query, keywords
    
    def _calculate_keyword_overlap(self, query_keywords: List[str], doc_keywords: List[str], text: str) -> float:
        """Calculate keyword overlap boost."""
        if not query_keywords:
            return 0.0
            
        # Check direct keyword matches
        direct_matches = sum(1 for qk in query_keywords if qk in doc_keywords)
        
        # Check for keyword presence in text
        text_lower = text.lower()
        text_matches = sum(1 for qk in query_keywords if qk in text_lower)
        
        # Calculate boost based on overlap
        direct_boost = direct_matches / len(query_keywords) * 0.1
        text_boost = text_matches / len(query_keywords) * 0.05
        
        return direct_boost + text_boost
    
    def _generate_highlights(self, text: str, query_keywords: List[str]) -> List[str]:
        """Generate text highlights for matching keywords."""
        if not query_keywords:
            return []
            
        highlights = []
        text_lower = text.lower()
        
        # Find sentences with keywords
        for keyword in query_keywords:
            if keyword in text_lower:
                # Simple highlighting - extract context around keyword
                start_pos = text_lower.find(keyword)
                if start_pos >= 0:
                    # Get context before and after (about 50 chars)
                    start = max(0, start_pos - 50)
                    end = min(len(text), start_pos + len(keyword) + 50)
                    
                    # Try to start/end at word boundaries
                    while start > 0 and text[start] != ' ':
                        start -= 1
                    while end < len(text) - 1 and text[end] != ' ':
                        end += 1
                        
                    context = text[start:end].strip()
                    # Bold the keyword (for markdown display)
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    highlighted = pattern.sub(f"**{keyword}**", context)
                    
                    highlights.append(f"...{highlighted}...")
        
        # Remove duplicates and limit number of highlights
        unique_highlights = list(set(highlights))
        return unique_highlights[:3]
    
    def multi_query_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Execute multiple query variations to improve recall.
        
        Args:
            query: Original user query
            limit: Maximum number of results to return
            
        Returns:
            List of matching chunks with scores and highlights
        """
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        # Execute searches for each variation
        all_results = []
        seen_texts = set()
        
        for q in query_variations:
            results = self.hybrid_search(q, limit=limit)
            
            # Add only unseen results
            for result in results:
                text = result["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(result)
        
        # Re-rank combined results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:limit]
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate variations of the query for better coverage."""
        variations = [query]  # Start with original query
        
        # Extract main keywords
        words = query.lower().split()
        if len(words) >= 3:
            # Add a variation with just the main keywords
            main_keywords = ' '.join([w for w in words if len(w) > 3])
            variations.append(main_keywords)
        
        # Add specific iridology terms if not present
        if 'iris' not in query.lower() and 'iridology' not in query.lower():
            variations.append(f"{query} iris")
            variations.append(f"{query} iridology")
        
        return variations
