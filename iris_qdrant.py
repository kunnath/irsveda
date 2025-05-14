from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any, Optional


class IrisQdrantClient:
    def __init__(self, host="localhost", port=6333, collection_name="iris_chunks"):
        """Initialize Qdrant client and model for embeddings."""
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
                )
            )
            print(f"Created new collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")
        
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Embed and store chunks in Qdrant.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            List of IDs for the stored points
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        
        # Create points with unique IDs
        point_ids = [str(uuid.uuid4()) for _ in chunks]
        points = [
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk
            )
            for point_id, vector, chunk in zip(point_ids, embeddings, chunks)
        ]
        
        # Upload points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return point_ids
        
    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for iris chunks matching the query.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        query_vector = self.model.encode(query).tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                "text": match.payload["text"],
                "page": match.payload.get("page", 0),
                "source": match.payload.get("source", ""),
                "score": match.score
            }
            for match in results
        ]
