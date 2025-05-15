from typing import Tuple, Dict

def verify_qdrant_collections(qdrant_client, enhanced_qdrant_client) -> Tuple[bool, bool]:
    """
    Verify if the Qdrant collections exist and have data.
    
    Args:
        qdrant_client: Standard IrisQdrantClient instance
        enhanced_qdrant_client: EnhancedIrisQdrantClient instance
        
    Returns:
        Tuple containing (standard_initialized, enhanced_initialized) status
    """
    standard_initialized = False
    enhanced_initialized = False
    
    try:
        # Check standard collection
        try:
            collections = qdrant_client.client.get_collections()
            collection_exists = any(collection.name == "iris_chunks" for collection in collections.collections)
            if collection_exists:
                # Check if it has data
                count = qdrant_client.client.count(
                    collection_name="iris_chunks"
                ).count
                if count > 0:
                    standard_initialized = True
        except Exception as e:
            print(f"Error checking standard collection: {str(e)}")
            
        # Check enhanced collection
        try:
            collections = enhanced_qdrant_client.client.get_collections()
            collection_exists = any(collection.name == "enhanced_iris_chunks" for collection in collections.collections)
            if collection_exists:
                # Check if it has data
                count = enhanced_qdrant_client.client.count(
                    collection_name="enhanced_iris_chunks"
                ).count
                if count > 0:
                    enhanced_initialized = True
        except Exception as e:
            print(f"Error checking enhanced collection: {str(e)}")
            
    except Exception as e:
        print(f"Error checking Qdrant status: {str(e)}")
        
    return standard_initialized, enhanced_initialized

def create_enhanced_chunk_from_standard(standard_chunk: Dict) -> Dict:
    """
    Convert a standard chunk to an enhanced chunk format.
    
    Args:
        standard_chunk: A standard chunk from the standard knowledge base
        
    Returns:
        An enhanced chunk with additional metadata
    """
    # Extract text for basic sentence splitting
    text = standard_chunk["text"]
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Create enhanced chunk
    enhanced_chunk = {
        "text": text,
        "page": standard_chunk["page"],
        "source": standard_chunk["source"],
        "extraction_method": standard_chunk.get("extraction_method", "standard"),
        "keywords": ["iris", "iridology"],  # Basic keywords
        "relevance_score": standard_chunk.get("score", 0.7),
        "sentences": sentences if sentences else [text],
        "sentence_count": len(sentences) if sentences else 1,
        "paragraph_idx": 0,
        "entities": []
    }
    
    return enhanced_chunk
