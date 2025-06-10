import sys
from qdrant_client import QdrantClient
import os

def test_qdrant_connection():
    # Connect to Qdrant - try with both localhost and Docker service name
    try:
        # First try with Docker service name if we're inside a container
        print("Attempting to connect to Qdrant using Docker service name...")
        client = QdrantClient(host="qdrant", port=6333)
        collections = client.get_collections()
        print(f"✅ Successfully connected to Qdrant (service name)!")
        print(f"Available collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect using Docker service name: {str(e)}")
        
        try:
            # Try with localhost for local development
            print("\nAttempting to connect to Qdrant using localhost...")
            client = QdrantClient(host="localhost", port=6333)
            collections = client.get_collections()
            print(f"✅ Successfully connected to Qdrant (localhost)!")
            print(f"Available collections: {[c.name for c in collections.collections]}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect using localhost: {str(e)}")
            
    return False

if __name__ == "__main__":
    print("Testing Qdrant connection...")
    if test_qdrant_connection():
        print("\nQdrant connection test passed!")
        sys.exit(0)
    else:
        print("\nQdrant connection test failed!")
        sys.exit(1)
