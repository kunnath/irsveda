#!/usr/bin/env python3
"""
Script to check Qdrant connection status and provide detailed diagnostics
"""

import sys
import logging
import time
import socket
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_port_open(host, port, timeout=2):
    """Check if a port is open on the given host"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def check_qdrant_connection(host="localhost", port=6333, retries=1, wait_time=2):
    """
    Check connection to Qdrant server with detailed diagnostics
    
    Args:
        host: Hostname or IP of Qdrant server
        port: Port number
        retries: Number of connection retries
        wait_time: Time to wait between retries in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    logger.info(f"Checking Qdrant connection to {host}:{port}")
    
    # First check if the port is open
    if not is_port_open(host, port):
        logger.error(f"Port {port} is not open on {host}")
        logger.info("Possible causes:")
        logger.info("1. Qdrant server is not running")
        logger.info("2. Qdrant is running on a different port")
        logger.info("3. Firewall is blocking the connection")
        logger.info("\nSuggested solutions:")
        logger.info("1. Start Qdrant server with: docker-compose up -d qdrant")
        logger.info("2. Check if Qdrant container is running with: docker ps")
        logger.info("3. Verify the port in docker-compose.yml")
        return False
    
    # Now try to connect with the client
    for attempt in range(retries):
        try:
            client = QdrantClient(host=host, port=port)
            # Try to get collections to verify connection
            collections = client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {host}:{port}")
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")
            return True
        except UnexpectedResponse as e:
            # This might indicate Qdrant is starting up
            logger.warning(f"Unexpected response from Qdrant: {str(e)}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {wait_time} seconds... ({attempt+1}/{retries})")
                time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {str(e)}")
            logger.info("\nPossible solutions if Qdrant is running:")
            logger.info("1. Check Qdrant logs with: docker logs qdrant")
            logger.info("2. Restart Qdrant with: docker-compose restart qdrant")
            logger.info("3. Verify host and port configuration")
            return False
    
    return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check Qdrant connection status")
    
    parser.add_argument("--host", default="localhost", help="Qdrant server host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--retries", type=int, default=3, help="Number of connection retries")
    parser.add_argument("--wait", type=int, default=2, help="Seconds to wait between retries")
    
    args = parser.parse_args()
    
    success = check_qdrant_connection(
        host=args.host,
        port=args.port,
        retries=args.retries,
        wait_time=args.wait
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
