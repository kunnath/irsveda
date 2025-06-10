#!/usr/bin/env python3
"""
Dosha Knowledge Base Checker

This utility script checks if dosha-related content (Vata, Pitta, Kapha) is available 
in the Qdrant vector database and retrieves sample entries for each dosha.
"""

import sys
import argparse
from typing import Dict, List, Any
from enhanced_iris_qdrant import EnhancedIrisQdrantClient
import json

def check_dosha_knowledge(qdrant_host="localhost", qdrant_port=6333, collection_name="enhanced_iris_chunks"):
    """
    Check if dosha-related knowledge is available in the Qdrant database.
    
    Args:
        qdrant_host: Host of the Qdrant server
        qdrant_port: Port of the Qdrant server
        collection_name: Name of the Qdrant collection to check
        
    Returns:
        Dictionary with dosha information statistics
    """
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}, collection: {collection_name}")
    
    # Connect to Qdrant
    try:
        client = EnhancedIrisQdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        # Check if collection exists
        collections = client.client.get_collections()
        collection_exists = any(collection.name == collection_name for collection in collections.collections)
        
        if not collection_exists:
            print(f"Error: Collection '{collection_name}' does not exist")
            return {
                "status": "error",
                "message": f"Collection '{collection_name}' does not exist"
            }
            
        print(f"Connected to Qdrant. Collection '{collection_name}' exists.")
        
        # Check for dosha-related content
        dosha_stats = {}
        dosha_examples = {}
        
        # Search for each dosha type
        for dosha in ["vata", "pitta", "kapha"]:
            print(f"\nChecking for {dosha.capitalize()} content...")
            
            # Search specifically for this dosha
            results = client.multi_query_search(
                query=f"{dosha} dosha characteristics in iris",
                limit=5
            )
            
            # Also search for dosha in iris analysis
            results2 = client.multi_query_search(
                query=f"{dosha} iris analysis",
                limit=5
            )
            
            # Combine results
            all_results = results + results2
            unique_results = []
            seen_texts = set()
            
            for result in all_results:
                text = result["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append(result)
            
            # Store statistics
            dosha_stats[dosha] = len(unique_results)
            
            # Store example snippets (first 3)
            dosha_examples[dosha] = []
            for result in unique_results[:3]:
                dosha_examples[dosha].append({
                    "text": result["text"][:200] + "...",  # First 200 chars
                    "score": result["score"],
                    "page": result["page"],
                    "source": result["source"].split("/")[-1],  # Just the filename
                    "keywords": result.get("keywords", [])[:10]  # First 10 keywords
                })
                
        # Summary
        total_chunks = sum(dosha_stats.values())
        
        summary = {
            "status": "success",
            "total_dosha_chunks_found": total_chunks,
            "dosha_distribution": dosha_stats,
            "has_vata": dosha_stats["vata"] > 0,
            "has_pitta": dosha_stats["pitta"] > 0,
            "has_kapha": dosha_stats["kapha"] > 0,
            "examples": dosha_examples
        }
        
        return summary
        
    except Exception as e:
        print(f"Error checking dosha knowledge: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check for dosha-related content in Qdrant')
    parser.add_argument('--host', default='localhost', help='Qdrant host (default: localhost)')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port (default: 6333)')
    parser.add_argument('--collection', default='enhanced_iris_chunks', 
                       help='Qdrant collection name (default: enhanced_iris_chunks)')
    parser.add_argument('--output', help='Output JSON file for detailed results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    
    # Check for dosha knowledge
    results = check_dosha_knowledge(
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection
    )
    
    # Print summary
    print("\n=== DOSHA KNOWLEDGE BASE SUMMARY ===")
    if results["status"] == "success":
        print(f"Total Dosha-Related Chunks: {results['total_dosha_chunks_found']}")
        print("\nDosha Distribution:")
        for dosha, count in results["dosha_distribution"].items():
            print(f"  - {dosha.capitalize()}: {count} chunks")
        
        print("\nDosha Availability:")
        for dosha in ["vata", "pitta", "kapha"]:
            status = "✅ Available" if results[f"has_{dosha}"] else "❌ Not Available"
            print(f"  - {dosha.capitalize()}: {status}")
        
        if args.verbose:
            # Print example snippets for each dosha
            for dosha in ["vata", "pitta", "kapha"]:
                print(f"\n{dosha.capitalize()} Examples:")
                for i, example in enumerate(results["examples"][dosha]):
                    print(f"  Example {i+1}:")
                    print(f"    Text: {example['text']}")
                    print(f"    Source: {example['source']}, Page: {example['page']}")
                    print(f"    Score: {example['score']:.2f}")
                    if example.get("keywords"):
                        print(f"    Keywords: {', '.join(example['keywords'])}")
                    print()
    else:
        print(f"Error: {results['message']}")
    
    # Write to output file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results written to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    
    # Return success/failure
    return 0 if results["status"] == "success" else 1

if __name__ == "__main__":
    sys.exit(main())
