#!/usr/bin/env python3
"""
Dosha Knowledge Query Tool

This script provides a way to query dosha-specific knowledge from the Qdrant vector database.
It enables targeted searches for Vata, Pitta, or Kapha content with various filtering options.
"""

import sys
import argparse
from typing import Dict, List, Any
from enhanced_iris_qdrant import EnhancedIrisQdrantClient
import json
import os

def query_dosha_knowledge(
    query: str,
    dosha: str = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "enhanced_iris_chunks",
    limit: int = 5,
    min_score: float = 0.6
):
    """
    Query dosha-related knowledge from the Qdrant database.
    
    Args:
        query: The search query
        dosha: Specific dosha to filter by (vata, pitta, kapha, or None for all)
        qdrant_host: Host of the Qdrant server
        qdrant_port: Port of the Qdrant server
        collection_name: Name of the Qdrant collection
        limit: Maximum number of results to return
        min_score: Minimum relevance score threshold
        
    Returns:
        Dictionary with query results
    """
    try:
        # Connect to Qdrant
        client = EnhancedIrisQdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        # Modify query to be dosha-specific if requested
        if dosha:
            # Validate dosha type
            if dosha.lower() not in ["vata", "pitta", "kapha"]:
                return {
                    "status": "error",
                    "message": f"Invalid dosha type: {dosha}. Must be one of: vata, pitta, kapha"
                }
                
            # Enhance query with dosha-specific terms
            enhanced_query = f"{dosha} {query}"
            print(f"Searching for: {enhanced_query}")
            
            # Perform search
            results = client.multi_query_search(
                query=enhanced_query,
                limit=limit
            )
            
            # Filter results to ensure they're actually about the requested dosha
            filtered_results = []
            for result in results:
                # Check if the result mentions the dosha
                if dosha.lower() in result["text"].lower():
                    filtered_results.append(result)
                # Also check dosha scores if available
                elif "dosha_scores" in result and result["dosha_scores"].get(dosha.lower(), 0) >= 0.3:
                    filtered_results.append(result)
                # Or check primary dosha if available
                elif "primary_dosha" in result and result["primary_dosha"] == dosha.lower():
                    filtered_results.append(result)
                    
            # Sort by relevance score
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit results
            filtered_results = filtered_results[:limit]
            
            return {
                "status": "success",
                "query": enhanced_query,
                "dosha": dosha.lower(),
                "results_count": len(filtered_results),
                "results": filtered_results
            }
            
        else:
            # General query without dosha filtering
            print(f"Searching for: {query}")
            
            # Perform search
            results = client.multi_query_search(
                query=query,
                limit=limit
            )
            
            # Check which results are dosha-related
            for result in results:
                # Add a flag for each dosha
                result["is_vata_related"] = False
                result["is_pitta_related"] = False
                result["is_kapha_related"] = False
                
                # Check text content
                text_lower = result["text"].lower()
                if "vata" in text_lower:
                    result["is_vata_related"] = True
                if "pitta" in text_lower:
                    result["is_pitta_related"] = True
                if "kapha" in text_lower:
                    result["is_kapha_related"] = True
                    
                # Also check dosha scores if available
                if "dosha_scores" in result:
                    if result["dosha_scores"].get("vata", 0) >= 0.3:
                        result["is_vata_related"] = True
                    if result["dosha_scores"].get("pitta", 0) >= 0.3:
                        result["is_pitta_related"] = True
                    if result["dosha_scores"].get("kapha", 0) >= 0.3:
                        result["is_kapha_related"] = True
                        
                # Check primary dosha if available
                if "primary_dosha" in result:
                    if result["primary_dosha"] == "vata":
                        result["is_vata_related"] = True
                    elif result["primary_dosha"] == "pitta":
                        result["is_pitta_related"] = True
                    elif result["primary_dosha"] == "kapha":
                        result["is_kapha_related"] = True
            
            return {
                "status": "success",
                "query": query,
                "dosha": None,
                "results_count": len(results),
                "results": results
            }
    
    except Exception as e:
        print(f"Error querying dosha knowledge: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def format_result(result, max_text_length=300):
    """Format a single result for display."""
    text = result["text"]
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
        
    # Create formatted output
    output = []
    output.append(f"Text: {text}")
    output.append(f"Source: {os.path.basename(result['source'])}, Page: {result['page']}")
    output.append(f"Relevance Score: {result['score']:.2f}")
    
    # Add dosha information if available
    if "primary_dosha" in result and result["primary_dosha"]:
        output.append(f"Primary Dosha: {result['primary_dosha'].capitalize()}")
        
    if "dosha_scores" in result:
        scores = result["dosha_scores"]
        output.append("Dosha Scores:")
        for dosha, score in scores.items():
            output.append(f"  - {dosha.capitalize()}: {score:.2f}")
            
    # Add dosha relevance flags
    related_doshas = []
    if result.get("is_vata_related"):
        related_doshas.append("Vata")
    if result.get("is_pitta_related"):
        related_doshas.append("Pitta")
    if result.get("is_kapha_related"):
        related_doshas.append("Kapha")
        
    if related_doshas:
        output.append(f"Related Doshas: {', '.join(related_doshas)}")
    
    # Add keywords
    if "keywords" in result and result["keywords"]:
        output.append(f"Keywords: {', '.join(result['keywords'][:10])}")
        
    return "\n".join(output)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Query dosha-related knowledge from Qdrant')
    parser.add_argument('query', help='The search query')
    parser.add_argument('--dosha', choices=['vata', 'pitta', 'kapha'], 
                       help='Filter results by specific dosha')
    parser.add_argument('--host', default='localhost', help='Qdrant host (default: localhost)')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port (default: 6333)')
    parser.add_argument('--collection', default='enhanced_iris_chunks', 
                       help='Qdrant collection name (default: enhanced_iris_chunks)')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of results (default: 5)')
    parser.add_argument('--min-score', type=float, default=0.6, 
                       help='Minimum relevance score (default: 0.6)')
    parser.add_argument('--output', help='Output JSON file for detailed results')
    
    args = parser.parse_args()
    
    # Query dosha knowledge
    results = query_dosha_knowledge(
        query=args.query,
        dosha=args.dosha,
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection,
        limit=args.limit,
        min_score=args.min_score
    )
    
    # Print results
    if results["status"] == "success":
        if args.dosha:
            print(f"\n=== {args.dosha.upper()} KNOWLEDGE QUERY RESULTS ===")
        else:
            print("\n=== DOSHA KNOWLEDGE QUERY RESULTS ===")
            
        print(f"Query: {results['query']}")
        print(f"Found {results['results_count']} matching results")
        
        if results["results_count"] > 0:
            # Print each result
            for i, result in enumerate(results["results"]):
                print(f"\n--- Result {i+1} ---")
                print(format_result(result))
        else:
            print("\nNo matching results found. Try a different query or dosha type.")
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
