from typing import List, Dict, Any
import re
from collections import Counter
import nltk

# Ensure NLTK packages are available
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ContextAwareAnswerGenerator:
    """Generate well-formed answers based on Qdrant search results."""
    
    def __init__(self):
        """Initialize the answer generator."""
        pass
    
    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer based on the search results.
        
        Args:
            query: The original user query
            search_results: List of search results from Qdrant
            
        Returns:
            Dict containing answer, sources, and other info
        """
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information on that topic in the iridology knowledge base.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Extract the most relevant passages
        passages = [result["text"] for result in search_results]
        sources = self._extract_sources(search_results)
        
        # Extract key insights from the passages
        insights = self._extract_key_insights(passages, query)
        
        # Generate a comprehensive answer
        answer = self._synthesize_answer(query, insights, search_results)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(search_results)
        
        return {
            "answer": answer,
            "sources": sources,
            "insights": insights,
            "confidence": confidence,
            "search_results": search_results[:3]  # Include top 3 raw results
        }
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and format source information."""
        sources = []
        seen_references = set()
        
        for result in search_results:
            source = result.get("source", "")
            page = result.get("page", 0)
            
            if source:
                # Extract filename from path
                source_name = source.split("/")[-1]
                reference = f"{source_name}, Page {page}"
                
                if reference not in seen_references:
                    seen_references.add(reference)
                    sources.append({
                        "title": source_name,
                        "page": page,
                        "score": result.get("score", 0)
                    })
        
        # Sort by score
        sources.sort(key=lambda x: x["score"], reverse=True)
        return sources
    
    def _extract_key_insights(self, passages: List[str], query: str) -> List[str]:
        """Extract key insights from passages that are relevant to the query."""
        if not passages:
            return []
            
        # Tokenize into sentences
        all_sentences = []
        for passage in passages:
            sentences = nltk.sent_tokenize(passage)
            all_sentences.extend(sentences)
        
        # Extract query keywords
        query_words = set(w.lower() for w in nltk.word_tokenize(query) if len(w) > 3)
        
        # Score sentences by relevance to query
        scored_sentences = []
        for sentence in all_sentences:
            sentence_words = set(w.lower() for w in nltk.word_tokenize(sentence) if len(w) > 3)
            overlap = len(query_words.intersection(sentence_words))
            score = overlap / max(1, len(query_words))
            
            # Boost sentences that appear to directly answer questions
            if re.search(r"(is|are|can|will|should|may|might|could|would|does|do|has|have)", sentence.lower()):
                score += 0.1
                
            scored_sentences.append((sentence, score))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences and deduplicate
        top_sentences = []
        seen_content = set()
        
        for sentence, score in scored_sentences:
            # Skip very short sentences
            if len(sentence) < 30:
                continue
                
            # Skip sentences that are too similar to ones we've already taken
            simplified = ' '.join(nltk.word_tokenize(sentence.lower()))
            is_duplicate = False
            
            for seen in seen_content:
                # Check similarity by token overlap
                s1 = set(simplified.split())
                s2 = set(seen.split())
                if len(s1) == 0:
                    continue
                    
                overlap_ratio = len(s1.intersection(s2)) / len(s1)
                if overlap_ratio > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate and score > 0.1:  # Minimum relevance threshold
                top_sentences.append(sentence)
                seen_content.add(simplified)
                
                if len(top_sentences) >= 5:  # Cap at 5 insights
                    break
        
        return top_sentences
    
    def _synthesize_answer(self, query: str, insights: List[str], search_results: List[Dict[str, Any]]) -> str:
        """Synthesize an answer from the insights and search results."""
        if not insights:
            return "I couldn't find specific information about that in the iridology knowledge base."
            
        # Combine insights into a coherent answer
        if len(insights) == 1:
            return insights[0]
        
        # Check if we have highlights for better answer generation
        highlights = []
        for result in search_results[:3]:  # Use top 3 results
            if "highlights" in result and result["highlights"]:
                highlights.extend(result["highlights"])
        
        # Start with an introduction
        answer_parts = []
        
        # Add insights
        for insight in insights:
            answer_parts.append(insight)
        
        # Combine everything
        answer = " ".join(answer_parts)
        
        return answer
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate a confidence score for the answer."""
        if not search_results:
            return 0.0
            
        # Average the top 3 scores
        top_scores = [result.get("score", 0) for result in search_results[:3]]
        if not top_scores:
            return 0.0
            
        return sum(top_scores) / len(top_scores)
