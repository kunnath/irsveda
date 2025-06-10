import fitz  # PyMuPDF
import re
import nltk
import numpy as np
from typing import List, Dict, Any, Union, Callable, Optional
import io
from PIL import Image
import os
import pytesseract
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import sys

# Flag to track if spaCy is available
SPACY_AVAILABLE = False
nlp = None

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize spaCy with fallback for Python 3.13 compatibility
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("spaCy loaded successfully!")
    except:
        try:
            print("Trying to download spaCy model...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
            print("spaCy model downloaded and loaded successfully!")
        except Exception as e:
            print(f"Could not install or load spaCy model: {e}")
            SPACY_AVAILABLE = False
except ImportError as e:
    print(f"spaCy import failed: {e}")
    print("Operating in NLTK-only mode with reduced NLP capabilities.")
    SPACY_AVAILABLE = False

# Keywords related to iris and iridology for better content filtering
IRIS_KEYWORDS = {
    'iris', 'iridology', 'iridologist', 'eye', 'pupil', 'sclera', 
    'cornea', 'irides', 'iris sign', 'iris interpretation', 'iris reading',
    'eye diagnosis', 'eye analysis', 'iris mapping', 'iris chart', 'iris zone', 
    'eye health', 'iris pattern', 'iris marking', 'eye marking', 'iris color', 
    'eye marking', 'organ mapping', 'ayurvedic', 'iridodiagnosis'
}

# Health-related keywords for context expansion
HEALTH_KEYWORDS = {
    'health', 'disease', 'symptom', 'ailment', 'condition', 'diagnosis',
    'treatment', 'healing', 'therapy', 'medicine', 'remedy', 'liver',
    'kidney', 'heart', 'lung', 'digestive', 'immune', 'nervous', 'pancreas',
    'gallbladder', 'stomach', 'thyroid', 'lymphatic', 'inflammation', 'detoxification',
    'chronic', 'acute', 'constitutional', 'inherent'
}

# Keywords specific to Ayurvedic doshas for better content categorization
DOSHA_KEYWORDS = {
    "vata": [
        "vata", "air", "ether", "dry", "light", "cold", "rough", "irregular", "movement", 
        "creative", "anxiety", "nervous", "variable energy", "spacey", "ungrounded",
        "thin frame", "insomnia", "blue iris", "light iris", "erratic", "variable"
    ],
    "pitta": [
        "pitta", "fire", "water", "hot", "sharp", "intense", "penetrating", "acidic", 
        "inflammation", "metabolic", "reddish", "anger", "impatience", "structured", 
        "efficient", "medium build", "focused", "amber iris", "reddish iris", "yellowish"
    ],
    "kapha": [
        "kapha", "earth", "water", "cold", "heavy", "slow", "stable", "dense", "static",
        "congestion", "attachment", "calm", "patient", "lethargic", "larger frame", 
        "retention", "brown iris", "dark iris", "thick fibers", "greenish"
    ]
}

def extract_structured_chunks(pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Extract iris-related chunks from PDF with advanced NLP processing for better context.
    
    Args:
        pdf_path: Path to the PDF file
        progress_callback: Optional function to report progress (page_num, total_pages)
        
    Returns:
        List of iris-related text chunks with metadata and enhanced features
    """
    doc = fitz.open(pdf_path)
    all_chunks = []
    
    # Get total page count
    total_pages = len(doc)
    print(f"PDF has {total_pages} pages. Beginning extraction...")
    
    # First pass: Standard text extraction with NLP enhancements
    print(f"Pass 1: Extracting text with NLP processing from all {total_pages} pages...")
    print(f"spaCy available: {SPACY_AVAILABLE}")
    
    # Initialize tracking variables used throughout the function
    # These will be used at the end for diagnostics
    pages_with_content = 0
    pages_with_iris_keywords = 0
    iris_keywords_found = set()
    
    for page_num, page in enumerate(doc):
        if progress_callback:
            progress_callback(page_num + 1, total_pages * 2)
            
        text = page.get_text()
        
        # Debug non-empty pages
        if text.strip():
            pages_with_content += 1
            
        # Debug keyword matches
        if any(keyword in text.lower() for keyword in IRIS_KEYWORDS):
            pages_with_iris_keywords += 1
            print(f"Found iris keywords on page {page_num+1}")
            
            # Show a sample of matching keywords for debugging
            found_keywords = [k for k in IRIS_KEYWORDS if k in text.lower()]
            if found_keywords:
                # Store all unique keywords found for later diagnostics
                iris_keywords_found.update(found_keywords)
                print(f"Keywords found: {', '.join(found_keywords[:5])}")
        
        # Check if the page has any relevant content before detailed processing
        if any(keyword in text.lower() for keyword in IRIS_KEYWORDS):
            # Extract paragraphs, sentences, and identify entities
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Process with spaCy if available, otherwise use basic text processing
            doc_nlp = None
            if SPACY_AVAILABLE and nlp is not None:
                try:
                    doc_nlp = nlp(text)
                except Exception as e:
                    print(f"Error processing text with spaCy: {e}")
            
            # Process each paragraph
            for para_idx, para in enumerate(paragraphs):
                # Check for relevance to iris or health topics
                if is_relevant_paragraph(para):
                    # Extract entities and keywords
                    entities = []
                    if SPACY_AVAILABLE and nlp is not None:
                        try:
                            doc_para = nlp(para)
                            entities = extract_entities(doc_para)
                        except Exception as e:
                            print(f"Error processing paragraph with spaCy: {e}")
                            entities = simple_entity_extraction(para)
                    else:
                        entities = simple_entity_extraction(para)
                    
                    keywords = extract_keywords(para)
                    
                    # Split into sentences for better context
                    sentences = sent_tokenize(para)
                    
                    # Calculate relevance score
                    relevance_score = calculate_relevance_score(para)
                    
                    # Detect dosha-related content
                    dosha_scores = detect_dosha_content(para)
                    
                    # Determine primary dosha (if any)
                    max_dosha_score = max(dosha_scores.values())
                    primary_dosha = None
                    if max_dosha_score >= 0.3:  # Only consider it a dosha chunk if score is significant
                        primary_dosha = max(dosha_scores, key=dosha_scores.get)
                    
                    # Add structured chunk
                    all_chunks.append({
                        "text": para.strip(),
                        "page": page_num + 1,  # 1-based page numbering
                        "source": pdf_path,
                        "extraction_method": "standard",
                        "paragraph_idx": para_idx,
                        "sentences": sentences,
                        "sentence_count": len(sentences),
                        "entities": entities,
                        "keywords": keywords,
                        "relevance_score": relevance_score,
                        "dosha_scores": dosha_scores,
                        "primary_dosha": primary_dosha,
                        "is_dosha_related": max_dosha_score >= 0.3
                    })
    
    # Second pass: OCR extraction for images and diagrams
    print(f"Pass 2: OCR processing for image content...")
    
    try:
        # Verify Tesseract is available
        pytesseract.get_tesseract_version()
        
        # Analyze fewer pages but with higher quality
        # Focus on pages with figures/diagrams and pages surrounding key content
        important_pages = identify_important_pages(doc, all_chunks)
        
        for page_idx, page_num in enumerate(important_pages):
            if progress_callback:
                progress_step = page_idx / len(important_pages) * total_pages
                progress_callback(total_pages + progress_step, total_pages * 2)
                
            print(f"OCR processing important page {page_num+1}/{total_pages}...")
            page = doc[page_num]
            
            # Get the page as an image at higher resolution for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # Higher resolution for better OCR
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Use OCR with custom configuration for better results
            ocr_text = pytesseract.image_to_string(
                img, 
                config='--psm 6 --oem 3'  # Assume a single uniform block of text
            )
            
            # Process OCR text
            if is_relevant_content(ocr_text):
                # Split into paragraphs
                ocr_paragraphs = [p.strip() for p in ocr_text.split('\n\n') if p.strip()]
                
                for para_idx, para in enumerate(ocr_paragraphs):
                    if is_relevant_paragraph(para):
                        # Skip duplicate content
                        if is_duplicate_content(para, all_chunks):
                            continue
                            
                        # Process with NLP
                        try:
                            # Use spaCy if available, otherwise use fallback methods
                            if SPACY_AVAILABLE and nlp is not None:
                                doc_para = nlp(para)
                                entities = extract_entities(doc_para)
                            else:
                                entities = simple_entity_extraction(para)
                                
                            keywords = extract_keywords(para)
                            sentences = sent_tokenize(para)
                            relevance_score = calculate_relevance_score(para)
                            
                            # Detect dosha-related content
                            dosha_scores = detect_dosha_content(para)
                            
                            # Determine primary dosha (if any)
                            max_dosha_score = max(dosha_scores.values())
                            primary_dosha = None
                            if max_dosha_score >= 0.3:  # Only consider it a dosha chunk if score is significant
                                primary_dosha = max(dosha_scores, key=dosha_scores.get)
                        except Exception as e:
                            print(f"NLP processing error for OCR text: {str(e)}")
                            entities = []
                            keywords = []
                            sentences = [para]
                            relevance_score = 0.5
                            dosha_scores = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
                            primary_dosha = None
                            max_dosha_score = 0.0
                        
                        # Add structured chunk
                        all_chunks.append({
                            "text": para.strip(),
                            "page": page_num + 1,  # 1-based page numbering
                            "source": pdf_path,
                            "extraction_method": "ocr",
                            "paragraph_idx": para_idx,
                            "sentences": sentences,
                            "sentence_count": len(sentences),
                            "entities": entities,
                            "keywords": keywords,
                            "relevance_score": relevance_score,
                            "dosha_scores": dosha_scores,
                            "primary_dosha": primary_dosha,
                            "is_dosha_related": max_dosha_score >= 0.3
                        })
                        
    except Exception as e:
        print(f"OCR processing error: {str(e)}")
    
    # Log the raw number of chunks found before post-processing
    print(f"Found {len(all_chunks)} potential chunks before post-processing")
    
    # If no chunks were found, try using broader criteria
    if len(all_chunks) == 0:
        print("No chunks found with standard criteria. Attempting broader matching...")
        # Retry with a broader content scan
        for page_num, page in enumerate(doc):
            text = page.get_text()
            # Use simpler matching to capture more content
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]  # Lower size threshold
            
            # Check if the page has almost no text - might need OCR
            if len(text) < 100 and page_num < min(20, total_pages):
                print(f"Page {page_num+1} has very little text. Might be an image-based PDF. Trying OCR...")
                try:
                    # Get the page as an image and run OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                    if len(ocr_text) > 100:
                        print(f"OCR extracted {len(ocr_text)} chars from page {page_num+1}")
                        paragraphs.append(ocr_text)
                except Exception as e:
                    print(f"OCR attempt failed: {e}")
            
            for para_idx, para in enumerate(paragraphs):
                # More permissive matching - any health or iris keyword
                if any(keyword in para.lower() for keyword in HEALTH_KEYWORDS) or \
                   any(keyword in para.lower() for keyword in IRIS_KEYWORDS):
                    print(f"Found broader match on page {page_num+1}")
                    # Add with basic processing
                    all_chunks.append({
                        "text": para.strip(),
                        "page": page_num + 1,
                        "source": pdf_path,
                        "extraction_method": "broad_match",
                        "paragraph_idx": para_idx,
                        "sentences": sent_tokenize(para),
                        "sentence_count": len(sent_tokenize(para)),
                        "entities": simple_entity_extraction(para),
                        "keywords": extract_keywords(para),
                        "relevance_score": 0.5  # Moderate score for broader matches
                    })
    
    # Post-processing to enhance chunks
    enhanced_chunks = post_process_chunks(all_chunks)
    
    # Final output with clear success/failure message
    if len(enhanced_chunks) > 0:
        print(f"SUCCESS: Extracted {len(enhanced_chunks)} structured iris-related chunks from {pdf_path}")
        # Show sample of the first extracted chunk
        if enhanced_chunks:
            print("\nSample of first extracted chunk:")
            print(f"Text (first 100 chars): {enhanced_chunks[0]['text'][:100]}...")
            print(f"Page: {enhanced_chunks[0]['page']}")
            print(f"Keywords: {', '.join(enhanced_chunks[0].get('keywords', [])[:5])}")
    else:
        print(f"WARNING: No iris-related chunks found in {pdf_path}. The document may not contain relevant content.")
        print("Try uploading a different document or check if the PDF has machine-readable text.")
        
        # Additional troubleshooting info
        print(f"\nTroubleshooting information:")
        print(f"- Pages with any content: {pages_with_content}/{total_pages}")  
        print(f"- Pages with iris keywords: {pages_with_iris_keywords}/{total_pages}")
        print(f"- IRIS_KEYWORDS found in document: {', '.join(list(iris_keywords_found)[:10])}")
        print(f"- All IRIS_KEYWORDS used for matching: {', '.join(list(IRIS_KEYWORDS)[:10])}...")
        
        # Dependency information
        print("\nSystem information:")
        print(f"- Python version: {sys.version}")
        print(f"- spaCy available: {SPACY_AVAILABLE}")
        if SPACY_AVAILABLE:
            import spacy
            print(f"- spaCy version: {spacy.__version__}")
            try:
                models = subprocess.check_output([sys.executable, "-m", "spacy", "info"]).decode('utf-8')
                print(f"- Available spaCy models: {models.split('===========')[1].strip() if '===========' in models else 'Unknown'}")
            except:
                print("- Could not retrieve spaCy model information")
        
        # Suggest a solution
        print("\nPossible solutions:")
        print("1. Check if the PDF has searchable text (not just images)")
        print("2. Try using OCR on the document first if it's image-based")
        print("3. Verify the document actually contains iridology or iris-related content")
        print("4. If running with Python 3.13, try using Python 3.10-3.12 for better compatibility with NLP libraries")
    
    return enhanced_chunks

def detect_dosha_content(text: str) -> Dict[str, float]:
    """
    Detect and score dosha-related content in the text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with dosha types as keys and relevance scores as values
    """
    text_lower = text.lower()
    dosha_scores = {"vata": 0.0, "pitta": 0.0, "kapha": 0.0}
    
    # Count occurrences of each dosha keyword
    for dosha, keywords in DOSHA_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Count occurrences and increase score
            count = text_lower.count(keyword)
            score += count
            
            # Give extra weight to explicit mentions of the dosha name
            if keyword.lower() == dosha.lower() and count > 0:
                score += 2 * count
                
            # Look for specific patterns with higher relevance
            # Format: "dosha_name + related concept"
            patterns = [
                f"{dosha} (constitution|type|dominant|dosha)",  # Direct dosha references
                f"{dosha} (iris|eye)",                         # Iris-specific references
                f"{dosha} (imbalance|excess|deficiency)",      # Health-related references
                f"(high|low|balanced) {dosha}"                 # Assessment references
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 3
    
        # Normalize score based on text length (to avoid bias toward longer texts)
        text_length = len(text_lower.split())
        normalized_score = score / max(10, text_length) * 10  # Scale to a reasonable range
        dosha_scores[dosha] = min(1.0, normalized_score)  # Cap at 1.0
    
    return dosha_scores

def is_relevant_content(text: str) -> bool:
    """Check if the text contains any iris or health-related content."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in IRIS_KEYWORDS.union(HEALTH_KEYWORDS))

def is_relevant_paragraph(para: str) -> bool:
    """More detailed check if paragraph is relevant to iris/iridology topics."""
    para_lower = para.lower()
    
    # Check for direct keyword matches first (most reliable)
    direct_match = any(keyword in para_lower for keyword in IRIS_KEYWORDS)
    
    # If no direct match, try more flexible matching
    if not direct_match:
        # Check for partial word matches (e.g., "irid" would match "iridology")
        partial_matches = ["iris", "irid", "eye", "pupil", "sclera", "cornea", "ayurved"]
        if any(partial in para_lower for partial in partial_matches):
            return True
            
        # Try regex pattern matching but with simpler patterns to avoid errors
        try:
            for keyword in IRIS_KEYWORDS:
                # Skip multi-word keywords for regex to avoid complex patterns
                if ' ' not in keyword and len(keyword) > 3:
                    if re.search(fr'\b{keyword}\w*\b', para_lower):
                        return True
        except Exception as e:
            print(f"Regex error in is_relevant_paragraph: {e}")
            # Fall back to basic matching if regex fails
            pass
            
        # If no matches yet, check if it has health keywords and is substantial
        if len(para_lower.split()) >= 20 and any(keyword in para_lower for keyword in HEALTH_KEYWORDS):
            print(f"Found health-related paragraph: {para[:50]}...")
            return True
            
        # No matches found
        return False
    
    # Paragraph must be substantive enough to be meaningful
    if len(para_lower.split()) < 8:  # Skip very short paragraphs, but lower threshold
        return False
        
    return True

def is_duplicate_content(text: str, existing_chunks: List[Dict[str, Any]]) -> bool:
    """Check if this text is a duplicate or near-duplicate of existing chunks."""
    # Simple check for exact substring
    for chunk in existing_chunks:
        if text in chunk["text"] or chunk["text"] in text:
            return True
            
    # Check for high similarity using a simple token overlap ratio
    text_tokens = set(word_tokenize(text.lower()))
    if len(text_tokens) < 5:  # Too short to reliably check for duplicates
        return False
        
    for chunk in existing_chunks:
        chunk_tokens = set(word_tokenize(chunk["text"].lower()))
        if len(chunk_tokens) < 5:
            continue
            
        # Calculate Jaccard similarity
        overlap = len(text_tokens.intersection(chunk_tokens))
        union = len(text_tokens.union(chunk_tokens))
        
        if overlap / union > 0.7:  # 70% similarity threshold
            return True
            
    return False

def extract_entities(doc_nlp) -> List[Dict[str, str]]:
    """Extract named entities from spaCy doc or use fallback with NLTK."""
    entities = []
    
    # Use spaCy if available
    if SPACY_AVAILABLE and doc_nlp is not None:
        try:
            for ent in doc_nlp.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
        except Exception as e:
            print(f"Error extracting entities with spaCy: {e}")
            # Fall back to simple extraction
            return simple_entity_extraction(str(doc_nlp))
    else:
        # Fallback for when spaCy is not available
        return simple_entity_extraction(doc_nlp if isinstance(doc_nlp, str) else str(doc_nlp))
            
    return entities

def simple_entity_extraction(text: str) -> List[Dict[str, str]]:
    """Simple entity extraction fallback using NLTK and regex patterns."""
    entities = []
    
    # Extract potential health terms
    health_terms = []
    for term in IRIS_KEYWORDS.union(HEALTH_KEYWORDS):
        if term in text.lower():
            health_terms.append({"text": term, "label": "HEALTH_TERM"})
    
    # Add any capitalized multi-word phrases as potential entities
    cap_phrases = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for phrase in cap_phrases:
        if len(phrase.split()) >= 2:  # At least two words
            entities.append({"text": phrase, "label": "PROPER_NOUN"})
    
    # Add health terms
    entities.extend(health_terms)
    
    return entities

def extract_keywords(text: str) -> List[str]:
    """Extract key terms from text."""
    try:
        # Process with basic NLP
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords, punctuation, and short words
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback if NLTK data is not available
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                         'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'of'}
        
        filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2 and w.isalnum()]
        
        # Get most common words
        word_freq = Counter(filtered_tokens)
        
        # Extract top keywords - ensure we get at least some keywords even if fewer than 10
        top_n = min(10, len(word_freq))
        keywords = [word for word, freq in word_freq.most_common(top_n)]
        
        # Add any iris or health keywords that appear in the text
        for keyword in IRIS_KEYWORDS:
            if keyword.lower() in text.lower() and keyword not in keywords:
                keywords.append(keyword)
        
        # Add dosha-related keywords
        for dosha, dosha_words in DOSHA_KEYWORDS.items():
            for keyword in dosha_words:
                if keyword.lower() in text.lower() and keyword not in keywords:
                    keywords.append(keyword)
        
        return keywords
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        # Return basic word extraction as fallback
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        return list(set(words))[:10] if words else ["no_keywords_extracted"]

def calculate_relevance_score(text: str) -> float:
    """Calculate a relevance score (0-1) for the text based on keyword density."""
    try:
        text_lower = text.lower()
        score = 0.0
        
        # Count iris keywords - with extra debug info
        iris_matches = [keyword for keyword in IRIS_KEYWORDS if keyword in text_lower]
        iris_count = len(iris_matches)
        
        # Count health keywords
        health_matches = [keyword for keyword in HEALTH_KEYWORDS if keyword in text_lower]
        health_count = len(health_matches)
        
        # Check for partial matches if no exact matches
        if iris_count == 0:
            # Check for word stems
            iris_stems = ["irid", "eye", "pupil", "scler"]
            for stem in iris_stems:
                if stem in text_lower:
                    iris_count += 0.5  # Partial match counts less
        
        # For very short texts, boost the score to avoid filtering out good content
        text_length = len(text_lower.split())
        
        # Calculate score based on keyword density with adjusted weights
        if text_length > 0:
            if text_length < 30:  # Very short paragraphs need higher density to matter
                iris_density = iris_count / min(text_length, 10) * 10  # Avoid division by very small numbers
                health_density = health_count / min(text_length, 10) * 3
            else:
                iris_density = iris_count / min(text_length, 100) * 12  # Cap at 100 words to avoid dilution
                health_density = health_count / min(text_length, 100) * 4
            
            # If we have iris keywords, boost the score
            if iris_count > 0:
                base_score = min(1.0, (iris_density + health_density))
                # Ensure at least 0.4 score if we have any iris keywords
                score = max(0.4, base_score) if iris_count > 0 else base_score
            else:
                # Lower score for health-only content
                score = min(0.6, (health_density * 1.5))
        
        # Debug info for low scores
        if score < 0.4 and (iris_count > 0 or health_count > 0):
            matched_keywords = iris_matches + health_matches
            print(f"Low score ({score:.2f}) despite keywords: {', '.join(matched_keywords[:3])}")
        
        return score
    except Exception as e:
        print(f"Error in calculate_relevance_score: {e}")
        # Fallback - if we have iris keywords, return higher score
        if any(keyword in text.lower() for keyword in IRIS_KEYWORDS):
            return 0.7
        elif any(keyword in text.lower() for keyword in HEALTH_KEYWORDS):
            return 0.5
        return 0.3

def identify_important_pages(doc, existing_chunks: List[Dict[str, Any]]) -> List[int]:
    """Identify important pages for OCR processing."""
    # Pages that already have content
    content_pages = set(chunk["page"] - 1 for chunk in existing_chunks)  # Convert to 0-based
    
    # Pages that likely contain images
    image_pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        if page.get_images():
            image_pages.append(page_num)
    
    # Pages surrounding content pages (context)
    context_pages = []
    for page in content_pages:
        if page > 0:
            context_pages.append(page - 1)
        if page < len(doc) - 1:
            context_pages.append(page + 1)
    
    # Combine and prioritize pages
    important_pages = list(set(content_pages).union(set(image_pages)).union(set(context_pages)))
    important_pages.sort()
    
    # If too many pages, sample them
    if len(important_pages) > 20:
        important_pages = sample_pages(important_pages, 20)
    
    return important_pages

def sample_pages(page_list: List[int], max_pages: int) -> List[int]:
    """Sample pages evenly throughout the document."""
    if len(page_list) <= max_pages:
        return page_list
        
    # Even sampling
    indices = np.linspace(0, len(page_list) - 1, max_pages, dtype=int)
    return [page_list[i] for i in indices]

def post_process_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Post-process chunks to enhance quality."""
    # Sort by relevance score
    chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Keep only the most relevant chunks if there are too many
    if len(chunks) > 200:  # Limit to avoid overwhelming the vector database
        chunks = chunks[:200]
    
    # Re-calculate citations and cross-references
    for i, chunk in enumerate(chunks):
        # Add a unique ID for the chunk
        chunk["chunk_id"] = f"chunk_{i+1}"
        
        # Set display priority based on position
        chunk["priority"] = i + 1
    
    return chunks
