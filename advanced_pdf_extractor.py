import fitz  # PyMuPDF
import re
import nltk
import numpy as np
from typing import List, Dict, Any, Union, Callable, Optional, Tuple
import io
from PIL import Image
import os
import pytesseract
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import sys
import hashlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_pdf_extractor")

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
        logger.info("spaCy loaded successfully!")
    except:
        try:
            logger.info("Trying to download spaCy model...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            SPACY_AVAILABLE = True
            logger.info("spaCy model downloaded and loaded successfully!")
        except Exception as e:
            logger.error(f"Could not install or load spaCy model: {e}")
            SPACY_AVAILABLE = False
except ImportError as e:
    logger.warning(f"spaCy import failed: {e}")
    logger.info("Operating in NLTK-only mode with reduced NLP capabilities.")
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

class AdvancedPdfExtractor:
    """Advanced PDF extractor with intelligent chunking and NLP processing."""
    
    def __init__(self, min_chunk_size=150, max_chunk_size=1500, overlap=50):
        """
        Initialize the advanced PDF extractor.
        
        Args:
            min_chunk_size: Minimum size (in chars) for a text chunk
            max_chunk_size: Maximum size (in chars) for a text chunk
            overlap: Number of characters to overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Create a content filter based on keywords
        self.keywords = IRIS_KEYWORDS.union(HEALTH_KEYWORDS)
        
        # Set up a hash set for duplicate detection
        self.content_hashes = set()
        
        # Statistics for reporting
        self.stats = {
            "total_pages": 0,
            "processed_pages": 0,
            "extracted_chunks": 0,
            "ocr_processed_pages": 0,
            "filtered_chunks": 0,
            "duplicate_chunks": 0
        }
    
    def extract_chunks_from_pdf(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Extract content chunks from PDF with intelligent chunking and NLP processing.
        
        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional function to report progress (current, total)
            
        Returns:
            List of text chunks with metadata and enhanced features
        """
        # Reset statistics and hash set
        self.stats = {
            "total_pages": 0,
            "processed_pages": 0,
            "extracted_chunks": 0,
            "ocr_processed_pages": 0,
            "filtered_chunks": 0,
            "duplicate_chunks": 0
        }
        self.content_hashes = set()
        
        # Initialize the result list
        all_chunks = []
        
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            self.stats["total_pages"] = total_pages
            
            logger.info(f"Processing PDF with {total_pages} pages: {pdf_path}")
            
            # PHASE 1: Text extraction and structure analysis
            text_chunks = self._extract_text_chunks(doc, pdf_path, progress_callback)
            all_chunks.extend(text_chunks)
            
            # PHASE 2: OCR processing for image-based content
            ocr_chunks = self._extract_ocr_chunks(doc, pdf_path, progress_callback)
            all_chunks.extend(ocr_chunks)
            
            # PHASE 3: Post-processing to enhance chunks
            processed_chunks = self._post_process_chunks(all_chunks)
            
            # Log extraction statistics
            logger.info(f"Extraction complete: {len(processed_chunks)} chunks from {pdf_path}")
            logger.info(f"Statistics: {self.stats}")
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error extracting chunks from PDF: {str(e)}", exc_info=True)
            return []
    
    def _extract_text_chunks(self, doc, pdf_path: str, progress_callback) -> List[Dict[str, Any]]:
        """Extract and process text chunks from PDF using advanced layout analysis."""
        chunks = []
        total_pages = len(doc)
        
        # First determine document structure (ToC, headers, normal content)
        structure_info = self._analyze_document_structure(doc)
        
        # Process each page
        for page_num, page in enumerate(doc):
            if progress_callback:
                progress_callback(page_num + 1, total_pages * 2)
            
            # Skip pages that are likely not content (e.g., TOC, index)
            if page_num in structure_info.get("skip_pages", []):
                continue
            
            # Extract text with layout information
            try:
                # Get text blocks with their bounding boxes
                blocks = page.get_text("blocks")
                
                # Process blocks into coherent chunks
                page_chunks = self._process_text_blocks(blocks, page_num, pdf_path)
                chunks.extend(page_chunks)
                
                self.stats["processed_pages"] += 1
                
            except Exception as e:
                logger.warning(f"Error processing page {page_num+1}: {str(e)}")
        
        return chunks
    
    def _analyze_document_structure(self, doc) -> Dict[str, Any]:
        """Analyze document structure to identify TOC, headers, and content areas."""
        structure_info = {
            "skip_pages": set(),
            "headers": {},
            "footer_height": 0
        }
        
        # Sample a few pages to determine structure
        sample_pages = min(10, len(doc))
        sample_indices = np.linspace(0, len(doc)-1, sample_pages, dtype=int)
        
        # Analyze text density and formatting patterns
        for idx in sample_indices:
            page = doc[int(idx)]
            
            # Check if page has very little text (possibly cover, blank, or image-only)
            text = page.get_text()
            if len(text.strip()) < 100:
                structure_info["skip_pages"].add(idx)
                continue
            
            # Check if page has TOC indicators
            toc_indicators = ["contents", "table of contents", "index", "chapter"]
            if any(indicator in text.lower() for indicator in toc_indicators):
                structure_info["skip_pages"].add(idx)
                continue
        
        return structure_info
    
    def _process_text_blocks(self, blocks, page_num: int, pdf_path: str) -> List[Dict[str, Any]]:
        """Process text blocks into coherent chunks with semantic boundaries."""
        chunks = []
        current_text = ""
        current_block_texts = []
        
        # Sort blocks by vertical position (top to bottom)
        blocks.sort(key=lambda b: b[1][1])  # Sort by y-coordinate of bbox
        
        for block in blocks:
            # Block structure: (x0, y0, x1, y1, "lines", block_no, block_type)
            if block[6] == 0:  # Text block
                block_text = block[4]
                
                # Skip if block text is too short (likely headers, page numbers, etc.)
                if len(block_text.strip()) < 10:
                    continue
                    
                # Add to current collection
                current_block_texts.append(block_text)
                current_text += block_text + " "
                
                # Check if we've reached a good chunk size
                if len(current_text) >= self.max_chunk_size:
                    # Create a chunk with the collected text
                    chunk = self._create_chunk_from_text(
                        current_text, 
                        page_num, 
                        pdf_path,
                        "text_block"
                    )
                    
                    if chunk:
                        chunks.append(chunk)
                        self.stats["extracted_chunks"] += 1
                    
                    # Start a new chunk with overlap
                    # Find a good sentence boundary for the overlap
                    overlap_text = self._find_sentence_boundary_overlap(current_text)
                    current_text = overlap_text
                    current_block_texts = [overlap_text]
        
        # Handle any remaining text
        if current_text and len(current_text.strip()) > self.min_chunk_size:
            chunk = self._create_chunk_from_text(
                current_text, 
                page_num, 
                pdf_path,
                "text_block"
            )
            
            if chunk:
                chunks.append(chunk)
                self.stats["extracted_chunks"] += 1
        
        return chunks
    
    def _find_sentence_boundary_overlap(self, text: str) -> str:
        """Find a good sentence boundary for chunk overlap."""
        # Try to find a sentence boundary near the desired overlap point
        if len(text) <= self.overlap:
            return text
            
        overlap_text = text[-self.overlap*2:]  # Get more than needed to find a good boundary
        sentences = sent_tokenize(overlap_text)
        
        if not sentences:
            return text[-self.overlap:]
            
        # Return the last 1-2 sentences that are closest to the overlap size
        overlap_result = ""
        for sentence in reversed(sentences):
            if len(overlap_result + sentence) <= self.overlap*2:
                overlap_result = sentence + " " + overlap_result
            else:
                break
                
        return overlap_result.strip()
    
    def _extract_ocr_chunks(self, doc, pdf_path: str, progress_callback) -> List[Dict[str, Any]]:
        """Extract content using OCR for image-based PDFs or pages with figures."""
        chunks = []
        total_pages = len(doc)
        
        try:
            # Verify Tesseract is available
            pytesseract.get_tesseract_version()
            
            # Identify important pages to OCR
            important_pages = self._identify_image_pages(doc)
            
            # Process these pages with OCR
            for i, page_num in enumerate(important_pages):
                if progress_callback:
                    progress_callback(total_pages + i, total_pages * 2)
                
                try:
                    logger.info(f"OCR processing page {page_num+1}")
                    page = doc[page_num]
                    
                    # Get high-resolution image of the page
                    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    
                    # Apply OCR with enhanced settings
                    ocr_text = pytesseract.image_to_string(
                        img, 
                        config='--psm 6 --oem 3'
                    )
                    
                    # Process the OCR text into chunks
                    if len(ocr_text.strip()) > self.min_chunk_size:
                        # Split into semantic chunks
                        ocr_chunks = self._split_text_into_chunks(ocr_text)
                        
                        for chunk_text in ocr_chunks:
                            chunk = self._create_chunk_from_text(
                                chunk_text,
                                page_num,
                                pdf_path,
                                "ocr"
                            )
                            
                            if chunk:
                                chunks.append(chunk)
                                self.stats["extracted_chunks"] += 1
                    
                    self.stats["ocr_processed_pages"] += 1
                    
                except Exception as e:
                    logger.warning(f"Error during OCR on page {page_num+1}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"OCR processing error: {str(e)}")
            
        return chunks
    
    def _identify_image_pages(self, doc) -> List[int]:
        """Identify pages that likely contain images or are image-based."""
        image_pages = []
        
        # Check all pages but limit to reasonable number (max 30) to avoid excessive processing
        max_pages = min(30, len(doc))
        
        for page_num in range(max_pages):
            page = doc[page_num]
            
            # Check if page has images
            image_list = page.get_images(full=True)
            
            # Check text content
            text = page.get_text()
            text_length = len(text.strip())
            
            # If page has images OR has very little text, include for OCR
            if len(image_list) > 0 or text_length < 200:
                image_pages.append(page_num)
                
        return image_pages
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks of appropriate size."""
        # First split on paragraph boundaries
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed max size, start a new chunk
            if len(current_chunk) + len(para) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it meets the minimum size
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
            
        # If no chunks were created and text is substantial, use the whole text
        if not chunks and len(text) >= self.min_chunk_size:
            chunks.append(text)
            
        return chunks
    
    def _create_chunk_from_text(self, text: str, page_num: int, source: str, extraction_method: str) -> Optional[Dict[str, Any]]:
        """Create a structured chunk from text if relevant to the topic."""
        # Skip if too short
        if len(text.strip()) < self.min_chunk_size:
            return None
            
        # Check for relevance to our topics
        if not self._is_relevant_content(text):
            self.stats["filtered_chunks"] += 1
            return None
            
        # Check for duplicates using content hashing
        text_hash = self._compute_content_hash(text)
        if text_hash in self.content_hashes:
            self.stats["duplicate_chunks"] += 1
            return None
            
        # Add hash to seen set
        self.content_hashes.add(text_hash)
        
        # Process with NLP
        keywords = self._extract_keywords(text)
        sentences = sent_tokenize(text)
        relevance_score = self._calculate_relevance_score(text)
        entities = self._extract_entities(text)
        
        # Detect dosha-related content
        dosha_scores = self._detect_dosha_content(text)
        max_dosha_score = max(dosha_scores.values())
        primary_dosha = max(dosha_scores, key=dosha_scores.get) if max_dosha_score >= 0.3 else None
        
        # Create the chunk
        chunk = {
            "text": text.strip(),
            "page": page_num + 1,  # 1-based page numbering
            "source": source,
            "extraction_method": extraction_method,
            "sentences": sentences,
            "sentence_count": len(sentences),
            "entities": entities,
            "keywords": keywords,
            "word_count": len(text.split()),
            "char_count": len(text),
            "relevance_score": relevance_score,
            "dosha_scores": dosha_scores,
            "primary_dosha": primary_dosha,
            "is_dosha_related": max_dosha_score >= 0.3,
            "hash": text_hash
        }
        
        return chunk
    
    def _is_relevant_content(self, text: str) -> bool:
        """Determine if content is relevant to our topics."""
        text_lower = text.lower()
        
        # Check for direct keyword matches first (most reliable)
        for keyword in IRIS_KEYWORDS:
            if keyword in text_lower:
                return True
                
        # If no direct iris match, check for health keywords with substantial content
        if len(text_lower.split()) >= 50:  # Longer text
            for keyword in HEALTH_KEYWORDS:
                if keyword in text_lower:
                    return True
                    
        # If still no match, check for dosha keywords for Ayurvedic content
        for dosha, keywords in DOSHA_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return True
        
        return False
    
    def _detect_dosha_content(self, text: str) -> Dict[str, float]:
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
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from the text."""
        # Process with spaCy if available
        if SPACY_AVAILABLE and nlp is not None:
            try:
                doc = nlp(text)
                
                # Extract noun phrases and named entities
                keywords = []
                for chunk in doc.noun_chunks:
                    if len(chunk.text.strip()) > 3:
                        keywords.append(chunk.text.strip().lower())
                        
                # Add named entities
                for ent in doc.ents:
                    if len(ent.text.strip()) > 3:
                        keywords.append(ent.text.strip().lower())
                
                # Add domain-specific keywords
                for keyword in IRIS_KEYWORDS:
                    if keyword in text.lower() and keyword not in keywords:
                        keywords.append(keyword)
                        
                # Add dosha-related keywords
                for dosha, dosha_words in DOSHA_KEYWORDS.items():
                    for keyword in dosha_words:
                        if keyword.lower() in text.lower() and keyword not in keywords:
                            keywords.append(keyword)
                
                # Remove duplicates and limit
                keywords = list(set(keywords))[:20]  # Limit to top 20
                return keywords
                
            except Exception as e:
                logger.warning(f"Error extracting keywords with spaCy: {e}")
                # Fall back to simpler extraction
        
        # Simple keyword extraction with NLTK
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
        
        # Get most common words
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(20)]
        
        # Add domain-specific keywords
        for keyword in IRIS_KEYWORDS:
            if keyword in text.lower() and keyword not in keywords:
                keywords.append(keyword)
                
        # Add dosha-related keywords
        for dosha, dosha_words in DOSHA_KEYWORDS.items():
            for keyword in dosha_words:
                if keyword.lower() in text.lower() and keyword not in keywords:
                    keywords.append(keyword)
        
        return keywords
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        entities = []
        
        # Use spaCy if available
        if SPACY_AVAILABLE and nlp is not None:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                return entities
            except Exception as e:
                logger.warning(f"Error extracting entities with spaCy: {e}")
                # Fall back to simple extraction
        
        # Simple extraction of capitalized multi-word phrases as potential entities
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            i = 0
            while i < len(words):
                if words[i] and words[i][0].isupper():
                    # Look for consecutive capitalized words
                    j = i + 1
                    while j < len(words) and words[j] and words[j][0].isupper():
                        j += 1
                    
                    if j > i + 1:  # At least 2 capitalized words in sequence
                        entity_text = ' '.join(words[i:j])
                        entities.append({
                            "text": entity_text,
                            "label": "MISC",
                            "start": 0,  # Simplified - not tracking exact positions
                            "end": 0
                        })
                    i = j
                else:
                    i += 1
        
        return entities
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score based on keyword density and content."""
        text_lower = text.lower()
        
        # Count iris-related keywords
        iris_count = sum(1 for keyword in IRIS_KEYWORDS if keyword in text_lower)
        
        # Count health-related keywords
        health_count = sum(1 for keyword in HEALTH_KEYWORDS if keyword in text_lower)
        
        # Count dosha-related keywords
        dosha_count = 0
        for dosha, keywords in DOSHA_KEYWORDS.items():
            dosha_count += sum(1 for keyword in keywords if keyword in text_lower)
        
        # Calculate total score
        word_count = len(text_lower.split())
        if word_count == 0:
            return 0.0
            
        iris_score = min(1.0, iris_count / 5)  # Cap at 1.0
        health_score = min(0.7, health_count / 10)  # Cap at 0.7
        dosha_score = min(0.8, dosha_count / 8)  # Cap at 0.8
        
        # Combine scores with weights
        total_score = (iris_score * 0.6) + (health_score * 0.2) + (dosha_score * 0.2)
        
        # Boost score for longer, more substantial content
        length_boost = min(0.2, word_count / 500)  # Up to 0.2 boost for 500+ words
        
        # Final score with length boost
        final_score = min(1.0, total_score + length_boost)
        
        return final_score
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute a hash of the content for duplicate detection."""
        # Normalize text: lowercase, remove extra spaces
        normalized_text = ' '.join(text.lower().split())
        
        # Create a MD5 hash
        hash_obj = hashlib.md5(normalized_text.encode())
        return hash_obj.hexdigest()
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process chunks for final quality improvements."""
        # Sort by relevance score
        chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Keep only the most relevant chunks if there are too many
        if len(chunks) > 300:  # Increase limit for larger documents
            chunks = chunks[:300]
        
        # Add unique IDs and citation info
        for i, chunk in enumerate(chunks):
            # Add a unique ID for the chunk
            chunk["chunk_id"] = f"chunk_{i+1}"
            
            # Add citation information for referencing
            chunk["citation"] = f"{os.path.basename(chunk['source'])}, Page {chunk['page']}"
            
            # Set display priority based on position
            chunk["priority"] = i + 1
            
            # Add context relations between chunks
            self._add_contextual_relations(chunk, chunks, i)
        
        return chunks
    
    def _add_contextual_relations(self, chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]], chunk_idx: int) -> None:
        """Add references to related chunks based on page proximity and content similarity."""
        related_chunks = []
        
        # Add page-based relations (previous and next chunks on same page)
        same_page_chunks = [c for c in all_chunks if c["page"] == chunk["page"] and c != chunk]
        if same_page_chunks:
            related_chunks.extend([c["chunk_id"] for c in same_page_chunks[:2]])
        
        # Add content-based relations
        chunk_keywords = set(chunk.get("keywords", []))
        if chunk_keywords:
            for other_chunk in all_chunks:
                if other_chunk == chunk:
                    continue
                    
                other_keywords = set(other_chunk.get("keywords", []))
                # Calculate Jaccard similarity
                if other_keywords:
                    overlap = len(chunk_keywords.intersection(other_keywords))
                    union = len(chunk_keywords.union(other_keywords))
                    similarity = overlap / union if union > 0 else 0
                    
                    if similarity > 0.3:  # Threshold for relatedness
                        related_chunks.append(other_chunk["chunk_id"])
                        
                        # Limit to 5 most related chunks
                        if len(related_chunks) >= 5:
                            break
        
        # Save related chunks
        chunk["related_chunks"] = related_chunks[:5]  # Limit to 5


def extract_advanced_chunks(pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Extract chunked content from PDF using advanced text processing.
    
    Args:
        pdf_path: Path to the PDF file
        progress_callback: Optional function to report progress
        
    Returns:
        List of text chunks with metadata and enhanced features
    """
    extractor = AdvancedPdfExtractor()
    return extractor.extract_chunks_from_pdf(pdf_path, progress_callback)


# For testing and demonstration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_pdf_extractor.py <pdf_file>")
        sys.exit(1)
        
    pdf_file = sys.argv[1]
    
    # Simple progress callback
    def report_progress(current, total):
        print(f"Processing: {current}/{total} ({current/total*100:.1f}%)")
    
    chunks = extract_advanced_chunks(pdf_file, report_progress)
    
    print(f"Extracted {len(chunks)} chunks")
    
    # Print sample of the first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk['text'][:150]}...")
        print(f"Page: {chunk['page']}")
        print(f"Relevance: {chunk['relevance_score']:.2f}")
        print(f"Keywords: {', '.join(chunk['keywords'][:10])}")
        print(f"Primary Dosha: {chunk.get('primary_dosha', 'None')}")
        if chunk.get('is_dosha_related'):
            print(f"Dosha Scores: {chunk['dosha_scores']}")
