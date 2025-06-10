from pathlib import Path
import tempfile
import streamlit as st
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
from PIL import Image

# Import our enhanced components
from enhanced_pdf_extractor import extract_structured_chunks
from enhanced_iris_qdrant import EnhancedIrisQdrantClient
from context_aware_answers import ContextAwareAnswerGenerator
from check_qdrant_status import verify_qdrant_collections, create_enhanced_chunk_from_standard
from advanced_iris_analyzer import AdvancedIrisAnalyzer
from iris_dataset_matcher import DatasetPatternMatcher
from suggested_queries import SuggestedQueriesGenerator

# Try to import enhanced iris image analyzer with fallback
try:
    from enhanced_iris_image_analyzer import EnhancedIrisImageAnalyzer
    ENHANCED_IMAGE_ANALYZER_AVAILABLE = True
    # Check if matplotlib is available for the enhanced analyzer
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
except ImportError as e:
    ENHANCED_IMAGE_ANALYZER_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = False
    print(f"Enhanced iris image analyzer not available: {e}")
    # Create a placeholder class
    class EnhancedIrisImageAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze_iris_image(self, *args, **kwargs):
            return {"error": "Enhanced image analyzer not available"}
        def set_detection_sensitivity(self, *args, **kwargs):
            pass

# Keep the original imports for backward compatibility
from pdf_extractor import extract_iris_chunks
from iris_qdrant import IrisQdrantClient
from iris_predictor import IrisPredictor

# Import the new IrisZone analyzer
from iris_zone_analyzer import IrisZoneAnalyzer
from iris_report_generator import IrisReportGenerator

# Import enhanced report generator
try:
    from enhanced_iris_report_generator import EnhancedIrisReportGenerator
    ENHANCED_REPORT_GENERATOR_AVAILABLE = True
except ImportError as e:
    ENHANCED_REPORT_GENERATOR_AVAILABLE = False
    print(f"Enhanced report generator not available: {e}")
    # Create placeholder
    class EnhancedIrisReportGenerator:
        def generate_enhanced_pdf_report(self, *args, **kwargs):
            return None
        def generate_enhanced_html_report(self, *args, **kwargs):
            return "Enhanced report generator not available"

# Define helper functions first, before they are used
def get_score_color(score):
    """Get color based on score."""
    if score >= 0.8:
        return "#28a745"  # Green
    elif score >= 0.6:
        return "#17a2b8"  # Blue
    elif score >= 0.4:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def show_results(results, standard=True):
    """Show standard search results."""
    # Display results
    st.subheader("Results:")
    
    if not results:
        st.info("No matching information found. Try rephrasing your question.")
        return
    
    for i, result in enumerate(results):
        with st.container():
            # Create expander for each result
            with st.expander(f"Result {i+1} (Page {result['page']}) - Relevance: {result['score']:.2f}", expanded=i==0):
                st.markdown(result['text'])
                st.caption(f"Source: {Path(result['source']).name}, Page: {result['page']}")

def show_enhanced_results(results):
    """Show enhanced search results with highlights."""
    st.subheader("Enhanced Results:")
    
    if not results:
        st.info("No matching information found. Try rephrasing your question or using different keywords.")
        return
    
    for i, result in enumerate(results):
        score_color = get_score_color(result['score'])
        
        with st.container():
            # Create expander for each result
            with st.expander(
                f"Result {i+1} (Page {result['page']}) - " +
                f"Relevance: {result['score']:.2f}", 
                expanded=i==0
            ):
                # Show highlights if available
                if "highlights" in result and result["highlights"]:
                    st.markdown("#### Highlights")
                    for highlight in result["highlights"]:
                        st.markdown(highlight)
                    st.markdown("---")
                
                # Show full text
                st.markdown(result['text'])
                
                # Show metadata
                metadata_cols = st.columns(3)
                with metadata_cols[0]:
                    st.markdown(f"Source: {Path(result['source']).name}")
                with metadata_cols[1]:
                    st.markdown(f"Page: {result['page']}")
                with metadata_cols[2]:
                    st.markdown(f"Method: {result.get('extraction_method', 'standard')}")
                
                # Show keywords if available
                if "keywords" in result and result["keywords"]:
                    st.markdown(f"**Keywords:** {', '.join(result.get('keywords', []))}")

def show_generated_answer(answer_data):
    """Show a generated answer from the search results."""
    # Skip if no answer generated
    if not answer_data or answer_data["confidence"] == 0:
        return
        
    st.markdown("---")
    st.subheader("Generated Answer")
    
    # Display confidence meter
    confidence = answer_data["confidence"]
    confidence_color = get_score_color(confidence)
    
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"**Answer Confidence:** {confidence:.2f}")
    with cols[1]:
        st.progress(confidence)
    
    # Display the answer
    st.markdown(answer_data["answer"])
    
    # Display sources
    if answer_data.get("sources"):
        with st.expander("Sources"):
            for src in answer_data["sources"]:
                st.markdown(f"- {src['title']}, Page {src['page']}")

def handle_suggested_query(query, query_mode, result_count, min_relevance):
    """
    Handle user clicking on a suggested query button.
    
    Args:
        query: The suggested query text
        query_mode: The current search mode
        result_count: Number of results to show
        min_relevance: Minimum relevance threshold
    """
    try:
        st.session_state.last_query = query
        st.subheader(f"Results for: '{query}'")
        
        # Choose search method based on mode
        if query_mode == "Standard Search":
            results = st.session_state.qdrant_client.search(query, limit=result_count)
            show_results(results, standard=True)
        elif query_mode == "Enhanced Search":
            results = st.session_state.enhanced_qdrant_client.hybrid_search(query, 
                                                                          limit=result_count, 
                                                                          min_score=min_relevance)
            show_enhanced_results(results)
        elif query_mode == "Multi-Query Search":
            results = st.session_state.enhanced_qdrant_client.multi_query_search(query, limit=result_count)
            show_enhanced_results(results)
            
            # Generate context-aware answer
            answer_data = st.session_state.answer_generator.generate_answer(query, results)
            show_generated_answer(answer_data)
            
    except Exception as e:
        st.error(f"Error processing suggested query: {str(e)}")

def check_qdrant_status():
    """Check Qdrant connection and initialize collection status."""
    try:
        # Check if collections exist and have data
        standard_initialized, enhanced_initialized = verify_qdrant_collections(
            st.session_state.qdrant_client, 
            st.session_state.enhanced_qdrant_client
        )
        
        st.session_state.is_initialized = standard_initialized
        st.session_state.is_enhanced_initialized = enhanced_initialized
    except Exception as e:
        print(f"Error checking Qdrant status: {str(e)}")

# Page title and configuration
st.set_page_config(
    page_title="IridoVeda",
    page_icon="👁️",
    layout="wide"
)

# Create sidebar with improved UI
st.sidebar.title("IridoVeda")
st.sidebar.image("static/iris_logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info(
    "This application extracts iris-related information from Ayurvedic/Iridology "
    "books and allows you to query the knowledge base using advanced NLP."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Dinexora](https://www.dinexora.de)")

# Add contact section
with st.sidebar.expander("Contact Us"):
    st.markdown("""
    **Get in touch with us**
    
    Have questions or need help with IridoVeda? Feel free to contact us.
    
    📧 Email: [contact@dinexora.de](mailto:contact@dinexora.de)
    
    🌐 Website: [www.dinexora.de](https://www.dinexora.de)
    """)

# Add settings in sidebar
with st.sidebar.expander("Advanced Settings"):
    use_enhanced_mode = st.checkbox("Use Enhanced NLP Mode", value=True, 
                                   help="Use advanced NLP processing for better results")
    result_count = st.slider("Result Count", min_value=3, max_value=10, value=5,
                            help="Number of results to return from the knowledge base")
    min_relevance = st.slider("Minimum Relevance", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                             help="Minimum relevance score (0-1) for search results")

# Initialize session state
if "enhanced_qdrant_client" not in st.session_state:
    # Use environment variables if available (for Docker)
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    # Initialize both regular and enhanced clients
    st.session_state.qdrant_client = IrisQdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="iris_chunks"
    )
    
    st.session_state.enhanced_qdrant_client = EnhancedIrisQdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="enhanced_iris_chunks"
    )
    
    # Check if collections exist and have data
    check_qdrant_status()

if "answer_generator" not in st.session_state:
    st.session_state.answer_generator = ContextAwareAnswerGenerator()

if "iris_predictor" not in st.session_state:
    st.session_state.iris_predictor = IrisPredictor()

if "iris_zone_analyzer" not in st.session_state:
    st.session_state.iris_zone_analyzer = IrisZoneAnalyzer()
    
if "iris_report_generator" not in st.session_state:
    st.session_state.iris_report_generator = IrisReportGenerator()

# Initialize enhanced report generator
if "enhanced_iris_report_generator" not in st.session_state and ENHANCED_REPORT_GENERATOR_AVAILABLE:
    st.session_state.enhanced_iris_report_generator = EnhancedIrisReportGenerator()

if "advanced_iris_analyzer" not in st.session_state:
    # Use environment variables if available (for Docker)
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    st.session_state.advanced_iris_analyzer = AdvancedIrisAnalyzer(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name="iris_patterns"
    )

if "enhanced_iris_image_analyzer" not in st.session_state:
    if ENHANCED_IMAGE_ANALYZER_AVAILABLE:
        st.session_state.enhanced_iris_image_analyzer = EnhancedIrisImageAnalyzer()
    else:
        st.session_state.enhanced_iris_image_analyzer = EnhancedIrisImageAnalyzer()  # placeholder

if "query_generator" not in st.session_state:
    st.session_state.query_generator = SuggestedQueriesGenerator()

if "extracted_chunks" not in st.session_state:
    st.session_state.extracted_chunks = []

if "enhanced_chunks" not in st.session_state:
    st.session_state.enhanced_chunks = []

if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False

if "is_enhanced_initialized" not in st.session_state:
    st.session_state.is_enhanced_initialized = False
# Main page content
st.title("IridoVeda")

# Tab layout
tabs = st.tabs(["📚 PDF Upload & Processing", "🔍 Knowledge Query", "👁️ Iris Analysis", "🔬 IrisZone", "🧬 Advanced Analysis", "🌿 Dosha Analysis", "📊 Statistics", "⚙️ Configuration"])

# First tab - PDF Upload with enhanced processing
with tabs[0]:
    st.header("Upload Ayurvedic/Iridology Books")
    
    # Enhanced processing toggle
    processing_mode = st.radio(
        "Select Processing Mode:",
        options=["Standard Processing", "Enhanced NLP Processing"],
        horizontal=True,
        help="Enhanced mode uses advanced NLP techniques for better extraction"
    )
    
    uploaded_pdf = st.file_uploader(
        "Select PDF file(s)", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload Ayurvedic or Iridology books in PDF format"
    )
    
    if uploaded_pdf:
        # Process each uploaded PDF file
        for pdf_file in uploaded_pdf:
            with st.expander(f"Processing: {pdf_file.name}", expanded=True):
                # Create uploads directory if it doesn't exist
                os.makedirs("uploads", exist_ok=True)
                
                # Save file to uploads directory
                temp_path = os.path.join("uploads", pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.read())
                
                # Extract iris-related chunks
                with st.spinner("Extracting iris-related information..."):
                    start_time = time.time()
                    
                    # Display status message
                    status_placeholder = st.empty()
                    status_placeholder.info("📄 Analyzing text and image content...")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Define a custom callback for progress updates
                    def update_progress(page_num, total_pages):
                        progress = min(page_num / total_pages, 1.0)
                        progress_bar.progress(progress)
                        status_placeholder.info(f"📄 Processing page {page_num}/{total_pages}...")
                    
                    # Choose extraction method based on mode
                    if processing_mode == "Enhanced NLP Processing":
                        chunks = extract_structured_chunks(temp_path, progress_callback=update_progress)
                        st.session_state.enhanced_chunks.extend(chunks)
                    else:
                        chunks = extract_iris_chunks(temp_path, progress_callback=update_progress)
                        st.session_state.extracted_chunks.extend(chunks)
                    
                    end_time = time.time()
                    
                    # Update status upon completion
                    progress_bar.progress(1.0)
                    
                    # Add extraction method info
                    if processing_mode == "Enhanced NLP Processing":
                        ocr_used = any(chunk.get("extraction_method") == "ocr" for chunk in chunks)
                        if ocr_used:
                            status_placeholder.success("🔍 Advanced NLP processing with OCR completed")
                        else:
                            status_placeholder.success("📄 Advanced NLP processing completed")
                    else:
                        ocr_used = any(chunk.get("extraction_method") == "ocr" for chunk in chunks)
                        if ocr_used:
                            status_placeholder.success("🔍 OCR processing completed successfully")
                        else:
                            status_placeholder.success("📄 Text extraction completed successfully")
                    
                    # Display extraction results
                    st.success(f"✅ Extracted {len(chunks)} iris-related chunks")
                    st.info(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
                    
                    # Display sample chunks
                    if chunks:
                        st.subheader("Sample extracted content:")
                        for i, chunk in enumerate(chunks[:3]):
                            with st.container():
                                st.markdown(f"**Chunk {i+1}** (Page {chunk['page']}):")
                                st.markdown(f"> {chunk['text']}")
                                
                                # Show additional metadata for enhanced chunks
                                if processing_mode == "Enhanced NLP Processing" and "keywords" in chunk:
                                    st.markdown(f"Keywords: {', '.join(chunk.get('keywords', []))}")
                                    st.markdown(f"Relevance Score: {chunk.get('relevance_score', 0):.2f}")
                                
                                st.markdown("---")
                    
                    # Clean up temp file (commented out to keep file for re-use if needed)
                    # os.unlink(temp_path)
        
        # Store chunks in Qdrant
        if processing_mode == "Enhanced NLP Processing":
            if st.session_state.enhanced_chunks and st.button("Store in Enhanced Knowledge Base"):
                with st.spinner("Storing chunks in enhanced knowledge base..."):
                    try:
                        # Initialize collection
                        st.session_state.enhanced_qdrant_client.create_collection()
                        
                        # Store chunks
                        point_ids = st.session_state.enhanced_qdrant_client.store_chunks(
                            st.session_state.enhanced_chunks
                        )
                        
                        # Update session state
                        st.session_state.is_enhanced_initialized = True
                        
                        # Success message
                        st.success(
                            f"✅ Successfully stored {len(point_ids)} chunks in the enhanced knowledge base"
                        )
                        
                        # Clear stored chunks to avoid duplicates
                        st.session_state.enhanced_chunks = []
                        
                    except Exception as e:
                        st.error(f"Error storing chunks: {str(e)}")
        else:
            if st.session_state.extracted_chunks and st.button("Store in Standard Knowledge Base"):
                with st.spinner("Storing chunks in standard knowledge base..."):
                    try:
                        # Initialize collection
                        st.session_state.qdrant_client.create_collection()
                        
                        # Store chunks
                        point_ids = st.session_state.qdrant_client.store_chunks(
                            st.session_state.extracted_chunks
                        )
                        
                        # Update session state
                        st.session_state.is_initialized = True
                        
                        # Success message
                        st.success(
                            f"✅ Successfully stored {len(point_ids)} chunks in the standard knowledge base"
                        )
                        
                        # Clear stored chunks to avoid duplicates
                        st.session_state.extracted_chunks = []
                        
                    except Exception as e:
                        st.error(f"Error storing chunks: {str(e)}")
# Second tab - Knowledge Query with enhanced search
with tabs[1]:
    st.header("Query the Iridology Knowledge Base")
    
    # Choose query mode
    query_mode = st.radio(
        "Select Query Mode:",
        options=["Standard Search", "Enhanced Search", "Multi-Query Search"],
        horizontal=True,
        help="Enhanced search uses advanced algorithms for better results"
    )
    
    # Check if knowledge base is initialized based on mode
    is_ready = False
    if query_mode == "Standard Search" and st.session_state.is_initialized:
        is_ready = True
    elif (query_mode in ["Enhanced Search", "Multi-Query Search"]):
        # If enhanced not initialized but standard is, offer to migrate the data
        if not st.session_state.is_enhanced_initialized and st.session_state.is_initialized:
            st.info("Enhanced knowledge base is not yet initialized, but standard knowledge base has data.")
            if st.button("Migrate data from standard to enhanced knowledge base"):
                with st.spinner("Migrating data to enhanced knowledge base..."):
                    try:
                        # Get data from standard knowledge base
                        standard_results = st.session_state.qdrant_client.search("iris", limit=1000)
                        if standard_results:
                            # Create enhanced collection
                            st.session_state.enhanced_qdrant_client.create_collection()
                            
                            # Process and enhance the standard results
                            enhanced_chunks = []
                            for chunk in standard_results:
                                # Create a basic enhanced chunk from standard chunk
                                enhanced_chunk = create_enhanced_chunk_from_standard(chunk)
                                enhanced_chunks.append(enhanced_chunk)
                            
                            # Store in enhanced knowledge base
                            point_ids = st.session_state.enhanced_qdrant_client.store_chunks(enhanced_chunks)
                            st.session_state.is_enhanced_initialized = True
                            st.success(f"Successfully migrated {len(point_ids)} chunks to enhanced knowledge base!")
                            is_ready = True
                    except Exception as e:
                        st.error(f"Error migrating data: {str(e)}")
        elif st.session_state.is_enhanced_initialized:
            is_ready = True
    
    if not is_ready:
        st.warning(f"⚠️ {query_mode} knowledge base is empty. Please upload and process PDFs first.")
    else:
        # Track if a suggested query was clicked
        if "suggested_query_clicked" not in st.session_state:
            st.session_state.suggested_query_clicked = False
            st.session_state.last_query = ""
            
        # Get suggested queries
        suggested_queries = st.session_state.query_generator.get_suggested_queries(5)
        
        # Create a container for suggested queries
        st.markdown("### Suggested Queries")
        st.markdown("Click on any query to see instant results:")
        
        # Display suggested queries as clickable buttons
        cols = st.columns(2)
        for i, suggested_query in enumerate(suggested_queries):
            col_index = i % 2
            with cols[col_index]:
                if st.button(suggested_query, key=f"suggested_query_{i}", use_container_width=True):
                    handle_suggested_query(suggested_query, query_mode, result_count, min_relevance)
                    st.session_state.suggested_query_clicked = True
                    st.session_state.last_query = suggested_query
        
        st.markdown("---")
        st.markdown("### Custom Query")
        
        # Get the value to display in the text input
        input_value = st.session_state.last_query if st.session_state.suggested_query_clicked else ""
        
        query = st.text_input(
            "Ask a question about iridology:",
            value=input_value,
            placeholder="How does the iris pattern reflect liver conditions in iridology?",
            key="main_query_input"
        )
        
        # Reset the clicked state if the user changes the query
        if query != st.session_state.last_query:
            st.session_state.suggested_query_clicked = False
        
        if query and not st.session_state.suggested_query_clicked:
            with st.spinner("Searching knowledge base..."):
                try:
                    # Choose search method based on mode
                    if query_mode == "Standard Search":
                        results = st.session_state.qdrant_client.search(query, limit=result_count)
                        show_results(results, standard=True)
                    elif query_mode == "Enhanced Search":
                        results = st.session_state.enhanced_qdrant_client.hybrid_search(query, 
                                                                                      limit=result_count, 
                                                                                      min_score=min_relevance)
                        show_enhanced_results(results)
                    elif query_mode == "Multi-Query Search":
                        results = st.session_state.enhanced_qdrant_client.multi_query_search(query, limit=result_count)
                        show_enhanced_results(results)
                        
                        # Generate context-aware answer
                        answer_data = st.session_state.answer_generator.generate_answer(query, results)
                        show_generated_answer(answer_data)
                        
                except Exception as e:
                    st.error(f"Error searching knowledge base: {str(e)}")
# Third tab - Iris Analysis
with tabs[2]:
    st.header("Iris Image Analysis")
    
    st.info(
        "Upload an iris image to analyze it and generate relevant health queries. "
        "This feature uses computer vision to identify potential health indicators in the iris."
    )
    
    uploaded_image = st.file_uploader(
        "Upload iris image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an iris"
    )
    
    if uploaded_image:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_image.read())
            temp_path = tmp_file.name
        
        try:
            # Process iris image
            with st.spinner("Analyzing iris image..."):
                results = st.session_state.iris_predictor.process_iris_image(temp_path)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    # Display original image with iris outline
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Uploaded Image")
                        st.image(results["image"], use_column_width=True)
                    
                    with col2:
                        st.subheader("Analysis Summary")
                        
                        # Display health indicators
                        for zone, info in results["analysis"]["zones"].items():
                            status_color = "green" if info["condition"] == "normal" else "orange" if info["condition"] == "stressed" else "red"
                            st.markdown(
                                f"**{zone.capitalize()}:** "
                                f"<span style='color:{status_color}'>{info['condition']}</span> "
                                f"(confidence: {info['confidence']:.0%})",
                                unsafe_allow_html=True
                            )
                        
                        st.markdown(f"**Overall Health:** {results['analysis']['overall_health'].capitalize()}")
                    
                    # Generate and display suggested queries based on iris features
                    st.subheader("Suggested Queries")
                    
                    # Generate iris-specific queries using our new module
                    if "features" in results:
                        iris_queries = st.session_state.query_generator.generate_query_from_iris_features(results["features"])
                    else:
                        iris_queries = st.session_state.query_generator.get_suggested_queries(5)
                    
                    query_cols = st.columns(2)
                    for i, query in enumerate(iris_queries):
                        col_idx = i % 2
                        with query_cols[col_idx]:
                            if st.button(f"🔍 {query}", key=f"query_{i}"):
                                # Instead of just setting the query and switching tabs,
                                # we'll now directly handle the query and show results
                                st.session_state.current_query = query
                                st.session_state.selected_iris_query = query
                                
                    st.markdown("---")
                    
                    # Run selected query if set
                    if "selected_iris_query" in st.session_state:
                        query = st.session_state.selected_iris_query
                        
                        # Check which knowledge base to use
                        if st.session_state.is_enhanced_initialized:
                            with st.spinner(f"Searching enhanced knowledge base: {query}"):
                                # Search with enhanced client
                                results = st.session_state.enhanced_qdrant_client.multi_query_search(query, limit=3)
                                
                                # Generate answer
                                answer_data = st.session_state.answer_generator.generate_answer(query, results)
                                
                                # Display results
                                st.subheader("Knowledge Base Results:")
                                if not results:
                                    st.info("No matching information found in the knowledge base.")
                                else:
                                    show_enhanced_results(results)
                                    show_generated_answer(answer_data)
                                
                        elif st.session_state.is_initialized:
                            with st.spinner(f"Searching knowledge base: {query}"):
                                # Search with standard client
                                results = st.session_state.qdrant_client.search(query, limit=3)
                                
                                # Display results
                                st.subheader("Knowledge Base Results:")
                                if not results:
                                    st.info("No matching information found in the knowledge base.")
                                else:
                                    show_results(results)
                        else:
                            st.warning("Please initialize a knowledge base by uploading and processing PDFs first.")
                    
        except Exception as e:
            st.error(f"Error analyzing iris image: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_path)

# Fourth tab - IrisZone Layered Analysis
with tabs[3]:
    st.header("Iris Zone Analysis")
    
    st.info(
        "Upload an iris image to analyze the different zones according to Ayurvedic principles. "
        "This feature uses computer vision to identify and map different iris zones to corresponding body systems."
    )
    
    uploaded_zone_image = st.file_uploader(
        "Upload iris image for zone analysis", 
        type=["jpg", "jpeg", "png"],
        key="zone_image_uploader",
        help="Upload a clear image of an iris for detailed zone mapping"
    )
    
    if uploaded_zone_image:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_zone_image.read())
            temp_zone_path = tmp_file.name
        
        try:
            # Process iris image for zone analysis
            with st.spinner("Analyzing iris zones..."):
                zone_results = st.session_state.iris_zone_analyzer.process_iris_image(temp_zone_path)
                
                if "error" in zone_results:
                    st.error(zone_results["error"])
                else:
                    # Create tabs for different visualizations
                    zone_vis_tabs = st.tabs(["Zone Map", "Detailed Analysis", "Ayurvedic Interpretation"])
                    
                    # Zone Map visualization tab
                    with zone_vis_tabs[0]:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original Image")
                            st.image(zone_results["original_image"], use_column_width=True)
                        
                        with col2:
                            st.subheader("Iris Zone Map")
                            st.image(zone_results["zone_map"], use_column_width=True)
                            st.caption("The color overlay shows different zones of the iris according to iridology principles")
                    
                    # Detailed Analysis tab
                    with zone_vis_tabs[1]:
                        st.subheader("Detailed Zone Analysis")
                        
                        # Show boundary detection
                        st.image(zone_results["boundary_image"], use_column_width=True)
                        st.caption("Iris and pupil boundaries detected")
                        
                        # Display each zone's data in expanders
                        for zone_name, zone_data in zone_results["zones_analysis"].items():
                            zone_display_name = zone_data["name"]
                            health_condition = zone_data["health_indication"]["condition"]
                            confidence = zone_data["health_indication"]["confidence"]
                            
                            # Determine color based on health condition
                            if health_condition == "normal":
                                condition_color = "green"
                            elif health_condition == "stressed":
                                condition_color = "orange"
                            else:
                                condition_color = "red"
                                
                            with st.expander(f"{zone_display_name} - {health_condition.capitalize()}"):
                                # Display zone systems and description
                                st.markdown(f"**Corresponds to:** {', '.join(zone_data['ayurvedic_mapping']['systems'])}")
                                st.markdown(f"**Description:** {zone_data['ayurvedic_mapping']['description']}")
                                
                                # Display health indication
                                st.markdown(f"**Condition:** <span style='color:{condition_color}'>{health_condition.capitalize()}</span> (Confidence: {confidence:.1%})", unsafe_allow_html=True)
                                st.markdown(f"**Suggestion:** {zone_data['health_indication']['suggestion']}")
                                
                                # Display dosha information
                                dosha = zone_data['ayurvedic_mapping']['dominant_dosha']
                                st.markdown(f"**Dominant Dosha:** {dosha.capitalize()}")
                                
                                if dosha != "unknown" and len(zone_data['ayurvedic_mapping']['dosha_qualities']) > 0:
                                    st.markdown(f"**Qualities:** {', '.join(zone_data['ayurvedic_mapping']['dosha_qualities'])}")
                    
                    # Ayurvedic Interpretation tab
                    with zone_vis_tabs[2]:
                        st.subheader("Ayurvedic Interpretation")
                        
                        # Create a nice container for overall balance
                        with st.container():
                            st.markdown("""
                            <div style="padding: 10px; border-radius: 10px; border: 1px solid #f0f2f6; margin-bottom: 20px;">
                                <h3 style="color: #1E88E5; text-align: center;">Overall Balance</h3>
                                <h4 style="text-align: center; padding: 10px;">
                                    {}
                                </h4>
                            </div>
                            """.format(zone_results['health_summary']['overall_health'].capitalize()), unsafe_allow_html=True)
                        
                        # Create dosha balance visualization
                        if 'dosha_balance' in zone_results['health_summary']:
                            dosha_balance = zone_results['health_summary']['dosha_balance']
                            
                            # Create a two-column layout for visualization and interpretation
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Create dosha pie chart in a container
                                with st.container():
                                    st.markdown("""
                                    <div style="padding: 10px; border-radius: 10px; border: 1px solid #f0f2f6;">
                                        <h3 style="color: #1E88E5; text-align: center;">Dosha Distribution</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if dosha_balance:
                                        dosha_labels = [f"{k.capitalize()} ({v:.0%})" for k, v in dosha_balance.items()]
                                        dosha_values = list(dosha_balance.values())
                                        
                                        fig, ax = plt.subplots(figsize=(5, 5))
                                        wedges, texts, autotexts = ax.pie(
                                            dosha_values, 
                                            labels=dosha_labels, 
                                            autopct='%1.1f%%', 
                                            startangle=90, 
                                            colors=['#a29bfe', '#ff7675', '#55efc4'],
                                            textprops={'fontsize': 12, 'weight': 'bold'}
                                        )
                                        
                                        # Make the percentage numbers larger and bold
                                        for text in autotexts:
                                            text.set_fontsize(12)
                                            text.set_weight('bold')
                                            
                                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                                        
                                        # Add a color legend
                                        st.markdown("""
                                        <div style="display: flex; justify-content: center; text-align: center; margin-top: 10px;">
                                          <div style="margin: 0 10px;"><span style="background-color: #a29bfe; padding: 2px 8px; border-radius: 3px;">⬤</span> Vata</div>
                                          <div style="margin: 0 10px;"><span style="background-color: #ff7675; padding: 2px 8px; border-radius: 3px;">⬤</span> Pitta</div>
                                          <div style="margin: 0 10px;"><span style="background-color: #55efc4; padding: 2px 8px; border-radius: 3px;">⬤</span> Kapha</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            with col2:
                                # Dosha interpretation heading with styling
                                st.markdown("""
                                <div style="padding: 10px; border-radius: 10px; border: 1px solid #f0f2f6;">
                                    <h3 style="color: #1E88E5; text-align: center;">Dosha Interpretation</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0] if dosha_balance else "unknown"
                            
                            # Use a styled container for the dosha interpretation
                            dosha_colors = {
                                "vata": "#a29bfe",
                                "pitta": "#ff7675",
                                "kapha": "#55efc4",
                                "unknown": "#b2bec3"
                            }
                            
                            # Get the color for the primary dosha
                            dosha_color = dosha_colors.get(primary_dosha, "#b2bec3")
                            
                            if primary_dosha == "vata":
                                st.markdown(f"""
                                <div style="background-color: rgba(162, 155, 254, 0.1); border-left: 5px solid {dosha_color}; 
                                     padding: 15px; border-radius: 5px; margin-top: 10px;">
                                    <h4 style="color: {dosha_color};">Vata Dominant Iris</h4>
                                    
                                    <p>The iris shows signs of <strong>Vata dominance</strong>, which relates to the air and ether elements.</p>
                                    
                                    <p><strong>This often manifests as:</strong></p>
                                    <ul>
                                        <li>Heightened nervous system activity</li>
                                        <li>Potential for dryness and variability in systems</li>
                                        <li>Need for grounding and stability</li>
                                    </ul>
                                    
                                    <p><strong>Balancing Suggestions:</strong></p>
                                    <ul>
                                        <li>Regular routines and rest patterns</li>
                                        <li>Warm, nourishing, and grounding foods</li>
                                        <li>Gentle oil massage (abhyanga) with sesame oil</li>
                                        <li>Meditation and stress reduction practices</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            elif primary_dosha == "pitta":
                                st.markdown(f"""
                                <div style="background-color: rgba(255, 118, 117, 0.1); border-left: 5px solid {dosha_color}; 
                                     padding: 15px; border-radius: 5px; margin-top: 10px;">
                                    <h4 style="color: {dosha_color};">Pitta Dominant Iris</h4>
                                    
                                    <p>The iris shows signs of <strong>Pitta dominance</strong>, which relates to the fire and water elements.</p>
                                    
                                    <p><strong>This often manifests as:</strong></p>
                                    <ul>
                                        <li>Strong digestive and metabolic functions</li>
                                        <li>Tendency toward heat and inflammation</li>
                                        <li>Potential for intensity and sharpness in bodily processes</li>
                                    </ul>
                                    
                                    <p><strong>Balancing Suggestions:</strong></p>
                                    <ul>
                                        <li>Cooling foods and herbs</li>
                                        <li>Avoiding excessive heat, sun exposure, and spicy foods</li>
                                        <li>Regular exercise that's not too intense</li>
                                        <li>Calming and cooling practices like moonlight walks</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            elif primary_dosha == "kapha":
                                st.markdown(f"""
                                <div style="background-color: rgba(85, 239, 196, 0.1); border-left: 5px solid {dosha_color}; 
                                     padding: 15px; border-radius: 5px; margin-top: 10px;">
                                    <h4 style="color: {dosha_color};">Kapha Dominant Iris</h4>
                                    
                                    <p>The iris shows signs of <strong>Kapha dominance</strong>, which relates to the earth and water elements.</p>
                                    
                                    <p><strong>This often manifests as:</strong></p>
                                    <ul>
                                        <li>Stable energy and strong immunity</li>
                                        <li>Potential for congestion or sluggishness</li>
                                        <li>Well-developed structural elements in the body</li>
                                    </ul>
                                    
                                    <p><strong>Balancing Suggestions:</strong></p>
                                    <ul>
                                        <li>Regular stimulating exercise</li>
                                        <li>Warm, light, and spiced foods</li>
                                        <li>Dry brushing and invigorating practices</li>
                                        <li>Varied routines to prevent stagnation</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add a section for secondary and tertiary doshas
                            st.markdown("<br>", unsafe_allow_html=True)
                            if len(dosha_balance) > 1:
                                sorted_doshas = sorted(dosha_balance.items(), key=lambda x: x[1], reverse=True)
                                
                                # Only display if there's at least one secondary dosha
                                if len(sorted_doshas) > 1 and sorted_doshas[1][1] > 0.15:  # Only show if secondary dosha is at least 15%
                                    secondary_dosha = sorted_doshas[1][0]
                                    secondary_color = dosha_colors.get(secondary_dosha, "#b2bec3")
                                    
                                    st.markdown(f"""
                                    <div style="background-color: rgba(240, 240, 240, 0.3); border: 1px solid #ddd; 
                                         padding: 15px; border-radius: 5px; margin-top: 10px;">
                                        <h4 style="color: {secondary_color}; margin-top: 0;">Secondary Influence: {secondary_dosha.capitalize()} ({(sorted_doshas[1][1]*100):.1f}%)</h4>
                                        <p>This secondary dosha influence suggests a {primary_dosha}-{secondary_dosha} dual influence in your constitution. 
                                        Balance both doshas for optimal results.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add a dosha relationship chart
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.markdown("""
                                    <div style="background-color: rgba(240, 240, 240, 0.3); border: 1px solid #ddd; 
                                         padding: 15px; border-radius: 5px; margin-top: 10px;">
                                        <h4 style="text-align: center; margin-top: 0;">Dosha Relationship</h4>
                                        <p style="text-align: center;">The three doshas work together in your system, with one or two typically more dominant.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                # Visual separator before the form section
                st.markdown("<hr style='margin-top: 30px; margin-bottom: 20px; border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0));'>", unsafe_allow_html=True)
                
                # Add user info form with improved styling
                with st.expander("📋 Add Personal Information to Report", expanded=False):
                    st.markdown("""
                    <div style="background-color: rgba(240, 240, 240, 0.3); border: 1px solid #ddd; 
                         padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                        <p style="color: #1E88E5; margin: 0;">
                            This information will be included in your report. All fields are optional.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    user_info = {}
                    col1, col2 = st.columns(2)
                    with col1:
                        user_info["Name"] = st.text_input("Name", "", key="zone_report_name")
                        user_info["Age"] = st.text_input("Age", "", key="zone_report_age")
                    with col2:
                        user_info["Gender"] = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="zone_report_gender")
                        user_info["Date"] = st.date_input("Date", datetime.now(), key="zone_report_date").strftime("%Y-%m-%d")
                    
                    # Remove empty fields
                    user_info = {k: v for k, v in user_info.items() if v}
                
                # Generate and offer reports for download
                try:
                    # Add a nice heading for reports section
                    st.markdown("""
                    <div style="padding: 10px; border-radius: 10px; border: 1px solid #f0f2f6; margin-top: 20px; margin-bottom: 20px;">
                        <h3 style="color: #1E88E5; text-align: center;">Download Analysis Reports</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create columns for download buttons
                    dl_col1, dl_col2 = st.columns(2)
                    
                    # First, always generate the HTML report
                    html_report = st.session_state.iris_report_generator.generate_html_report(zone_results, user_info)
                    
                    with dl_col1:
                        st.markdown("""
                        <div style="background-color: rgba(30, 136, 229, 0.1); border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px;">
                            <h4 style="margin-top: 0;">HTML Report</h4>
                            <p>Interactive report for viewing in web browsers</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=html_report,
                            file_name=f"iris_zone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            key="html_report_button",
                            use_container_width=True
                        )
                    
                    # Try to generate PDF report if available
                    pdf_report = st.session_state.iris_report_generator.generate_report(zone_results, user_info)
                    
                    with dl_col2:
                        st.markdown("""
                        <div style="background-color: rgba(30, 136, 229, 0.1); border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px;">
                            <h4 style="margin-top: 0;">PDF Report</h4>
                            <p>Printable document with analysis details</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if pdf_report is not None:
                            st.download_button(
                                label="📑 Download PDF Report",
                                data=pdf_report,
                                file_name=f"iris_zone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key="pdf_report_button",
                                use_container_width=True
                            )
                        else:
                            st.warning("PDF generation is not available. To enable PDF reports, install the 'fpdf' package.")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                
        except Exception as e:
            st.error(f"Error analyzing iris zones: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_zone_path)

# Sixth tab - Dosha Analysis
with tabs[5]:
    st.header("🌿 Ayurvedic Dosha Analysis")
    
    st.info(
        "Comprehensive Ayurvedic dosha analysis using iris image analysis. "
        "This feature quantifies Vata, Pitta, and Kapha doshas, provides metabolic health indicators, "
        "and offers personalized Ayurvedic recommendations based on your constitutional makeup."
    )
    
    # Add explanation for dosha analysis
    with st.expander("Understanding Dosha Analysis", expanded=True):
        st.markdown("""
        **Ayurvedic Dosha Analysis includes:**
        
        * **Constitutional Assessment** - Determine your Prakriti (original constitution) and Vikriti (current state)
        * **Dosha Quantification** - Precise percentage calculation of Vata, Pitta, and Kapha
        * **Metabolic Health Metrics** - BMR, lipid profiles, inflammation markers, and gut health indicators
        * **Organ Health Assessment** - Evaluation of major organs based on iris zones and dosha influences
        * **Personalized Recommendations** - Diet, lifestyle, and treatment suggestions based on your dosha profile
        * **Imbalance Detection** - Identify specific dosha imbalances and their severity
        """)
    
    # Create uploader for dosha analysis
    uploaded_dosha_image = st.file_uploader(
        "Upload iris image for dosha analysis", 
        type=["jpg", "jpeg", "png"],
        key="dosha_image_uploader",
        help="Upload a clear and high-resolution iris image for comprehensive dosha analysis"
    )
    
    if uploaded_dosha_image:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_dosha_image.read())
            temp_dosha_path = tmp_file.name
        
        try:
            # Import dosha quantification model
            from dosha_quantification_model import DoshaQuantificationModel, MetabolicIndicators, OrganHealthAssessment
            
            # Initialize dosha model
            if 'dosha_model' not in st.session_state:
                st.session_state.dosha_model = DoshaQuantificationModel()
            
            # Process iris image for dosha analysis
            with st.spinner("Performing comprehensive dosha analysis..."):
                dosha_results = st.session_state.dosha_model.analyze_iris_for_doshas(temp_dosha_path)
                
                if "error" in dosha_results:
                    st.error(dosha_results["error"])
                else:
                    # Create tabs for different dosha analysis views
                    dosha_tabs = st.tabs([
                        "Constitutional Profile", 
                        "Dosha Distribution", 
                        "Metabolic Health", 
                        "Organ Assessment", 
                        "Recommendations", 
                        "Detailed Report"
                    ])
                    
                    # Constitutional Profile tab
                    with dosha_tabs[0]:
                        st.subheader("Your Ayurvedic Constitutional Profile")
                        
                        dosha_profile = dosha_results.get("dosha_profile", {})
                        
                        if dosha_profile:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Display primary constitution
                                primary_dosha = dosha_profile.get("primary_dosha", "unknown")
                                secondary_dosha = dosha_profile.get("secondary_dosha", None)
                                
                                st.markdown(f"### Primary Constitution: **{primary_dosha.upper()}**")
                                if secondary_dosha:
                                    st.markdown(f"### Secondary Constitution: **{secondary_dosha.upper()}**")
                                
                                # Constitution description
                                constitution_desc = {
                                    "vata": "Air and ether elements. Characterized by movement, creativity, and quick thinking. Tends toward dryness, coldness, and variability.",
                                    "pitta": "Fire and water elements. Characterized by metabolism, digestion, and transformation. Tends toward heat, intensity, and precision.",
                                    "kapha": "Water and earth elements. Characterized by structure, stability, and immunity. Tends toward moisture, coolness, and steadiness."
                                }
                                
                                if primary_dosha in constitution_desc:
                                    st.markdown(f"**Description:** {constitution_desc[primary_dosha]}")
                                
                                # Balance assessment
                                balance_score = dosha_profile.get("balance_score", 0)
                                imbalance_type = dosha_profile.get("imbalance_type", None)
                                
                                st.markdown("### Constitutional Balance")
                                balance_color = "green" if balance_score > 0.7 else "orange" if balance_score > 0.4 else "red"
                                st.markdown(f"**Balance Score:** <span style='color: {balance_color}'>{balance_score:.2f}</span>", unsafe_allow_html=True)
                                st.progress(balance_score)
                                
                                if imbalance_type:
                                    st.warning(f"**Detected Imbalance:** {imbalance_type}")
                            
                            with col2:
                                # Display constitutional percentages
                                st.markdown("### Dosha Percentages")
                                
                                vata_pct = dosha_profile.get("vata_percentage", 0) * 100
                                pitta_pct = dosha_profile.get("pitta_percentage", 0) * 100
                                kapha_pct = dosha_profile.get("kapha_percentage", 0) * 100
                                
                                # Create dosha percentage visualization
                                fig, ax = plt.subplots(figsize=(6, 6))
                                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Vata, Pitta, Kapha colors
                                sizes = [vata_pct, pitta_pct, kapha_pct]
                                labels = [f'Vata\n{vata_pct:.1f}%', f'Pitta\n{pitta_pct:.1f}%', f'Kapha\n{kapha_pct:.1f}%']
                                
                                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
                                ax.set_title('Constitutional Distribution', fontsize=14, fontweight='bold')
                                
                                st.pyplot(fig)
                    
                    # Dosha Distribution tab
                    with dosha_tabs[1]:
                        st.subheader("Detailed Dosha Analysis")
                        
                        dosha_profile = dosha_results.get("dosha_profile", {})
                        
                        if dosha_profile:
                            # Create three columns for each dosha
                            vata_col, pitta_col, kapha_col = st.columns(3)
                            
                            with vata_col:
                                st.markdown("#### 💨 VATA (Air + Ether)")
                                vata_score = dosha_profile.get("vata_score", 0)
                                vata_pct = dosha_profile.get("vata_percentage", 0) * 100
                                
                                st.metric("Score", f"{vata_score:.2f}")
                                st.metric("Percentage", f"{vata_pct:.1f}%")
                                st.progress(vata_pct/100)
                                
                                st.markdown("**Qualities:**")
                                st.markdown("- Movement & circulation")
                                st.markdown("- Nervous system function")
                                st.markdown("- Mental activity & creativity")
                                st.markdown("- Elimination processes")
                            
                            with pitta_col:
                                st.markdown("#### 🔥 PITTA (Fire + Water)")
                                pitta_score = dosha_profile.get("pitta_score", 0)
                                pitta_pct = dosha_profile.get("pitta_percentage", 0) * 100
                                
                                st.metric("Score", f"{pitta_score:.2f}")
                                st.metric("Percentage", f"{pitta_pct:.1f}%")
                                st.progress(pitta_pct/100)
                                
                                st.markdown("**Qualities:**")
                                st.markdown("- Digestion & metabolism")
                                st.markdown("- Body temperature regulation")
                                st.markdown("- Intelligence & focus")
                                st.markdown("- Hormone production")
                            
                            with kapha_col:
                                st.markdown("#### 🌊 KAPHA (Water + Earth)")
                                kapha_score = dosha_profile.get("kapha_score", 0)
                                kapha_pct = dosha_profile.get("kapha_percentage", 0) * 100
                                
                                st.metric("Score", f"{kapha_score:.2f}")
                                st.metric("Percentage", f"{kapha_pct:.1f}%")
                                st.progress(kapha_pct/100)
                                
                                st.markdown("**Qualities:**")
                                st.markdown("- Structure & stability")
                                st.markdown("- Immunity & strength")
                                st.markdown("- Lubrication & moisture")
                                st.markdown("- Growth & repair")
                    
                    # Metabolic Health tab
                    with dosha_tabs[2]:
                        st.subheader("Metabolic Health Indicators")
                        
                        metabolic_data = dosha_results.get("metabolic_indicators", {})
                        
                        if metabolic_data:
                            # Create metrics display
                            met_col1, met_col2 = st.columns(2)
                            
                            with met_col1:
                                st.markdown("#### Energy & Metabolism")
                                
                                bmr = metabolic_data.get("basal_metabolic_rate", 0)
                                st.metric("Basal Metabolic Rate", f"{bmr:.0f} kcal/day")
                                
                                metabolic_efficiency = metabolic_data.get("metabolic_efficiency", 0)
                                st.metric("Metabolic Efficiency", f"{metabolic_efficiency:.1%}")
                                
                                energy_level = metabolic_data.get("energy_level", 0)
                                st.metric("Energy Level", f"{energy_level:.1f}/10")
                                
                                st.markdown("#### Digestive Health")
                                
                                digestive_fire = metabolic_data.get("digestive_fire", 0)
                                st.metric("Digestive Fire (Agni)", f"{digestive_fire:.1f}/10")
                                
                                gut_health = metabolic_data.get("gut_diversity", 0)
                                st.metric("Gut Health Score", f"{gut_health:.1f}/10")
                            
                            with met_col2:
                                st.markdown("#### Cardiovascular & Inflammatory Markers")
                                
                                serum_lipid = metabolic_data.get("serum_lipid", 0)
                                st.metric("Serum Lipid", f"{serum_lipid:.0f} mg/dL")
                                
                                triglycerides = metabolic_data.get("triglycerides", 0)
                                st.metric("Triglycerides", f"{triglycerides:.0f} mg/dL")
                                
                                crp_level = metabolic_data.get("crp_level", 0)
                                st.metric("Inflammation (CRP)", f"{crp_level:.2f} mg/L")
                                
                                st.markdown("#### Stress & Recovery")
                                
                                stress_level = metabolic_data.get("stress_indicators", {}).get("overall_stress", 0)
                                st.metric("Stress Level", f"{stress_level:.1f}/10")
                                
                                recovery_capacity = metabolic_data.get("recovery_capacity", 0)
                                st.metric("Recovery Capacity", f"{recovery_capacity:.1f}/10")
                    
                    # Organ Assessment tab
                    with dosha_tabs[3]:
                        st.subheader("Organ Health Assessment")
                        
                        organ_assessment = dosha_results.get("organ_health", {})
                        
                        if organ_assessment:
                            # Create organ system cards
                            organs = ["heart", "liver", "kidneys", "lungs", "digestive_system", "nervous_system"]
                            organ_names = ["Heart", "Liver", "Kidneys", "Lungs", "Digestive System", "Nervous System"]
                            
                            for i in range(0, len(organs), 2):
                                cols = st.columns(2)
                                
                                for j, col in enumerate(cols):
                                    if i + j < len(organs):
                                        organ = organs[i + j]
                                        organ_name = organ_names[i + j]
                                        organ_data = organ_assessment.get(organ, {})
                                        
                                        with col:
                                            with st.container():
                                                st.markdown(f"#### {organ_name}")
                                                
                                                health_score = organ_data.get("health_score", 0)
                                                vitality = organ_data.get("vitality", 0)
                                                dosha_influence = organ_data.get("dominant_dosha", "unknown")
                                                
                                                # Health score with color coding
                                                score_color = "green" if health_score > 0.7 else "orange" if health_score > 0.4 else "red"
                                                st.markdown(f"**Health Score:** <span style='color: {score_color}'>{health_score:.2f}/1.0</span>", unsafe_allow_html=True)
                                                st.progress(health_score)
                                                
                                                st.metric("Vitality", f"{vitality:.1f}/10")
                                                st.markdown(f"**Dominant Dosha:** {dosha_influence.capitalize()}")
                                                
                                                # Show recommendations if available
                                                recommendations = organ_data.get("recommendations", [])
                                                if recommendations:
                                                    st.markdown("**Recommendations:**")
                                                    for rec in recommendations[:2]:  # Show top 2 recommendations
                                                        st.markdown(f"• {rec}")
                    
                    # Recommendations tab
                    with dosha_tabs[4]:
                        st.subheader("Personalized Ayurvedic Recommendations")
                        
                        recommendations = dosha_results.get("recommendations", {})
                        
                        if recommendations:
                            # Diet recommendations
                            if "diet" in recommendations:
                                st.markdown("### 🍽️ Dietary Recommendations")
                                diet_recs = recommendations["diet"]
                                
                                diet_col1, diet_col2 = st.columns(2)
                                
                                with diet_col1:
                                    st.markdown("#### Foods to Favor")
                                    favorable_foods = diet_recs.get("favorable_foods", [])
                                    for food in favorable_foods:
                                        st.markdown(f"✅ {food}")
                                
                                with diet_col2:
                                    st.markdown("#### Foods to Avoid")
                                    avoid_foods = diet_recs.get("avoid_foods", [])
                                    for food in avoid_foods:
                                        st.markdown(f"❌ {food}")
                            
                            # Lifestyle recommendations
                            if "lifestyle" in recommendations:
                                st.markdown("### 🧘 Lifestyle Recommendations")
                                lifestyle_recs = recommendations["lifestyle"]
                                
                                for category, advice_list in lifestyle_recs.items():
                                    if advice_list:
                                        st.markdown(f"#### {category.replace('_', ' ').title()}")
                                        for advice in advice_list:
                                            st.markdown(f"• {advice}")
                            
                            # Herbal recommendations
                            if "herbs" in recommendations:
                                st.markdown("### 🌿 Herbal Recommendations")
                                herb_recs = recommendations["herbs"]
                                
                                for herb_info in herb_recs:
                                    st.markdown(f"**{herb_info.get('name', 'Unknown')}**: {herb_info.get('benefit', 'No description available')}")
                    
                    # Detailed Report tab
                    with dosha_tabs[5]:
                        st.subheader("Comprehensive Analysis Report")
                        
                        # Generate downloadable report
                        if st.button("🔄 Generate Comprehensive Report", use_container_width=True):
                            with st.spinner("Generating comprehensive dosha analysis report..."):
                                try:
                                    # Generate PDF report using iris report generator
                                    if 'iris_report_generator' not in st.session_state:
                                        from iris_report_generator import IrisReportGenerator
                                        st.session_state.iris_report_generator = IrisReportGenerator()
                                    
                                    # Create a comprehensive report data structure
                                    report_data = {
                                        "analysis_type": "dosha_analysis",
                                        "dosha_profile": dosha_results.get("dosha_profile", {}),
                                        "metabolic_indicators": dosha_results.get("metabolic_indicators", {}),
                                        "organ_health": dosha_results.get("organ_health", {}),
                                        "recommendations": dosha_results.get("recommendations", {}),
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    
                                    pdf_report = st.session_state.iris_report_generator.generate_dosha_report(report_data)
                                    
                                    if pdf_report is not None:
                                        st.download_button(
                                            label="📑 Download Dosha Analysis Report",
                                            data=pdf_report,
                                            file_name=f"dosha_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            key="dosha_pdf_report_button",
                                            use_container_width=True
                                        )
                                        st.success("✅ Comprehensive dosha analysis report generated successfully!")
                                    else:
                                        st.warning("PDF generation is not available. To enable PDF reports, install the 'fpdf' package.")
                                except Exception as e:
                                    st.error(f"Error generating report: {str(e)}")
                        
                        # Display raw analysis data for advanced users
                        with st.expander("Raw Analysis Data (Advanced)", expanded=False):
                            st.json(dosha_results)
                        
        except ImportError:
            st.error("Dosha quantification model is not available. Please ensure all required modules are installed.")
        except Exception as e:
            st.error(f"Error performing dosha analysis: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_dosha_path)

# Seventh tab - Statistics with enhanced metrics
with tabs[6]:
    st.header("Knowledge Base Statistics")
    
    # Create tabs for different stats
    stat_tabs = st.tabs(["Overview", "Content Analysis", "Performance Metrics"])
    
    with stat_tabs[0]:
        # Display basic stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Standard Chunks", len(st.session_state.extracted_chunks))
        
        with col2:
            st.metric("Enhanced Chunks", len(st.session_state.enhanced_chunks))
        
        with col3:
            status1 = "Initialized" if st.session_state.is_initialized else "Not Initialized"
            status2 = "Initialized" if st.session_state.is_enhanced_initialized else "Not Initialized"
            st.metric("Knowledge Base Status", f"Standard: {status1}\nEnhanced: {status2}")

        # Display source breakdown
        if st.session_state.extracted_chunks or st.session_state.enhanced_chunks:
            # Get unique sources
            sources = {}
            
            # Add from standard chunks
            for chunk in st.session_state.extracted_chunks:
                source = Path(chunk.get('source', '')).name
                if source in sources:
                    sources[source] = sources[source] + 1
                else:
                    sources[source] = 1
                    
            # Add from enhanced chunks
            for chunk in st.session_state.enhanced_chunks:
                source = Path(chunk.get('source', '')).name
                if source in sources:
                    sources[source] = sources[source] + 1
                else:
                    sources[source] = 1
            
            # Display as a table
            st.subheader("Source Distribution")
            source_data = {"Source": list(sources.keys()), "Chunks": list(sources.values())}
            st.dataframe(source_data, use_container_width=True)
            
            # Add a bar chart
            if sources:
                st.bar_chart(sources)
    
    with stat_tabs[1]:
        st.subheader("Content Analysis")
        
        # Display extraction methods breakdown if enhanced chunks exist
        if st.session_state.enhanced_chunks:
            # Count by extraction method
            extraction_methods = {}
            for chunk in st.session_state.enhanced_chunks:
                method = chunk.get('extraction_method', 'unknown')
                if method in extraction_methods:
                    extraction_methods[method] += 1
                else:
                    extraction_methods[method] = 1
            
            # Display as a pie chart
            st.subheader("Extraction Methods")
            
            extraction_df = pd.DataFrame({
                'Method': list(extraction_methods.keys()),
                'Count': list(extraction_methods.values())
            })
            
            if extraction_methods:
                st.bar_chart(extraction_methods)
            st.dataframe(extraction_df, use_container_width=True)
            
            # Show relevance score distribution
            if any('relevance_score' in chunk for chunk in st.session_state.enhanced_chunks):
                relevance_scores = [chunk.get('relevance_score', 0) for chunk in st.session_state.enhanced_chunks 
                                   if 'relevance_score' in chunk]
                
                if relevance_scores:
                    st.subheader("Relevance Score Distribution")
                    fig, ax = plt.subplots()
                    ax.hist(relevance_scores, bins=10, alpha=0.7)
                    ax.set_xlabel('Relevance Score')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    
                    # Display average relevance
                    st.metric("Average Relevance Score", f"{sum(relevance_scores) / len(relevance_scores):.2f}")
    
    with stat_tabs[2]:
        st.subheader("Performance Metrics")
        
        # Create some demo metrics (replace with real measurements in production)
        st.info("These are sample metrics for demonstration. Implement actual measurements in production.")
        
        # Create a dataframe with performance metrics
        metrics_data = {
            'Processing Type': ['Standard', 'Enhanced NLP'],
            'Average Processing Time (sec/page)': [0.8, 1.5],
            'Content Extraction Rate (chunks/page)': [2.3, 4.1],
            'Average Relevance Score': [0.65, 0.78],
            'Query Performance (ms)': [120, 150]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

# Fifth tab - Advanced Analysis
with tabs[4]:
    st.header("Advanced Iris Analysis")
    
    st.info(
        "This feature uses advanced computer vision and machine learning techniques to perform "
        "detailed analysis of iris colors, spots, textures, and patterns. The system can also match "
        "patterns with similar irises in the database."
    )
    
    # Add explanation for features
    with st.expander("What does this analysis include?", expanded=True):
        st.markdown("""
        **Advanced Iris Analysis includes:**
        
        * **High-Resolution Processing** - Enhanced image processing for detailed feature extraction
        * **Color Analysis** - Identification and quantification of dominant colors in the iris
        * **Spot Detection** - Identification of spots, freckles, and markings in the iris
        * **Texture Analysis** - Assessment of iris texture patterns and their significance
        * **Pattern Matching** - Comparison with similar iris patterns from database
        * **Health Insights** - Generation of health indicators based on combined analysis
        """)
    
    # Create uploader for iris image
    uploaded_advanced_image = st.file_uploader(
        "Upload iris image for advanced analysis", 
        type=["jpg", "jpeg", "png"],
        key="advanced_image_uploader",
        help="Upload a clear and high-resolution image of an iris"
    )
    
    if uploaded_advanced_image:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_advanced_image.read())
            temp_advanced_path = tmp_file.name
        
        try:
            # Process iris image using the advanced analyzer
            with st.spinner("Performing advanced iris analysis..."):
                analysis_results = st.session_state.advanced_iris_analyzer.analyze_iris(
                    temp_advanced_path,
                    enhanced_qdrant_client=st.session_state.enhanced_qdrant_client if st.session_state.is_enhanced_initialized else None
                )
                
                if "error" in analysis_results:
                    st.error(analysis_results["error"])
                else:
                    # Create tabs for different analysis views
                    advanced_tabs = st.tabs(["Overview", "Enhanced Spot Detection", "Color Analysis", "Spot Detection", "Texture Analysis", "Pattern Matching", "Health Insights", "Health Queries"])
                    
                    # Overview tab
                    with advanced_tabs[0]:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Original Image")
                            
                            # Display original image with segmentation outline
                            if "image_paths" in analysis_results and "segmentation" in analysis_results["image_paths"]:
                                st.image("data:image/png;base64," + analysis_results["image_paths"]["segmentation"], use_column_width=True)
                                st.caption("Iris and pupil boundaries detected")
                            elif "image_paths" in analysis_results and "original" in analysis_results["image_paths"]:
                                st.image("data:image/png;base64," + analysis_results["image_paths"]["original"], use_column_width=True)
                        
                        with col2:
                            st.subheader("Analysis Summary")
                            
                            # Display processing time
                            processing_time = analysis_results.get("processing_time", 0)
                            st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                            
                            # Display iris segmentation data
                            segmentation = analysis_results.get("segmentation", {})
                            if segmentation:
                                st.markdown("#### Structural Parameters:")
                                
                                # Calculate pupil/iris ratio
                                pupil_radius = segmentation.get("pupil_radius", 0)
                                iris_radius = segmentation.get("iris_radius", 0)
                                
                                if iris_radius > 0 and pupil_radius is not None:
                                    pupil_iris_ratio = pupil_radius / iris_radius
                                    st.markdown(f"**Pupil/Iris Ratio:** {pupil_iris_ratio:.2f}")
                                
                                # Show counts of detected features
                                features = analysis_results.get("features", {})
                                spot_count = features.get("spot_count", 0)
                                st.markdown(f"**Detected Spots/Markings:** {spot_count}")
                                
                                # Display dominant color count
                                colors = features.get("color_summary", [])
                                st.markdown(f"**Dominant Colors:** {len(colors)}")
                            
                            # If zone analysis is available
                            if "zone_analysis" in analysis_results:
                                st.markdown("#### Health Assessment:")
                                
                                # Display overall health
                                overall_health = analysis_results["zone_analysis"].get("health_summary", {}).get("overall_health", "unknown")
                                st.markdown(f"**Overall Assessment:** {overall_health.capitalize()}")
                                
                                # Display dosha balance if available
                                dosha_balance = analysis_results["zone_analysis"].get("health_summary", {}).get("dosha_balance", {})
                                if dosha_balance:
                                    primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0]
                                    st.markdown(f"**Primary Dosha:** {primary_dosha.capitalize()}")
                    
                    # Enhanced Spot Detection tab
                    with advanced_tabs[1]:
                        st.subheader("🔬 Enhanced Iris Spot Detection & Analysis")
                        
                        if not ENHANCED_IMAGE_ANALYZER_AVAILABLE:
                            st.error("Enhanced iris image analyzer is not available. Please check that all required dependencies are installed.")
                            st.info("Required packages: opencv-python, matplotlib, pandas, scikit-image")
                        else:
                            # Add sensitivity controls
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sensitivity = st.selectbox(
                                    "Detection Sensitivity",
                                    ["low", "medium", "high"],
                                    index=1,
                                    help="Higher sensitivity detects more spots but may include false positives"
                                )
                            
                            with col2:
                                iris_side = st.selectbox(
                                    "Iris Side",
                                    ["left", "right", "unknown"],
                                    index=2,
                                    help="Select the iris side for accurate organ mapping"
                                )
                            
                            with col3:
                                if st.button("🔄 Re-analyze with Enhanced Detection", use_container_width=True):
                                    st.session_state.enhanced_iris_image_analyzer.set_detection_sensitivity(sensitivity)
                                    
                                    with st.spinner("Performing enhanced spot detection..."):
                                        enhanced_results = st.session_state.enhanced_iris_image_analyzer.analyze_iris_image(
                                            temp_advanced_path, iris_side
                                        )
                                        
                                        if "error" not in enhanced_results:
                                            st.session_state.enhanced_analysis_results = enhanced_results
                                            st.success(f"✅ Enhanced analysis complete! Detected {enhanced_results['total_spots']} spots using {len(enhanced_results['detection_methods_used'])} different methods.")
                                        else:
                                            st.error(f"Enhanced analysis failed: {enhanced_results['error']}")
                            
                            # Display enhanced results if available
                            if hasattr(st.session_state, 'enhanced_analysis_results'):
                                enhanced_results = st.session_state.enhanced_analysis_results
                                
                                # Create sub-tabs for enhanced analysis
                                enhanced_subtabs = st.tabs(["Annotated Image", "Spot Details", "Organ Mapping", "Health Assessment"])
                                
                                with enhanced_subtabs[0]:
                                    st.subheader("Annotated Iris Image")
                                    
                                    if "annotated_image" in enhanced_results:
                                        st.image(
                                            f"data:image/png;base64,{enhanced_results['annotated_image']}", 
                                            caption="Enhanced spot detection with organ mapping",
                                            use_column_width=True
                                        )
                                    
                                    # Display legend for detection methods
                                    st.markdown("#### Detection Method Legend")
                                    
                                    methods_used = enhanced_results.get('detection_methods_used', [])
                                    method_info = {
                                        'edge': ('E', 'Edge Detection', 'Green'),
                                        'dark_spot': ('D', 'Dark Spots', 'Red'),
                                        'light_spot': ('L', 'Light Spots', 'Orange'),
                                        'blob': ('B', 'Blob Detection', 'Yellow'),
                                        'adaptive': ('A', 'Adaptive Threshold', 'Pink'),
                                        'contour': ('C', 'Contour-based', 'Magenta'),
                                        'watershed': ('W', 'Watershed', 'Cyan')
                                    }
                                    
                                    legend_cols = st.columns(min(4, len(methods_used)))
                                    for i, method in enumerate(methods_used):
                                        if method in method_info:
                                            symbol, name, color = method_info[method]
                                            with legend_cols[i % len(legend_cols)]:
                                                st.markdown(f"**{symbol}** - {name} ({color})")
                                
                                with enhanced_subtabs[1]:
                                    st.subheader("Detailed Spot Analysis")
                                    
                                    spots = enhanced_results.get('spots', [])
                                    
                                    if spots:
                                        # Create DataFrame for display
                                        spot_data = []
                                        for spot in spots:
                                            spot_data.append({
                                                'ID': spot['segment_id'],
                                                'Method': spot['detection_method'],
                                                'Area (px²)': spot['area'],
                                                'Zone': spot.get('iris_zone', 'Unknown'),
                                                'Organ': spot.get('corresponding_organ', 'Unknown'),
                                                'Angle (°)': f"{spot.get('angle_degrees', 0):.1f}",
                                                'Distance (%)': f"{spot.get('distance_from_center_pct', 0):.1f}",
                                                'Visibility': f"{spot.get('visibility', 0):.1f}",
                                                'Brightness': f"{spot.get('avg_brightness', 0):.1f}"
                                            })
                                        
                                        df_spots = pd.DataFrame(spot_data)
                                        st.dataframe(df_spots, use_container_width=True)
                                        
                                        # Summary statistics
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            st.metric("Total Spots", len(spots))
                                        
                                        with col2:
                                            avg_area = np.mean([spot['area'] for spot in spots])
                                            st.metric("Avg Area", f"{avg_area:.0f} px²")
                                        
                                        with col3:
                                            avg_visibility = np.mean([spot.get('visibility', 0) for spot in spots])
                                            st.metric("Avg Visibility", f"{avg_visibility:.1f}")
                                        
                                        with col4:
                                            methods_count = len(enhanced_results.get('detection_methods_used', []))
                                            st.metric("Detection Methods", methods_count)
                                    else:
                                        st.info("No spots detected with current sensitivity settings.")
                                
                                with enhanced_subtabs[2]:
                                    st.subheader("Organ System Mapping")
                                    
                                    analysis_summary = enhanced_results.get('analysis_summary', {})
                                    organs = analysis_summary.get('organs', {})
                                    
                                    if organs and MATPLOTLIB_AVAILABLE:
                                        # Create organ distribution chart
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        
                                        organ_names = list(organs.keys())
                                        organ_counts = list(organs.values())
                                        
                                        # Limit to top 10 organs for readability
                                        if len(organ_names) > 10:
                                            sorted_organs = sorted(organs.items(), key=lambda x: x[1], reverse=True)[:10]
                                            organ_names = [item[0] for item in sorted_organs]
                                            organ_counts = [item[1] for item in sorted_organs]
                                        
                                        bars = ax.barh(organ_names, organ_counts, color='skyblue')
                                        ax.set_xlabel('Number of Spots')
                                        ax.set_title('Spot Distribution by Organ System')
                                        
                                        # Add count labels on bars
                                        for bar in bars:
                                            width = bar.get_width()
                                            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2., 
                                                   f'{int(width)}', ha='left', va='center')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Display organ-specific recommendations
                                        st.markdown("#### Organ-Specific Observations")
                                        
                                        for organ, count in sorted(organs.items(), key=lambda x: x[1], reverse=True)[:5]:
                                            if count > 1:
                                                st.markdown(f"**{organ}**: {count} spots detected - Consider supporting this system")
                                    elif organs:
                                        # Fallback display without matplotlib
                                        st.markdown("#### Organ Distribution")
                                        for organ, count in sorted(organs.items(), key=lambda x: x[1], reverse=True):
                                            st.markdown(f"**{organ}**: {count} spots")
                                    else:
                                        st.info("No organ mapping data available.")
                                
                                with enhanced_subtabs[3]:
                                    st.subheader("Health Assessment & Recommendations")
                                    
                                    analysis_summary = enhanced_results.get('analysis_summary', {})
                                    health_indicators = analysis_summary.get('health_indicators', {})
                                    
                                    if health_indicators:
                                        # Overall assessment
                                        if 'overall' in health_indicators:
                                            st.markdown("### Overall Assessment")
                                            assessment = health_indicators['overall']
                                            
                                            if "good" in assessment.lower() or "minimal" in assessment.lower():
                                                st.success(f"✅ {assessment}")
                                            elif "moderate" in assessment.lower():
                                                st.warning(f"⚠️ {assessment}")
                                            else:
                                                st.error(f"🔴 {assessment}")
                                        
                                        # Zone-specific assessments
                                        st.markdown("### Zone-Specific Analysis")
                                        
                                        zones = analysis_summary.get('zones', {})
                                        total_spots = analysis_summary.get('total_spots', 0)
                                        
                                        for zone, count in zones.items():
                                            percentage = (count / total_spots * 100) if total_spots > 0 else 0
                                            
                                            col1, col2 = st.columns([3, 1])
                                            with col1:
                                                st.markdown(f"**{zone}**: {count} spots ({percentage:.1f}%)")
                                            with col2:
                                                st.progress(percentage / 100)
                                        
                                        # Specific recommendations
                                        st.markdown("### Recommendations")
                                        
                                        for key, indicator in health_indicators.items():
                                            if key != 'overall' and indicator:
                                                st.markdown(f"• {indicator}")
                                        
                                        # General Ayurvedic recommendations
                                        st.markdown("### General Ayurvedic Guidelines")
                                        st.markdown("""
                                        - **Detoxification**: Consider gentle cleansing practices (Panchakarma)
                                        - **Diet**: Focus on fresh, whole foods and proper food combining
                                        - **Lifestyle**: Maintain regular sleep patterns and stress management
                                        - **Herbs**: Consult with an Ayurvedic practitioner for personalized herbal support
                                        - **Follow-up**: Regular iris analysis can track improvements over time
                                        """)
                                    else:
                                        st.info("No specific health indicators generated. This may indicate minimal tissue stress.")
                            else:
                                st.info("👆 Click 'Re-analyze with Enhanced Detection' above to perform detailed spot analysis.")
                    
                    # Color Analysis tab
                    with advanced_tabs[2]:
                        st.subheader("Iris Color Analysis")
                        
                        features = analysis_results.get("features", {})
                        colors = features.get("color_summary", [])
                        
                        if colors:
                            # Create a visual representation of the color palette
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                # Show color distribution
                                st.markdown("#### Color Distribution")
                                
                                # Create a visualization of the color percentages
                                fig, ax = plt.subplots(figsize=(8, 5))
                                
                                color_rgb = []
                                color_names = []
                                color_values = []
                                
                                for i, color_info in enumerate(colors):
                                    # Extract RGB values and normalize to 0-1 for matplotlib
                                    rgb_str = color_info["color"].replace("rgb(", "").replace(")", "")
                                    r, g, b = map(int, rgb_str.split(", "))
                                    rgb = (r/255, g/255, b/255)
                                    color_rgb.append(rgb)
                                    
                                    # Create labels and values
                                    color_names.append(f"Color {i+1}")
                                    color_values.append(color_info["percentage"])
                                
                                # Create bar chart
                                bars = ax.bar(color_names, color_values, color=color_rgb)
                                
                                # Add percentage labels on top of bars
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                            f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
                                
                                ax.set_ylim(0, max(color_values) * 1.2)  # Add some space for labels
                                ax.set_ylabel('Percentage')
                                ax.set_title('Iris Color Distribution')
                                
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown("#### Color Palette")
                                
                                # Create color palette with rectangles
                                for i, color_info in enumerate(colors):
                                    rgb_str = color_info["color"]
                                    percentage = color_info["percentage"]
                                    
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: {rgb_str}; 
                                            width: 100%; 
                                            height: 40px; 
                                            margin-bottom: 10px;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            color: {'#fff' if sum(map(int, rgb_str.replace("rgb(", "").replace(")", "").split(", "))) < 380 else '#000'};
                                            font-weight: bold;
                                            border-radius: 5px;
                                        ">
                                            {percentage:.1%}
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                
                                # Add ayurvedic color interpretation
                                st.markdown("#### Ayurvedic Color Interpretation")
                                st.markdown("""
                                In Ayurvedic iridology, different iris colors are associated with different constitutions:
                                
                                - **Brown/Dark Tones**: Primarily associated with Pitta dosha (fire element)
                                - **Blue/Light Tones**: Associated with Vata dosha (air and ether elements)
                                - **Green/Mixed Tones**: Often indicates a Kapha-Pitta combination
                                - **Gray/Pale Tones**: Strong association with Kapha dosha (water and earth elements)
                                """)
                        else:
                            st.warning("No color analysis data available")
                    
                    # Spot Detection tab
                    with advanced_tabs[3]:
                        st.subheader("Iris Spot Analysis")
                        
                        # Get spot data
                        features = analysis_results.get("features", {})
                        spot_count = features.get("spot_count", 0)
                        
                        # Display spot visualization if available
                        if "image_paths" in analysis_results and "zones" in analysis_results["image_paths"]:
                            st.image("data:image/png;base64," + analysis_results["image_paths"]["zones"], use_column_width=True)
                            st.caption("Iris zones with spot detection overlay")
                        
                        # Display spot count and interpretation
                        st.markdown(f"#### Total Spots Detected: {spot_count}")
                        
                        # Provide interpretation
                        if spot_count > 0:
                            spot_interpretation = ""
                            if spot_count < 5:
                                spot_interpretation = "Low spot count - typically indicates a cleaner system with minimal toxin accumulation."
                            elif spot_count < 10:
                                spot_interpretation = "Moderate spot count - may indicate some toxin accumulation in tissues."
                            else:
                                spot_interpretation = "High spot count - could indicate significant toxin accumulation and potential inflammatory processes."
                            
                            st.markdown(f"**Interpretation:** {spot_interpretation}")
                            
                            # Add Ayurvedic context
                            st.markdown("""
                            #### Ayurvedic Significance of Iris Spots
                            
                            In Ayurvedic iridology, spots (known as "ama" indicators) represent:
                            
                            - **Toxin accumulation** in corresponding body tissues
                            - Potential **weak spots** in the system
                            - Areas that may need **detoxification**
                            - Possible **digestive system imbalances**
                            
                            The location of spots corresponds to specific organs and systems according to the iris chart.
                            """)
                        else:
                            st.success("No significant spots detected in the iris - this generally indicates good elimination of toxins from the system.")
                    
                    # Texture Analysis tab
                    with advanced_tabs[4]:
                        st.subheader("Iris Texture Analysis")
                        
                        # Get texture data
                        features = analysis_results.get("features", {})
                        texture_stats = features.get("texture_stats", {})
                        
                        if texture_stats:
                            # Display texture statistics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Texture Metrics")
                                
                                # Create a metrics display for texture parameters
                                metrics = [
                                    ("Contrast", texture_stats.get("contrast", 0), "Higher values indicate more variation in texture"),
                                    ("Uniformity", texture_stats.get("uniformity", 0), "Higher values indicate more uniform texture"),
                                    ("Homogeneity", texture_stats.get("homogeneity", 0), "Higher values indicate more homogeneous texture"),
                                    ("Energy", texture_stats.get("energy", 0), "Higher values indicate more regular patterns"),
                                    ("Entropy", texture_stats.get("entropy", 0), "Higher values indicate more randomness"),
                                ]
                                
                                for name, value, description in metrics:
                                    # Calculate normalized value for progress bar (0-1)
                                    normalized = min(1.0, max(0.0, value / 10.0 if name == "Contrast" or name == "Entropy" else value))
                                    
                                    st.markdown(f"**{name}:** {value:.4f}")
                                    st.progress(normalized)
                                    st.caption(description)
                                    st.markdown("---")
                            
                            with col2:
                                st.markdown("#### Ayurvedic Interpretation")
                                
                                # Interpret texture based on metrics
                                contrast = texture_stats.get("contrast", 0)
                                uniformity = texture_stats.get("uniformity", 0)
                                entropy = texture_stats.get("entropy", 0)
                                
                                if contrast > 5 and entropy > 4:
                                    dosha = "Vata"
                                    interpretation = "High contrast and high entropy patterns suggest Vata dominance, indicating more variable and changeable bodily processes."
                                elif contrast > 3 and uniformity < 0.3:
                                    dosha = "Pitta"
                                    interpretation = "Moderate contrast with moderate texture organization suggests Pitta dominance, indicating transformative and metabolic processes."
                                elif uniformity > 0.4:
                                    dosha = "Kapha"
                                    interpretation = "High uniformity with lower entropy suggests Kapha dominance, indicating stable and consistent bodily processes."
                                else:
                                    dosha = "Mixed"
                                    interpretation = "The texture patterns suggest a more balanced constitution without strong dominance of any single dosha."
                                
                                st.markdown(f"**Texture-Based Dosha Assessment:** {dosha}")
                                st.markdown(f"**Interpretation:** {interpretation}")
                                
                                # Add general context
                                st.markdown("""
                                #### Texture and Constitution
                                
                                In Ayurvedic iridology, iris texture relates to foundational aspects of constitution:
                                
                                - **Dense, tightly-packed fibers**: Kapha dominance (earth/water elements)
                                - **Moderate density with radial patterns**: Pitta dominance (fire element)
                                - **Looser, more variable patterns**: Vata dominance (air/ether elements)
                                
                                Texture analysis provides insights into the inherent nature of one's body-mind type.
                                """)
                        else:
                            st.warning("No texture analysis data available")
                    
                    # Pattern Matching tab
                    with advanced_tabs[5]:
                        st.subheader("Iris Pattern Matching")
                        
                        # Create tabs for user patterns and research datasets
                        pattern_tabs = st.tabs(["User Patterns", "Research Datasets"])
                        
                        # User Patterns Tab - Show patterns from previous user uploads
                        with pattern_tabs[0]:
                            # Get similar patterns
                            similar_patterns = analysis_results.get("similar_patterns", [])
                            
                            if similar_patterns:
                                st.markdown(f"Found {len(similar_patterns)} similar iris patterns in the user database")
                                
                                # Display similar patterns
                                for i, pattern in enumerate(similar_patterns):
                                    similarity = pattern.get("similarity", 0)
                                    metadata = pattern.get("metadata", {})
                                    
                                    with st.expander(f"Similar Pattern #{i+1} - Similarity: {similarity:.2f}", expanded=i==0):
                                        # Display metadata
                                        st.markdown(f"**Timestamp:** {metadata.get('timestamp', 'Unknown')}")
                                        st.markdown(f"**Source:** {metadata.get('filename', 'Unknown')}")
                            else:
                                st.info("No similar patterns found in user database. Upload more iris images to build your pattern database.")
                        
                        # Research Datasets Tab - Show patterns from research datasets
                        with pattern_tabs[1]:
                            st.markdown("### Research Dataset Matches")
                            
                            # Select dataset
                            dataset_options = {
                                "casia_thousand": "CASIA-Iris-Thousand (1,000 subjects, 20,000 images)",
                                "nd_iris_0405": "ND-IRIS-0405 (356 subjects, 64,980 images)",
                                "ubiris_v2": "UBIRIS.v2 (261 subjects, non-ideal conditions)"
                            }
                            
                            selected_dataset = st.selectbox(
                                "Select research dataset for comparison",
                                options=list(dataset_options.keys()),
                                format_func=lambda x: dataset_options[x]
                            )
                            
                            # Initialize dataset matcher
                            if st.button("Find Matches in Research Dataset"):
                                with st.spinner(f"Comparing with {dataset_options[selected_dataset]}..."):
                                    try:
                                        # Get the path of the uploaded image
                                        if not 'temp_advanced_path' in locals() and not 'temp_advanced_path' in globals():
                                            st.error("Error: No image has been uploaded for analysis.")
                                            st.info("Please upload an iris image before matching with research datasets.")
                                            raise Exception("No image uploaded")
                                            
                                        tmp_img_path = temp_advanced_path
                                        
                                        # Verify that the file exists and is accessible
                                        if not os.path.exists(tmp_img_path):
                                            st.error("Error: The uploaded iris image is not accessible.")
                                            st.info("Please try uploading the image again.")
                                            raise Exception("Image not accessible")
                                            
                                        # Check if dataset exists
                                        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                                  "datasets", selected_dataset)
                                        metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                                   "datasets", "dataset_metadata.csv")
                                        
                                        if not os.path.exists(dataset_path) or not os.listdir(dataset_path) or not os.path.exists(metadata_path):
                                            st.warning(f"""
                                            Dataset {dataset_options[selected_dataset]} is not set up yet.
                                            
                                            To use this feature:
                                            1. Obtain the dataset from the official source
                                            2. Run the setup script: `python dataset_downloader.py --dataset {selected_dataset}`
                                            
                                            For testing, you can create a simulated dataset:
                                            `python dataset_downloader.py --simulate --dataset {selected_dataset} --samples 20`
                                            """)
                                        else:
                                            # Try to match with dataset
                                            try:
                                                dataset_matcher = DatasetPatternMatcher(dataset_key=selected_dataset)
                                                dataset_matches = dataset_matcher.match_iris_with_dataset(tmp_img_path)
                                                
                                                if "error" in dataset_matches:
                                                    st.error(f"Error matching with dataset: {dataset_matches['error']}")
                                                
                                                # Suggest dataset setup if it's missing
                                                if "not found" in str(dataset_matches['error']) or "empty" in str(dataset_matches['error']):
                                                    st.info("""
                                                    ### Dataset Setup Instructions
                                                    
                                                    To use research datasets for iris pattern matching:
                                                    
                                                    1. Run the dataset downloader with the simulate option for testing:
                                                       ```
                                                       python dataset_downloader.py --simulate --dataset casia_thousand --samples 20
                                                       ```
                                                       
                                                    2. For actual research datasets, follow the setup instructions in ENHANCED_README.md
                                                    """)
                                            except Exception as e:
                                                st.error(f"Error accessing dataset: {str(e)}")
                                                
                                                # Display helpful setup instructions
                                                st.info("""
                                                ### Dataset Setup Required
                                                
                                                The research dataset functionality requires additional setup:
                                                
                                                1. Make sure the Qdrant server is running (check docker-compose up -d)
                                                2. Run the dataset simulation script for testing:
                                                   ```
                                                   python dataset_downloader.py --simulate --dataset casia_thousand
                                                   ```
                                                """)
                                            else:
                                                # Get statistics about the dataset
                                                stats = dataset_matcher.get_dataset_statistics()
                                                
                                                # Display stats
                                                if "error" not in stats:
                                                    st.markdown(f"**Dataset:** {stats['dataset_name']}")
                                                    st.markdown(f"**Total subjects:** {stats['subjects']}")
                                                    st.markdown(f"**Total patterns:** {stats['total_patterns']}")
                                                
                                                # Display matches
                                                matches = dataset_matches.get("matches", [])
                                                if matches:
                                                    st.success(f"Found {len(matches)} similar patterns in the research dataset")
                                                    
                                                    # Create columns for matches
                                                    for i, match in enumerate(matches):
                                                        score = match["score"]
                                                        # Higher score means lower similarity in cosine distance
                                                        similarity = max(0, 1.0 - score)
                                                        
                                                        with st.expander(f"Research Match #{i+1} - Similarity: {similarity:.2f}", expanded=i==0):
                                                            cols = st.columns(2)
                                                            with cols[0]:
                                                                st.markdown(f"**Subject ID:** {match['subject_id']}")
                                                                st.markdown(f"**Eye:** {match['eye']}")
                                                                st.markdown(f"**Dataset:** {match['dataset']}")
                                                            
                                                            with cols[1]:
                                                                # Extract features for display
                                                                features = match.get('features', {})
                                                                if features:
                                                                    st.markdown("#### Feature Comparison")
                                                                    if "main_color" in features and features["main_color"]:
                                                                        color = features["main_color"]
                                                                        st.markdown(f"**Main Color:** RGB({color[0]}, {color[1]}, {color[2]})")
                                                                    
                                                                    if "num_spots" in features:
                                                                        st.markdown(f"**Spot Count:** {features['num_spots']}")
                                                                        
                                                                    if "contrast" in features:
                                                                        st.markdown(f"**Contrast:** {features['contrast']:.2f}")
                                                            
                                                if not matches:
                                                    st.warning("No matches found in the research dataset. This could be because the dataset is not yet set up or the pattern is unique.")
                                    
                                    except Exception as e:
                                        st.error(f"Error during dataset matching: {str(e)}")
                            
                            # Show info about datasets
                            st.markdown("### About Research Datasets")
                            st.markdown("""
                            The research datasets need to be separately downloaded and prepared due to their large size and licensing requirements.
                            
                            To set up the datasets for pattern matching:
                            1. Obtain the dataset from the official source
                            2. Run the dataset setup script: `python iris_dataset_matcher.py --setup --dataset [dataset_name]`
                            3. Once set up, you can match patterns against the research dataset
                            """)
                            
                            st.markdown("**Note:** First-time setup of research datasets may take several hours depending on the dataset size.")
                        
                        if similar_patterns:
                            # Show feature comparison if available
                            for i, pattern in enumerate(similar_patterns):
                                if "feature_comparison" in pattern:
                                    feature_comp = pattern["feature_comparison"]
                                    
                                    st.markdown("#### Feature Comparison")
                                    
                                    # Create a comparison table
                                    data = []
                                    for feature_name, values in feature_comp.items():
                                        this_value = values.get("current", "N/A")
                                        other_value = values.get("matched", "N/A")
                                        data.append([feature_name, this_value, other_value])
                                    
                                    df = pd.DataFrame(data, columns=["Feature", "Current Iris", "Matched Iris"])
                                    st.table(df)
                        else:
                            st.info("No similar patterns found in the database. This might be the first iris with these characteristics.")
                            
                            # Add explanation
                            st.markdown("""
                            As more iris images are analyzed, the system will build a database of patterns that can be used for comparison.
                            This is useful for tracking changes in your own iris over time, or comparing with similar constitutions.
                            """)
                    
                    # Health Insights tab
                    with advanced_tabs[6]:
                        st.subheader("Comprehensive Health Insights")
                        
                        # Generate health insights
                        with st.spinner("Generating health insights..."):
                            insights = st.session_state.advanced_iris_analyzer.generate_health_insights(analysis_results)
                            
                            if "error" in insights:
                                st.error(insights["error"])
                            else:
                                # Display overall health assessment
                                overall_health = insights.get("overall_health", "unknown")
                                
                                # Create a styled assessment container
                                st.markdown(f"""
                                <div style="
                                    background-color: rgba(49, 51, 63, 0.1);
                                    padding: 20px;
                                    border-radius: 10px;
                                    margin-bottom: 20px;
                                    text-align: center;
                                ">
                                    <h3>Overall Assessment</h3>
                                    <h2 style="color: {'#28a745' if overall_health == 'balanced' else '#ffc107' if overall_health == 'mildly imbalanced' else '#dc3545'};">
                                        {overall_health.upper()}
                                    </h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display dosha balance if available
                                dosha_balance = insights.get("dosha_balance", {})
                                if dosha_balance:
                                    st.markdown("#### Dosha Distribution")
                                    
                                    # Create a two-column layout for a more compact presentation
                                    cols = st.columns([2, 3])
                                    
                                    with cols[0]:
                                        # Find primary dosha
                                        primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0]
                                        
                                        # Display primary dosha prominently 
                                        st.markdown(f"""
                                        <div style="border-radius: 5px; padding: 10px; text-align: center; 
                                                 background-color: rgba(75, 75, 75, 0.1);">
                                            <h4>Primary Dosha</h4>
                                            <h3 style="color: {'#9b59b6' if primary_dosha == 'vata' else 
                                                          '#e74c3c' if primary_dosha == 'pitta' else '#27ae60'};">
                                                {primary_dosha.upper()}
                                            </h3>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Display percentages in a simple list
                                        st.markdown("##### Constitution")
                                        for dosha, value in dosha_balance.items():
                                            st.markdown(f"**{dosha.capitalize()}:** {value:.1%}")
                                    
                                    with cols[1]:
                                        # Prepare data for chart
                                        dosha_labels = list(dosha_balance.keys())
                                        dosha_values = list(dosha_balance.values())
                                        
                                        # Set colors for doshas
                                        dosha_colors = ['#a29bfe', '#ff7675', '#55efc4']
                                        
                                        # Create smaller pie chart
                                        fig, ax = plt.subplots(figsize=(3.5, 3))
                                        wedges, texts, autotexts = ax.pie(
                                            dosha_values, 
                                            labels=[f"{d.capitalize()}" for d in dosha_labels], 
                                            autopct='%1.1f%%', 
                                            startangle=90, 
                                            colors=dosha_colors,
                                            textprops={'fontsize': 9}
                                        )
                                        
                                        # Style the chart - smaller font for better fit in reports
                                        for text in autotexts:
                                            text.set_fontsize(8)
                                            
                                        ax.axis('equal')
                                        plt.tight_layout()
                                        
                                        # Display the smaller chart
                                        st.pyplot(fig)
                                
                                # Display key findings
                                key_findings = insights.get("key_findings", [])
                                
                                if key_findings:
                                    st.markdown("#### Key Findings")
                                    
                                    for i, finding in enumerate(key_findings):
                                        if "zone" in finding:
                                            # Zone-based finding
                                            zone = finding.get("zone", "Unknown zone")
                                            condition = finding.get("condition", "unknown")
                                            confidence = finding.get("confidence", 0)
                                            suggestion = finding.get("suggestion", "")
                                            
                                            # Determine color based on condition
                                            color = "#28a745" if condition == "normal" else "#ffc107" if condition == "stressed" else "#dc3545"
                                            
                                            st.markdown(f"""
                                            <div style="
                                                border-left: 5px solid {color};
                                                padding: 10px 15px;
                                                margin-bottom: 10px;
                                                background-color: rgba(49, 51, 63, 0.05);
                                                border-radius: 0 5px 5px 0;
                                            ">
                                                <strong>{zone}:</strong> {condition.capitalize()} (Confidence: {confidence:.0%})
                                                <br/><em>{suggestion}</em>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        elif "type" in finding:
                                            # Feature-based finding
                                            finding_text = finding.get("finding", "")
                                            suggestion = finding.get("suggestion", "")
                                            
                                            st.markdown(f"""
                                            <div style="
                                                border-left: 5px solid #17a2b8;
                                                padding: 10px 15px;
                                                margin-bottom: 10px;
                                                background-color: rgba(49, 51, 63, 0.05);
                                                border-radius: 0 5px 5px 0;
                                            ">
                                                <strong>{finding_text}</strong>
                                                <br/><em>{suggestion}</em>
                                            </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.success("No significant issues detected in the iris analysis.")
                                
                                # Add customized recommendations based on the analysis
                                st.markdown("#### Ayurvedic Recommendations")
                                
                                # Determine primary dosha if available
                                primary_dosha = "balanced"
                                if dosha_balance:
                                    primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0]
                                
                                # Create recommendations based on dosha
                                if primary_dosha == "vata":
                                    st.markdown("""
                                    **For Vata Constitution:**
                                    
                                    - Maintain regular routines for meals, sleep, and activities
                                    - Favor warm, cooked, slightly oily foods
                                    - Include sweet, sour, and salty tastes
                                    - Practice gentle, grounding yoga
                                    - Use warm sesame oil for self-massage (abhyanga)
                                    - Protect from cold, dry, and windy conditions
                                    """)
                                elif primary_dosha == "pitta":
                                    st.markdown("""
                                    **For Pitta Constitution:**
                                    
                                    - Avoid excessive heat, sun exposure, and hot foods
                                    - Favor cooling foods like sweet fruits, vegetables, and grains
                                    - Include sweet, bitter, and astringent tastes
                                    - Practice moderate, cooling exercise like swimming
                                    - Use cooling coconut or sunflower oil for self-massage
                                    - Maintain emotional balance and avoid intense anger or frustration
                                    """)
                                elif primary_dosha == "kapha":
                                    st.markdown("""
                                    **For Kapha Constitution:**
                                    
                                    - Maintain an active lifestyle with regular exercise
                                    - Favor light, warm, and dry foods
                                    - Include pungent, bitter, and astringent tastes
                                    - Practice stimulating, warming yoga
                                    - Use stimulating oils like mustard or eucalyptus for massage
                                    - Seek variety and stimulation to avoid stagnation
                                    """)
                                else:
                                    st.markdown("""
                                    **For Balanced Constitution:**
                                    
                                    - Maintain a seasonal routine adjusted to your local climate
                                    - Include all six tastes (sweet, sour, salty, pungent, bitter, astringent)
                                    - Practice balanced exercise combining strength and flexibility
                                    - Use appropriate oils and practices based on seasonal needs
                                    - Regular detoxification practices like seasonal cleanses
                                    """)
                    
                    # Add button to generate comprehensive report
                    st.markdown("---")
                    st.subheader("📄 Generate Comprehensive Report")
                    
                    # Report format selection
                    col1, col2 = st.columns(2)
                    with col1:
                        report_format = st.selectbox(
                            "Report Format",
                            ["PDF", "HTML"],
                            help="Choose the format for your comprehensive iris analysis report"
                        )
                    
                    with col2:
                        include_enhanced = st.checkbox(
                            "Include Enhanced Spot Analysis",
                            value=hasattr(st.session_state, 'enhanced_analysis_results'),
                            help="Include detailed spot detection results in the report",
                            disabled=not hasattr(st.session_state, 'enhanced_analysis_results')
                        )
                    
                    # User information for report
                    with st.expander("Optional: Add Personal Information to Report"):
                        user_name = st.text_input("Name", key="enhanced_report_name")
                        user_age = st.number_input("Age", min_value=1, max_value=120, value=30, key="enhanced_report_age")
                        user_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="enhanced_report_gender")
                        user_email = st.text_input("Email", key="enhanced_report_email")
                        user_concerns = st.text_area("Health Concerns or Goals", key="enhanced_report_concerns")
                    
                    # Collect user info
                    user_info = {}
                    if user_name:
                        user_info["Name"] = user_name
                    if user_age:
                        user_info["Age"] = str(user_age)
                    if user_gender:
                        user_info["Gender"] = user_gender
                    if user_email:
                        user_info["Email"] = user_email
                    if user_concerns:
                        user_info["Health Concerns"] = user_concerns
                    
                    if st.button("🎯 Generate Comprehensive Iris Analysis Report", use_container_width=True):
                        with st.spinner("Generating comprehensive iris analysis report..."):
                            try:
                                if "zone_analysis" in analysis_results:
                                    # Prepare zone results data
                                    zone_results = analysis_results["zone_analysis"].copy()
                                    
                                    # Add original image to zone results
                                    if uploaded_advanced_image is not None:
                                        # Convert uploaded file to PIL Image
                                        from PIL import Image
                                        original_img = Image.open(uploaded_advanced_image)
                                        zone_results["original_image"] = original_img
                                    
                                    # Get enhanced results if available and requested
                                    enhanced_results = None
                                    if include_enhanced and hasattr(st.session_state, 'enhanced_analysis_results'):
                                        enhanced_results = st.session_state.enhanced_analysis_results
                                    
                                    # Generate report using enhanced generator if available
                                    if ENHANCED_REPORT_GENERATOR_AVAILABLE and hasattr(st.session_state, 'enhanced_iris_report_generator'):
                                        if report_format == "PDF":
                                            report_bytes = st.session_state.enhanced_iris_report_generator.generate_enhanced_pdf_report(
                                                zone_results,
                                                enhanced_results,
                                                user_info if user_info else None
                                            )
                                            
                                            if report_bytes:
                                                st.success("✅ Enhanced PDF report generated successfully!")
                                                
                                                # Generate filename with timestamp
                                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                filename = f"enhanced_iris_analysis_report_{timestamp}.pdf"
                                                
                                                st.download_button(
                                                    label="📥 Download Enhanced Iris Analysis Report (PDF)",
                                                    data=report_bytes,
                                                    file_name=filename,
                                                    mime="application/pdf",
                                                    use_container_width=True
                                                )
                                                
                                                # Show report preview info
                                                st.info(f"""
                                                **Report Generated Successfully!**
                                                - Format: PDF
                                                - Enhanced Analysis: {'Yes' if enhanced_results else 'No'}
                                                - Total Spots Detected: {len(enhanced_results.get('spots', [])) if enhanced_results else 'N/A'}
                                                - Personal Info: {'Included' if user_info else 'Not included'}
                                                """)
                                            else:
                                                st.error("Failed to generate PDF report. Please try again.")
                                                
                                        else:  # HTML format
                                            html_report = st.session_state.enhanced_iris_report_generator.generate_enhanced_html_report(
                                                zone_results,
                                                enhanced_results,
                                                user_info if user_info else None
                                            )
                                            
                                            st.success("✅ Enhanced HTML report generated successfully!")
                                            
                                            # Show HTML report in expandable section
                                            with st.expander("📋 View HTML Report", expanded=True):
                                                st.components.v1.html(html_report, height=800, scrolling=True)
                                            
                                            # Provide download option for HTML
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            filename = f"enhanced_iris_analysis_report_{timestamp}.html"
                                            
                                            st.download_button(
                                                label="📥 Download Enhanced Iris Analysis Report (HTML)",
                                                data=html_report.encode('utf-8'),
                                                file_name=filename,
                                                mime="text/html",
                                                use_container_width=True
                                            )
                                    
                                    else:
                                        # Fallback to original report generator
                                        st.warning("Enhanced report generator not available. Using standard report generator.")
                                        
                                        if report_format == "PDF":
                                            pdf_report = st.session_state.iris_report_generator.generate_report(
                                                zone_results,
                                                user_info if user_info else None
                                            )
                                            
                                            if pdf_report:
                                                st.success("✅ Standard PDF report generated successfully!")
                                                
                                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                filename = f"iris_analysis_report_{timestamp}.pdf"
                                                
                                                st.download_button(
                                                    label="📥 Download Iris Analysis Report (PDF)",
                                                    data=pdf_report,
                                                    file_name=filename,
                                                    mime="application/pdf",
                                                    use_container_width=True
                                                )
                                            else:
                                                st.error("Failed to generate PDF report.")
                                        else:
                                            # HTML report
                                            html_report = st.session_state.iris_report_generator.generate_html_report(
                                                zone_results,
                                                user_info if user_info else None
                                            )
                                            
                                            st.success("✅ Standard HTML report generated successfully!")
                                            
                                            with st.expander("📋 View HTML Report", expanded=True):
                                                st.components.v1.html(html_report, height=600, scrolling=True)
                                            
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            filename = f"iris_analysis_report_{timestamp}.html"
                                            
                                            st.download_button(
                                                label="📥 Download Iris Analysis Report (HTML)",
                                                data=html_report.encode('utf-8'),
                                                file_name=filename,
                                                mime="text/html",
                                                use_container_width=True
                                            )
                                
                                else:
                                    st.error("No analysis results available. Please perform iris analysis first.")
                                    
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")
                                st.info("Please try again or contact support if the issue persists.")
                                # Show debug info in development
                                if st.checkbox("Show Debug Info"):
                                    st.exception(e)
        
        except Exception as e:
            st.error(f"Error in advanced iris analysis: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_advanced_path)
            except:
                pass
                
            # Health Queries tab
            with advanced_tabs[7]:
                st.subheader("Personalized Health Queries")
                
                st.markdown("""
                <div style="
                    background-color: rgba(49, 51, 63, 0.1);
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                ">
                    <p>Based on your iris analysis, here are personalized questions to help you learn more about potential health implications. 
                    Click on any question to get an informative response from our knowledge base.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate queries based on iris features
                iris_queries = st.session_state.query_generator.generate_query_from_iris_features(analysis_results.get("features", {}))
                
                # Add container for query results
                query_result_container = st.container()
                
                # Set default state for advanced query handling
                if "advanced_query_clicked" not in st.session_state:
                    st.session_state.advanced_query_clicked = False
                    st.session_state.current_advanced_query = ""
                
                # Display queries in a nice layout with cards
                for i, query in enumerate(iris_queries):
                    card_color = "#f0f2f6"
                    border_color = "#e0e2e6" 
                    
                    if st.session_state.current_advanced_query == query:
                        card_color = "rgba(151, 187, 205, 0.2)"
                        border_color = "#97bbcd"
                        
                    # Create a styled card for each query
                    st.markdown(f"""
                    <div style="
                        background-color: {card_color};
                        border: 1px solid {border_color};
                        border-radius: 8px;
                        padding: 10px 15px;
                        margin-bottom: 10px;
                        cursor: pointer;
                    ">
                        <p style="margin: 0; font-size: 16px;">🔍 {query}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a button with the same query text (hidden visually but functional)
                    if st.button(f"Query: {query}", key=f"adv_query_{i}", help="Click to answer this query"):
                        st.session_state.advanced_query_clicked = True
                        st.session_state.current_advanced_query = query
                
                # If a query is clicked, show the results
                if st.session_state.advanced_query_clicked:
                    query = st.session_state.current_advanced_query
                    
                    with query_result_container:
                        st.markdown("---")
                        st.subheader(f"Results for: '{query}'")
                        
                        with st.spinner("Searching for the best answer..."):
                            try:
                                # Use the best search method based on availability
                                if st.session_state.is_enhanced_initialized:
                                    # Use multi-query search for better results
                                    results = st.session_state.enhanced_qdrant_client.multi_query_search(
                                        query, 
                                        limit=5
                                    )
                                    
                                    # Generate a comprehensive answer
                                    answer_data = st.session_state.answer_generator.generate_answer(
                                        query, 
                                        results
                                    )
                                    
                                    # Display the answer first for better UX
                                    if answer_data and answer_data["confidence"] > 0:
                                        st.markdown("### Answer")
                                        st.markdown(answer_data["answer"])
                                        
                                        # Show confidence with colored indicator
                                        conf = answer_data["confidence"]
                                        conf_color = get_score_color(conf)
                                        st.markdown(f"""
                                            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                                <span style="margin-right: 10px;">Confidence:</span>
                                                <div style="background-color: #f0f2f6; flex-grow: 1; height: 10px; border-radius: 5px; overflow: hidden;">
                                                    <div style="background-color: {conf_color}; width: {conf*100}%; height: 100%;"></div>
                                                </div>
                                                <span style="margin-left: 10px; font-weight: bold;">{conf:.2f}</span>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                        # Show sources
                                        if answer_data.get("sources"):
                                            with st.expander("View Sources"):
                                                for src in answer_data["sources"]:
                                                    st.markdown(f"- {src.get('title', 'Unknown source')}, Page: {src.get('page', 'N/A')}")
                                    
                                    # Show supporting results
                                    with st.expander("View Search Results", expanded=not answer_data or answer_data["confidence"] < 0.7):
                                        show_enhanced_results(results)
                                elif st.session_state.is_initialized:
                                    # Fallback to standard search
                                    results = st.session_state.qdrant_client.search(query, limit=5)
                                    show_results(results, standard=True)
                                else:
                                    st.warning("Knowledge base is not initialized. Please upload PDF documents first to enable querying.")
                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")
                                st.info("Try another query or check if the knowledge base is properly initialized.")
                else:
                    with query_result_container:
                        st.info("👆 Click on any question above to see the answer from our knowledge base.")
                        
                        # Show sample query topic buttons for exploration
                        st.markdown("### Explore Topics")
                        st.markdown("Or explore these general topics related to iridology:")
                        
                        # Create columns for topic buttons
                        topic_cols = st.columns(3)
                        
                        topics = [
                            "Iris Zones", "Constitutional Types", 
                            "Digestive System", "Nervous System", 
                            "Detoxification", "Ayurvedic Connection"
                        ]
                        
                        for i, topic in enumerate(topics):
                            with topic_cols[i % 3]:
                                if st.button(topic, key=f"topic_{i}"):
                                    # Generate a query about this topic
                                    topic_query = f"What does iridology tell us about {topic.lower()}?"
                                    st.session_state.advanced_query_clicked = True
                                    st.session_state.current_advanced_query = topic_query
                                    st.rerun()
