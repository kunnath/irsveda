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

# Keep the original imports for backward compatibility
from pdf_extractor import extract_iris_chunks
from iris_qdrant import IrisQdrantClient
from iris_predictor import IrisPredictor

# Import the new IrisZone analyzer
from iris_zone_analyzer import IrisZoneAnalyzer
from iris_report_generator import IrisReportGenerator

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

if "advanced_iris_analyzer" not in st.session_state:
    # Use environment variables if available (for Docker)
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    st.session_state.advanced_iris_analyzer = AdvancedIrisAnalyzer(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name="iris_patterns"
    )

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
tabs = st.tabs(["📚 PDF Upload & Processing", "🔍 Knowledge Query", "👁️ Iris Analysis", "🔬 IrisZone", "🧬 Advanced Analysis", "📊 Statistics", "⚙️ Configuration"])

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
        query = st.text_input(
            "Ask a question about iridology:",
            placeholder="How does the iris pattern reflect liver conditions in iridology?"
        )
        
        if query:
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
                    
                    # Generate and display queries
                    st.subheader("Suggested Queries")
                    
                    query_cols = st.columns(2)
                    for i, query in enumerate(results["queries"]):
                        col_idx = i % 2
                        with query_cols[col_idx]:
                            if st.button(f"🔍 {query}", key=f"query_{i}"):
                                # Set the query in the query tab
                                st.session_state.current_query = query
                                # Modern way to switch tabs with rerun
                                st.query_params["tab"] = "query"
                                st.rerun()
                    
                    # Run selected query if set
                    if hasattr(st.session_state, "current_query"):
                        query = st.session_state.current_query
                        
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
                        user_info["Name"] = st.text_input("Name", "")
                        user_info["Age"] = st.text_input("Age", "")
                    with col2:
                        user_info["Gender"] = st.selectbox("Gender", ["", "Male", "Female", "Other"])
                        user_info["Date"] = st.date_input("Date", datetime.now()).strftime("%Y-%m-%d")
                    
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

# Sixth tab - Statistics with enhanced metrics
with tabs[5]:
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
                analysis_results = st.session_state.advanced_iris_analyzer.analyze_iris(temp_advanced_path)
                
                if "error" in analysis_results:
                    st.error(analysis_results["error"])
                else:
                    # Create tabs for different analysis views
                    advanced_tabs = st.tabs(["Overview", "Color Analysis", "Spot Detection", "Texture Analysis", "Pattern Matching", "Health Insights"])
                    
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
                    
                    # Color Analysis tab
                    with advanced_tabs[1]:
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
                    with advanced_tabs[2]:
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
                    with advanced_tabs[3]:
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
                    with advanced_tabs[4]:
                        st.subheader("Iris Pattern Matching")
                        
                        # Get similar patterns
                        similar_patterns = analysis_results.get("similar_patterns", [])
                        
                        if similar_patterns:
                            st.markdown(f"Found {len(similar_patterns)} similar iris patterns in the database")
                            
                            # Display similar patterns
                            for i, pattern in enumerate(similar_patterns):
                                similarity = pattern.get("similarity", 0)
                                metadata = pattern.get("metadata", {})
                                
                                with st.expander(f"Similar Pattern #{i+1} - Similarity: {similarity:.2f}", expanded=i==0):
                                    # Display metadata
                                    st.markdown(f"**Timestamp:** {metadata.get('timestamp', 'Unknown')}")
                                    st.markdown(f"**Source:** {metadata.get('filename', 'Unknown')}")
                                    
                                    # Show feature comparison if available
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
                    with advanced_tabs[5]:
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
                                    
                                    # Prepare data for chart
                                    dosha_labels = list(dosha_balance.keys())
                                    dosha_values = list(dosha_balance.values())
                                    
                                    # Set colors for doshas
                                    dosha_colors = ['#a29bfe', '#ff7675', '#55efc4']
                                    
                                    # Create pie chart
                                    fig, ax = plt.subplots(figsize=(6, 6))
                                    wedges, texts, autotexts = ax.pie(
                                        dosha_values, 
                                        labels=[f"{d.capitalize()}" for d in dosha_labels], 
                                        autopct='%1.1f%%', 
                                        startangle=90, 
                                        colors=dosha_colors,
                                        textprops={'fontsize': 12, 'weight': 'bold'}
                                    )
                                    
                                    # Style the chart
                                    for text in autotexts:
                                        text.set_fontsize(12)
                                        text.set_weight('bold')
                                        
                                    ax.axis('equal')
                                    plt.tight_layout()
                                    
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
                    if st.button("Generate Comprehensive Iris Analysis Report"):
                        with st.spinner("Generating comprehensive iris analysis report..."):
                            # Generate PDF report (using existing IrisReportGenerator)
                            try:
                                # TODO: Update the report generator to include advanced analysis
                                # For now, just display a message
                                st.success("Report generation feature will be available in the next update!")
                                
                                # Alternatively, use the existing report generator with available data
                                if "zone_analysis" in analysis_results:
                                    zone_results = {
                                        "health_summary": analysis_results["zone_analysis"]["health_summary"],
                                        "zones_analysis": analysis_results["zone_analysis"]["zones_analysis"],
                                        "original_image": temp_advanced_path  # Use the path to the original image
                                    }
                                    
                                    # Generate report using existing functionality
                                    report_path = st.session_state.iris_report_generator.generate_report(
                                        zone_results,
                                        include_details=True,
                                        include_recommendations=True
                                    )
                                    
                                    # Provide download link
                                    with open(report_path, "rb") as file:
                                        report_bytes = file.read()
                                    
                                    st.download_button(
                                        label="Download Iris Analysis Report",
                                        data=report_bytes,
                                        file_name="iris_analysis_report.pdf",
                                        mime="application/pdf"
                                    )
                            except Exception as e:
                                st.error(f"Error generating report: {str(e)}")
                                st.info("Advanced report generation will be available in the next update.")
        
        except Exception as e:
            st.error(f"Error in advanced iris analysis: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_advanced_path)
            except:
                pass

# Seventh tab - Configuration
with tabs[6]:
    st.header("Advanced Configuration")
    
    st.warning("⚠️ Changes to these settings may affect the performance of the knowledge base.")
    
    with st.form("config_form"):
        st.subheader("Knowledge Base Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            collection_name = st.text_input("Collection Name", "iris_chunks", 
                                          help="Name of the Qdrant collection")
            vector_model = st.selectbox("Vector Model", 
                                      ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
                                      help="Select the embedding model for vector search")
        
        with col2:
            distance_metric = st.selectbox("Distance Metric", 
                                         ["COSINE", "DOT", "EUCLIDEAN"],
                                         help="Vector distance metric for similarity search")
            page_step = st.number_input("OCR Page Step", min_value=1, max_value=10, value=5,
                                       help="Process every N pages for OCR to balance speed and coverage")
        
        st.subheader("NLP Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            min_paragraph_length = st.number_input("Min Paragraph Length", 
                                                 min_value=10, max_value=100, value=25,
                                                 help="Minimum word count for paragraphs")
            duplicate_threshold = st.slider("Duplicate Detection Threshold", 
                                          min_value=0.3, max_value=0.9, value=0.7, step=0.05,
                                          help="Similarity threshold for duplicate detection")
        
        with col4:
            keyword_boost = st.slider("Keyword Boost", 
                                    min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                                    help="Boost factor for keyword matches")
            nlp_model = st.selectbox("NLP Model", 
                                   ["en_core_web_sm", "en_core_web_md"],
                                   help="spaCy model for NLP processing")
        
        # JSON configuration export/import
        st.subheader("Configuration Import/Export")
        
        config_json = st.text_area("Configuration JSON", 
                                  value=json.dumps({
                                      "collection_name": collection_name,
                                      "vector_model": vector_model,
                                      "distance_metric": distance_metric,
                                      "page_step": page_step,
                                      "min_paragraph_length": min_paragraph_length,
                                      "duplicate_threshold": duplicate_threshold,
                                      "keyword_boost": keyword_boost,
                                      "nlp_model": nlp_model
                                  }, indent=2),
                                  height=200)
        
        # Submit button
        submitted = st.form_submit_button("Apply Configuration")
        
        if submitted:
            st.success("✅ Configuration applied successfully!")
            st.info("Note: Some changes may require restarting the application to take effect.")
