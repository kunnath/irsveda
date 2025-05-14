from pathlib import Path
import tempfile
import streamlit as st
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from pdf_extractor import extract_iris_chunks
from iris_qdrant import IrisQdrantClient
from iris_predictor import IrisPredictor

# Page title and configuration
st.set_page_config(
    page_title="AyushIris - Iridology Knowledge Base",
    page_icon="👁️",
    layout="wide"
)

# Create sidebar
st.sidebar.title("AyushIris")
st.sidebar.image("static/iris_logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info(
    "This application extracts iris-related information from Ayurvedic/Iridology "
    "books and allows you to query the knowledge base."
)

# Initialize session state
if "qdrant_client" not in st.session_state:
    # Use environment variables if available (for Docker)
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", 6333))
    
    st.session_state.qdrant_client = IrisQdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="iris_chunks"
    )

if "iris_predictor" not in st.session_state:
    st.session_state.iris_predictor = IrisPredictor()

if "extracted_chunks" not in st.session_state:
    st.session_state.extracted_chunks = []

if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False

# Main page content
st.title("Ayurvedic Iridology Knowledge Base")

# Tab layout
tabs = st.tabs(["📚 PDF Upload & Processing", "🔍 Knowledge Query", "👁️ Iris Analysis", "📊 Statistics"])

# First tab - PDF Upload
with tabs[0]:
    st.header("Upload Ayurvedic/Iridology Books")
    
    uploaded_pdf = st.file_uploader(
        "Select PDF file(s)", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload Ayurvedic or Iridology books in PDF format"
    )
    
    if uploaded_pdf:
        # Process each uploaded PDF file
        for pdf_file in uploaded_pdf:
            with st.expander(f"Processing: {pdf_file.name}"):
                # Create uploads directory if it doesn't exist
                os.makedirs("uploads", exist_ok=True)
                
                # Save file to uploads directory
                temp_path = os.path.join("uploads", pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.read())
                
                # Extract iris-related chunks
                with st.spinner("Extracting iris-related information..."):
                    start_time = time.time()
                    
                    # Display OCR status message
                    ocr_status = st.empty()
                    ocr_status.info("📄 Analyzing text content...")
                    chunks = extract_iris_chunks(temp_path)
                    end_time = time.time()
                    
                    # Add OCR extraction method info if applicable
                    ocr_used = any(chunk.get("extraction_method") == "ocr" for chunk in chunks)
                    if ocr_used:
                        ocr_status.success("🔍 OCR processing completed successfully")
                    else:
                        ocr_status.empty()
                    
                    st.session_state.extracted_chunks.extend(chunks)
                    
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
                                st.markdown("---")
                    
                    # Clean up temp file
                    os.unlink(temp_path)
        
        # Store chunks in Qdrant
        if st.session_state.extracted_chunks and st.button("Store in Knowledge Base"):
            with st.spinner("Storing chunks in knowledge base..."):
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
                        f"✅ Successfully stored {len(point_ids)} chunks in the knowledge base"
                    )
                except Exception as e:
                    st.error(f"Error storing chunks: {str(e)}")

# Second tab - Knowledge Query
with tabs[1]:
    st.header("Query the Iridology Knowledge Base")
    
    if not st.session_state.is_initialized:
        st.warning("⚠️ Knowledge base is empty. Please upload and process PDFs first.")
    else:
        query = st.text_input(
            "Ask a question about iridology:",
            placeholder="How does the iris reflect liver conditions?"
        )
        
        if query:
            with st.spinner("Searching knowledge base..."):
                try:
                    # Search Qdrant
                    results = st.session_state.qdrant_client.search(query, limit=5)
                    
                    # Display results
                    st.subheader("Results:")
                    
                    if not results:
                        st.info("No matching information found. Try rephrasing your question.")
                    
                    for i, result in enumerate(results):
                        with st.container():
                            # Create expander for each result
                            with st.expander(f"Result {i+1} (Page {result['page']}) - Relevance: {result['score']:.2f}"):
                                st.markdown(result['text'])
                                st.caption(f"Source: {Path(result['source']).name}, Page: {result['page']}")
                
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
                    
                    for query in results["queries"]:
                        if st.button(f"🔍 {query}"):
                            # Set the query in the second tab
                            st.session_state.current_query = query
                            # Switch to the second tab
                            st.experimental_set_query_params(tab="query")
                            st.experimental_rerun()
                    
                    # Run selected query if set
                    if hasattr(st.session_state, "current_query") and st.session_state.is_initialized:
                        query = st.session_state.current_query
                        
                        with st.spinner(f"Searching: {query}"):
                            # Search Qdrant
                            results = st.session_state.qdrant_client.search(query, limit=3)
                            
                            # Display results
                            st.subheader("Knowledge Base Results:")
                            
                            if not results:
                                st.info("No matching information found in the knowledge base.")
                            
                            for i, result in enumerate(results):
                                with st.expander(f"Result {i+1} - Relevance: {result['score']:.2f}"):
                                    st.markdown(result['text'])
                                    st.caption(f"Source: {Path(result['source']).name}, Page: {result['page']}")
                    
        except Exception as e:
            st.error(f"Error analyzing iris image: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_path)

# Fourth tab - Statistics
with tabs[3]:
    st.header("Knowledge Base Statistics")
    
    # Display basic stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Extracted Chunks", len(st.session_state.extracted_chunks))
    
    with col2:
        st.metric("Knowledge Base Status", 
                 "Initialized" if st.session_state.is_initialized else "Not Initialized")

    # Display source breakdown
    if st.session_state.extracted_chunks:
        # Get unique sources
        sources = {}
        for chunk in st.session_state.extracted_chunks:
            source = Path(chunk.get('source', '')).name
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        # Display as a table
        st.subheader("Source Distribution")
        source_data = {"Source": list(sources.keys()), "Chunks": list(sources.values())}
        st.dataframe(source_data, use_container_width=True)
        
        # Add a simple bar chart
        st.bar_chart(sources)
