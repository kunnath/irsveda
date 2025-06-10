from pathlib import Path
import tempfile
import streamlit as st
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from PIL import Image
from pdf_extractor import extract_iris_chunks
from iris_qdrant import IrisQdrantClient
from iris_predictor import IrisPredictor

# Initialize NLTK resources
def initialize_nltk_resources():
    """Download required NLTK resources if they're not already available."""
    try:
        # Check if punkt is already downloaded
        nltk.data.find('tokenizers/punkt')
        print("NLTK punkt tokenizer already downloaded.")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    try:
        # Check for punkt_tab resource (required for some advanced tokenization)
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK punkt_tab tokenizer already downloaded.")
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        # punkt_tab is included in the 'all' package
        nltk.download('all')
        
    try:
        # Check if stopwords are already downloaded
        nltk.data.find('corpora/stopwords')
        print("NLTK stopwords already downloaded.")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    try:
        # Check if wordnet is already downloaded
        nltk.data.find('corpora/wordnet')
        print("NLTK wordnet already downloaded.")
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

# Initialize NLTK resources at startup
initialize_nltk_resources()

# Page title and configuration
st.set_page_config(
    page_title="IridoVeda",
    page_icon="👁️",
    layout="wide"
)

# Create sidebar
st.sidebar.title("irsveda")
st.sidebar.image("static/iris_logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info(
    "This application extracts iris-related information from Ayurvedic/Iridology "
    "books and allows you to query the knowledge base."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Dinexora](https://www.dinexora.de)")

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
st.title("IridoVeda")

# Tab layout
tabs = st.tabs(["📚 PDF Upload & Processing", "🔍 Knowledge Query", "👁️ Iris Analysis", "🔬 Dosha Analysis", "📊 Statistics"])

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
                    ocr_status.info("📄 Analyzing text and image content...")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Define a custom callback for progress updates
                    def update_progress(page_num, total_pages):
                        progress = min(page_num / total_pages, 1.0)
                        progress_bar.progress(progress)
                        ocr_status.info(f"📄 Processing page {page_num}/{total_pages}...")
                    
                    # Extract chunks with progress updates
                    chunks = extract_iris_chunks(temp_path, progress_callback=update_progress)
                    end_time = time.time()
                    
                    # Update status upon completion
                    progress_bar.progress(1.0)
                    
                    # Add OCR extraction method info if applicable
                    ocr_used = any(chunk.get("extraction_method") == "ocr" for chunk in chunks)
                    if ocr_used:
                        ocr_status.success("🔍 OCR processing completed successfully")
                    else:
                        ocr_status.success("📄 Text extraction completed successfully")
                    
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

# Third tab - Enhanced Iris Analysis
with tabs[2]:
    st.header("Enhanced Iris Image Analysis")
    
    st.info(
        "Upload an iris image for comprehensive analysis including advanced segmentation, "
        "zone mapping, and detailed spot detection. Choose your analysis preferences below."
    )
    
    # Initialize enhanced analyzer in session state
    if "enhanced_iris_analyzer" not in st.session_state:
        from enhanced_iris_analysis_service import EnhancedIrisAnalysisService
        st.session_state.enhanced_iris_analyzer = EnhancedIrisAnalysisService()
    
    # Analysis configuration
    st.subheader("Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sensitivity = st.selectbox(
            "Detection Sensitivity",
            ["low", "medium", "high"],
            index=1,
            help="Higher sensitivity detects more subtle features"
        )
    
    with col2:
        include_zones = st.checkbox("Include Zone Analysis", value=True)
        include_doshas = st.checkbox("Include Dosha Analysis", value=True)
    
    with col3:
        include_segmentation = st.checkbox("Include Detailed Segmentation", value=True)
        export_data = st.checkbox("Enable Data Export", value=False)
    
    # Advanced parameters (collapsible)
    with st.expander("Advanced Detection Parameters", expanded=False):
        st.markdown("Fine-tune detection parameters for specialized analysis:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_area = st.slider("Minimum Spot Area", 10, 200, 50)
            max_area = st.slider("Maximum Spot Area", 500, 5000, 2000)
            min_circularity = st.slider("Minimum Circularity", 0.1, 1.0, 0.3)
            detect_dark_spots = st.checkbox("Detect Dark Spots", True)
        
        with col2:
            brightness_threshold = st.slider("Brightness Threshold", 5, 50, 20)
            contrast_enhancement = st.slider("Contrast Enhancement", 1.0, 5.0, 2.0)
            detect_light_spots = st.checkbox("Detect Light Spots", True)
            gaussian_blur = st.slider("Gaussian Blur", 1, 9, 3)
        
        if st.button("Apply Custom Parameters"):
            st.session_state.enhanced_iris_analyzer.update_detection_params(
                min_area=min_area,
                max_area=max_area,
                min_circularity=min_circularity,
                brightness_threshold=brightness_threshold,
                contrast_enhancement=contrast_enhancement,
                detect_dark_spots=detect_dark_spots,
                detect_light_spots=detect_light_spots,
                gaussian_blur=gaussian_blur
            )
            st.success("Custom parameters applied!")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "Upload iris image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, high-resolution image of an iris",
        key="enhanced_iris_analysis_uploader"
    )
    
    if uploaded_image:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_image.read())
            temp_path = tmp_file.name
        
        try:
            # Set sensitivity preset
            st.session_state.enhanced_iris_analyzer.set_sensitivity_preset(sensitivity)
            
            # Perform comprehensive analysis
            with st.spinner("Performing comprehensive iris analysis..."):
                enhanced_results = st.session_state.enhanced_iris_analyzer.analyze_iris_comprehensive(
                    temp_path,
                    include_zones=include_zones,
                    include_doshas=include_doshas,
                    include_segmentation=include_segmentation
                )
            
            if "error" in enhanced_results:
                st.error(f"Analysis error: {enhanced_results['error']}")
            else:
                # Display results in tabs
                analysis_tabs = st.tabs(["📊 Overview", "🔬 Segmentation", "🗺️ Zone Analysis", "🧬 Dosha Analysis", "📈 Data Export"])
                
                # Overview tab
                with analysis_tabs[0]:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Original Image")
                        original_img = Image.open(temp_path)
                        st.image(original_img, use_column_width=True)
                        
                        # Show basic analysis if available
                        if 'basic_analysis' in enhanced_results and 'analysis' in enhanced_results['basic_analysis']:
                            basic = enhanced_results['basic_analysis']['analysis']
                            st.markdown(f"**Overall Health:** {basic.get('overall_health', 'Unknown').capitalize()}")
                    
                    with col2:
                        st.subheader("Analysis Summary")
                        
                        # Show comprehensive report
                        if 'comprehensive_report' in enhanced_results:
                            report = enhanced_results['comprehensive_report']
                            
                            if 'summary' in report:
                                summary = report['summary']
                                
                                # Display metrics
                                metrics_col1, metrics_col2 = st.columns(2)
                                with metrics_col1:
                                    st.metric("Segments Detected", summary.get('segments_detected', 0))
                                    st.metric("Zones Analyzed", summary.get('zones_analyzed', 0))
                                
                                with metrics_col2:
                                    if 'detection_methods' in summary:
                                        methods = summary['detection_methods']
                                        st.metric("Detection Methods", len(methods))
                                        st.caption(f"Methods: {', '.join(methods)}")
                            
                            # Show recommendations
                            if 'recommendations' in report and report['recommendations']:
                                st.subheader("Health Recommendations")
                                for i, rec in enumerate(report['recommendations'][:5], 1):
                                    st.markdown(f"{i}. {rec}")
                
                # Segmentation tab
                with analysis_tabs[1]:
                    if include_segmentation and 'segmentation_analysis' in enhanced_results:
                        seg_results = enhanced_results['segmentation_analysis']
                        
                        st.subheader("Detailed Segmentation Analysis")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Show annotated image
                            if 'annotated_image' in seg_results:
                                st.subheader("Annotated Image")
                                st.markdown("*Color-coded detection methods: Green=Edge, Red=Dark Spots, Orange=Light Spots, Yellow=Blob*")
                                # Display base64 image
                                st.markdown(f'<img src="{seg_results["annotated_image"]}" style="max-width: 100%;">', 
                                          unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader("Detection Summary")
                            st.metric("Total Segments", seg_results.get('total_segments', 0))
                            
                            # Show detection methods used
                            if 'detection_methods_used' in seg_results:
                                st.write("**Methods Used:**")
                                for method in seg_results['detection_methods_used']:
                                    st.write(f"• {method.replace('_', ' ').title()}")
                            
                            # Show segment statistics
                            if 'segments' in seg_results and seg_results['segments']:
                                segments_df = pd.DataFrame(seg_results['segments'])
                                
                                st.write("**Segment Statistics:**")
                                st.write(f"Average Area: {segments_df['area'].mean():.1f}")
                                st.write(f"Average Brightness: {segments_df['avg_brightness'].mean():.1f}")
                                
                                # Show distribution by detection method
                                method_counts = segments_df['detection_method'].value_counts()
                                st.write("**Detection Method Distribution:**")
                                for method, count in method_counts.items():
                                    st.write(f"• {method.replace('_', ' ').title()}: {count}")
                        
                        # Detailed segments table
                        if 'segments' in seg_results and seg_results['segments']:
                            st.subheader("Detailed Segment Data")
                            segments_df = pd.DataFrame(seg_results['segments'])
                            
                            # Select columns to display
                            display_columns = ['segment_id', 'detection_method', 'area', 'avg_brightness', 
                                             'sharpness', 'visibility', 'contrast']
                            
                            if all(col in segments_df.columns for col in display_columns):
                                st.dataframe(segments_df[display_columns], use_container_width=True)
                    else:
                        st.info("Segmentation analysis not performed. Enable it in the configuration above.")
                
                # Zone Analysis tab
                with analysis_tabs[2]:
                    if include_zones:
                        if 'zone_analysis' in enhanced_results:
                            zone_results = enhanced_results['zone_analysis']
                            st.subheader("Iris Zone Analysis")
                            
                            # Display zone analysis results
                            if isinstance(zone_results, dict) and 'error' not in zone_results:
                                st.json(zone_results)
                            else:
                                st.info("Zone analysis results not available or encountered an error.")
                        
                        # Also show basic zone analysis from original results
                        if 'basic_analysis' in enhanced_results and 'analysis' in enhanced_results['basic_analysis']:
                            basic = enhanced_results['basic_analysis']['analysis']
                            if 'zones' in basic:
                                st.subheader("Basic Zone Assessment")
                                
                                for zone, info in basic['zones'].items():
                                    status_color = "green" if info["condition"] == "normal" else "orange" if info["condition"] == "stressed" else "red"
                                    st.markdown(
                                        f"**{zone.capitalize()}:** "
                                        f"<span style='color:{status_color}'>{info['condition']}</span> "
                                        f"(confidence: {info['confidence']:.0%})",
                                        unsafe_allow_html=True
                                    )
                    else:
                        st.info("Zone analysis not performed. Enable it in the configuration above.")
                
                # Dosha Analysis tab
                with analysis_tabs[3]:
                    if include_doshas and 'dosha_analysis' in enhanced_results['basic_analysis']:
                        dosha_analysis = enhanced_results['basic_analysis']['dosha_analysis']
                        st.subheader("Ayurvedic Dosha Analysis")
                        
                        if 'error' not in dosha_analysis:
                            # Display dosha profile if available
                            dosha_profile = dosha_analysis.get("dosha_profile", {})
                            
                            if dosha_profile:
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    # Dosha distribution pie chart
                                    fig, ax = plt.subplots(figsize=(6, 6))
                                    labels = ['Vata', 'Pitta', 'Kapha']
                                    sizes = [
                                        dosha_profile.get("vata_percentage", 33.3),
                                        dosha_profile.get("pitta_percentage", 33.3),
                                        dosha_profile.get("kapha_percentage", 33.3)
                                    ]
                                    colors = ['#AED6F1', '#F5B041', '#A9DFBF']
                                    
                                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                                    ax.axis('equal')
                                    plt.title('Dosha Distribution')
                                    st.pyplot(fig)
                                
                                with col2:
                                    # Dosha details
                                    primary_dosha = dosha_profile.get("primary_dosha", "unknown")
                                    secondary_dosha = dosha_profile.get("secondary_dosha", "none")
                                    balance_score = dosha_profile.get("balance_score", 0)
                                    
                                    st.markdown(f"**Primary Dosha:** {primary_dosha.capitalize()}")
                                    if secondary_dosha and secondary_dosha != "none":
                                        st.markdown(f"**Secondary Dosha:** {secondary_dosha.capitalize()}")
                                    
                                    st.markdown(f"**Balance Score:** {balance_score:.2f}/1.00")
                                    st.progress(balance_score)
                                    
                                    # Show detailed percentages
                                    st.markdown("**Detailed Percentages:**")
                                    for dosha, percentage in zip(labels, sizes):
                                        st.markdown(f"• {dosha}: {percentage:.1f}%")
                            else:
                                st.info("Dosha profile data not available.")
                        else:
                            st.error(f"Dosha analysis error: {dosha_analysis['error']}")
                    else:
                        st.info("Dosha analysis not performed or not available.")
                
                # Data Export tab
                with analysis_tabs[4]:
                    if export_data:
                        st.subheader("Data Export")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📊 Export Segments CSV"):
                                csv_path = st.session_state.enhanced_iris_analyzer.export_segments_csv()
                                if csv_path:
                                    st.success(f"Exported segments data to: {Path(csv_path).name}")
                                    
                                    # Show download link
                                    with open(csv_path, 'rb') as f:
                                        st.download_button(
                                            label="Download CSV",
                                            data=f.read(),
                                            file_name=Path(csv_path).name,
                                            mime='text/csv'
                                        )
                        
                        with col2:
                            if st.button("📈 Show Analysis Summary"):
                                summary = st.session_state.enhanced_iris_analyzer.get_analysis_summary()
                                st.json(summary)
                        
                        # Display current segments dataframe
                        st.subheader("Current Session Data")
                        segments_df = st.session_state.enhanced_iris_analyzer.get_segments_dataframe()
                        if not segments_df.empty:
                            st.dataframe(segments_df, use_container_width=True)
                        else:
                            st.info("No segment data available for current session.")
                    else:
                        st.info("Data export not enabled. Check 'Enable Data Export' in the configuration above.")
                
                # Generate and display suggested queries
                st.subheader("Suggested Knowledge Base Queries")
                
                if 'basic_analysis' in enhanced_results and 'queries' in enhanced_results['basic_analysis']:
                    queries = enhanced_results['basic_analysis']['queries']
                    
                    for query in queries:
                        if st.button(f"🔍 {query}", key=f"query_{hash(query)}"):
                            # Set the query for knowledge base search
                            st.session_state.current_query = query
                            
                            # Perform search if knowledge base is initialized
                            if st.session_state.is_initialized:
                                with st.spinner(f"Searching: {query}"):
                                    search_results = st.session_state.qdrant_client.search(query, limit=3)
                                    
                                    if search_results:
                                        st.subheader("Knowledge Base Results:")
                                        for i, result in enumerate(search_results):
                                            with st.expander(f"Result {i+1} - Relevance: {result['score']:.2f}"):
                                                st.markdown(result['text'])
                                                st.caption(f"Source: {Path(result['source']).name}, Page: {result['page']}")
                                    else:
                                        st.info("No matching information found in the knowledge base.")
                            else:
                                st.warning("Knowledge base not initialized. Please upload PDFs in the first tab.")
        
        except Exception as e:
            st.error(f"Error in enhanced iris analysis: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Fourth tab - Dosha Analysis
with tabs[3]:
    st.header("Ayurvedic Dosha Analysis")
    
    st.info(
        "Upload an iris image to perform comprehensive Ayurvedic dosha analysis. "
        "This analysis quantifies Vata, Pitta, and Kapha doshas and connects them to specific health metrics and organ assessments."
    )
    
    uploaded_dosha_image = st.file_uploader(
        "Upload iris image for dosha analysis", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an iris",
        key="dosha_analysis_uploader"
    )
    
    if uploaded_dosha_image:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_dosha_image.read())
            temp_path = tmp_file.name
        
        try:
            # Process iris image
            with st.spinner("Performing deep dosha analysis..."):
                results = st.session_state.iris_predictor.process_iris_image(temp_path)
                
                if "error" in results:
                    st.error(results["error"])
                elif "error" in results.get("dosha_analysis", {}):
                    st.error(results["dosha_analysis"]["error"])
                else:
                    dosha_analysis = results.get("dosha_analysis", {})
                    
                    # Display original image with iris outline
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("Iris Image")
                        st.image(results["image"], use_column_width=True)
                    
                    with col2:
                        st.subheader("Dosha Profile")
                        
                        # Get dosha profile data
                        dosha_profile = dosha_analysis.get("dosha_profile", {})
                        
                        # Create a pie chart for dosha distribution
                        if dosha_profile:
                            fig, ax = plt.subplots(figsize=(4, 4))
                            labels = ['Vata', 'Pitta', 'Kapha']
                            sizes = [
                                dosha_profile.get("vata_percentage", 33.3),
                                dosha_profile.get("pitta_percentage", 33.3),
                                dosha_profile.get("kapha_percentage", 33.3)
                            ]
                            colors = ['#AED6F1', '#F5B041', '#A9DFBF']
                            
                            wedges, texts, autotexts = ax.pie(
                                sizes, 
                                labels=labels, 
                                colors=colors,
                                autopct='%1.1f%%', 
                                startangle=90,
                                textprops={'fontsize': 12, 'weight': 'bold'}
                            )
                            
                            # Equal aspect ratio ensures that pie is drawn as a circle
                            ax.axis('equal')
                            plt.title('Dosha Distribution', fontsize=14, pad=20)
                            st.pyplot(fig)
                            
                            # Display primary and secondary doshas
                            primary_dosha = dosha_profile.get("primary_dosha", "unknown")
                            secondary_dosha = dosha_profile.get("secondary_dosha", "none")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Primary Dosha:** {primary_dosha.capitalize()}")
                            with col2:
                                if secondary_dosha and secondary_dosha != "none":
                                    st.markdown(f"**Secondary Dosha:** {secondary_dosha.capitalize()}")
                            
                            # Display balance score
                            balance_score = dosha_profile.get("balance_score", 0)
                            st.markdown(f"**Dosha Balance Score:** {balance_score:.2f}/1.00")
                            st.progress(balance_score)
                            
                            # Display imbalance type if any
                            imbalance_type = dosha_profile.get("imbalance_type", None)
                            if imbalance_type:
                                st.markdown(f"**Imbalance Type:** {imbalance_type.capitalize()}")
                    
                    # Display metabolic indicators
                    st.subheader("Metabolic Indicators")
                    
                    metabolic_indicators = dosha_analysis.get("metabolic_indicators", {})
                    if metabolic_indicators:
                        # Create columns for metabolic indicators
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Primary Metrics")
                            st.markdown(f"**Basal Metabolic Rate:** {metabolic_indicators.get('basal_metabolic_rate', 0):.1f} kcal/day")
                            st.markdown(f"**Serum Lipid (Cholesterol):** {metabolic_indicators.get('serum_lipid', 0):.1f} mg/dL")
                            st.markdown(f"**Triglycerides:** {metabolic_indicators.get('triglycerides', 0):.1f} mg/dL")
                            st.markdown(f"**CRP Level:** {metabolic_indicators.get('crp_level', 0):.2f} mg/L")
                        
                        with col2:
                            st.markdown("#### Secondary Metrics")
                            st.markdown(f"**Gut Diversity Score:** {metabolic_indicators.get('gut_diversity', 0):.2f}/1.00")
                            st.progress(metabolic_indicators.get('gut_diversity', 0))
                            
                            st.markdown(f"**Enzyme Activity Index:** {metabolic_indicators.get('enzyme_activity', 0):.2f}/1.00")
                            st.progress(metabolic_indicators.get('enzyme_activity', 0))
                            
                            st.markdown(f"**Appetite Score:** {metabolic_indicators.get('appetite_score', 0):.1f}/10.0")
                            st.progress(metabolic_indicators.get('appetite_score', 0) / 10)
                            
                            st.markdown(f"**Metabolism Variability:** {metabolic_indicators.get('metabolism_variability', 0):.2f}/1.00")
                            st.progress(metabolic_indicators.get('metabolism_variability', 0))
                    
                    # Display organ health assessment
                    st.subheader("Organ Health Assessment")
                    
                    organ_assessments = dosha_analysis.get("organ_assessments", [])
                    if organ_assessments:
                        # Create a dataframe for the organ assessments
                        organ_data = []
                        
                        for assessment in organ_assessments:
                            organ_name = assessment.get("organ_name", "unknown")
                            health_score = assessment.get("health_score", 0)
                            warning_level = assessment.get("warning_level", "normal")
                            
                            # Determine color based on warning level
                            if warning_level == "critical":
                                color = "#d9534f"  # Red
                            elif warning_level == "warning":
                                color = "#f0ad4e"  # Orange
                            elif warning_level == "attention":
                                color = "#5bc0de"  # Blue
                            else:
                                color = "#5cb85c"  # Green
                            
                            organ_data.append({
                                "Organ": organ_name.capitalize(),
                                "Health Score": health_score,
                                "Status": warning_level.upper(),
                                "Color": color
                            })
                        
                        # Convert to dataframe
                        organ_df = pd.DataFrame(organ_data)
                        
                        # Display organs that need attention first
                        attention_organs = organ_df[organ_df["Status"] != "NORMAL"].sort_values("Health Score")
                        if not attention_organs.empty:
                            st.markdown("#### Organs Needing Attention")
                            
                            # Display each organ with a progress bar
                            for _, row in attention_organs.iterrows():
                                st.markdown(f"**{row['Organ']}** - {row['Status']}")
                                st.progress(row['Health Score'])
                        
                        # Create bar chart for all organs
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Sort by health score
                        organ_df = organ_df.sort_values("Health Score")
                        
                        # Create horizontal bar chart
                        bars = ax.barh(organ_df["Organ"], organ_df["Health Score"], color=organ_df["Color"])
                        
                        # Add value labels
                        for i, bar in enumerate(bars):
                            ax.text(
                                bar.get_width() + 0.01,
                                bar.get_y() + bar.get_height()/2,
                                f"{organ_df.iloc[i]['Health Score']:.2f}",
                                va='center'
                            )
                        
                        # Add labels and title
                        ax.set_xlabel('Health Score (0-1)')
                        ax.set_title('Organ Health Assessment')
                        ax.set_xlim(0, 1.1)  # Set x-axis limit
                        
                        # Display the chart
                        st.pyplot(fig)
                    
                    # Display key recommendations
                    st.subheader("Health Recommendations")
                    
                    key_recommendations = dosha_analysis.get("key_recommendations", [])
                    if key_recommendations:
                        for i, recommendation in enumerate(key_recommendations, 1):
                            st.markdown(f"{i}. {recommendation}")
                    
                    # Display overall health assessment
                    st.subheader("Overall Health Assessment")
                    
                    health_status = dosha_analysis.get("health_status", "unknown")
                    overall_health_score = dosha_analysis.get("overall_health_score", 0)
                    
                    # Determine color based on health status
                    if health_status == "excellent":
                        status_color = "#5cb85c"  # Green
                    elif health_status == "good":
                        status_color = "#5bc0de"  # Blue
                    elif health_status == "fair":
                        status_color = "#f0ad4e"  # Orange
                    else:
                        status_color = "#d9534f"  # Red
                    
                    st.markdown(
                        f"**Overall Health Status:** "
                        f"<span style='color:{status_color}'>{health_status.upper()}</span> "
                        f"(Score: {overall_health_score:.2f}/1.00)",
                        unsafe_allow_html=True
                    )
                    st.progress(overall_health_score)
                    
                    # Add a button to generate a detailed PDF report
                    if st.button("Generate Detailed Health Report (PDF)"):
                        st.info("This feature will generate a comprehensive health report based on your iris analysis. Coming soon!")
                    
        except Exception as e:
            st.error(f"Error in dosha analysis: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(temp_path)

# Fifth tab - Statistics
with tabs[4]:
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
