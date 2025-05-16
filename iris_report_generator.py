import os
import base64
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# Import FPDF conditionally to avoid errors if the module is not installed
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: fpdf module not installed. PDF generation will be disabled.")
    PDF_AVAILABLE = False
from PIL import Image
from typing import Dict, Any, List, Optional


class IrisReportGenerator:
    """Generate PDF reports from iris analysis results."""
    
    def __init__(self):
        """Initialize report generator."""
        self.width = 210  # A4 width in mm
        self.margin = 10
    
    def _fig_to_img(self, fig):
        """Convert a matplotlib figure to a temporary file path for FPDF."""
        import tempfile
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Save the figure to the temporary file
        fig.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        
        return temp_filename
    
    def _fig_to_base64(self, fig):
        """Convert a matplotlib figure to base64 string for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def _np_array_to_img(self, array_or_img):
        """Convert a numpy array or PIL Image to a temporary file path for FPDF."""
        import tempfile
        
        if isinstance(array_or_img, np.ndarray):
            img = Image.fromarray(array_or_img.astype('uint8'))
        elif isinstance(array_or_img, Image.Image):
            img = array_or_img
        else:
            raise ValueError(f"Expected numpy array or PIL Image, got {type(array_or_img)}")
            
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Save the image to the temporary file
        img.save(temp_filename, format='PNG')
        
        return temp_filename
    
    def _np_array_to_base64(self, array_or_img):
        """Convert a numpy array or PIL Image to base64 string for HTML embedding."""
        if isinstance(array_or_img, np.ndarray):
            img = Image.fromarray(array_or_img.astype('uint8'))
        elif isinstance(array_or_img, Image.Image):
            img = array_or_img
        else:
            raise ValueError(f"Expected numpy array or PIL Image, got {type(array_or_img)}")
            
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def _create_dosha_chart(self, dosha_balance):
        """Create dosha balance chart."""
        if not dosha_balance:
            return None
            
        dosha_labels = [f"{k.capitalize()}" for k in dosha_balance.keys()]
        dosha_values = list(dosha_balance.values())
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(dosha_values, labels=dosha_labels, autopct='%1.1f%%', 
               startangle=90, colors=['#a29bfe', '#ff7675', '#55efc4'])
        ax.axis('equal')
        ax.set_title('Dosha Distribution in Iris')
        
        return self._fig_to_img(fig)
    
    def _create_zone_distribution_chart(self, zones_analysis):
        """Create a chart showing health condition distribution across zones."""
        conditions = {"normal": 0, "stressed": 0, "compromised": 0}
        
        for zone_data in zones_analysis.values():
            condition = zone_data["health_indication"]["condition"]
            conditions[condition] = conditions.get(condition, 0) + 1
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        y_pos = np.arange(len(conditions))
        values = list(conditions.values())
        
        bars = ax.barh(y_pos, values, align='center', 
                color=['#28a745', '#ffc107', '#dc3545'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([k.capitalize() for k in conditions.keys()])
        ax.invert_yaxis()
        ax.set_xlabel('Number of Zones')
        ax.set_title('Health Condition Distribution')
        
        # Add values on bars
        for i, v in enumerate(values):
            if v > 0:
                ax.text(v + 0.1, i, str(v), color='black', va='center')
        
        return self._fig_to_img(fig)
    
    def generate_report(self, zone_results: Dict[str, Any], user_info: Dict[str, str] = None) -> Optional[bytes]:
        """
        Generate a PDF report from iris analysis results.
        
        Args:
            zone_results: Results from IrisZoneAnalyzer
            user_info: Optional user information dictionary
            
        Returns:
            PDF report as bytes or None if PDF generation is not available
        """
        if not PDF_AVAILABLE:
            print("PDF generation is not available because the fpdf module is not installed.")
            return None
            
        if "error" in zone_results:
            raise ValueError(f"Cannot generate report from error results: {zone_results['error']}")
        
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Use built-in fonts instead of DejaVu fonts to avoid font issues
        # Note: Built-in fonts don't support unicode characters
        
        # Title and header
        pdf.set_font('helvetica', 'B', 20)
        pdf.cell(0, 10, 'Iris Zone Analysis Report', 0, 1, 'C')
        pdf.set_font('helvetica', '', 11)
        pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M")}', 0, 1, 'C')
        pdf.ln(5)
        
        # Add user info if provided
        if user_info:
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Client Information', 0, 1, 'L')
            pdf.set_font('helvetica', '', 11)
            
            for key, value in user_info.items():
                pdf.cell(40, 10, f"{key}:", 0, 0, 'L')
                pdf.cell(0, 10, f"{value}", 0, 1, 'L')
            pdf.ln(5)
        
        # Add iris images
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, 'Iris Analysis Visualizations', 0, 1, 'L')
        
        # Original and zone map images side by side
        pdf.set_font('helvetica', '', 10)
        img_width = (self.width - 2*self.margin) / 2 - 5  # Allow some spacing
        
        # Original image
        original_img = self._np_array_to_img(zone_results["original_image"])
        pdf.cell(img_width, 10, 'Original Iris Image', 0, 0, 'C')
        pdf.cell(img_width, 10, 'Zone Map Visualization', 0, 1, 'C')
        
        y_position = pdf.get_y()
        pdf.image(original_img, x=self.margin, y=y_position, w=img_width)
        
        # Zone map image
        zone_map_img = self._np_array_to_img(zone_results["zone_map"])
        pdf.image(zone_map_img, x=self.margin + img_width + 5, y=y_position, w=img_width)
        
        # Ensure we move past the images
        pdf.ln(75)
        
        # Overall health summary
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, 'Health Summary', 0, 1, 'L')
        pdf.set_font('helvetica', '', 11)
        
        overall_health = zone_results['health_summary']['overall_health'].capitalize()
        pdf.cell(0, 10, f'Overall Balance: {overall_health}', 0, 1, 'L')
        
        # Add dosha balance chart
        if 'dosha_balance' in zone_results['health_summary'] and zone_results['health_summary']['dosha_balance']:
            dosha_chart = self._create_dosha_chart(zone_results['health_summary']['dosha_balance'])
            if dosha_chart:
                pdf.image(dosha_chart, x=self.margin + 25, w=self.width / 2)
                pdf.ln(5)
        
        # Add zone distribution chart
        health_chart = self._create_zone_distribution_chart(zone_results["zones_analysis"])
        if health_chart:
            pdf.image(health_chart, x=self.margin + 25, w=self.width / 2)
            pdf.ln(10)
        
        # Dosha interpretation
        if 'dosha_balance' in zone_results['health_summary'] and zone_results['health_summary']['dosha_balance']:
            dosha_balance = zone_results['health_summary']['dosha_balance']
            primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0] if dosha_balance else "unknown"
            
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, 'Dosha Interpretation', 0, 1, 'L')
            pdf.set_font('helvetica', '', 11)
            
            if primary_dosha == "vata":
                pdf.multi_cell(0, 7, "Vata Dominant Iris\n\n"
                              "The iris shows signs of Vata dominance, which relates to the air and ether elements. "
                              "This often manifests as:\n"
                              "- Heightened nervous system activity\n"
                              "- Potential for dryness and variability in systems\n"
                              "- Need for grounding and stability\n\n"
                              "Balancing Suggestions:\n"
                              "- Regular routines and rest patterns\n"
                              "- Warm, nourishing, and grounding foods\n"
                              "- Gentle oil massage (abhyanga) with sesame oil\n"
                              "- Meditation and stress reduction practices")
            elif primary_dosha == "pitta":
                pdf.multi_cell(0, 7, "Pitta Dominant Iris\n\n"
                              "The iris shows signs of Pitta dominance, which relates to the fire and water elements. "
                              "This often manifests as:\n"
                              "- Strong digestive and metabolic functions\n"
                              "- Tendency toward heat and inflammation\n"
                              "- Potential for intensity and sharpness in bodily processes\n\n"
                              "Balancing Suggestions:\n"
                              "- Cooling foods and herbs\n"
                              "- Avoiding excessive heat, sun exposure, and spicy foods\n"
                              "- Regular exercise that's not too intense\n"
                              "- Calming and cooling practices like moonlight walks")
            elif primary_dosha == "kapha":
                pdf.multi_cell(0, 7, "Kapha Dominant Iris\n\n"
                              "The iris shows signs of Kapha dominance, which relates to the earth and water elements. "
                              "This often manifests as:\n"
                              "- Stable energy and strong immunity\n"
                              "- Potential for congestion or sluggishness\n"
                              "- Well-developed structural elements in the body\n\n"
                              "Balancing Suggestions:\n"
                              "- Regular stimulating exercise\n"
                              "- Warm, light, and spiced foods\n"
                              "- Dry brushing and invigorating practices\n"
                              "- Varied routines to prevent stagnation")
        
        # Detailed zone analysis
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, 'Detailed Zone Analysis', 0, 1, 'L')
        
        # Loop through each zone and add its details
        for zone_name, zone_data in zone_results["zones_analysis"].items():
            zone_display_name = zone_data["name"]
            health_condition = zone_data["health_indication"]["condition"]
            confidence = zone_data["health_indication"]["confidence"]
            
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, f"{zone_display_name} - {health_condition.capitalize()}", 0, 1, 'L')
            
            pdf.set_font('helvetica', '', 11)
            pdf.multi_cell(0, 7, f"Corresponds to: {', '.join(zone_data['ayurvedic_mapping']['systems'])}")
            pdf.multi_cell(0, 7, f"Description: {zone_data['ayurvedic_mapping']['description']}")
            pdf.multi_cell(0, 7, f"Condition: {health_condition.capitalize()} (Confidence: {confidence:.1%})")
            pdf.multi_cell(0, 7, f"Suggestion: {zone_data['health_indication']['suggestion']}")
            
            # Display dosha information
            dosha = zone_data['ayurvedic_mapping']['dominant_dosha']
            pdf.multi_cell(0, 7, f"Dominant Dosha: {dosha.capitalize()}")
            
            if dosha != "unknown" and len(zone_data['ayurvedic_mapping']['dosha_qualities']) > 0:
                pdf.multi_cell(0, 7, f"Qualities: {', '.join(zone_data['ayurvedic_mapping']['dosha_qualities'])}")
            
            pdf.ln(5)
            
        # Disclaimer
        pdf.set_font('helvetica', 'I', 9)
        pdf.ln(10)
        pdf.multi_cell(0, 5, "Disclaimer: This iris analysis is provided for educational and informational purposes only. "
                    "It is not intended to diagnose, treat, cure, or prevent any disease. "
                    "Please consult with a qualified healthcare provider for medical advice.")

        # Get PDF as bytes
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        
        # Clean up temporary files
        temp_files = []
        if 'original_img' in locals(): temp_files.append(original_img)
        if 'zone_map_img' in locals(): temp_files.append(zone_map_img)
        if 'dosha_chart' in locals() and dosha_chart: temp_files.append(dosha_chart)
        if 'health_chart' in locals() and health_chart: temp_files.append(health_chart)
        
        self._cleanup_temp_files(temp_files)
        
        return pdf_bytes
        
    def generate_html_report(self, zone_results: Dict[str, Any], user_info: Dict[str, str] = None) -> str:
        """
        Generate an HTML report from iris analysis results.
        
        Args:
            zone_results: Results from IrisZoneAnalyzer
            user_info: Optional user information dictionary
            
        Returns:
            HTML report as string
        """
        if "error" in zone_results:
            raise ValueError(f"Cannot generate report from error results: {zone_results['error']}")
        
        # Convert images to base64 for HTML embedding
        original_img = self._np_array_to_base64(zone_results["original_image"])
        zone_map_img = self._np_array_to_base64(zone_results["zone_map"])
        boundary_img = self._np_array_to_base64(zone_results["boundary_image"])
        
        # Generate dosha chart
        dosha_chart = None
        if 'dosha_balance' in zone_results['health_summary'] and zone_results['health_summary']['dosha_balance']:
            dosha_balance = zone_results['health_summary']['dosha_balance']
            dosha_labels = [f"{k.capitalize()}" for k in dosha_balance.keys()]
            dosha_values = list(dosha_balance.values())
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(dosha_values, labels=dosha_labels, autopct='%1.1f%%', 
                  startangle=90, colors=['#a29bfe', '#ff7675', '#55efc4'])
            ax.axis('equal')
            ax.set_title('Dosha Distribution in Iris')
            
            dosha_chart = self._fig_to_base64(fig)
        
        # Generate zone distribution chart
        health_chart = None
        conditions = {"normal": 0, "stressed": 0, "compromised": 0}
        for zone_data in zone_results["zones_analysis"].values():
            condition = zone_data["health_indication"]["condition"]
            conditions[condition] = conditions.get(condition, 0) + 1
            
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        y_pos = np.arange(len(conditions))
        values = list(conditions.values())
        
        bars = ax.barh(y_pos, values, align='center', 
                color=['#28a745', '#ffc107', '#dc3545'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([k.capitalize() for k in conditions.keys()])
        ax.invert_yaxis()
        ax.set_xlabel('Number of Zones')
        ax.set_title('Health Condition Distribution')
        
        # Add values on bars
        for i, v in enumerate(values):
            if v > 0:
                ax.text(v + 0.1, i, str(v), color='black', va='center')
        
        health_chart = self._fig_to_base64(fig)
        
        # Get primary dosha
        primary_dosha = "unknown"
        if 'dosha_balance' in zone_results['health_summary'] and zone_results['health_summary']['dosha_balance']:
            dosha_balance = zone_results['health_summary']['dosha_balance']
            primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0] if dosha_balance else "unknown"
        
        # Build HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Iris Zone Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }}
                .image-container {{
                    display: flex;
                    justify-content: space-between;
                    margin: 20px 0;
                }}
                .image-box {{
                    width: 48%;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                }}
                .zone-box {{
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 15px 0;
                    background-color: #f9f9f9;
                }}
                .normal {{
                    border-left: 5px solid #28a745;
                }}
                .stressed {{
                    border-left: 5px solid #ffc107;
                }}
                .compromised {{
                    border-left: 5px solid #dc3545;
                }}
                .charts {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
                }}
                .chart {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .disclaimer {{
                    font-size: 0.8em;
                    font-style: italic;
                    color: #6c757d;
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Iris Zone Analysis Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M")}</p>
            </div>
        """
        
        # Add user info if provided
        if user_info:
            html += "<h2>Client Information</h2><ul>"
            for key, value in user_info.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        # Add iris images
        html += f"""
            <h2>Iris Analysis Visualizations</h2>
            <div class="image-container">
                <div class="image-box">
                    <h3>Original Iris Image</h3>
                    <img src="data:image/png;base64,{original_img}">
                </div>
                <div class="image-box">
                    <h3>Zone Map Visualization</h3>
                    <img src="data:image/png;base64,{zone_map_img}">
                </div>
            </div>
            <div style="text-align: center; margin: 20px 0;">
                <h3>Boundary Detection</h3>
                <img src="data:image/png;base64,{boundary_img}" style="max-width: 70%;">
                <p><em>Iris and pupil boundaries detected</em></p>
            </div>
        """
        
        # Add health summary and charts
        html += f"""
            <h2>Health Summary</h2>
            <p><strong>Overall Balance:</strong> {zone_results['health_summary']['overall_health'].capitalize()}</p>
            
            <div class="charts">
        """
        
        # Add dosha chart if available
        if dosha_chart:
            html += f"""
                <div class="chart">
                    <h3>Dosha Distribution</h3>
                    <img src="data:image/png;base64,{dosha_chart}" style="max-width: 90%;">
                </div>
            """
        
        # Add health distribution chart
        if health_chart:
            html += f"""
                <div class="chart">
                    <h3>Health Condition Distribution</h3>
                    <img src="data:image/png;base64,{health_chart}" style="max-width: 90%;">
                </div>
            """
        
        html += "</div>"  # Close charts div
        
        # Add dosha interpretation
        html += "<h2>Dosha Interpretation</h2>"
        if primary_dosha == "vata":
            html += """
                <h3>Vata Dominant Iris</h3>
                <p>The iris shows signs of Vata dominance, which relates to the air and ether elements.
                This often manifests as:</p>
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
            """
        elif primary_dosha == "pitta":
            html += """
                <h3>Pitta Dominant Iris</h3>
                <p>The iris shows signs of Pitta dominance, which relates to the fire and water elements.
                This often manifests as:</p>
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
            """
        elif primary_dosha == "kapha":
            html += """
                <h3>Kapha Dominant Iris</h3>
                <p>The iris shows signs of Kapha dominance, which relates to the earth and water elements.
                This often manifests as:</p>
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
            """
        
        # Add detailed zone analysis
        html += """
            <h2>Detailed Zone Analysis</h2>
        """
        
        for zone_name, zone_data in zone_results["zones_analysis"].items():
            zone_display_name = zone_data["name"]
            health_condition = zone_data["health_indication"]["condition"]
            confidence = zone_data["health_indication"]["confidence"]
            
            html += f"""
                <div class="zone-box {health_condition}">
                    <h3>{zone_display_name} - {health_condition.capitalize()}</h3>
                    <p><strong>Corresponds to:</strong> {', '.join(zone_data['ayurvedic_mapping']['systems'])}</p>
                    <p><strong>Description:</strong> {zone_data['ayurvedic_mapping']['description']}</p>
                    <p><strong>Condition:</strong> {health_condition.capitalize()} (Confidence: {confidence:.1%})</p>
                    <p><strong>Suggestion:</strong> {zone_data['health_indication']['suggestion']}</p>
            """
            
            # Display dosha information
            dosha = zone_data['ayurvedic_mapping']['dominant_dosha']
            html += f"<p><strong>Dominant Dosha:</strong> {dosha.capitalize()}</p>"
            
            if dosha != "unknown" and len(zone_data['ayurvedic_mapping']['dosha_qualities']) > 0:
                html += f"<p><strong>Qualities:</strong> {', '.join(zone_data['ayurvedic_mapping']['dosha_qualities'])}</p>"
            
            html += "</div>"
        
        # Add disclaimer
        html += """
            <div class="disclaimer">
                <p>Disclaimer: This iris analysis is provided for educational and informational purposes only.
                It is not intended to diagnose, treat, cure, or prevent any disease.
                Please consult with a qualified healthcare provider for medical advice.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _cleanup_temp_files(self, file_list):
        """Clean up temporary files created during report generation."""
        import os
        for file_path in file_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_path}: {e}")
