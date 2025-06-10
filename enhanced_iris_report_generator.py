"""
Enhanced Iris Report Generator
Generates comprehensive PDF reports with annotated iris images, health assessments, organ mapping, and spot details.
"""

import os
import base64
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import json

# Import FPDF conditionally
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: fpdf module not installed. PDF generation will be disabled.")
    PDF_AVAILABLE = False


class EnhancedIrisReportGenerator:
    """Enhanced iris report generator with comprehensive analysis features."""
    
    def __init__(self):
        """Initialize the enhanced report generator."""
        self.width = 210  # A4 width in mm
        self.height = 297  # A4 height in mm
        self.margin = 20  # Increased margin for better layout
        self.temp_files = []  # Track temporary files for cleanup
        
        # Color scheme for different health conditions
        self.condition_colors = {
            'normal': '#28a745',
            'stressed': '#ffc107', 
            'compromised': '#dc3545',
            'unknown': '#6c757d'
        }
        
        # Dosha colors
        self.dosha_colors = {
            'vata': '#a29bfe',
            'pitta': '#ff7675',
            'kapha': '#55efc4'
        }
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_path}: {e}")
        self.temp_files = []
    
    def _save_temp_image(self, img_data, format='PNG') -> str:
        """Save image data to temporary file and track it."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format.lower()}')
        temp_filename = temp_file.name
        temp_file.close()
        
        if isinstance(img_data, np.ndarray):
            img = Image.fromarray(img_data.astype('uint8'))
        elif isinstance(img_data, Image.Image):
            img = img_data
        else:
            raise ValueError(f"Expected numpy array or PIL Image, got {type(img_data)}")
        
        img.save(temp_filename, format=format)
        self.temp_files.append(temp_filename)
        return temp_filename
    
    def _fig_to_temp_file(self, fig) -> str:
        """Save matplotlib figure to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        temp_file.close()
        
        fig.savefig(temp_filename, format='png', dpi=300, bbox_inches='tight')
        self.temp_files.append(temp_filename)
        return temp_filename
    
    def _create_enhanced_dosha_chart(self, dosha_balance: Dict[str, float]) -> str:
        """Create an enhanced dosha distribution chart."""
        if not dosha_balance:
            return None
            
        dosha_labels = [f"{k.capitalize()}" for k in dosha_balance.keys()]
        dosha_values = list(dosha_balance.values())
        colors = [self.dosha_colors.get(k.lower(), '#6c757d') for k in dosha_balance.keys()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            dosha_values, 
            labels=dosha_labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            textprops={'fontsize': 10}
        )
        ax1.set_title('Dosha Distribution', fontsize=12, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(dosha_labels, dosha_values, color=colors)
        ax2.set_title('Dosha Percentages', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentage')
        ax2.set_ylim(0, max(dosha_values) * 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, dosha_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_temp_file(fig)
    
    def _create_spot_distribution_chart(self, spots: List[Dict]) -> str:
        """Create spot distribution charts by organ system."""
        if not spots:
            return None
        
        # Count spots by organ
        organ_counts = {}
        zone_counts = {}
        method_counts = {}
        
        for spot in spots:
            organ = spot.get('corresponding_organ', 'Unknown')
            zone = spot.get('iris_zone', 'Unknown')
            method = spot.get('detection_method', 'Unknown')
            
            organ_counts[organ] = organ_counts.get(organ, 0) + 1
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Organ distribution
        if organ_counts:
            organs = list(organ_counts.keys())[:10]  # Top 10
            counts = [organ_counts[org] for org in organs]
            ax1.barh(organs, counts, color='lightblue')
            ax1.set_title('Spots by Organ System', fontweight='bold')
            ax1.set_xlabel('Number of Spots')
        
        # Zone distribution
        if zone_counts:
            zones = list(zone_counts.keys())
            counts = list(zone_counts.values())
            ax2.pie(counts, labels=zones, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Spots by Iris Zone', fontweight='bold')
        
        # Detection method distribution
        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
            ax3.bar(methods, counts, color=colors[:len(methods)])
            ax3.set_title('Detection Methods Used', fontweight='bold')
            ax3.set_ylabel('Number of Spots')
            ax3.tick_params(axis='x', rotation=45)
        
        # Spot size distribution
        areas = [spot.get('area', 0) for spot in spots if spot.get('area', 0) > 0]
        if areas:
            ax4.hist(areas, bins=20, color='lightgreen', alpha=0.7)
            ax4.set_title('Spot Size Distribution', fontweight='bold')
            ax4.set_xlabel('Area (pixels²)')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        return self._fig_to_temp_file(fig)
    
    def _create_health_assessment_chart(self, zones_analysis: Dict) -> str:
        """Create health assessment visualization."""
        conditions = {"normal": 0, "stressed": 0, "compromised": 0}
        
        for zone_data in zones_analysis.values():
            condition = zone_data.get("health_indication", {}).get("condition", "unknown")
            if condition in conditions:
                conditions[condition] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Horizontal bar chart
        y_pos = np.arange(len(conditions))
        values = list(conditions.values())
        colors = [self.condition_colors[k] for k in conditions.keys()]
        
        bars = ax1.barh(y_pos, values, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([k.capitalize() for k in conditions.keys()])
        ax1.invert_yaxis()
        ax1.set_xlabel('Number of Zones')
        ax1.set_title('Health Condition Distribution')
        
        # Add value labels
        for i, v in enumerate(values):
            if v > 0:
                ax1.text(v + 0.1, i, str(v), va='center')
        
        # Pie chart
        if sum(values) > 0:
            ax2.pie(values, labels=[k.capitalize() for k in conditions.keys()], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Health Distribution')
        
        plt.tight_layout()
        return self._fig_to_temp_file(fig)
    
    def generate_enhanced_pdf_report(self, 
                                   zone_results: Dict[str, Any], 
                                   enhanced_results: Optional[Dict[str, Any]] = None,
                                   user_info: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        """
        Generate enhanced PDF report with comprehensive iris analysis.
        
        Args:
            zone_results: Results from iris zone analysis
            enhanced_results: Results from enhanced spot detection (optional)
            user_info: User information dictionary (optional)
            
        Returns:
            PDF report as bytes or None if generation fails
        """
        if not PDF_AVAILABLE:
            print("PDF generation is not available because the fpdf module is not installed.")
            return None
            
        if "error" in zone_results:
            raise ValueError(f"Cannot generate report from error results: {zone_results['error']}")
        
        try:
            # Initialize PDF with more conservative settings
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=25)
            
            # Use Arial font which has better Unicode support
            pdf.set_font('Arial', 'B', 20)
            pdf.set_text_color(0, 0, 0)
            
            # Title and header
            pdf.cell(0, 15, 'Enhanced Iris Analysis Report', 0, 1, 'C')
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M")}', 0, 1, 'C')
            pdf.ln(10)
            
            # Client Information
            if user_info:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Client Information', 0, 1, 'L')
                pdf.set_font('Arial', '', 11)
                
                for key, value in user_info.items():
                    pdf.cell(50, 8, f"{key}:", 0, 0, 'L')
                    pdf.cell(0, 8, str(value), 0, 1, 'L')
                pdf.ln(5)
            
            # Executive Summary
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
            pdf.set_font('Arial', '', 11)
            
            # Overall health assessment
            overall_health = zone_results.get('health_summary', {}).get('overall_health', 'unknown')
            pdf.cell(0, 6, f"Overall Health Assessment: {overall_health.title()}", 0, 1, 'L')
            
            # Spot count summary
            total_spots = 0
            if enhanced_results and 'spots' in enhanced_results:
                total_spots = len(enhanced_results['spots'])
                pdf.cell(0, 6, f"Total Spots Detected: {total_spots}", 0, 1, 'L')
                
                if total_spots > 0:
                    # Detection methods used
                    methods = enhanced_results.get('detection_methods_used', [])
                    pdf.cell(0, 6, f"Detection Methods Used: {', '.join(methods)}", 0, 1, 'L')
            
            pdf.ln(10)
            
            # Dosha Analysis
            if 'dosha_balance' in zone_results.get('health_summary', {}):
                dosha_balance = zone_results['health_summary']['dosha_balance']
                if dosha_balance:
                    primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0]
                    
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, 'Dosha Analysis', 0, 1, 'L')
                    pdf.set_font('Arial', '', 11)
                    
                    pdf.cell(0, 8, f'Primary Dosha: {primary_dosha.title()}', 0, 1, 'L')
                    
                    # Dosha percentages
                    for dosha, value in dosha_balance.items():
                        pdf.cell(0, 6, f"  {dosha.title()}: {value:.1%}", 0, 1, 'L')
                    
                    pdf.ln(5)
            
            # Enhanced Spot Analysis Section
            if enhanced_results and enhanced_results.get('spots'):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 12, 'Enhanced Spot Detection Analysis', 0, 1, 'L')
                
                spots = enhanced_results['spots']
                
                # Spot statistics
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, 'Spot Detection Summary', 0, 1, 'L')
                pdf.set_font('Arial', '', 10)
                
                pdf.cell(0, 6, f"Total Spots Detected: {len(spots)}", 0, 1, 'L')
                
                # Average spot size
                if spots:
                    avg_area = sum(spot.get('area', 0) for spot in spots) / len(spots)
                    pdf.cell(0, 6, f"Average Spot Size: {avg_area:.1f} pixels", 0, 1, 'L')
                
                # Detection methods
                methods = enhanced_results.get('detection_methods_used', [])
                pdf.cell(0, 6, f"Detection Methods: {', '.join(methods)}", 0, 1, 'L')
                pdf.ln(5)
                
                # Top organs affected
                organ_counts = {}
                for spot in spots:
                    organ = spot.get('corresponding_organ', 'Unknown')
                    organ_counts[organ] = organ_counts.get(organ, 0) + 1
                
                if organ_counts:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 8, 'Most Affected Organ Systems', 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    
                    sorted_organs = sorted(organ_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    for organ, count in sorted_organs:
                        pdf.cell(0, 6, f"  {organ}: {count} spots", 0, 1, 'L')
                    pdf.ln(5)
                
                # Detailed spot information (top 10)
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, 'Detailed Spot Analysis (Top 10 by Size)', 0, 1, 'L')
                pdf.set_font('Arial', '', 9)
                
                # Table header
                pdf.cell(30, 6, 'Zone', 1, 0, 'C')
                pdf.cell(40, 6, 'Organ System', 1, 0, 'C')
                pdf.cell(25, 6, 'Area', 1, 0, 'C')
                pdf.cell(30, 6, 'Method', 1, 0, 'C')
                pdf.cell(30, 6, 'Position', 1, 1, 'C')
                
                # Sort spots by area and show top 10
                sorted_spots = sorted(spots, key=lambda x: x.get('area', 0), reverse=True)[:10]
                for spot in sorted_spots:
                    zone = spot.get('iris_zone', 'Unknown')[:8]  # Truncate for space
                    organ = spot.get('corresponding_organ', 'Unknown')[:12]  # Truncate
                    area = spot.get('area', 0)
                    method = spot.get('detection_method', 'Unknown')[:8]  # Truncate
                    angle = spot.get('angle_degrees', 0)
                    distance = spot.get('distance_from_center_pct', 0)
                    position = f"{angle:.0f}° {distance:.0f}%"[:10]  # Truncate
                    
                    pdf.cell(30, 6, zone, 1, 0, 'C')
                    pdf.cell(40, 6, organ, 1, 0, 'C')
                    pdf.cell(25, 6, f"{area:.0f}", 1, 0, 'C')
                    pdf.cell(30, 6, method, 1, 0, 'C')
                    pdf.cell(30, 6, position, 1, 1, 'C')
                
                pdf.ln(5)
            
            # Zone Analysis Section
            if "zones_analysis" in zone_results:
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 12, 'Detailed Zone Analysis', 0, 1, 'L')
                
                for zone_name, zone_data in zone_results["zones_analysis"].items():
                    zone_display_name = zone_data.get("name", zone_name)
                    health_condition = zone_data.get("health_indication", {}).get("condition", "unknown")
                    confidence = zone_data.get("health_indication", {}).get("confidence", 0)
                    
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 8, f"{zone_display_name} - {health_condition.title()}", 0, 1, 'L')
                    
                    pdf.set_font('Arial', '', 10)
                    systems = zone_data.get('ayurvedic_mapping', {}).get('systems', [])
                    if systems:
                        pdf.multi_cell(0, 5, f"Organ Systems: {', '.join(systems)}")
                    
                    description = zone_data.get('ayurvedic_mapping', {}).get('description', '')
                    if description:
                        pdf.multi_cell(0, 5, f"Description: {description}")
                    
                    pdf.cell(0, 5, f"Condition: {health_condition.title()} (Confidence: {confidence:.1%})", 0, 1, 'L')
                    
                    suggestion = zone_data.get('health_indication', {}).get('suggestion', '')
                    if suggestion:
                        pdf.multi_cell(0, 5, f"Suggestion: {suggestion}")
                    
                    # Dosha information
                    dosha = zone_data.get('ayurvedic_mapping', {}).get('dominant_dosha', 'unknown')
                    if dosha != "unknown":
                        pdf.cell(0, 5, f"Dominant Dosha: {dosha.title()}", 0, 1, 'L')
                    
                    pdf.ln(5)
            
            # Recommendations Section
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, 'Personalized Recommendations', 0, 1, 'L')
            
            # General health recommendations
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'General Health Recommendations', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            if enhanced_results and enhanced_results.get('spots'):
                spot_count = len(enhanced_results['spots'])
                if spot_count > 10:
                    recommendations = [
                        "High spot count detected - consider detoxification protocols",
                        "Focus on liver and kidney support",
                        "Reduce inflammatory foods and toxin exposure"
                    ]
                elif spot_count > 5:
                    recommendations = [
                        "Moderate spot count - maintain healthy elimination",
                        "Support digestive health"
                    ]
                else:
                    recommendations = [
                        "Low spot count indicates good elimination",
                        "Maintain current healthy practices"
                    ]
                
                for rec in recommendations:
                    pdf.multi_cell(0, 5, f"- {rec}")
            
            # Dosha-specific recommendations
            if 'dosha_balance' in zone_results.get('health_summary', {}):
                dosha_balance = zone_results['health_summary']['dosha_balance']
                if dosha_balance:
                    primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0]
                    
                    pdf.ln(5)
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 8, f'{primary_dosha.title()} Dosha Recommendations', 0, 1, 'L')
                    pdf.set_font('Arial', '', 10)
                    
                    if primary_dosha == "vata":
                        recommendations = [
                            "Regular routines and consistent meal times",
                            "Warm, nourishing, and grounding foods",
                            "Oil massage with sesame oil",
                            "Meditation and stress reduction practices",
                            "Avoid cold, dry, and raw foods"
                        ]
                    elif primary_dosha == "pitta":
                        recommendations = [
                            "Cooling foods and herbs",
                            "Avoid excessive heat and spicy foods",
                            "Regular moderate exercise",
                            "Cooling practices like moonlight walks",
                            "Manage stress and anger"
                        ]
                    else:  # kapha
                        recommendations = [
                            "Regular stimulating exercise",
                            "Warm, light, and spiced foods",
                            "Dry brushing and invigorating practices",
                            "Varied routines to prevent stagnation",
                            "Avoid heavy, oily, and cold foods"
                        ]
                    
                    for rec in recommendations:
                        pdf.multi_cell(0, 5, f"- {rec}")
            
            # Follow-up recommendations
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Follow-up Recommendations', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            follow_up = [
                "Consult with an Ayurvedic practitioner for personalized guidance",
                "Consider follow-up iris analysis in 3-6 months",
                "Implement dietary and lifestyle changes gradually",
                "Track symptoms and energy levels",
                "Regular health check-ups with healthcare providers"
            ]
            
            for rec in follow_up:
                pdf.multi_cell(0, 5, f"- {rec}")
            
            # Disclaimer
            pdf.ln(10)
            pdf.set_font('Arial', 'I', 9)
            pdf.multi_cell(0, 4, 
                "Disclaimer: This iris analysis is provided for educational and informational purposes only. "
                "It is not intended to diagnose, treat, cure, or prevent any disease. "
                "Please consult with a qualified healthcare provider for medical advice and before making "
                "any changes to your health regimen.")
            
            # Get PDF as bytes
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            
            return pdf_bytes
            
        except Exception as e:
            print(f"Error generating enhanced PDF report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def generate_enhanced_html_report(self,
                                    zone_results: Dict[str, Any],
                                    enhanced_results: Optional[Dict[str, Any]] = None,
                                    user_info: Optional[Dict[str, str]] = None) -> str:
        """
        Generate enhanced HTML report with comprehensive analysis.
        
        Args:
            zone_results: Results from iris zone analysis
            enhanced_results: Results from enhanced spot detection (optional)
            user_info: User information dictionary (optional)
            
        Returns:
            HTML report as string
        """
        if "error" in zone_results:
            raise ValueError(f"Cannot generate report from error results: {zone_results['error']}")
        
        # Convert images to base64
        def img_to_base64(img_data):
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'))
            elif isinstance(img_data, Image.Image):
                img = img_data
            else:
                return ""
            
            buf = BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        
        # Get images
        original_img = img_to_base64(zone_results.get("original_image", ""))
        zone_map_img = img_to_base64(zone_results.get("zone_map", ""))
        boundary_img = img_to_base64(zone_results.get("boundary_image", ""))
        
        # Enhanced annotated image
        annotated_img = ""
        if enhanced_results and "annotated_image" in enhanced_results:
            annotated_img = enhanced_results["annotated_image"]
        
        # Generate charts
        dosha_chart = ""
        health_chart = ""
        spot_chart = ""
        
        if 'dosha_balance' in zone_results.get('health_summary', {}):
            dosha_balance = zone_results['health_summary']['dosha_balance']
            if dosha_balance:
                dosha_chart_path = self._create_enhanced_dosha_chart(dosha_balance)
                if dosha_chart_path:
                    with open(dosha_chart_path, 'rb') as f:
                        dosha_chart = base64.b64encode(f.read()).decode('utf-8')
        
        if "zones_analysis" in zone_results:
            health_chart_path = self._create_health_assessment_chart(zone_results["zones_analysis"])
            if health_chart_path:
                with open(health_chart_path, 'rb') as f:
                    health_chart = base64.b64encode(f.read()).decode('utf-8')
        
        if enhanced_results and enhanced_results.get('spots'):
            spot_chart_path = self._create_spot_distribution_chart(enhanced_results['spots'])
            if spot_chart_path:
                with open(spot_chart_path, 'rb') as f:
                    spot_chart = base64.b64encode(f.read()).decode('utf-8')
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Iris Analysis Report</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                
                .section {{
                    background: white;
                    margin: 20px 0;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .section-title {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                
                .image-gallery {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .image-card {{
                    text-align: center;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }}
                
                .image-card img {{
                    max-width: 100%;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .metric-card {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                
                .zone-analysis {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 15px;
                }}
                
                .zone-card {{
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 15px;
                    background: #f8f9fa;
                }}
                
                .zone-card.normal {{ border-left: 5px solid #28a745; }}
                .zone-card.stressed {{ border-left: 5px solid #ffc107; }}
                .zone-card.compromised {{ border-left: 5px solid #dc3545; }}
                
                .spot-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                
                .spot-table th, .spot-table td {{
                    border: 1px solid #dee2e6;
                    padding: 8px 12px;
                    text-align: left;
                }}
                
                .spot-table th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                
                .recommendation-list {{
                    list-style: none;
                    padding: 0;
                }}
                
                .recommendation-list li {{
                    background: #e8f5e8;
                    margin: 8px 0;
                    padding: 12px;
                    border-left: 4px solid #28a745;
                    border-radius: 4px;
                }}
                
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .chart-container img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                
                .disclaimer {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 0.9em;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Iris Analysis Report</h1>
                <p>Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
            </div>
        """
        
        # Client Information
        if user_info:
            html += """
            <div class="section">
                <h2 class="section-title">Client Information</h2>
            """
            for key, value in user_info.items():
                html += f"<p><strong>{key.title()}:</strong> {value}</p>"
            html += "</div>"
        
        # Executive Summary
        overall_health = zone_results.get('health_summary', {}).get('overall_health', 'unknown')
        total_spots = len(enhanced_results.get('spots', [])) if enhanced_results else 0
        
        html += f"""
        <div class="section">
            <h2 class="section-title">Executive Summary</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{overall_health.title()}</div>
                    <div class="metric-label">Overall Health</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_spots}</div>
                    <div class="metric-label">Spots Detected</div>
                </div>
        """
        
        if enhanced_results:
            methods_count = len(enhanced_results.get('detection_methods_used', []))
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{methods_count}</div>
                    <div class="metric-label">Detection Methods</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        # Image Gallery
        html += """
        <div class="section">
            <h2 class="section-title">Iris Analysis Visualizations</h2>
            <div class="image-gallery">
        """
        
        if original_img:
            html += f"""
            <div class="image-card">
                <h3>Original Iris Image</h3>
                <img src="data:image/png;base64,{original_img}" alt="Original Iris">
            </div>
            """
        
        if annotated_img:
            html += f"""
            <div class="image-card">
                <h3>Enhanced Spot Detection</h3>
                <img src="data:image/png;base64,{annotated_img}" alt="Annotated Iris">
            </div>
            """
        elif zone_map_img:
            html += f"""
            <div class="image-card">
                <h3>Zone Map Analysis</h3>
                <img src="data:image/png;base64,{zone_map_img}" alt="Zone Map">
            </div>
            """
        
        if boundary_img:
            html += f"""
            <div class="image-card">
                <h3>Boundary Detection</h3>
                <img src="data:image/png;base64,{boundary_img}" alt="Boundary Detection">
            </div>
            """
        
        html += "</div></div>"
        
        # Charts Section
        if dosha_chart or health_chart or spot_chart:
            html += """
            <div class="section">
                <h2 class="section-title">Analysis Charts</h2>
            """
            
            if dosha_chart:
                html += f"""
                <div class="chart-container">
                    <h3>Dosha Distribution Analysis</h3>
                    <img src="data:image/png;base64,{dosha_chart}" alt="Dosha Chart">
                </div>
                """
            
            if health_chart:
                html += f"""
                <div class="chart-container">
                    <h3>Health Condition Distribution</h3>
                    <img src="data:image/png;base64,{health_chart}" alt="Health Chart">
                </div>
                """
            
            if spot_chart:
                html += f"""
                <div class="chart-container">
                    <h3>Spot Distribution Analysis</h3>
                    <img src="data:image/png;base64,{spot_chart}" alt="Spot Chart">
                </div>
                """
            
            html += "</div>"
        
        # Enhanced Spot Analysis
        if enhanced_results and enhanced_results.get('spots'):
            spots = enhanced_results['spots']
            html += f"""
            <div class="section">
                <h2 class="section-title">Enhanced Spot Detection Results</h2>
                <p><strong>Total Spots Detected:</strong> {len(spots)}</p>
                <p><strong>Detection Methods Used:</strong> {', '.join(enhanced_results.get('detection_methods_used', []))}</p>
                
                <h3>Top Affected Organ Systems</h3>
            """
            
            # Count spots by organ
            organ_counts = {}
            for spot in spots:
                organ = spot.get('corresponding_organ', 'Unknown')
                organ_counts[organ] = organ_counts.get(organ, 0) + 1
            
            sorted_organs = sorted(organ_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            html += "<ul class='recommendation-list'>"
            for organ, count in sorted_organs:
                html += f"<li><strong>{organ}:</strong> {count} spots detected</li>"
            html += "</ul>"
            
            # Spot details table (top 10)
            html += """
            <h3>Detailed Spot Analysis (Top 10 by Size)</h3>
            <table class="spot-table">
                <tr>
                    <th>Zone</th>
                    <th>Organ System</th>
                    <th>Area (px²)</th>
                    <th>Detection Method</th>
                    <th>Position</th>
                </tr>
            """
            
            sorted_spots = sorted(spots, key=lambda x: x.get('area', 0), reverse=True)[:10]
            for spot in sorted_spots:
                zone = spot.get('iris_zone', 'Unknown')
                organ = spot.get('corresponding_organ', 'Unknown')
                area = spot.get('area', 0)
                method = spot.get('detection_method', 'Unknown')
                angle = spot.get('angle_degrees', 0)
                distance = spot.get('distance_from_center_pct', 0)
                
                html += f"""
                <tr>
                    <td>{zone}</td>
                    <td>{organ}</td>
                    <td>{area:.0f}</td>
                    <td>{method}</td>
                    <td>{angle:.1f}° at {distance:.1f}%</td>
                </tr>
                """
            
            html += "</table></div>"
        
        # Zone Analysis
        if "zones_analysis" in zone_results:
            html += """
            <div class="section">
                <h2 class="section-title">Detailed Zone Analysis</h2>
                <div class="zone-analysis">
            """
            
            for zone_name, zone_data in zone_results["zones_analysis"].items():
                zone_display_name = zone_data.get("name", zone_name)
                health_condition = zone_data.get("health_indication", {}).get("condition", "unknown")
                confidence = zone_data.get("health_indication", {}).get("confidence", 0)
                
                html += f"""
                <div class="zone-card {health_condition}">
                    <h3>{zone_display_name}</h3>
                    <p><strong>Status:</strong> {health_condition.capitalize()} ({confidence:.1%} confidence)</p>
                """
                
                systems = zone_data.get('ayurvedic_mapping', {}).get('systems', [])
                if systems:
                    html += f"<p><strong>Organ Systems:</strong> {', '.join(systems)}</p>"
                
                description = zone_data.get('ayurvedic_mapping', {}).get('description', '')
                if description:
                    html += f"<p><strong>Description:</strong> {description}</p>"
                
                suggestion = zone_data.get('health_indication', {}).get('suggestion', '')
                if suggestion:
                    html += f"<p><strong>Recommendation:</strong> {suggestion}</p>"
                
                html += "</div>"
            
            html += "</div></div>"
        
        # Recommendations
        html += """
        <div class="section">
            <h2 class="section-title">Personalized Recommendations</h2>
        """
        
        # Dosha-specific recommendations
        if 'dosha_balance' in zone_results.get('health_summary', {}):
            dosha_balance = zone_results['health_summary']['dosha_balance']
            if dosha_balance:
                primary_dosha = max(dosha_balance.items(), key=lambda x: x[1])[0]
                
                html += f"<h3>{primary_dosha.capitalize()} Dosha Balancing</h3>"
                
                if primary_dosha == "vata":
                    recommendations = [
                        "Regular routines and consistent meal times",
                        "Warm, nourishing, and grounding foods",
                        "Oil massage (abhyanga) with sesame oil",
                        "Meditation and stress reduction practices",
                        "Avoid cold, dry, and raw foods"
                    ]
                elif primary_dosha == "pitta":
                    recommendations = [
                        "Cooling foods and herbs",
                        "Avoid excessive heat and spicy foods",
                        "Regular moderate exercise",
                        "Cooling practices like moonlight walks",
                        "Manage stress and anger"
                    ]
                else:  # kapha
                    recommendations = [
                        "Regular stimulating exercise",
                        "Warm, light, and spiced foods",
                        "Dry brushing and invigorating practices",
                        "Varied routines to prevent stagnation",
                        "Avoid heavy, oily, and cold foods"
                    ]
                
                html += "<ul class='recommendation-list'>"
                for rec in recommendations:
                    html += f"<li>{rec}</li>"
                html += "</ul>"
        
        # General recommendations based on spot analysis
        if enhanced_results and enhanced_results.get('spots'):
            spot_count = len(enhanced_results['spots'])
            html += "<h3>Detoxification Recommendations</h3>"
            html += "<ul class='recommendation-list'>"
            
            if spot_count > 10:
                html += "<li>High spot count detected - consider comprehensive detoxification protocols</li>"
                html += "<li>Focus on liver and kidney support with herbs and supplements</li>"
                html += "<li>Reduce inflammatory foods and environmental toxin exposure</li>"
                html += "<li>Consider professional guidance for detox program</li>"
            elif spot_count > 5:
                html += "<li>Moderate spot count - maintain healthy elimination pathways</li>"
                html += "<li>Support digestive health with probiotics and fiber</li>"
                html += "<li>Regular exercise and hydration</li>"
            else:
                html += "<li>Low spot count indicates good elimination - maintain current practices</li>"
                html += "<li>Continue healthy lifestyle habits</li>"
            
            html += "</ul>"
        
        html += "</div>"
        
        # Disclaimer
        html += """
        <div class="disclaimer">
            <strong>Important Disclaimer:</strong> This iris analysis is provided for educational and informational purposes only. 
            It is not intended to diagnose, treat, cure, or prevent any disease. 
            Please consult with a qualified healthcare provider for medical advice and before making 
            any changes to your health regimen. This analysis should be used as a complementary tool 
            alongside conventional medical care.
        </div>
        """
        
        html += """
        </body>
        </html>
        """
        
        # Clean up temporary files
        self._cleanup_temp_files()
        
        return html


# Convenience function for backward compatibility
def generate_enhanced_report(zone_results, enhanced_results=None, user_info=None, format='pdf'):
    """
    Convenience function to generate enhanced iris reports.
    
    Args:
        zone_results: Results from iris zone analysis
        enhanced_results: Results from enhanced spot detection (optional)
        user_info: User information (optional)
        format: 'pdf' or 'html'
        
    Returns:
        Report as bytes (PDF) or string (HTML)
    """
    generator = EnhancedIrisReportGenerator()
    
    if format.lower() == 'pdf':
        return generator.generate_enhanced_pdf_report(zone_results, enhanced_results, user_info)
    else:
        return generator.generate_enhanced_html_report(zone_results, enhanced_results, user_info)
