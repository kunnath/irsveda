#!/usr/bin/env python3
"""
Test script for Enhanced Iris Report Generator
Tests the enhanced report generation features with mock data.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Add the project directory to the path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def create_mock_zone_results():
    """Create mock zone analysis results for testing."""
    # Create a mock iris image
    mock_image = Image.new('RGB', (300, 300), color='lightblue')
    
    # Create mock zone results
    zone_results = {
        "original_image": mock_image,
        "zone_map": mock_image,
        "boundary_image": mock_image,
        "health_summary": {
            "overall_health": "good",
            "dosha_balance": {
                "vata": 0.35,
                "pitta": 0.45,
                "kapha": 0.20
            }
        },
        "zones_analysis": {
            "zone_1": {
                "name": "Brain & Nervous System",
                "health_indication": {
                    "condition": "normal",
                    "confidence": 0.85,
                    "suggestion": "Maintain current healthy practices for nervous system support"
                },
                "ayurvedic_mapping": {
                    "systems": ["Nervous System", "Brain"],
                    "description": "Controls cognitive function and neural coordination",
                    "dominant_dosha": "vata",
                    "dosha_qualities": ["light", "mobile", "clear"]
                }
            },
            "zone_2": {
                "name": "Digestive System",
                "health_indication": {
                    "condition": "stressed",
                    "confidence": 0.75,
                    "suggestion": "Support digestive fire with warming spices and regular meals"
                },
                "ayurvedic_mapping": {
                    "systems": ["Digestive System", "Stomach"],
                    "description": "Governs digestion and metabolism",
                    "dominant_dosha": "pitta",
                    "dosha_qualities": ["hot", "sharp", "oily"]
                }
            },
            "zone_3": {
                "name": "Lymphatic System",
                "health_indication": {
                    "condition": "compromised",
                    "confidence": 0.70,
                    "suggestion": "Support lymphatic drainage with dry brushing and movement"
                },
                "ayurvedic_mapping": {
                    "systems": ["Lymphatic System", "Immunity"],
                    "description": "Manages immune function and fluid balance",
                    "dominant_dosha": "kapha",
                    "dosha_qualities": ["heavy", "slow", "stable"]
                }
            }
        }
    }
    
    return zone_results

def create_mock_enhanced_results():
    """Create mock enhanced analysis results for testing."""
    # Create a mock annotated image (base64 encoded)
    mock_annotated = Image.new('RGB', (300, 300), color='lightgreen')
    buf = BytesIO()
    mock_annotated.save(buf, format='PNG')
    buf.seek(0)
    annotated_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    enhanced_results = {
        "annotated_image": annotated_base64,
        "total_spots": 12,
        "detection_methods_used": ["edge", "dark_spot", "blob", "adaptive"],
        "spots": [
            {
                "segment_id": 1,
                "detection_method": "edge",
                "area": 45,
                "iris_zone": "Zone 1",
                "corresponding_organ": "Brain",
                "angle_degrees": 45.5,
                "distance_from_center_pct": 65.2,
                "visibility": 8.5,
                "avg_brightness": 125.3
            },
            {
                "segment_id": 2,
                "detection_method": "dark_spot",
                "area": 32,
                "iris_zone": "Zone 2",
                "corresponding_organ": "Stomach",
                "angle_degrees": 120.0,
                "distance_from_center_pct": 72.1,
                "visibility": 9.2,
                "avg_brightness": 98.7
            },
            {
                "segment_id": 3,
                "detection_method": "blob",
                "area": 28,
                "iris_zone": "Zone 3",
                "corresponding_organ": "Liver",
                "angle_degrees": 200.5,
                "distance_from_center_pct": 58.9,
                "visibility": 7.8,
                "avg_brightness": 145.6
            },
            {
                "segment_id": 4,
                "detection_method": "adaptive",
                "area": 51,
                "iris_zone": "Zone 4",
                "corresponding_organ": "Kidneys",
                "angle_degrees": 315.2,
                "distance_from_center_pct": 81.3,
                "visibility": 8.9,
                "avg_brightness": 110.4
            },
            {
                "segment_id": 5,
                "detection_method": "edge",
                "area": 38,
                "iris_zone": "Zone 2",
                "corresponding_organ": "Pancreas",
                "angle_degrees": 90.0,
                "distance_from_center_pct": 69.7,
                "visibility": 8.1,
                "avg_brightness": 132.8
            }
        ],
        "analysis_summary": {
            "organs": {
                "Brain": 1,
                "Stomach": 2,
                "Liver": 1,
                "Kidneys": 1,
                "Pancreas": 1
            }
        }
    }
    
    return enhanced_results

def test_enhanced_report_generator():
    """Test the enhanced iris report generator."""
    print("Testing Enhanced Iris Report Generator...")
    print("=" * 50)
    
    try:
        # Import the enhanced report generator
        from enhanced_iris_report_generator import EnhancedIrisReportGenerator
        print("✅ Enhanced report generator imported successfully")
        
        # Create test data
        zone_results = create_mock_zone_results()
        enhanced_results = create_mock_enhanced_results()
        user_info = {
            "Name": "Test User",
            "Age": "35",
            "Gender": "Female",
            "Email": "test@example.com",
            "Health Concerns": "General wellness assessment"
        }
        
        print("✅ Mock data created successfully")
        
        # Initialize the generator
        generator = EnhancedIrisReportGenerator()
        print("✅ Enhanced report generator initialized")
        
        # Test PDF generation
        print("\n📄 Testing PDF Report Generation...")
        try:
            pdf_bytes = generator.generate_enhanced_pdf_report(
                zone_results, 
                enhanced_results, 
                user_info
            )
            
            if pdf_bytes:
                # Save the PDF for inspection
                output_path = project_dir / "test_enhanced_iris_report.pdf"
                with open(output_path, 'wb') as f:
                    f.write(pdf_bytes)
                print(f"✅ PDF report generated successfully: {output_path}")
                print(f"   Report size: {len(pdf_bytes):,} bytes")
            else:
                print("❌ PDF generation returned None")
                
        except Exception as e:
            print(f"❌ PDF generation failed: {str(e)}")
        
        # Test HTML generation
        print("\n🌐 Testing HTML Report Generation...")
        try:
            html_report = generator.generate_enhanced_html_report(
                zone_results,
                enhanced_results,
                user_info
            )
            
            if html_report:
                # Save the HTML for inspection
                output_path = project_dir / "test_enhanced_iris_report.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                print(f"✅ HTML report generated successfully: {output_path}")
                print(f"   Report size: {len(html_report):,} characters")
            else:
                print("❌ HTML generation returned empty string")
                
        except Exception as e:
            print(f"❌ HTML generation failed: {str(e)}")
        
        # Test without enhanced results (backward compatibility)
        print("\n🔄 Testing Backward Compatibility (without enhanced results)...")
        try:
            pdf_basic = generator.generate_enhanced_pdf_report(zone_results, None, user_info)
            html_basic = generator.generate_enhanced_html_report(zone_results, None, user_info)
            
            if pdf_basic and html_basic:
                print("✅ Backward compatibility test passed")
            else:
                print("❌ Backward compatibility test failed")
                
        except Exception as e:
            print(f"❌ Backward compatibility test failed: {str(e)}")
        
        print("\n" + "=" * 50)
        print("✅ Enhanced Iris Report Generator testing completed!")
        print("\nGenerated test files:")
        print("  - test_enhanced_iris_report.pdf")
        print("  - test_enhanced_iris_report.html")
        print("\nYou can open these files to review the report quality.")
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced report generator: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install fpdf2 matplotlib pillow pandas")
        
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_report_generator()
