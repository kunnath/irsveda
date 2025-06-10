#!/usr/bin/env python
"""
Test script for Dosha Quantification Model

This script tests the dosha quantification model with a sample image
and prints out the detailed analysis results.
"""

import os
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dosha_quantification_model import DoshaQuantificationModel

def plot_dosha_distribution(dosha_profile):
    """Plot the dosha distribution as a pie chart."""
    plt.figure(figsize=(8, 5))
    
    labels = ['Vata', 'Pitta', 'Kapha']
    sizes = [
        dosha_profile.get("vata_percentage", 33.3),
        dosha_profile.get("pitta_percentage", 33.3),
        dosha_profile.get("kapha_percentage", 33.3)
    ]
    colors = ['#AED6F1', '#F5B041', '#A9DFBF']
    
    plt.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%', 
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title('Dosha Distribution', fontsize=14, pad=20)
    
    # Save the plot
    plt.savefig('dosha_distribution.png', bbox_inches='tight', dpi=300)
    print(f"Dosha distribution chart saved as 'dosha_distribution.png'")
    
    # Display if in interactive mode
    if plt.isinteractive():
        plt.show()
    plt.close()

def plot_organ_health(organ_assessments):
    """Plot the organ health assessment as a horizontal bar chart."""
    plt.figure(figsize=(10, 6))
    
    # Extract organ data
    organ_names = []
    health_scores = []
    colors = []
    
    for assessment in organ_assessments:
        organ_name = assessment.get("organ_name", "unknown")
        health_score = assessment.get("health_score", 0)
        warning_level = assessment.get("warning_level", "normal")
        
        organ_names.append(organ_name.capitalize())
        health_scores.append(health_score)
        
        # Determine color based on warning level
        if warning_level == "critical":
            color = "#d9534f"  # Red
        elif warning_level == "warning":
            color = "#f0ad4e"  # Orange
        elif warning_level == "attention":
            color = "#5bc0de"  # Blue
        else:
            color = "#5cb85c"  # Green
        
        colors.append(color)
    
    # Sort by health score
    sorted_indices = np.argsort(health_scores)
    sorted_organ_names = [organ_names[i] for i in sorted_indices]
    sorted_health_scores = [health_scores[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = plt.barh(sorted_organ_names, sorted_health_scores, color=sorted_colors)
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{sorted_health_scores[i]:.2f}",
            va='center'
        )
    
    # Add labels and title
    plt.xlabel('Health Score (0-1)')
    plt.title('Organ Health Assessment')
    plt.xlim(0, 1.1)  # Set x-axis limit
    
    # Save the plot
    plt.savefig('organ_health.png', bbox_inches='tight', dpi=300)
    print(f"Organ health chart saved as 'organ_health.png'")
    
    # Display if in interactive mode
    if plt.isinteractive():
        plt.show()
    plt.close()

def main():
    """Run the test script."""
    # Initialize the model
    model = DoshaQuantificationModel()
    
    # Get sample image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a sample image in the sample_images directory
        sample_dir = Path("sample_images")
        if sample_dir.exists() and sample_dir.is_dir():
            sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if sample_files:
                image_path = str(sample_files[0])
                print(f"Using sample image: {image_path}")
            else:
                print("No sample images found. Please provide an image path.")
                return
        else:
            print("Sample images directory not found. Please provide an image path.")
            return
    
    # Ensure image exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Process the image
    print(f"Processing iris image: {image_path}")
    report = model.process_iris_image(image_path)
    
    if "error" in report:
        print(f"Error: {report['error']}")
        return
    
    # Display results
    print("\n" + "="*50)
    print("DOSHA QUANTIFICATION RESULTS")
    print("="*50)
    
    # Dosha Profile
    print("\n=== DOSHA PROFILE ===")
    dosha_profile = report['dosha_profile']
    primary_dosha = dosha_profile.get('primary_dosha', 'unknown').capitalize()
    secondary_dosha = dosha_profile.get('secondary_dosha', None)
    if secondary_dosha:
        secondary_dosha = secondary_dosha.capitalize()
    
    print(f"Primary Dosha: {primary_dosha}")
    if secondary_dosha:
        print(f"Secondary Dosha: {secondary_dosha}")
    
    print(f"Vata: {dosha_profile.get('vata_percentage', 0):.1f}%")
    print(f"Pitta: {dosha_profile.get('pitta_percentage', 0):.1f}%")
    print(f"Kapha: {dosha_profile.get('kapha_percentage', 0):.1f}%")
    
    print(f"Balance Score: {dosha_profile.get('balance_score', 0):.2f}")
    
    imbalance_type = dosha_profile.get('imbalance_type', None)
    if imbalance_type:
        print(f"Imbalance Type: {imbalance_type.capitalize()}")
    
    # Plot dosha distribution
    plot_dosha_distribution(dosha_profile)
    
    # Metabolic Indicators
    print("\n=== METABOLIC INDICATORS ===")
    metabolic_indicators = report.get('metabolic_indicators', {})
    
    for key, value in metabolic_indicators.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Organ Health
    print("\n=== ORGAN HEALTH ===")
    print("Concerning Organs:")
    
    concerning_organs = report.get('concerning_organs', [])
    for organ in concerning_organs:
        print(f"- {organ.get('organ_name', 'unknown').capitalize()}: " +
              f"{organ.get('health_score', 0):.2f} ({organ.get('warning_level', 'normal').upper()})")
    
    # Plot organ health
    organ_assessments = report.get('organ_assessments', [])
    if organ_assessments:
        plot_organ_health(organ_assessments)
    
    # Key Recommendations
    print("\n=== KEY RECOMMENDATIONS ===")
    key_recommendations = report.get('key_recommendations', [])
    
    for i, rec in enumerate(key_recommendations, 1):
        print(f"{i}. {rec}")
    
    # Overall Health
    print(f"\nOverall Health Score: {report.get('overall_health_score', 0):.2f} - " +
          f"{report.get('health_status', 'unknown').upper()}")
    
    # Save full report as JSON
    output_file = "dosha_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to {output_file}")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
