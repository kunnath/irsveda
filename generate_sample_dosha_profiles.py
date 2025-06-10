#!/usr/bin/env python
"""
Generate sample dosha profile data for testing

This script generates sample dosha profiles based on preset parameters
to help with testing the dosha quantification model.
"""

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the datasets directory exists
os.makedirs("datasets/dosha_profiles", exist_ok=True)

# Define preset dosha profiles for testing
PRESET_PROFILES = [
    {
        "name": "vata_dominant",
        "description": "Vata dominant profile (thin build, active, creative)",
        "dosha_distribution": {
            "vata": 65,
            "pitta": 20,
            "kapha": 15
        },
        "characteristics": {
            "build": "thin",
            "energy": "variable",
            "temperament": "creative",
            "digestion": "irregular"
        }
    },
    {
        "name": "pitta_dominant",
        "description": "Pitta dominant profile (medium build, focused, intense)",
        "dosha_distribution": {
            "vata": 20,
            "pitta": 65,
            "kapha": 15
        },
        "characteristics": {
            "build": "medium",
            "energy": "intense",
            "temperament": "focused",
            "digestion": "strong"
        }
    },
    {
        "name": "kapha_dominant",
        "description": "Kapha dominant profile (solid build, calm, methodical)",
        "dosha_distribution": {
            "vata": 15,
            "pitta": 20,
            "kapha": 65
        },
        "characteristics": {
            "build": "heavy",
            "energy": "steady",
            "temperament": "calm",
            "digestion": "slow"
        }
    },
    {
        "name": "vata_pitta",
        "description": "Vata-Pitta dual profile (light build, energetic, creative)",
        "dosha_distribution": {
            "vata": 45,
            "pitta": 45,
            "kapha": 10
        },
        "characteristics": {
            "build": "light to medium",
            "energy": "high",
            "temperament": "creative and focused",
            "digestion": "variable"
        }
    },
    {
        "name": "pitta_kapha",
        "description": "Pitta-Kapha dual profile (athletic build, determined, methodical)",
        "dosha_distribution": {
            "vata": 10,
            "pitta": 45,
            "kapha": 45
        },
        "characteristics": {
            "build": "medium to heavy",
            "energy": "moderate to high",
            "temperament": "determined",
            "digestion": "good"
        }
    },
    {
        "name": "vata_kapha",
        "description": "Vata-Kapha dual profile (variable build, mixed energy, creative but methodical)",
        "dosha_distribution": {
            "vata": 45,
            "pitta": 10,
            "kapha": 45
        },
        "characteristics": {
            "build": "thin to heavy",
            "energy": "variable",
            "temperament": "creative but methodical",
            "digestion": "variable"
        }
    },
    {
        "name": "tridoshic",
        "description": "Balanced tridoshic profile (well-proportioned, balanced energy and temperament)",
        "dosha_distribution": {
            "vata": 33,
            "pitta": 33,
            "kapha": 34
        },
        "characteristics": {
            "build": "proportionate",
            "energy": "balanced",
            "temperament": "adaptable",
            "digestion": "good"
        }
    }
]

def generate_dosha_profile(preset_name=None):
    """
    Generate a sample dosha profile based on a preset or randomly.
    
    Args:
        preset_name: Optional name of preset profile to use
        
    Returns:
        Dictionary with dosha profile data
    """
    # Select a preset if specified
    if preset_name:
        preset = next((p for p in PRESET_PROFILES if p["name"] == preset_name), None)
        if preset:
            return create_profile_from_preset(preset)
    
    # Random selection if no preset specified or not found
    preset = random.choice(PRESET_PROFILES)
    return create_profile_from_preset(preset)

def create_profile_from_preset(preset):
    """
    Create a dosha profile from a preset with some randomization.
    
    Args:
        preset: Preset profile dictionary
        
    Returns:
        Dictionary with dosha profile data
    """
    # Add some randomness to the distribution while maintaining dominance
    base_distribution = preset["dosha_distribution"]
    
    # Calculate random variations (±5% max)
    variation = random.uniform(-5, 5)
    
    # Adjust vata and pitta with the variation
    vata = max(5, min(90, base_distribution["vata"] + variation))
    pitta = max(5, min(90, base_distribution["pitta"] - variation/2))
    
    # Ensure they sum to 100%
    kapha = 100 - vata - pitta
    kapha = max(5, min(90, kapha))
    
    # Recalculate to ensure exactly 100%
    total = vata + pitta + kapha
    vata = (vata / total) * 100
    pitta = (pitta / total) * 100
    kapha = (kapha / total) * 100
    
    # Create profile
    profile = {
        "profile_type": preset["name"],
        "description": preset["description"],
        "dosha_distribution": {
            "vata_percentage": vata,
            "pitta_percentage": pitta,
            "kapha_percentage": kapha
        },
        "primary_dosha": determine_primary_dosha(vata, pitta, kapha),
        "secondary_dosha": determine_secondary_dosha(vata, pitta, kapha),
        "characteristics": preset["characteristics"],
        "balance_score": calculate_balance_score(vata, pitta, kapha)
    }
    
    # Determine imbalance type
    profile["imbalance_type"] = determine_imbalance_type(profile)
    
    return profile

def determine_primary_dosha(vata, pitta, kapha):
    """Determine the primary dosha based on percentages."""
    max_val = max(vata, pitta, kapha)
    if max_val == vata:
        return "vata"
    elif max_val == pitta:
        return "pitta"
    else:
        return "kapha"

def determine_secondary_dosha(vata, pitta, kapha):
    """Determine the secondary dosha based on percentages."""
    values = [("vata", vata), ("pitta", pitta), ("kapha", kapha)]
    sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
    
    # If the top two are very close (within 5%), consider it dual
    if abs(sorted_values[0][1] - sorted_values[1][1]) < 5:
        return sorted_values[1][0]
    
    # If primary is very dominant, may not have a clear secondary
    if sorted_values[0][1] > 60 and sorted_values[1][1] < 25:
        return None
    
    return sorted_values[1][0]

def calculate_balance_score(vata, pitta, kapha):
    """
    Calculate balance score (0-1) based on how evenly distributed the doshas are.
    1.0 means perfect balance (33.3% each).
    """
    ideal = 33.33
    deviations = [
        abs(vata - ideal),
        abs(pitta - ideal),
        abs(kapha - ideal)
    ]
    
    # Calculate average deviation and normalize
    avg_deviation = sum(deviations) / 3
    max_deviation = 100 - ideal  # Maximum possible deviation from ideal
    
    # Convert to a score where 1.0 is perfect balance
    balance_score = 1 - (avg_deviation / max_deviation)
    return round(balance_score, 2)

def determine_imbalance_type(profile):
    """Determine if there's a significant imbalance and what type."""
    vata = profile["dosha_distribution"]["vata_percentage"]
    pitta = profile["dosha_distribution"]["pitta_percentage"]
    kapha = profile["dosha_distribution"]["kapha_percentage"]
    
    # No significant imbalance if balance score is high
    if profile["balance_score"] > 0.75:
        return None
    
    # Check for excesses (>50%)
    if vata > 50:
        return "vata excess"
    elif pitta > 50:
        return "pitta excess"
    elif kapha > 50:
        return "kapha excess"
    
    # Check for deficiencies (<15%)
    if vata < 15:
        return "vata deficiency"
    elif pitta < 15:
        return "pitta deficiency"
    elif kapha < 15:
        return "kapha deficiency"
    
    return None

def plot_dosha_distribution(profile, save_path=None):
    """Generate a pie chart of dosha distribution."""
    plt.figure(figsize=(8, 6))
    
    # Get distribution data
    distribution = profile["dosha_distribution"]
    labels = ['Vata', 'Pitta', 'Kapha']
    sizes = [
        distribution["vata_percentage"],
        distribution["pitta_percentage"],
        distribution["kapha_percentage"]
    ]
    colors = ['#AED6F1', '#F5B041', '#A9DFBF']
    
    # Create pie chart
    plt.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%', 
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    
    # Add title with profile type
    profile_type = profile["profile_type"].replace('_', '-').title()
    plt.title(f'{profile_type} Dosha Profile', fontsize=14, pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close()

def main():
    """Generate and save sample dosha profiles."""
    print("Generating sample dosha profiles...")
    
    # Create a profile for each preset
    for preset in PRESET_PROFILES:
        preset_name = preset["name"]
        profile = generate_dosha_profile(preset_name)
        
        # Save as JSON
        json_path = f"datasets/dosha_profiles/{preset_name}_profile.json"
        with open(json_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        # Generate and save chart
        chart_path = f"datasets/dosha_profiles/{preset_name}_chart.png"
        plot_dosha_distribution(profile, chart_path)
        
        print(f"Generated {preset_name} profile: {profile['primary_dosha'].capitalize()} dominant")
    
    print(f"\nAll profiles saved to datasets/dosha_profiles/")

if __name__ == "__main__":
    main()
