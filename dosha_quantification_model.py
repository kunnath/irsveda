"""
Dosha Quantification Model

This module implements a comprehensive model that quantifies Vata, Pitta, and Kapha doshas
from iris analysis and links these measurements to specific organ health indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import cv2
import logging
from dataclasses import dataclass
import json
import os
from sklearn.preprocessing import MinMaxScaler
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dosha_quantification")

# Define dosha types as enum for type safety
class DoshaType(str, Enum):
    VATA = "vata"
    PITTA = "pitta"
    KAPHA = "kapha"

@dataclass
class DoshaProfile:
    """Represents a complete dosha profile with quantified values."""
    vata_score: float
    pitta_score: float
    kapha_score: float
    primary_dosha: DoshaType
    secondary_dosha: Optional[DoshaType]
    vata_percentage: float
    pitta_percentage: float
    kapha_percentage: float
    balance_score: float  # 0-1 scale, 1 being perfectly balanced
    imbalance_type: Optional[str]  # e.g., "vata excess", "pitta deficiency"

@dataclass
class MetabolicIndicators:
    """Metabolic indicators derived from iris analysis."""
    basal_metabolic_rate: float  # kcal/day
    serum_lipid: float  # mg/dL
    triglycerides: float  # mg/dL
    crp_level: float  # mg/L (inflammatory marker)
    gut_diversity: float  # 0-1 scale
    enzyme_activity: float  # relative scale
    appetite_score: float  # 1-10 scale
    metabolism_variability: float  # standard deviation (0.1-1.0)

@dataclass
class OrganHealthAssessment:
    """Assessment of organ health based on iris and dosha analysis."""
    organ_name: str
    health_score: float  # 0-1 scale, 1 being optimal health
    dosha_influence: Dict[str, float]  # Influence of each dosha on this organ
    iris_indicators: List[Dict[str, Any]]  # Specific iris signs related to this organ
    recommendations: List[str]  # Health recommendations
    warning_level: str  # "normal", "attention", "warning", "critical"

class DoshaQuantificationModel:
    """
    Model for quantifying Vata, Pitta, and Kapha doshas from iris analysis
    and connecting to organ health indicators.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the dosha quantification model.
        
        Args:
            model_path: Optional path to model parameters
        """
        # Load iris-dosha correlation reference data
        self.iris_dosha_mapping = self._load_iris_dosha_mapping()
        
        # Load organ-dosha correlation reference data
        self.organ_dosha_mapping = self._load_organ_dosha_mapping()
        
        # Initialize scalers for normalizing features
        self.feature_scaler = MinMaxScaler()
        
        # Load model parameters if provided
        self.model_params = {}
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    self.model_params = json.load(f)
                logger.info(f"Loaded model parameters from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model parameters: {e}")
    
    def _load_iris_dosha_mapping(self) -> Dict[str, Any]:
        """
        Load reference data for iris-dosha correlations.
        This would typically be loaded from a JSON file, but for now we'll define it inline.
        """
        # This mapping connects iris features to dosha characteristics
        return {
            "iris_colors": {
                "blue": {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
                "light_blue": {"vata": 0.8, "pitta": 0.15, "kapha": 0.05},
                "gray": {"vata": 0.6, "pitta": 0.3, "kapha": 0.1},
                "green": {"vata": 0.3, "pitta": 0.6, "kapha": 0.1},
                "light_brown": {"vata": 0.2, "pitta": 0.7, "kapha": 0.1},
                "brown": {"vata": 0.1, "pitta": 0.3, "kapha": 0.6},
                "dark_brown": {"vata": 0.05, "pitta": 0.25, "kapha": 0.7}
            },
            "pupil_size": {
                "small": {"vata": 0.2, "pitta": 0.3, "kapha": 0.5},
                "medium": {"vata": 0.3, "pitta": 0.5, "kapha": 0.2},
                "large": {"vata": 0.6, "pitta": 0.3, "kapha": 0.1}
            },
            "iris_fiber_density": {
                "sparse": {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
                "medium": {"vata": 0.3, "pitta": 0.6, "kapha": 0.1},
                "dense": {"vata": 0.1, "pitta": 0.2, "kapha": 0.7}
            },
            "iris_fiber_pattern": {
                "straight": {"vata": 0.1, "pitta": 0.6, "kapha": 0.3},
                "wavy": {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
                "mixed": {"vata": 0.4, "pitta": 0.4, "kapha": 0.2},
                "circular": {"vata": 0.2, "pitta": 0.2, "kapha": 0.6}
            },
            "iris_spots": {
                "few": {"vata": 0.3, "pitta": 0.3, "kapha": 0.4},
                "moderate": {"vata": 0.3, "pitta": 0.6, "kapha": 0.1},
                "many": {"vata": 0.6, "pitta": 0.3, "kapha": 0.1}
            },
            "iris_pigmentation": {
                "light": {"vata": 0.6, "pitta": 0.3, "kapha": 0.1},
                "medium": {"vata": 0.3, "pitta": 0.5, "kapha": 0.2},
                "heavy": {"vata": 0.1, "pitta": 0.3, "kapha": 0.6}
            },
            "collarette_type": {
                "thin": {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
                "normal": {"vata": 0.3, "pitta": 0.4, "kapha": 0.3},
                "thick": {"vata": 0.1, "pitta": 0.3, "kapha": 0.6}
            },
            "sclera_color": {
                "clear_white": {"vata": 0.3, "pitta": 0.4, "kapha": 0.3},
                "yellowish": {"vata": 0.1, "pitta": 0.8, "kapha": 0.1},
                "reddish": {"vata": 0.1, "pitta": 0.8, "kapha": 0.1},
                "bluish": {"vata": 0.7, "pitta": 0.1, "kapha": 0.2}
            }
        }
    
    def _load_organ_dosha_mapping(self) -> Dict[str, Any]:
        """
        Load reference data for organ-dosha correlations.
        This would typically be loaded from a JSON file, but for now we'll define it inline.
        """
        return {
            "liver": {
                "primary_dosha": "pitta",
                "dosha_influence": {"vata": 0.2, "pitta": 0.7, "kapha": 0.1},
                "iris_zone": "lower_right_quadrant",
                "metabolic_indicators": ["serum_lipid", "enzyme_activity"]
            },
            "stomach": {
                "primary_dosha": "pitta",
                "dosha_influence": {"vata": 0.3, "pitta": 0.6, "kapha": 0.1},
                "iris_zone": "lower_inner_quadrant",
                "metabolic_indicators": ["appetite_score", "enzyme_activity"]
            },
            "intestines": {
                "primary_dosha": "vata",
                "dosha_influence": {"vata": 0.6, "pitta": 0.3, "kapha": 0.1},
                "iris_zone": "lower_quadrants",
                "metabolic_indicators": ["gut_diversity", "metabolism_variability"]
            },
            "lungs": {
                "primary_dosha": "kapha",
                "dosha_influence": {"vata": 0.3, "pitta": 0.1, "kapha": 0.6},
                "iris_zone": "upper_left_quadrant",
                "metabolic_indicators": ["basal_metabolic_rate"]
            },
            "heart": {
                "primary_dosha": "pitta",
                "dosha_influence": {"vata": 0.2, "pitta": 0.5, "kapha": 0.3},
                "iris_zone": "upper_inner_quadrant_left",
                "metabolic_indicators": ["triglycerides", "crp_level"]
            },
            "kidneys": {
                "primary_dosha": "vata",
                "dosha_influence": {"vata": 0.6, "pitta": 0.2, "kapha": 0.2},
                "iris_zone": "lower_middle_quadrants",
                "metabolic_indicators": ["crp_level", "metabolism_variability"]
            },
            "brain": {
                "primary_dosha": "vata",
                "dosha_influence": {"vata": 0.7, "pitta": 0.2, "kapha": 0.1},
                "iris_zone": "upper_most_region",
                "metabolic_indicators": ["basal_metabolic_rate", "metabolism_variability"]
            },
            "thyroid": {
                "primary_dosha": "pitta",
                "dosha_influence": {"vata": 0.3, "pitta": 0.6, "kapha": 0.1},
                "iris_zone": "throat_region",
                "metabolic_indicators": ["basal_metabolic_rate", "metabolism_variability"]
            },
            "pancreas": {
                "primary_dosha": "kapha",
                "dosha_influence": {"vata": 0.1, "pitta": 0.4, "kapha": 0.5},
                "iris_zone": "lower_left_quadrant",
                "metabolic_indicators": ["serum_lipid", "triglycerides", "enzyme_activity"]
            },
            "lymphatic_system": {
                "primary_dosha": "kapha",
                "dosha_influence": {"vata": 0.1, "pitta": 0.1, "kapha": 0.8},
                "iris_zone": "all_quadrants_lymphatic_ring",
                "metabolic_indicators": ["crp_level", "gut_diversity"]
            }
        }
    
    def analyze_iris_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract iris features from an image for dosha analysis.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Dictionary of extracted iris features
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert to RGB for better color analysis
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract features (in a real implementation, this would use sophisticated 
            # computer vision techniques)
            
            # For now, use a simplified approach for demonstration
            features = self._extract_iris_features_demo(img_rgb)
            
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing iris image: {e}")
            return {"error": str(e)}
    
    def _extract_iris_features_demo(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Demo version of feature extraction - in a real implementation,
        this would use sophisticated computer vision techniques.
        
        Args:
            img_rgb: RGB image array
            
        Returns:
            Dictionary of extracted features
        """
        # Get image dimensions
        height, width, _ = img_rgb.shape
        
        # Assume the iris is roughly in the center of the image
        center_x, center_y = width // 2, height // 2
        
        # Create simple masks for different regions (this is simplified)
        # In a real implementation, proper iris segmentation would be performed
        
        # Calculate average color in the iris region (simplified)
        iris_region = img_rgb[center_y-50:center_y+50, center_x-50:center_x+50]
        avg_color = np.mean(iris_region, axis=(0, 1))
        
        # Determine iris color category (simplified)
        r, g, b = avg_color
        
        # Simple color classification logic
        if b > r and b > g:
            if b > 150:
                iris_color = "light_blue"
            else:
                iris_color = "blue"
        elif g > r and g > b:
            iris_color = "green"
        elif r > 150 and g > 100 and b < 100:
            iris_color = "light_brown"
        elif r > 100 and g > 70 and b < 70:
            iris_color = "brown"
        elif r < 100 and g < 100 and b < 100:
            iris_color = "dark_brown"
        else:
            iris_color = "gray"
        
        # Mock pupil detection (simplified)
        # In a real implementation, proper pupil detection would be performed
        pupil_region = img_rgb[center_y-20:center_y+20, center_x-20:center_x+20]
        darkness = 255 - np.mean(pupil_region)
        
        if darkness > 200:
            pupil_size = "large"
        elif darkness > 150:
            pupil_size = "medium"
        else:
            pupil_size = "small"
        
        # Mock fiber pattern detection (simplified)
        # In a real implementation, texture analysis would be performed
        iris_texture = img_rgb[center_y-40:center_y+40, center_x-40:center_x+40]
        texture_variance = np.var(iris_texture)
        
        if texture_variance > 1500:
            fiber_pattern = "wavy"
        elif texture_variance > 1000:
            fiber_pattern = "mixed"
        elif texture_variance > 500:
            fiber_pattern = "straight"
        else:
            fiber_pattern = "circular"
        
        # Mock fiber density calculation (simplified)
        gray_iris = cv2.cvtColor(iris_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_iris, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 30:
            fiber_density = "dense"
        elif edge_density > 15:
            fiber_density = "medium"
        else:
            fiber_density = "sparse"
        
        # Mock spot detection (simplified)
        # In a real implementation, proper spot/marking detection would be performed
        spots_count = int(texture_variance / 100)  # Just a mock value
        
        if spots_count > 15:
            spots = "many"
        elif spots_count > 7:
            spots = "moderate"
        else:
            spots = "few"
        
        # Return extracted features
        return {
            "iris_color": iris_color,
            "pupil_size": pupil_size,
            "fiber_pattern": fiber_pattern,
            "fiber_density": fiber_density,
            "spots": spots,
            "pigmentation": "medium",  # Mock value
            "collarette_type": "normal",  # Mock value
            "sclera_color": "clear_white",  # Mock value
            "raw_features": {
                "avg_color": avg_color.tolist(),
                "texture_variance": float(texture_variance),
                "edge_density": float(edge_density),
                "spots_count": spots_count
            }
        }
    
    def quantify_doshas(self, iris_features: Dict[str, Any]) -> DoshaProfile:
        """
        Quantify the three doshas based on iris features.
        
        Args:
            iris_features: Dictionary of iris features
            
        Returns:
            DoshaProfile object with quantified values
        """
        if "error" in iris_features:
            raise ValueError(f"Cannot quantify doshas: {iris_features['error']}")
        
        # Initialize scores
        vata_score = 0.0
        pitta_score = 0.0
        kapha_score = 0.0
        
        # Calculate weighted scores based on iris features
        for feature_name, feature_value in iris_features.items():
            if feature_name in self.iris_dosha_mapping and feature_value in self.iris_dosha_mapping[feature_name]:
                dosha_weights = self.iris_dosha_mapping[feature_name][feature_value]
                vata_score += dosha_weights["vata"]
                pitta_score += dosha_weights["pitta"]
                kapha_score += dosha_weights["kapha"]
        
        # Normalize scores
        total_score = vata_score + pitta_score + kapha_score
        if total_score > 0:
            vata_percentage = (vata_score / total_score) * 100
            pitta_percentage = (pitta_score / total_score) * 100
            kapha_percentage = (kapha_score / total_score) * 100
        else:
            vata_percentage = pitta_percentage = kapha_percentage = 33.33
        
        # Determine primary and secondary doshas
        dosha_scores = {
            "vata": vata_score,
            "pitta": pitta_score,
            "kapha": kapha_score
        }
        
        sorted_doshas = sorted(dosha_scores.items(), key=lambda x: x[1], reverse=True)
        primary_dosha = DoshaType(sorted_doshas[0][0])
        secondary_dosha = DoshaType(sorted_doshas[1][0]) if len(sorted_doshas) > 1 else None
        
        # Calculate balance score (how balanced the doshas are)
        ideal_distribution = 33.33  # Equal distribution of all three doshas
        deviations = [
            abs(vata_percentage - ideal_distribution),
            abs(pitta_percentage - ideal_distribution),
            abs(kapha_percentage - ideal_distribution)
        ]
        max_possible_deviation = 100 - ideal_distribution  # If one dosha is 100% and others are 0%
        avg_deviation = sum(deviations) / 3
        balance_score = 1 - (avg_deviation / max_possible_deviation)
        
        # Determine imbalance type
        imbalance_type = None
        if balance_score < 0.7:  # Threshold for considering an imbalance
            if primary_dosha == DoshaType.VATA and vata_percentage > 50:
                imbalance_type = "vata excess"
            elif primary_dosha == DoshaType.PITTA and pitta_percentage > 50:
                imbalance_type = "pitta excess"
            elif primary_dosha == DoshaType.KAPHA and kapha_percentage > 50:
                imbalance_type = "kapha excess"
            elif vata_percentage < 20:
                imbalance_type = "vata deficiency"
            elif pitta_percentage < 20:
                imbalance_type = "pitta deficiency"
            elif kapha_percentage < 20:
                imbalance_type = "kapha deficiency"
        
        # Create dosha profile
        dosha_profile = DoshaProfile(
            vata_score=vata_score,
            pitta_score=pitta_score,
            kapha_score=kapha_score,
            primary_dosha=primary_dosha,
            secondary_dosha=secondary_dosha,
            vata_percentage=vata_percentage,
            pitta_percentage=pitta_percentage,
            kapha_percentage=kapha_percentage,
            balance_score=balance_score,
            imbalance_type=imbalance_type
        )
        
        return dosha_profile
    
    def calculate_metabolic_indicators(self, 
                                      iris_features: Dict[str, Any], 
                                      dosha_profile: DoshaProfile) -> MetabolicIndicators:
        """
        Calculate metabolic indicators based on iris features and dosha profile.
        
        Args:
            iris_features: Dictionary of iris features
            dosha_profile: Quantified dosha profile
            
        Returns:
            MetabolicIndicators object
        """
        # Note: In a real implementation, this would use sophisticated algorithms
        # Here we provide a simplified calculation for demonstration
        
        # Calculate base values based on dosha percentages
        vata_pct = dosha_profile.vata_percentage / 100
        pitta_pct = dosha_profile.pitta_percentage / 100
        kapha_pct = dosha_profile.kapha_percentage / 100
        
        # Base metabolic rate (higher in pitta, lower in kapha)
        # Normal BMR range: 1200-2000 kcal/day
        bmr_base = 1200 + (800 * pitta_pct) - (400 * kapha_pct)
        
        # Serum lipids (higher in kapha, lower in vata)
        # Normal range: 150-200 mg/dL
        serum_lipid = 150 + (50 * kapha_pct) - (30 * vata_pct)
        
        # Triglycerides (higher in kapha, lower in vata)
        # Normal range: 50-150 mg/dL
        triglycerides = 50 + (100 * kapha_pct) - (30 * vata_pct)
        
        # CRP - inflammation marker (higher in pitta, lower in kapha)
        # Normal range: 0-3 mg/L
        crp_level = 0.5 + (2.5 * pitta_pct) - (0.5 * kapha_pct)
        
        # Gut diversity (better in balanced doshas, worse with high vata)
        # Range: 0-1
        gut_diversity = 0.7 * dosha_profile.balance_score - (0.2 * vata_pct)
        gut_diversity = max(0.1, min(1.0, gut_diversity))  # Clamp to valid range
        
        # Enzyme activity (higher in pitta, lower in kapha)
        # Range: 0-1
        enzyme_activity = 0.3 + (0.6 * pitta_pct) - (0.3 * kapha_pct)
        enzyme_activity = max(0.1, min(1.0, enzyme_activity))  # Clamp to valid range
        
        # Appetite score (higher in pitta, lower in kapha and vata)
        # Range: 1-10
        appetite_score = 3 + (7 * pitta_pct) - (2 * kapha_pct) - (1 * vata_pct)
        appetite_score = max(1, min(10, appetite_score))  # Clamp to valid range
        
        # Metabolism variability (higher in vata, lower in kapha)
        # Range: 0.1-1.0
        metabolism_var = 0.1 + (0.9 * vata_pct) - (0.3 * kapha_pct)
        metabolism_var = max(0.1, min(1.0, metabolism_var))  # Clamp to valid range
        
        # Adjust based on iris features
        if iris_features["pupil_size"] == "large":
            bmr_base *= 1.1  # Larger pupil suggests higher metabolism
            metabolism_var *= 1.2
        elif iris_features["pupil_size"] == "small":
            bmr_base *= 0.9
            metabolism_var *= 0.8
        
        if iris_features["fiber_density"] == "dense":
            serum_lipid *= 0.9  # Denser fibers suggest better lipid handling
            triglycerides *= 0.9
        elif iris_features["fiber_density"] == "sparse":
            serum_lipid *= 1.1
            triglycerides *= 1.1
        
        if iris_features["spots"] == "many":
            crp_level *= 1.3  # More spots suggest more inflammation
            gut_diversity *= 0.8
        
        # Return metabolic indicators
        return MetabolicIndicators(
            basal_metabolic_rate=round(bmr_base, 1),
            serum_lipid=round(serum_lipid, 1),
            triglycerides=round(triglycerides, 1),
            crp_level=round(crp_level, 2),
            gut_diversity=round(gut_diversity, 2),
            enzyme_activity=round(enzyme_activity, 2),
            appetite_score=round(appetite_score, 1),
            metabolism_variability=round(metabolism_var, 2)
        )
    
    def assess_organ_health(self, 
                           iris_features: Dict[str, Any],
                           dosha_profile: DoshaProfile,
                           metabolic_indicators: MetabolicIndicators) -> List[OrganHealthAssessment]:
        """
        Assess organ health based on iris features, dosha profile, and metabolic indicators.
        
        Args:
            iris_features: Dictionary of iris features
            dosha_profile: Quantified dosha profile
            metabolic_indicators: Calculated metabolic indicators
            
        Returns:
            List of OrganHealthAssessment objects
        """
        organ_assessments = []
        
        # Get simplified mock iris zones from features
        # In a real implementation, this would use sophisticated iris zone mapping
        iris_zones = self._extract_iris_zones_demo(iris_features)
        
        # Assess each organ
        for organ_name, organ_data in self.organ_dosha_mapping.items():
            # Get organ's primary dosha and influence percentages
            primary_dosha = organ_data["primary_dosha"]
            dosha_influence = organ_data["dosha_influence"]
            
            # Check if organ's zone has any abnormalities in the iris
            zone_name = organ_data["iris_zone"]
            zone_issues = iris_zones.get(zone_name, {"health_score": 0.9, "issues": []})
            
            # Calculate base health score
            # 1. Start with the zone's health score from iris analysis
            health_score = zone_issues["health_score"]
            
            # 2. Adjust based on dosha balance/imbalance for this organ
            # If the organ's primary dosha matches the person's primary dosha, check for excess
            if primary_dosha == dosha_profile.primary_dosha.value:
                if dosha_profile.imbalance_type and "excess" in dosha_profile.imbalance_type:
                    health_score *= 0.9  # Reduce score for excess of organ's primary dosha
            
            # 3. Adjust based on metabolic indicators relevant to this organ
            relevant_indicators = organ_data.get("metabolic_indicators", [])
            for indicator in relevant_indicators:
                if indicator == "basal_metabolic_rate":
                    # Check if BMR is in healthy range for this organ
                    bmr = metabolic_indicators.basal_metabolic_rate
                    if primary_dosha == "pitta" and bmr < 1400:
                        health_score *= 0.95  # Too low for pitta organ
                    elif primary_dosha == "kapha" and bmr > 1800:
                        health_score *= 0.95  # Too high for kapha organ
                
                elif indicator == "serum_lipid":
                    # Check if serum lipid is in healthy range
                    lipid = metabolic_indicators.serum_lipid
                    if lipid > 200:
                        health_score *= 0.9  # High cholesterol affects organ health
                
                elif indicator == "crp_level":
                    # Check inflammation levels
                    crp = metabolic_indicators.crp_level
                    if crp > 3.0:
                        health_score *= 0.85  # High inflammation significantly affects health
                
                elif indicator == "gut_diversity":
                    # Check gut diversity for digestive organs
                    diversity = metabolic_indicators.gut_diversity
                    if diversity < 0.5 and organ_name in ["intestines", "stomach"]:
                        health_score *= 0.9
                
                # Add similar checks for other indicators
            
            # Generate recommendations based on health score and dominant doshas
            recommendations = self._generate_recommendations(
                organ_name, 
                health_score, 
                dosha_profile, 
                iris_zones.get(zone_name, {}).get("issues", [])
            )
            
            # Determine warning level
            warning_level = "normal"
            if health_score < 0.7:
                warning_level = "critical"
            elif health_score < 0.8:
                warning_level = "warning"
            elif health_score < 0.9:
                warning_level = "attention"
            
            # Create organ assessment
            assessment = OrganHealthAssessment(
                organ_name=organ_name,
                health_score=round(health_score, 2),
                dosha_influence=dosha_influence,
                iris_indicators=iris_zones.get(zone_name, {}).get("issues", []),
                recommendations=recommendations,
                warning_level=warning_level
            )
            
            organ_assessments.append(assessment)
        
        # Sort by health score (ascending) so most concerning organs are first
        organ_assessments.sort(key=lambda x: x.health_score)
        
        return organ_assessments
    
    def _extract_iris_zones_demo(self, iris_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Demo implementation of iris zone extraction.
        In a real implementation, this would analyze specific zones of the iris.
        
        Args:
            iris_features: Dictionary of iris features
            
        Returns:
            Dictionary mapping iris zones to health indicators
        """
        # This is a simplified mock implementation
        # In a real system, this would use sophisticated zone mapping and analysis
        
        zones = {}
        
        # Generate some mock zone data based on overall iris features
        if iris_features["spots"] == "many":
            # Add issues to some zones
            zones["lower_right_quadrant"] = {
                "health_score": 0.7,
                "issues": [
                    {"type": "spot", "severity": "moderate", "description": "Dark spot in liver zone"}
                ]
            }
            
            zones["lower_quadrants"] = {
                "health_score": 0.8,
                "issues": [
                    {"type": "radii", "severity": "mild", "description": "Radii lines in intestinal zone"}
                ]
            }
        
        if iris_features["fiber_pattern"] == "wavy":
            zones["upper_inner_quadrant_left"] = {
                "health_score": 0.85,
                "issues": [
                    {"type": "fiber_disruption", "severity": "mild", "description": "Wavy fiber pattern in heart zone"}
                ]
            }
        
        if iris_features["pupil_size"] == "large":
            zones["upper_most_region"] = {
                "health_score": 0.75,
                "issues": [
                    {"type": "nerve_tension", "severity": "moderate", "description": "Indications of nervous system tension"}
                ]
            }
        
        # Generate default values for zones not explicitly defined
        all_zones = [
            "lower_right_quadrant", "lower_left_quadrant", "upper_right_quadrant", "upper_left_quadrant",
            "lower_inner_quadrant", "upper_inner_quadrant_left", "upper_inner_quadrant_right",
            "lower_middle_quadrants", "upper_most_region", "throat_region", "all_quadrants_lymphatic_ring"
        ]
        
        for zone in all_zones:
            if zone not in zones:
                # Generate a slightly random health score between 0.85 and 1.0
                random_score = 0.85 + (np.random.random() * 0.15)
                zones[zone] = {
                    "health_score": round(random_score, 2),
                    "issues": []
                }
        
        return zones
    
    def _generate_recommendations(self, 
                                organ_name: str, 
                                health_score: float,
                                dosha_profile: DoshaProfile,
                                iris_issues: List[Dict[str, Any]]) -> List[str]:
        """
        Generate health recommendations based on organ analysis.
        
        Args:
            organ_name: Name of the organ
            health_score: Calculated health score (0-1)
            dosha_profile: The person's dosha profile
            iris_issues: Issues detected in the iris for this organ
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Base recommendations by organ
        organ_base_recs = {
            "liver": [
                "Limit processed foods and alcohol consumption",
                "Include bitter greens like dandelion and arugula in your diet",
                "Consider liver-supporting herbs like milk thistle and turmeric"
            ],
            "stomach": [
                "Eat smaller, more frequent meals",
                "Avoid eating late at night",
                "Include ginger or mint tea to support digestion"
            ],
            "intestines": [
                "Increase dietary fiber intake",
                "Stay well-hydrated throughout the day",
                "Consider probiotics to support gut health"
            ],
            "lungs": [
                "Practice deep breathing exercises daily",
                "Ensure adequate ventilation in living spaces",
                "Consider steam inhalation with eucalyptus for respiratory support"
            ],
            "heart": [
                "Maintain regular cardiovascular exercise",
                "Include omega-3 rich foods in your diet",
                "Practice stress reduction techniques"
            ],
            "kidneys": [
                "Ensure adequate water intake",
                "Reduce sodium consumption",
                "Consider cranberry or dandelion tea for kidney support"
            ],
            "brain": [
                "Ensure quality sleep",
                "Practice meditation or mindfulness",
                "Include omega-3 fatty acids and antioxidant-rich foods in diet"
            ],
            "thyroid": [
                "Ensure adequate iodine in diet",
                "Include selenium-rich foods like Brazil nuts",
                "Manage stress levels with relaxation techniques"
            ],
            "pancreas": [
                "Limit refined sugar consumption",
                "Include cinnamon in your diet",
                "Eat meals at regular times"
            ],
            "lymphatic_system": [
                "Practice regular physical activity",
                "Consider dry brushing to stimulate lymphatic flow",
                "Stay well-hydrated throughout the day"
            ]
        }
        
        # Add base recommendations for the organ
        base_recs = organ_base_recs.get(organ_name, ["Support overall health with a balanced diet and lifestyle"])
        
        # Select recommendations based on health score
        if health_score < 0.7:
            # Critical condition - add all recommendations
            recommendations.extend(base_recs)
            recommendations.append(f"Consider consulting with a healthcare provider about your {organ_name}")
        elif health_score < 0.8:
            # Warning condition - add most recommendations
            recommendations.extend(base_recs[:2])
            recommendations.append(f"Monitor your {organ_name} health closely")
        elif health_score < 0.9:
            # Attention needed - add top recommendation
            recommendations.append(base_recs[0])
        
        # Add dosha-specific recommendations
        primary_dosha = dosha_profile.primary_dosha.value
        
        if primary_dosha == "vata":
            if organ_name in ["intestines", "brain", "kidneys"]:
                recommendations.append("Follow a vata-pacifying diet with warm, cooked, and oily foods")
                recommendations.append("Maintain regular daily routines to balance vata")
        
        elif primary_dosha == "pitta":
            if organ_name in ["liver", "stomach", "heart", "thyroid"]:
                recommendations.append("Follow a pitta-pacifying diet with cooling, sweet, and bitter foods")
                recommendations.append("Avoid excessive heat and intense exercise during hot weather")
        
        elif primary_dosha == "kapha":
            if organ_name in ["lungs", "pancreas", "lymphatic_system"]:
                recommendations.append("Follow a kapha-pacifying diet with light, dry, and spicy foods")
                recommendations.append("Include regular vigorous exercise in your routine")
        
        # Add recommendations based on specific iris issues
        for issue in iris_issues:
            if issue["type"] == "spot" and issue["severity"] in ["moderate", "severe"]:
                recommendations.append("Consider a gentle detoxification protocol")
            
            elif issue["type"] == "radii" and issue["severity"] in ["moderate", "severe"]:
                recommendations.append("Support connective tissue health with collagen-rich foods")
            
            elif issue["type"] == "fiber_disruption":
                recommendations.append("Support tissue regeneration with adequate protein and zinc-rich foods")
            
            elif issue["type"] == "nerve_tension":
                recommendations.append("Incorporate stress-reduction practices like yoga or meditation")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def generate_comprehensive_report(self, 
                                    iris_features: Dict[str, Any],
                                    dosha_profile: DoshaProfile,
                                    metabolic_indicators: MetabolicIndicators,
                                    organ_assessments: List[OrganHealthAssessment]) -> Dict[str, Any]:
        """
        Generate a comprehensive health report based on all analyses.
        
        Args:
            iris_features: Extracted iris features
            dosha_profile: Quantified dosha profile
            metabolic_indicators: Calculated metabolic indicators
            organ_assessments: List of organ health assessments
            
        Returns:
            Dictionary with comprehensive report data
        """
        # Calculate overall health score (weighted average of organ scores)
        organ_scores = [assessment.health_score for assessment in organ_assessments]
        overall_health_score = sum(organ_scores) / len(organ_scores) if organ_scores else 0.0
        
        # Adjust overall score based on dosha balance
        overall_health_score = overall_health_score * 0.8 + dosha_profile.balance_score * 0.2
        
        # Determine health status
        if overall_health_score >= 0.9:
            health_status = "excellent"
        elif overall_health_score >= 0.8:
            health_status = "good"
        elif overall_health_score >= 0.7:
            health_status = "fair"
        else:
            health_status = "needs attention"
        
        # Get most concerning organs (lowest health scores)
        concerning_organs = sorted(organ_assessments, key=lambda x: x.health_score)[:3]
        
        # Compile key recommendations across all organs, prioritizing more critical ones
        all_recommendations = []
        for assessment in concerning_organs:
            all_recommendations.extend(assessment.recommendations)
        
        # Remove duplicates while preserving order
        key_recommendations = []
        for rec in all_recommendations:
            if rec not in key_recommendations:
                key_recommendations.append(rec)
        
        # Limit to top recommendations
        key_recommendations = key_recommendations[:7]
        
        # Compile report
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_health_score": round(overall_health_score, 2),
            "health_status": health_status,
            "dosha_profile": {
                "primary_dosha": dosha_profile.primary_dosha.value,
                "secondary_dosha": dosha_profile.secondary_dosha.value if dosha_profile.secondary_dosha else None,
                "vata_percentage": round(dosha_profile.vata_percentage, 1),
                "pitta_percentage": round(dosha_profile.pitta_percentage, 1),
                "kapha_percentage": round(dosha_profile.kapha_percentage, 1),
                "balance_score": round(dosha_profile.balance_score, 2),
                "imbalance_type": dosha_profile.imbalance_type
            },
            "metabolic_indicators": {
                "basal_metabolic_rate": metabolic_indicators.basal_metabolic_rate,
                "serum_lipid": metabolic_indicators.serum_lipid,
                "triglycerides": metabolic_indicators.triglycerides,
                "crp_level": metabolic_indicators.crp_level,
                "gut_diversity": metabolic_indicators.gut_diversity,
                "enzyme_activity": metabolic_indicators.enzyme_activity,
                "appetite_score": metabolic_indicators.appetite_score,
                "metabolism_variability": metabolic_indicators.metabolism_variability
            },
            "iris_features": iris_features,
            "organ_assessments": [
                {
                    "organ_name": assessment.organ_name,
                    "health_score": assessment.health_score,
                    "warning_level": assessment.warning_level,
                    "dosha_influence": assessment.dosha_influence,
                    "issues": assessment.iris_indicators,
                    "recommendations": assessment.recommendations
                }
                for assessment in organ_assessments
            ],
            "key_recommendations": key_recommendations,
            "concerning_organs": [
                {
                    "organ_name": assessment.organ_name,
                    "health_score": assessment.health_score,
                    "warning_level": assessment.warning_level
                }
                for assessment in concerning_organs
            ]
        }
        
        return report
    
    def process_iris_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete end-to-end processing of an iris image.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Comprehensive analysis report
        """
        try:
            # Step 1: Extract iris features
            iris_features = self.analyze_iris_image(image_path)
            if "error" in iris_features:
                return {"error": iris_features["error"]}
            
            # Step 2: Quantify doshas
            dosha_profile = self.quantify_doshas(iris_features)
            
            # Step 3: Calculate metabolic indicators
            metabolic_indicators = self.calculate_metabolic_indicators(iris_features, dosha_profile)
            
            # Step 4: Assess organ health
            organ_assessments = self.assess_organ_health(iris_features, dosha_profile, metabolic_indicators)
            
            # Step 5: Generate comprehensive report
            report = self.generate_comprehensive_report(
                iris_features, dosha_profile, metabolic_indicators, organ_assessments
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error processing iris image: {e}")
            return {"error": str(e)}


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dosha_quantification_model.py <iris_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model = DoshaQuantificationModel()
    
    print(f"Processing iris image: {image_path}")
    report = model.process_iris_image(image_path)
    
    if "error" in report:
        print(f"Error: {report['error']}")
    else:
        print("\n=== DOSHA PROFILE ===")
        print(f"Primary Dosha: {report['dosha_profile']['primary_dosha'].capitalize()}")
        print(f"Secondary Dosha: {report['dosha_profile']['secondary_dosha'].capitalize() if report['dosha_profile']['secondary_dosha'] else 'None'}")
        print(f"Vata: {report['dosha_profile']['vata_percentage']:.1f}%")
        print(f"Pitta: {report['dosha_profile']['pitta_percentage']:.1f}%")
        print(f"Kapha: {report['dosha_profile']['kapha_percentage']:.1f}%")
        print(f"Balance Score: {report['dosha_profile']['balance_score']:.2f}")
        if report['dosha_profile']['imbalance_type']:
            print(f"Imbalance Type: {report['dosha_profile']['imbalance_type'].capitalize()}")
        
        print("\n=== METABOLIC INDICATORS ===")
        for key, value in report['metabolic_indicators'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n=== ORGAN HEALTH ===")
        print("Concerning Organs:")
        for organ in report['concerning_organs']:
            print(f"- {organ['organ_name'].capitalize()}: {organ['health_score']:.2f} ({organ['warning_level'].upper()})")
        
        print("\n=== KEY RECOMMENDATIONS ===")
        for i, rec in enumerate(report['key_recommendations'], 1):
            print(f"{i}. {rec}")
        
        print(f"\nOverall Health Score: {report['overall_health_score']:.2f} - {report['health_status'].upper()}")
