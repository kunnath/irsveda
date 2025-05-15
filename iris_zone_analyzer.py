import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from scipy.ndimage import gaussian_filter

class IrisZoneAnalyzer:
    """Class for analyzing iris zones and mapping them to ayurvedic concepts."""
    
    def __init__(self):
        """Initialize the iris zone analyzer."""
        # Define standard zone mapping
        self.zones = {
            "pupillary_zone": {
                "name": "Pupillary Zone",
                "radius_ratio": 0.2,
                "color": (255, 215, 0, 128),  # Gold with transparency
                "ayurvedic_systems": ["Digestive Tract"],
                "description": "Represents the digestive system and intestines."
            },
            "ciliary_zone": {
                "name": "Ciliary Zone",
                "radius_ratio": 0.5,
                "color": (46, 139, 87, 128),  # Sea Green with transparency
                "ayurvedic_systems": ["Respiratory System", "Circulatory System"],
                "description": "Corresponds to the respiratory and circulatory systems, including heart and lungs."
            },
            "autonomic_nerve_wreath": {
                "name": "Autonomic Nerve Wreath",
                "radius_ratio": 0.65,
                "color": (106, 90, 205, 128),  # Slate Blue with transparency
                "ayurvedic_systems": ["Nervous System"],
                "description": "Represents the nervous system and neural activity."
            },
            "middle_zone": {
                "name": "Middle Zone",
                "radius_ratio": 0.8,
                "color": (70, 130, 180, 128),  # Steel Blue with transparency
                "ayurvedic_systems": ["Musculoskeletal System", "Endocrine System"],
                "description": "Associated with muscles, bones, and glandular functions."
            },
            "peripheral_zone": {
                "name": "Peripheral Zone",
                "radius_ratio": 0.95,
                "color": (128, 0, 128, 128),  # Purple with transparency
                "ayurvedic_systems": ["Skin", "Lymphatic System"],
                "description": "Relates to the skin, lymphatics, and extremities."
            }
        }
        
        # Ayurvedic dosha mapping (Vata, Pitta, Kapha)
        self.dosha_mapping = {
            "vata": {
                "color_range": [(100, 100, 100), (150, 150, 150)],  # Grayish colors
                "characteristics": ["Dry", "Light", "Cold", "Rough", "Subtle", "Mobile"],
                "description": "Associated with air and ether elements, controlling movement and nervous system functions."
            },
            "pitta": {
                "color_range": [(0, 0, 100), (50, 50, 255)],  # Reddish/yellowish colors
                "characteristics": ["Hot", "Sharp", "Light", "Liquid", "Spreading", "Oily"],
                "description": "Associated with fire and water elements, governing metabolism and transformation."
            },
            "kapha": {
                "color_range": [(0, 100, 0), (100, 255, 100)],  # Whitish/greenish colors
                "characteristics": ["Heavy", "Slow", "Cold", "Oily", "Smooth", "Dense", "Soft", "Static"],
                "description": "Associated with earth and water elements, maintaining structure and fluid balance."
            }
        }

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess the iris image for analysis.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply some basic enhancements for better iris visibility
        # Histogram equalization on value channel
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Gentle Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img
    
    def detect_iris_and_pupil(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Detect iris and pupil boundaries using Hough Circle Transform.
        
        Args:
            image: Input eye image
            
        Returns:
            Boundaries information and visualization image
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # In a production system, we'd use sophisticated eye detection methods
        # For this demo, we'll approximate with Hough circle detection
        
        # Try to detect the pupil first (smaller, darker circle)
        min_radius = min(width, height) // 12
        max_radius = min(width, height) // 6
        
        try:
            # Preprocess for pupil detection
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            pupil_circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(width, height),
                param1=50,
                param2=30,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            if pupil_circles is not None and len(pupil_circles[0]) > 0:
                # Get the most prominent pupil circle
                pupil_x, pupil_y, pupil_r = pupil_circles[0][0].astype(int)
            else:
                # Fallback if pupil not detected
                pupil_x, pupil_y = width // 2, height // 2
                pupil_r = min(width, height) // 10
            
            # Now detect the iris (larger circle concentric to pupil)
            min_radius = pupil_r * 2
            max_radius = min(width, height) // 2
            
            iris_circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min(width, height),
                param1=100,
                param2=30,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            if iris_circles is not None and len(iris_circles[0]) > 0:
                # Get the most prominent iris circle
                iris_x, iris_y, iris_r = iris_circles[0][0].astype(int)
            else:
                # Fallback if iris not detected
                iris_x, iris_y = pupil_x, pupil_y
                iris_r = min(width, height) // 3
            
            # Use the pupil center for better accuracy
            iris_x, iris_y = pupil_x, pupil_y
            
        except Exception as e:
            print(f"Error detecting iris and pupil: {e}")
            # Fallback to educated guesses
            pupil_x, pupil_y = width // 2, height // 2
            pupil_r = min(width, height) // 10
            iris_x, iris_y = pupil_x, pupil_y
            iris_r = min(width, height) // 3
            
        # Create visualization with detected circles
        vis_image = image.copy()
        # Draw pupil
        cv2.circle(vis_image, (pupil_x, pupil_y), pupil_r, (0, 0, 255), 2)
        # Draw iris
        cv2.circle(vis_image, (iris_x, iris_y), iris_r, (255, 0, 0), 2)
        
        boundaries = {
            "pupil": {
                "center": (pupil_x, pupil_y),
                "radius": pupil_r
            },
            "iris": {
                "center": (iris_x, iris_y),
                "radius": iris_r
            }
        }
        
        return boundaries, vis_image
    
    def generate_zone_map(self, image: np.ndarray, boundaries: Dict[str, Any]) -> np.ndarray:
        """
        Generate a visual map of iris zones.
        
        Args:
            image: Input eye image
            boundaries: Detected iris and pupil boundaries
            
        Returns:
            Image with overlaid zone map
        """
        # Convert to PIL for easier text drawing
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Get iris boundaries
        iris_center = boundaries["iris"]["center"]
        iris_radius = boundaries["iris"]["radius"]
        pupil_radius = boundaries["pupil"]["radius"]
        
        # Draw each zone
        for zone_name, zone_info in self.zones.items():
            # Calculate this zone's radius
            outer_radius = int(pupil_radius + (iris_radius - pupil_radius) * zone_info["radius_ratio"])
            inner_radius = int(pupil_radius) if zone_name == "pupillary_zone" else int(
                pupil_radius + (iris_radius - pupil_radius) * 
                self.get_previous_zone_ratio(zone_name)
            )
            
            # Draw the zone as a filled ring
            for r in range(inner_radius, outer_radius + 1):
                draw.ellipse(
                    [(iris_center[0] - r, iris_center[1] - r), 
                     (iris_center[0] + r, iris_center[1] + r)],
                    outline=zone_info["color"],
                    fill=None
                )
                
        return np.array(pil_image)
    
    def get_previous_zone_ratio(self, zone_name: str) -> float:
        """Get the radius ratio of the zone before the current one."""
        zones_list = list(self.zones.keys())
        if zone_name not in zones_list or zones_list.index(zone_name) == 0:
            return 0.0
        previous_zone = zones_list[zones_list.index(zone_name) - 1]
        return self.zones[previous_zone]["radius_ratio"]
        
    def analyze_iris_zones(self, image: np.ndarray, boundaries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the texture and color characteristics of each iris zone.
        
        Args:
            image: Input eye image
            boundaries: Detected iris and pupil boundaries
            
        Returns:
            Analysis of each zone with ayurvedic interpretations
        """
        # Extract features from each zone
        zones_analysis = {}
        
        # Create HSV version for better color analysis
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Get iris boundaries
        iris_center = boundaries["iris"]["center"]
        iris_radius = boundaries["iris"]["radius"]
        pupil_radius = boundaries["pupil"]["radius"]
        
        # Create masks and analyze each zone
        for zone_name, zone_info in self.zones.items():
            # Calculate this zone's radius
            outer_radius = int(pupil_radius + (iris_radius - pupil_radius) * zone_info["radius_ratio"])
            inner_radius = int(pupil_radius) if zone_name == "pupillary_zone" else int(
                pupil_radius + (iris_radius - pupil_radius) * 
                self.get_previous_zone_ratio(zone_name)
            )
            
            # Create a mask for this zone
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, iris_center, outer_radius, 255, -1)
            cv2.circle(mask, iris_center, inner_radius, 0, -1)
            
            # Apply mask to both RGB and HSV images
            masked_rgb = cv2.bitwise_and(image, image, mask=mask)
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
            
            # Extract features (simplistic approach for demonstration)
            # In a real system, you would use more sophisticated feature extraction
            
            # Only consider non-zero pixels
            non_zero_mask = mask > 0
            
            # Color features (from HSV)
            if np.sum(non_zero_mask) > 0:
                mean_hue = np.mean(masked_hsv[:,:,0][non_zero_mask])
                mean_saturation = np.mean(masked_hsv[:,:,1][non_zero_mask])
                mean_value = np.mean(masked_hsv[:,:,2][non_zero_mask])
                
                # Texture features (simple approximation using gradients)
                gray_zone = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
                grad_x = cv2.Sobel(gray_zone, cv2.CV_16S, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_zone, cv2.CV_16S, 0, 1, ksize=3)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                texture_intensity = np.mean(grad[non_zero_mask])
                
                # Count "anomalies" - simple approach using local maxima
                smoothed = gaussian_filter(gray_zone, sigma=1)
                threshold_value = np.percentile(smoothed[non_zero_mask], 90)
                anomalies = np.sum(smoothed > threshold_value) / np.sum(non_zero_mask)
                
                # Dosha mapping (based on color)
                rgb_mean = [
                    np.mean(masked_rgb[:,:,0][non_zero_mask]),
                    np.mean(masked_rgb[:,:,1][non_zero_mask]),
                    np.mean(masked_rgb[:,:,2][non_zero_mask])
                ]
                dominant_dosha = self.map_color_to_dosha(rgb_mean)
            else:
                mean_hue = 0
                mean_saturation = 0
                mean_value = 0
                texture_intensity = 0
                anomalies = 0
                dominant_dosha = "unknown"
            
            # Store analysis results
            zones_analysis[zone_name] = {
                "name": zone_info["name"],
                "color_features": {
                    "hue": float(mean_hue),
                    "saturation": float(mean_saturation),
                    "value": float(mean_value)
                },
                "texture_features": {
                    "intensity": float(texture_intensity),
                    "anomaly_ratio": float(anomalies)
                },
                "ayurvedic_mapping": {
                    "systems": zone_info["ayurvedic_systems"],
                    "dominant_dosha": dominant_dosha,
                    "dosha_qualities": self.dosha_mapping.get(dominant_dosha, {}).get("characteristics", []),
                    "description": zone_info["description"]
                },
                "health_indication": self.interpret_health_state(texture_intensity, anomalies)
            }
            
        return zones_analysis
    
    def map_color_to_dosha(self, rgb_color: List[float]) -> str:
        """Map RGB color to dominant dosha."""
        min_distance = float('inf')
        dominant_dosha = "unknown"
        
        for dosha, info in self.dosha_mapping.items():
            color_range = info["color_range"]
            # Calculate distance to middle of color range
            mid_color = [
                (color_range[0][0] + color_range[1][0]) / 2,
                (color_range[0][1] + color_range[1][1]) / 2,
                (color_range[0][2] + color_range[1][2]) / 2
            ]
            
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_color, mid_color)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                dominant_dosha = dosha
                
        return dominant_dosha
    
    def interpret_health_state(self, texture_intensity: float, anomaly_ratio: float) -> Dict[str, Any]:
        """
        Interpret texture features to determine health state.
        
        This is a simplified interpretation for demonstration purposes.
        Real iridology would use much more nuanced analysis.
        """
        # Normalize values
        norm_texture = min(texture_intensity / 50, 1.0)
        
        if anomaly_ratio > 0.2:
            condition = "compromised"
            confidence = min(anomaly_ratio * 1.5, 1.0)
            suggestion = "Consider detailed examination by an Ayurvedic practitioner"
        elif norm_texture > 0.7:
            condition = "stressed"
            confidence = norm_texture
            suggestion = "May benefit from balancing practices"
        else:
            condition = "normal"
            confidence = 1.0 - norm_texture
            suggestion = "Maintain current health practices"
            
        return {
            "condition": condition,
            "confidence": float(confidence),
            "suggestion": suggestion
        }
    
    def process_iris_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an iris image and generate a complete zone analysis.
        
        Args:
            image_path: Path to the iris image
            
        Returns:
            Complete analysis with zone mapping and health interpretations
        """
        try:
            # Preprocess image
            preprocessed = self.preprocess_image(image_path)
            
            # Detect iris and pupil
            boundaries, boundary_image = self.detect_iris_and_pupil(preprocessed)
            
            # Generate zone map visualization
            zone_map = self.generate_zone_map(preprocessed, boundaries)
            
            # Analyze each zone
            zones_analysis = self.analyze_iris_zones(preprocessed, boundaries)
            
            # Create overall health summary
            health_states = [zone["health_indication"]["condition"] for zone in zones_analysis.values()]
            if "compromised" in health_states:
                overall_health = "needs attention"
            elif "stressed" in health_states:
                overall_health = "moderately balanced"
            else:
                overall_health = "well balanced"
                
            # Calculate dosha balance percentages
            dosha_counts = {"vata": 0, "pitta": 0, "kapha": 0, "unknown": 0}
            for zone in zones_analysis.values():
                dosha = zone["ayurvedic_mapping"]["dominant_dosha"]
                dosha_counts[dosha] = dosha_counts.get(dosha, 0) + 1
                
            total_zones = len(zones_analysis)
            dosha_balance = {
                dosha: count / total_zones
                for dosha, count in dosha_counts.items()
                if dosha != "unknown"
            }
            
            # Complete results
            return {
                "original_image": preprocessed,
                "boundary_image": boundary_image,
                "zone_map": zone_map,
                "zones_analysis": zones_analysis,
                "health_summary": {
                    "overall_health": overall_health,
                    "dosha_balance": dosha_balance
                }
            }
            
        except Exception as e:
            return {
                "error": f"Error analyzing iris image: {str(e)}"
            }
