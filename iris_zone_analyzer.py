import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import math
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation

class IrisZoneAnalyzer:
    """Class for analyzing iris zones and mapping them to ayurvedic concepts."""
    
    def __init__(self):
        """Initialize the iris zone analyzer."""
        # Define standard zone mapping with enhanced detail
        self.zones = {
            "pupillary_zone": {
                "name": "Pupillary Zone (Digestive Ring)",
                "radius_ratio": 0.2,
                "color": (255, 215, 0, 128),  # Gold with transparency
                "ayurvedic_systems": ["Digestive Tract", "Stomach", "Intestines"],
                "description": "Represents the digestive system and intestines. The innermost ring around the pupil.",
                "detailed_mapping": {
                    "upper": "Upper digestive tract, esophagus, and stomach",
                    "lower": "Lower digestive tract and intestines",
                    "left": "Left side digestive organs including spleen",
                    "right": "Right side digestive organs including liver and gallbladder"
                },
                "physiological_indicators": {
                    "radial_lines": "Indicates stress in digestive system",
                    "dark_spots": "May indicate toxin accumulation",
                    "white_spots": "Could indicate inflammation or injury"
                }
            },
            "ciliary_zone": {
                "name": "Ciliary Zone",
                "radius_ratio": 0.5,
                "color": (46, 139, 87, 128),  # Sea Green with transparency
                "ayurvedic_systems": ["Respiratory System", "Circulatory System"],
                "description": "Corresponds to the respiratory and circulatory systems, including heart and lungs.",
                "detailed_mapping": {
                    "upper": "Respiratory organs - lungs, bronchi",
                    "lower": "Circulatory organs - heart, major blood vessels",
                    "left": "Left lung and heart",
                    "right": "Right lung and heart chambers"
                },
                "physiological_indicators": {
                    "small_dark_spots": "Potential circulation issues",
                    "cloudy_areas": "May indicate respiratory congestion", 
                    "yellow_tint": "Could indicate excess mucus production"
                }
            },
            "autonomic_nerve_wreath": {
                "name": "Autonomic Nerve Wreath",
                "radius_ratio": 0.65,
                "color": (106, 90, 205, 128),  # Slate Blue with transparency
                "ayurvedic_systems": ["Nervous System", "Brain Function"],
                "description": "Represents the nervous system and neural activity. Appears as a zigzag or scalloped pattern.",
                "detailed_mapping": {
                    "shape": "Regular shape indicates balanced nervous system",
                    "irregular": "Irregularities suggest nervous system stress",
                    "breaks": "Breaks in the wreath may indicate nervous exhaustion"
                },
                "physiological_indicators": {
                    "clearly_defined": "Well-functioning autonomic response",
                    "fuzzy_appearance": "Potential autonomic dysfunction",
                    "color_variation": "Different states of nervous system function"
                }
            },
            "middle_zone": {
                "name": "Middle Zone",
                "radius_ratio": 0.8,
                "color": (70, 130, 180, 128),  # Steel Blue with transparency
                "ayurvedic_systems": ["Musculoskeletal System", "Endocrine System", "Metabolic Functions"],
                "description": "Associated with muscles, bones, and glandular functions.",
                "detailed_mapping": {
                    "upper": "Endocrine glands - thyroid, pituitary",
                    "lower": "Lower metabolic and reproductive systems",
                    "outer_ring": "Muscular tissues and structural elements",
                    "inner_ring": "Glandular and metabolic functions"
                },
                "physiological_indicators": {
                    "dense_fibers": "Strong constitution and good structural integrity",
                    "pale_areas": "Potential mineral deficiencies",
                    "discolorations": "Hormonal imbalances or metabolic issues"
                }
            },
            "peripheral_zone": {
                "name": "Peripheral Zone",
                "radius_ratio": 0.95,
                "color": (128, 0, 128, 128),  # Purple with transparency
                "ayurvedic_systems": ["Skin", "Lymphatic System", "Extremities"],
                "description": "Relates to the skin, lymphatics, and extremities. The outermost area of the iris.",
                "detailed_mapping": {
                    "outer_edge": "Skin and exterior tissues",
                    "inner_boundary": "Lymphatic system",
                    "peripheral_structures": "Relates to extremities - hands, feet, skin"
                },
                "physiological_indicators": {
                    "clear_demarcation": "Good lymphatic circulation",
                    "fuzzy_boundary": "Potential skin or lymphatic congestion",
                    "spots_or_marks": "May indicate issues in specific regions of extremities"
                }
            },
            "sclera_zone": {
                "name": "Sclera",
                "color": (255, 255, 255, 60),  # White with transparency
                "ayurvedic_systems": ["Overall Constitution", "Systemic Balance"],
                "description": "The white part of the eye provides additional diagnostic information in Ayurvedic iridology.",
                "detailed_mapping": {
                    "color": "General constitutional state",
                    "blood_vessels": "Circulation and vascular health",
                    "markings": "Specific imbalances or systemic conditions"
                },
                "physiological_indicators": {
                    "clear_white": "Good overall health",
                    "yellowish": "Potential liver/gallbladder issues or Pitta imbalance",
                    "redness": "Inflammation or irritation",
                    "bluish": "Potential oxygen issues or Vata imbalance"
                }
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
        Preprocess the iris image for analysis with enhanced techniques.
        
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
        
        # Get dimensions
        height, width = img.shape[:2]
        
        # Resize image if it's too large or too small
        target_size = 800  # Target width
        if width > target_size or width < 400:
            aspect_ratio = height / width
            new_width = target_size
            new_height = int(target_size * aspect_ratio)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA if width > target_size else cv2.INTER_CUBIC)
        
        # Store the original for reference
        original_img = img.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better detail
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Convert to LAB color space for better color enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))  # Convert tuple to list so we can modify elements
        lab_planes[0] = clahe.apply(lab_planes[0])  # Apply CLAHE to L-channel
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Gentle Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Sharpen the image for better feature detection
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def detect_iris_and_pupil(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Enhanced detection of iris, pupil, and sclera using advanced image processing.
        
        Args:
            image: Input eye image
            
        Returns:
            Boundaries information and visualization image
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create normalized version for better detection
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply different preprocessing techniques for pupil vs iris detection
        # For pupil: dark area, more contrast
        pupil_img = cv2.medianBlur(normalized, 7)
        _, pupil_thresh = cv2.threshold(pupil_img, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((5, 5), np.uint8)
        pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_OPEN, kernel)
        pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_CLOSE, kernel)
        
        # First try to detect the pupil using contours on the binary image
        contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Default fallback values
        pupil_x, pupil_y = width // 2, height // 2
        pupil_r = min(width, height) // 10
        
        # Try to find the pupil from contours
        if contours:
            # Sort contours by area and take the largest
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            
            # Check if it's a reasonable pupil size
            if 10 < radius < min(width, height) // 5:
                pupil_x, pupil_y, pupil_r = int(x), int(y), int(radius)
        
        # If contour method doesn't work well, try Hough circles as backup
        if pupil_r == min(width, height) // 10:  # If we're still using default
            try:
                # Preprocess for pupil detection with Hough
                blurred = cv2.GaussianBlur(gray, (9, 9), 2)
                min_radius = min(width, height) // 15
                max_radius = min(width, height) // 5
                
                pupil_circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=min(width, height),
                    param1=50,
                    param2=30,
                    minRadius=min_radius,
                    maxRadius=max_radius
                )
                
                if pupil_circles is not None and len(pupil_circles[0]) > 0:
                    # Get the most prominent pupil circle
                    pupil_x, pupil_y, pupil_r = pupil_circles[0][0].astype(int)
            except Exception as e:
                print(f"Hough circle pupil detection failed: {e}")
                # Keep the default values
        
        # For iris: use edge detection approach
        # Apply bilateral filter to preserve edges
        iris_img = cv2.bilateralFilter(normalized, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(iris_img, 30, 70)
        
        # Try to detect the iris using Hough circles
        iris_x, iris_y = pupil_x, pupil_y  # Start with pupil center
        iris_r = min(width, height) // 3   # Default fallback
        
        try:
            # Estimated min and max radius based on pupil radius
            min_radius = int(pupil_r * 1.8)
            max_radius = min(int(pupil_r * 5), min(width, height) // 2)
            
            iris_circles = cv2.HoughCircles(
                iris_img,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min(width, height),
                param1=80,
                param2=30,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            if iris_circles is not None and len(iris_circles[0]) > 0:
                # Find the most concentric circle with the pupil
                best_circle = None
                min_center_dist = float('inf')
                
                for circle in iris_circles[0]:
                    cx, cy, cr = circle
                    # Calculate distance between centers
                    center_dist = math.sqrt((cx - pupil_x)**2 + (cy - pupil_y)**2)
                    
                    if center_dist < min_center_dist:
                        min_center_dist = center_dist
                        best_circle = circle
                
                if best_circle is not None:
                    iris_x, iris_y, iris_r = best_circle.astype(int)
                    
                    # Adjust center to be more concentric with pupil if close enough
                    if min_center_dist < pupil_r:
                        iris_x, iris_y = pupil_x, pupil_y
        
        except Exception as e:
            print(f"Iris detection error: {e}")
            # Fallback to estimated values
            iris_x, iris_y = pupil_x, pupil_y
            iris_r = pupil_r * 3
        
        # Create visualization with detected circles and segments
        vis_image = image.copy()
        
        # Calculate sclera boundary (estimated from iris)
        sclera_r = int(iris_r * 1.3)
        
        # Draw sclera
        cv2.circle(vis_image, (iris_x, iris_y), sclera_r, (255, 255, 255), 2)
        
        # Draw iris
        cv2.circle(vis_image, (iris_x, iris_y), iris_r, (255, 0, 0), 2)
        
        # Draw pupil
        cv2.circle(vis_image, (pupil_x, pupil_y), pupil_r, (0, 0, 255), 2)
        
        # Draw zone boundaries (different layers of the iris)
        colors = [(46, 139, 87), (106, 90, 205), (70, 130, 180), (128, 0, 128)]  # Different colors for zones
        
        # Draw zones for visualization
        for i, zone_name in enumerate(list(self.zones.keys())[:5]):  # Excluding sclera
            zone_info = self.zones[zone_name]
            zone_radius = int(pupil_r + (iris_r - pupil_r) * zone_info["radius_ratio"])
            cv2.circle(vis_image, (iris_x, iris_y), zone_radius, colors[i % len(colors)], 1)
        
        # Add detailed information to the boundaries
        boundaries = {
            "pupil": {
                "center": (pupil_x, pupil_y),
                "radius": pupil_r
            },
            "iris": {
                "center": (iris_x, iris_y),
                "radius": iris_r,
                "inner_radius": pupil_r,  # Add inner radius for convenience
                "outer_radius": iris_r     # Outer radius is the iris radius
            },
            "sclera": {
                "center": (iris_x, iris_y),
                "radius": sclera_r,
                "inner_radius": iris_r   # Inner radius of sclera is the outer radius of iris
            },
            "zones": {}
        }
        
        # Add zone boundaries to the result
        for zone_name, zone_info in self.zones.items():
            if zone_name == "sclera_zone":
                continue  # Skip sclera as it's already defined
                
            outer_radius = int(pupil_r + (iris_r - pupil_r) * zone_info["radius_ratio"])
            inner_radius = int(pupil_r) if zone_name == "pupillary_zone" else int(
                pupil_r + (iris_r - pupil_r) * self.get_previous_zone_ratio(zone_name)
            )
            
            boundaries["zones"][zone_name] = {
                "center": (iris_x, iris_y),
                "inner_radius": inner_radius,
                "outer_radius": outer_radius
            }
        
        return boundaries, vis_image
    
    def generate_zone_map(self, image: np.ndarray, boundaries: Dict[str, Any]) -> np.ndarray:
        """
        Generate a detailed visual map of all eye segments including pupil, iris zones, and sclera.
        
        Args:
            image: Input eye image
            boundaries: Detected eye segment boundaries
            
        Returns:
            Image with overlaid zone map and labels
        """
        # Convert to PIL for easier text drawing and transparency
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Get boundaries
        iris_center = boundaries["iris"]["center"]
        iris_radius = boundaries["iris"]["radius"]
        pupil_radius = boundaries["pupil"]["radius"]
        sclera_radius = boundaries["sclera"]["radius"]
        
        # Create a font for labels
        try:
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts/DejaVuSans.ttf')
            font = ImageFont.truetype(font_path, 14)
            small_font = ImageFont.truetype(font_path, 10)
        except IOError:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            
        # First draw the sclera zone with very light overlay
        sclera_color = self.zones["sclera_zone"]["color"]
        sclera_mask = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        sclera_draw = ImageDraw.Draw(sclera_mask, 'RGBA')
        sclera_draw.ellipse(
            [(iris_center[0] - sclera_radius, iris_center[1] - sclera_radius), 
             (iris_center[0] + sclera_radius, iris_center[1] + sclera_radius)],
            fill=sclera_color
        )
        # Create a mask for the iris area to avoid overlapping
        iris_mask = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        iris_draw = ImageDraw.Draw(iris_mask, 'RGBA')
        iris_draw.ellipse(
            [(iris_center[0] - iris_radius, iris_center[1] - iris_radius), 
             (iris_center[0] + iris_radius, iris_center[1] + iris_radius)],
            fill=(0, 0, 0, 255)  # Fully opaque
        )
        
        # Cut out iris area from sclera mask
        sclera_mask_array = np.array(sclera_mask, dtype=np.uint8).copy()
        iris_mask_array = np.array(iris_mask, dtype=np.uint8).copy()
        for c in range(3):  # RGB channels
            sclera_mask_array[:, :, c] = np.where(iris_mask_array[:, :, 3] > 0, 0, sclera_mask_array[:, :, c])
        # Alpha channel handling
        sclera_mask_array[:, :, 3] = np.where(iris_mask_array[:, :, 3] > 0, 0, sclera_mask_array[:, :, 3])
        
        # Convert back to PIL and composite onto image
        sclera_mask = Image.fromarray(sclera_mask_array)
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), sclera_mask)
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Draw a clear border around sclera for visibility
        draw.ellipse(
            [(iris_center[0] - sclera_radius, iris_center[1] - sclera_radius), 
             (iris_center[0] + sclera_radius, iris_center[1] + sclera_radius)],
            outline=(255, 255, 255, 200),
            width=2
        )
        
        # Draw iris zones from outer to inner (to allow inner zones to overlay)
        zone_names = list(self.zones.keys())[:5]  # Exclude sclera
        zone_names.reverse()  # Start from outermost iris zone
        
        # Draw each zone with semitransparent color
        for zone_name in zone_names:
            zone_info = self.zones[zone_name]
            outer_radius = int(pupil_radius + (iris_radius - pupil_radius) * zone_info["radius_ratio"])
            inner_radius = int(pupil_radius) if zone_name == "pupillary_zone" else int(
                pupil_radius + (iris_radius - pupil_radius) * 
                self.get_previous_zone_ratio(zone_name)
            )
            
            # Create a zone mask
            zone_color = zone_info["color"]
            zone_mask = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            zone_draw = ImageDraw.Draw(zone_mask, 'RGBA')
            
            # Draw the zone as a filled ring
            zone_draw.ellipse(
                [(iris_center[0] - outer_radius, iris_center[1] - outer_radius), 
                 (iris_center[0] + outer_radius, iris_center[1] + outer_radius)],
                fill=zone_color
            )
            
            # Cut out the inner circle if not pupillary zone
            if zone_name != "pupillary_zone":
                zone_draw.ellipse(
                    [(iris_center[0] - inner_radius, iris_center[1] - inner_radius), 
                     (iris_center[0] + inner_radius, iris_center[1] + inner_radius)],
                    fill=(0, 0, 0, 0)  # Transparent
                )
                
            # Composite zone mask onto image
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), zone_mask)
            draw = ImageDraw.Draw(pil_image, 'RGBA')
            
            # Draw zone border for clarity
            draw.ellipse(
                [(iris_center[0] - outer_radius, iris_center[1] - outer_radius), 
                 (iris_center[0] + outer_radius, iris_center[1] + outer_radius)],
                outline=(255, 255, 255, 200),
                width=1
            )
            
            # Add zone label
            label_angle = -45  # Position at upper-right
            label_radius = (inner_radius + outer_radius) / 2
            label_x = iris_center[0] + int(label_radius * math.cos(math.radians(label_angle)))
            label_y = iris_center[1] + int(label_radius * math.sin(math.radians(label_angle)))
            
            # Draw text with background for readability
            # Short label due to space constraints
            short_name = zone_info["name"].split('(')[0].strip()
            
            # Measure text size
            text_bbox = draw.textbbox((label_x, label_y), short_name, font=small_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw text background
            draw.rectangle(
                [label_x - 2, label_y - 2, label_x + text_width + 2, label_y + text_height + 2],
                fill=(0, 0, 0, 128)
            )
            
            # Draw text
            draw.text((label_x, label_y), short_name, font=small_font, fill=(255, 255, 255, 255))
            
        # Finally, draw the pupil border
        draw.ellipse(
            [(iris_center[0] - pupil_radius, iris_center[1] - pupil_radius), 
             (iris_center[0] + pupil_radius, iris_center[1] + pupil_radius)],
            outline=(255, 255, 255, 200),
            width=2
        )
        
        # Add legend in bottom-right corner
        legend_x = image.shape[1] - 150
        legend_y = image.shape[0] - 120
        
        # Draw legend background
        draw.rectangle(
            [legend_x - 5, legend_y - 5, legend_x + 145, legend_y + 115],
            fill=(0, 0, 0, 150)
        )
        
        # Add legend title
        draw.text((legend_x, legend_y), "IRIS ZONES", font=font, fill=(255, 255, 255, 255))
        legend_y += 20
        
        # Add legend entries for each zone
        for i, zone_name in enumerate(list(self.zones.keys())[:5]):  # Main iris zones
            zone_info = self.zones[zone_name]
            color = list(zone_info["color"][:3])  # Convert RGB part to a list
            
            # Draw color box
            draw.rectangle(
                [legend_x, legend_y + i*18, legend_x + 12, legend_y + i*18 + 12],
                fill=tuple(color) + (200,)  # Convert back to tuple and add alpha
            )
            
            # Draw zone name
            short_name = zone_info["name"].split('(')[0].strip()
            draw.text((legend_x + 18, legend_y + i*18), short_name, font=small_font, fill=(255, 255, 255, 255))
            
        return np.array(pil_image, dtype=np.uint8)
    
    def get_previous_zone_ratio(self, zone_name: str) -> float:
        """Get the radius ratio of the zone before the current one."""
        zones_list = list(self.zones.keys())
        if zone_name not in zones_list or zones_list.index(zone_name) == 0:
            return 0.0
        previous_zone = zones_list[zones_list.index(zone_name) - 1]
        # Safety check in case the previous zone doesn't have radius_ratio (sclera_zone)
        return self.zones[previous_zone].get("radius_ratio", 0.0)
        
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
            # Special handling for sclera zone which doesn't have radius_ratio
            if zone_name == "sclera_zone":
                outer_radius = int(iris_radius * 1.3)  # Same as sclera_r in detect_iris_and_pupil
                inner_radius = iris_radius
            else:
                # Calculate normal zone radius
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
            
            # Convert numpy arrays to PIL images for better compatibility with web frameworks
            from PIL import Image
            import io
            
            # Convert arrays to PIL Images
            def array_to_pil(img_array):
                if isinstance(img_array, np.ndarray):
                    return Image.fromarray(img_array)
                return img_array
            
            # Complete results with converted images
            return {
                "original_image": array_to_pil(preprocessed),
                "boundary_image": array_to_pil(boundary_image),
                "zone_map": array_to_pil(zone_map),
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
