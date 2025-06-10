"""
Enhanced Iris Image Analyzer
Integrates advanced spot detection and segmentation capabilities for better iris analysis
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime
import math
import tempfile
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import matplotlib with fallback
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - some visualization features will be disabled")

class EnhancedIrisImageAnalyzer:
    """
    Enhanced iris image analyzer with advanced spot detection and iris identification capabilities.
    Integrates features from the irisimage web app for better spot detection and iris segmentation.
    """
    
    def __init__(self, detection_params=None):
        """
        Initialize the Enhanced Iris Image Analyzer
        
        Args:
            detection_params (dict): Parameters for spot detection and analysis
        """
        # Enhanced detection parameters with more comprehensive options
        self.detection_params = detection_params or {
            'min_area': 20,                    # Minimum spot area (pixels)
            'max_area': 2000,                  # Maximum spot area (pixels)
            'min_circularity': 0.2,            # Minimum circularity (0-1)
            'gaussian_blur': 3,                # Gaussian blur kernel size
            'canny_low': 30,                   # Canny edge detection low threshold
            'canny_high': 100,                 # Canny edge detection high threshold
            'morphology_kernel': 2,            # Morphological operation kernel size
            'brightness_threshold': 20,        # Brightness difference threshold
            'contrast_enhancement': 2.0,       # CLAHE contrast enhancement
            'detect_dark_spots': True,         # Detect dark spots
            'detect_light_spots': True,        # Detect light spots
            'sharpness_threshold': 0.1,        # Minimum sharpness for spot detection
            'visibility_threshold': 10,        # Minimum visibility (contrast) threshold
            'adaptive_threshold_block': 11,    # Adaptive threshold block size
            'adaptive_threshold_c': 2,         # Adaptive threshold constant
            'watershed_min_distance': 10,      # Watershed minimum distance
            'bilateral_d': 9,                  # Bilateral filter diameter
            'bilateral_sigma_color': 75,       # Bilateral filter sigma color
            'bilateral_sigma_space': 75        # Bilateral filter sigma space
        }
        
        # Initialize organ mapping based on iridology charts
        self.init_organ_mapping()
        
        # Color coding for different detection methods
        self.method_colors = {
            'edge': (0, 255, 0),           # Bright Green for edge detection
            'dark_spot': (0, 0, 255),      # Red for dark spots  
            'light_spot': (255, 165, 0),   # Orange for light spots
            'blob': (255, 255, 0),         # Yellow for blob detection
            'contour': (255, 0, 255),      # Magenta for contour-based
            'watershed': (0, 255, 255),    # Cyan for watershed
            'threshold': (128, 0, 128),    # Purple for threshold-based
            'adaptive': (255, 192, 203),   # Pink for adaptive methods
            'unknown': (128, 128, 128)     # Gray for unknown methods
        }
        
        # Method symbols for better identification
        self.method_symbols = {
            'edge': 'E',
            'dark_spot': 'D', 
            'light_spot': 'L',
            'blob': 'B',
            'contour': 'C',
            'watershed': 'W', 
            'threshold': 'T',
            'adaptive': 'A',
            'unknown': '?'
        }
    
    def init_organ_mapping(self):
        """
        Initialize organ mapping based on iridology charts.
        Maps angles (degrees) to corresponding organs/body systems.
        """
        # Right iris organ mapping (angles in degrees, 0° = 3 o'clock position)
        self.right_iris_organs = {
            (0, 15): "Lymphatic System",
            (15, 30): "Kidney",
            (30, 45): "Adrenal Glands", 
            (45, 60): "Pancreas/Spleen",
            (60, 75): "Heart",
            (75, 90): "Lung",
            (90, 105): "Bronchi",
            (105, 120): "Throat/Thyroid",
            (120, 135): "Brain/Cerebrum",
            (135, 150): "Pituitary",
            (150, 165): "Pineal",
            (165, 180): "Cerebellum",
            (180, 195): "Medulla",
            (195, 210): "Cervical Spine",
            (210, 225): "Thoracic Spine",
            (225, 240): "Lumbar Spine",
            (240, 255): "Sacral/Coccyx",
            (255, 270): "Reproductive Organs",
            (270, 285): "Prostate/Uterus",
            (285, 300): "Bladder",
            (300, 315): "Appendix",
            (315, 330): "Ascending Colon",
            (330, 345): "Liver",
            (345, 360): "Gall Bladder"
        }
        
        # Left iris organ mapping (mirror of right iris with some differences)
        self.left_iris_organs = {
            (0, 15): "Lymphatic System",
            (15, 30): "Kidney",
            (30, 45): "Adrenal Glands",
            (45, 60): "Stomach",
            (60, 75): "Heart",
            (75, 90): "Lung",
            (90, 105): "Bronchi",
            (105, 120): "Throat/Thyroid",
            (120, 135): "Brain/Cerebrum",
            (135, 150): "Pituitary",
            (150, 165): "Pineal",
            (165, 180): "Cerebellum",
            (180, 195): "Medulla",
            (195, 210): "Cervical Spine",
            (210, 225): "Thoracic Spine",
            (225, 240): "Lumbar Spine",
            (240, 255): "Sacral/Coccyx",
            (255, 270): "Reproductive Organs",
            (270, 285): "Prostate/Uterus",
            (285, 300): "Bladder",
            (300, 315): "Sigmoid Colon",
            (315, 330): "Descending Colon",
            (330, 345): "Spleen",
            (345, 360): "Pancreas"
        }
    
    def analyze_iris_image(self, image_path: str, iris_side: str = "unknown") -> Dict[str, Any]:
        """
        Comprehensive analysis of iris image with enhanced spot detection
        
        Args:
            image_path: Path to the iris image
            iris_side: "left", "right", or "unknown"
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load and preprocess image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return {"error": "Could not load image"}
            
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect iris center and radius
            iris_center, iris_radius = self.detect_iris_center_and_radius(gray_image)
            
            if iris_center is None:
                return {"error": "Could not detect iris in the image"}
            
            # Preprocess image for better segmentation
            preprocessed = self.preprocess_image(gray_image)
            
            # Perform comprehensive spot detection
            spots = self.segment_iris_features(preprocessed, rgb_image, iris_center, iris_radius)
            
            # Map spots to organs based on their location
            enhanced_spots = []
            for spot in spots:
                # Calculate angle and distance from center
                angle = self.calculate_angle_from_center(spot['centroid_x'], spot['centroid_y'], iris_center)
                distance_pct, zone = self.calculate_distance_from_center(
                    spot['centroid_x'], spot['centroid_y'], iris_center, iris_radius
                )
                
                # Map to organ system
                organ = self.map_angle_to_organ(angle, iris_side)
                
                # Enhance spot data
                enhanced_spot = {
                    **spot,
                    'angle_degrees': angle,
                    'distance_from_center_pct': distance_pct,
                    'iris_zone': zone,
                    'corresponding_organ': organ,
                    'iris_side': iris_side
                }
                enhanced_spots.append(enhanced_spot)
            
            # Generate annotated image
            annotated_image = self.create_annotated_image(image_bgr, enhanced_spots)
            
            # Convert annotated image to base64 for display
            annotated_base64 = self.image_to_base64(annotated_image)
            
            # Generate comprehensive analysis summary
            analysis_summary = self.generate_analysis_summary(enhanced_spots, iris_center, iris_radius)
            
            return {
                "spots": enhanced_spots,
                "iris_center": iris_center,
                "iris_radius": iris_radius,
                "annotated_image": annotated_base64,
                "analysis_summary": analysis_summary,
                "total_spots": len(enhanced_spots),
                "detection_methods_used": list(set(spot['detection_method'] for spot in enhanced_spots))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing iris image: {str(e)}")
            return {"error": str(e)}
    
    def preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for better spot detection
        """
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(
            gray_image, 
            self.detection_params['bilateral_d'],
            self.detection_params['bilateral_sigma_color'],
            self.detection_params['bilateral_sigma_space']
        )
        
        # Apply Gaussian blur to reduce noise
        blur_size = self.detection_params['gaussian_blur']
        if blur_size > 0:
            blurred = cv2.GaussianBlur(filtered, (blur_size, blur_size), 0)
        else:
            blurred = filtered.copy()
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.detection_params['contrast_enhancement'], 
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_iris_center_and_radius(self, gray_image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        Detect iris center and radius using enhanced circle detection
        """
        # Apply preprocessing
        preprocessed = self.preprocess_image(gray_image)
        
        # Use HoughCircles to detect circular boundaries
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=min(preprocessed.shape) // 2
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Get the largest circle (likely the iris boundary)
            largest_circle = circles[np.argmax(circles[:, 2])]
            center = (largest_circle[0], largest_circle[1])
            radius = largest_circle[2]
            return center, radius
        
        # Fallback: use image center
        h, w = gray_image.shape
        return (w // 2, h // 2), min(w, h) // 4
    
    def segment_iris_features(self, preprocessed: np.ndarray, rgb_image: np.ndarray, 
                            iris_center: Tuple[int, int], iris_radius: int) -> List[Dict[str, Any]]:
        """
        Enhanced iris feature segmentation with multiple detection methods
        """
        valid_segments = []
        
        # Method 1: Edge-based detection
        edges = cv2.Canny(
            preprocessed, 
            self.detection_params['canny_low'], 
            self.detection_params['canny_high']
        )
        
        kernel_size = self.detection_params['morphology_kernel']
        if kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_segments.extend(self._filter_and_analyze_contours(contours, preprocessed, rgb_image, "edge"))
        
        # Method 2: Dark spot detection
        if self.detection_params['detect_dark_spots']:
            mean_intensity = np.mean(preprocessed)
            dark_threshold = mean_intensity - self.detection_params['brightness_threshold']
            _, dark_binary = cv2.threshold(preprocessed, dark_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up the binary image
            kernel = np.ones((2, 2), np.uint8)
            dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_CLOSE, kernel)
            dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, kernel)
            
            dark_contours, _ = cv2.findContours(dark_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_segments.extend(self._filter_and_analyze_contours(dark_contours, preprocessed, rgb_image, "dark_spot"))
        
        # Method 3: Light spot detection
        if self.detection_params['detect_light_spots']:
            mean_intensity = np.mean(preprocessed)
            light_threshold = mean_intensity + self.detection_params['brightness_threshold']
            _, light_binary = cv2.threshold(preprocessed, light_threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up the binary image
            kernel = np.ones((2, 2), np.uint8)
            light_binary = cv2.morphologyEx(light_binary, cv2.MORPH_CLOSE, kernel)
            light_binary = cv2.morphologyEx(light_binary, cv2.MORPH_OPEN, kernel)
            
            light_contours, _ = cv2.findContours(light_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_segments.extend(self._filter_and_analyze_contours(light_contours, preprocessed, rgb_image, "light_spot"))
        
        # Method 4: Adaptive threshold detection
        adaptive_binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.detection_params['adaptive_threshold_block'], 
            self.detection_params['adaptive_threshold_c']
        )
        
        adaptive_contours, _ = cv2.findContours(adaptive_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_segments.extend(self._filter_and_analyze_contours(adaptive_contours, preprocessed, rgb_image, "adaptive"))
        
        # Method 5: Blob detection for circular spots
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.detection_params['min_area']
        params.maxArea = self.detection_params['max_area']
        params.filterByCircularity = True
        params.minCircularity = self.detection_params['min_circularity']
        params.filterByConvexity = False
        params.filterByInertia = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(preprocessed)
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            
            # Create bounding box
            bbox = (max(0, x - radius), max(0, y - radius), 
                   min(preprocessed.shape[1] - x + radius, 2 * radius), 
                   min(preprocessed.shape[0] - y + radius, 2 * radius))
            
            # Calculate characteristics
            characteristics = self.calculate_spot_characteristics_from_bbox(bbox, preprocessed)
            
            if characteristics['visibility'] >= self.detection_params['visibility_threshold']:
                # Calculate dominant color from RGB image
                segment_region = rgb_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                if segment_region.size > 0:
                    dominant_color = np.mean(segment_region.reshape(-1, 3), axis=0)
                else:
                    dominant_color = [128, 128, 128]
                
                valid_segments.append({
                    'segment_id': len(valid_segments) + 1,
                    'x': bbox[0],
                    'y': bbox[1],
                    'width': bbox[2],
                    'height': bbox[3],
                    'area': bbox[2] * bbox[3],
                    'dominant_color_r': int(dominant_color[0]),
                    'dominant_color_g': int(dominant_color[1]),
                    'dominant_color_b': int(dominant_color[2]),
                    'avg_brightness': characteristics['mean_intensity'],
                    'detection_method': 'blob',
                    'sharpness': characteristics['sharpness'],
                    'visibility': characteristics['visibility'],
                    'contrast': characteristics['contrast'],
                    'circularity': 1.0,  # Blob detection ensures circularity
                    'centroid_x': x,
                    'centroid_y': y,
                    'timestamp': datetime.now().isoformat()
                })
        
        return valid_segments
    
    def _filter_and_analyze_contours(self, contours: List, preprocessed: np.ndarray, 
                                   rgb_image: np.ndarray, method: str) -> List[Dict[str, Any]]:
        """
        Filter and analyze contours based on detection parameters
        """
        valid_segments = []
        
        for i, contour in enumerate(contours):
            # Calculate area and filter
            area = cv2.contourArea(contour)
            if area < self.detection_params['min_area'] or area > self.detection_params['max_area']:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.detection_params['min_circularity']:
                continue
            
            # Get bounding box and characteristics
            x, y, w, h = cv2.boundingRect(contour)
            characteristics = self.calculate_spot_characteristics_from_contour(contour, preprocessed)
            
            # Filter by visibility and sharpness
            if (characteristics['visibility'] < self.detection_params['visibility_threshold'] or
                characteristics['sharpness'] < self.detection_params['sharpness_threshold']):
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
            else:
                centroid_x = x + w // 2
                centroid_y = y + h // 2
            
            # Calculate dominant color from RGB image
            segment_region = rgb_image[y:y+h, x:x+w]
            if segment_region.size > 0:
                dominant_color = np.mean(segment_region.reshape(-1, 3), axis=0)
            else:
                dominant_color = [128, 128, 128]
            
            valid_segments.append({
                'segment_id': len(valid_segments) + 1,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': int(area),
                'dominant_color_r': int(dominant_color[0]),
                'dominant_color_g': int(dominant_color[1]),
                'dominant_color_b': int(dominant_color[2]),
                'avg_brightness': characteristics['mean_intensity'],
                'detection_method': method,
                'sharpness': characteristics['sharpness'],
                'visibility': characteristics['visibility'],
                'contrast': characteristics['contrast'],
                'circularity': circularity,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'timestamp': datetime.now().isoformat()
            })
        
        return valid_segments
    
    def calculate_spot_characteristics_from_contour(self, contour: np.ndarray, image: np.ndarray) -> Dict[str, float]:
        """Calculate characteristics from contour"""
        # Get bounding box and create mask
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return self._calculate_characteristics(mask, image, x, y, w, h)
    
    def calculate_spot_characteristics_from_bbox(self, bbox: Tuple[int, int, int, int], image: np.ndarray) -> Dict[str, float]:
        """Calculate characteristics from bounding box"""
        x, y, w, h = bbox
        mask = np.zeros(image.shape, np.uint8)
        mask[y:y+h, x:x+w] = 255
        
        return self._calculate_characteristics(mask, image, x, y, w, h)
    
    def _calculate_characteristics(self, mask: np.ndarray, image: np.ndarray, 
                                 x: int, y: int, w: int, h: int) -> Dict[str, float]:
        """Internal method to calculate spot characteristics"""
        # Get region of interest
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0 or roi_mask.sum() == 0:
            return {'sharpness': 0, 'visibility': 0, 'mean_intensity': 0, 'contrast': 0}
        
        # Calculate sharpness using Laplacian variance
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Calculate visibility (local contrast)
        spot_pixels = image[mask == 255]
        if len(spot_pixels) == 0:
            return {'sharpness': 0, 'visibility': 0, 'mean_intensity': 0, 'contrast': 0}
        
        spot_mean = np.mean(spot_pixels)
        
        # Get surrounding area for contrast calculation
        kernel = np.ones((max(w, h) + 10, max(w, h) + 10), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        surrounding_mask = dilated_mask - mask
        surrounding_pixels = image[surrounding_mask == 255]
        
        if len(surrounding_pixels) > 0:
            surrounding_mean = np.mean(surrounding_pixels)
            visibility = abs(spot_mean - surrounding_mean)
            contrast = abs(spot_mean - surrounding_mean) / (spot_mean + surrounding_mean + 1e-6)
        else:
            visibility = 0
            contrast = 0
        
        return {
            'sharpness': sharpness,
            'visibility': visibility,
            'mean_intensity': spot_mean,
            'contrast': contrast
        }
    
    def calculate_angle_from_center(self, x: int, y: int, iris_center: Tuple[int, int]) -> float:
        """Calculate angle from iris center in degrees"""
        center_x, center_y = iris_center
        angle_rad = math.atan2(y - center_y, x - center_x)
        angle_deg = math.degrees(angle_rad)
        # Normalize to 0-360 degrees
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg
    
    def calculate_distance_from_center(self, x: int, y: int, iris_center: Tuple[int, int], 
                                     iris_radius: int) -> Tuple[float, str]:
        """Calculate distance from center as percentage and determine zone"""
        center_x, center_y = iris_center
        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        distance_percentage = (distance / iris_radius) * 100 if iris_radius > 0 else 0
        
        # Classify zone based on distance
        if distance_percentage <= 30:
            zone = "Pupillary Zone"
        elif distance_percentage <= 70:
            zone = "Collarette Zone" 
        else:
            zone = "Ciliary Zone"
            
        return distance_percentage, zone
    
    def map_angle_to_organ(self, angle_deg: float, iris_side: str) -> str:
        """Map angle to corresponding organ based on iridology charts"""
        organ_map = self.right_iris_organs if iris_side == "right" else self.left_iris_organs
        
        for (start_angle, end_angle), organ in organ_map.items():
            if start_angle <= angle_deg < end_angle:
                return organ
        
        # Handle the wraparound case (345-360 and 0-15)
        for (start_angle, end_angle), organ in organ_map.items():
            if start_angle > end_angle:  # Wraparound case
                if angle_deg >= start_angle or angle_deg < end_angle:
                    return organ
        
        return "Unknown Region"
    
    def create_annotated_image(self, image_bgr: np.ndarray, spots: List[Dict[str, Any]]) -> np.ndarray:
        """Create annotated image with enhanced visualization"""
        annotated = image_bgr.copy()
        
        for spot in spots:
            x, y, w, h = spot['x'], spot['y'], spot['width'], spot['height']
            method = spot['detection_method']
            
            # Get color and symbol for this detection method
            color = self.method_colors.get(method, self.method_colors['unknown'])
            symbol = self.method_symbols.get(method, '?')
            
            # Draw bounding rectangle with thickness based on confidence
            confidence = spot.get('visibility', 0)
            thickness = max(2, min(5, int(2 + confidence * 0.03)))
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            
            # Draw filled circle at center for method identification
            center_x, center_y = spot['centroid_x'], spot['centroid_y']
            cv2.circle(annotated, (center_x, center_y), 8, color, -1)
            cv2.circle(annotated, (center_x, center_y), 8, (255, 255, 255), 1)
            
            # Add method symbol in the circle
            font_scale = 0.4
            text_size = cv2.getTextSize(symbol, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(annotated, symbol, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
            # Add segment ID and organ information as label
            organ = spot.get('corresponding_organ', 'Unknown')[:15]  # Truncate long names
            label = f"#{spot['segment_id']}: {organ}"
            label_y = y - 15 if y > 20 else y + h + 20
            
            # Add background rectangle for better text visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(annotated, (x, label_y - 15), (x + label_size[0] + 10, label_y + 5), 
                         (0, 0, 0), -1)  # Black background
            cv2.rectangle(annotated, (x, label_y - 15), (x + label_size[0] + 10, label_y + 5), 
                         color, 2)  # Colored border
            
            cv2.putText(annotated, label, (x + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert image to base64 string for web display"""
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def generate_analysis_summary(self, spots: List[Dict[str, Any]], 
                                iris_center: Tuple[int, int], iris_radius: int) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        if not spots:
            return {
                "total_spots": 0,
                "zones": {},
                "organs": {},
                "detection_methods": {},
                "health_indicators": {"overall": "No significant spots detected"}
            }
        
        # Analyze by zones
        zone_counts = {}
        for spot in spots:
            zone = spot.get('iris_zone', 'Unknown')
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        # Analyze by organs
        organ_counts = {}
        for spot in spots:
            organ = spot.get('corresponding_organ', 'Unknown')
            organ_counts[organ] = organ_counts.get(organ, 0) + 1
        
        # Analyze by detection methods
        method_counts = {}
        for spot in spots:
            method = spot.get('detection_method', 'Unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Generate health indicators
        health_indicators = self._generate_health_indicators(spots, zone_counts, organ_counts)
        
        return {
            "total_spots": len(spots),
            "zones": zone_counts,
            "organs": organ_counts,
            "detection_methods": method_counts,
            "health_indicators": health_indicators,
            "iris_center": iris_center,
            "iris_radius": iris_radius
        }
    
    def _generate_health_indicators(self, spots: List[Dict[str, Any]], 
                                  zone_counts: Dict[str, int], 
                                  organ_counts: Dict[str, int]) -> Dict[str, str]:
        """Generate health indicators based on spot analysis"""
        total_spots = len(spots)
        
        if total_spots == 0:
            return {"overall": "No significant spots detected - generally good tissue integrity"}
        
        indicators = {}
        
        # Overall assessment
        if total_spots < 5:
            indicators["overall"] = "Low spot count - minimal tissue stress indicated"
        elif total_spots < 15:
            indicators["overall"] = "Moderate spot count - some areas may need attention"
        else:
            indicators["overall"] = "High spot count - significant tissue stress indicated"
        
        # Zone-specific analysis
        if "Pupillary Zone" in zone_counts:
            pupillary_spots = zone_counts["Pupillary Zone"]
            if pupillary_spots > total_spots * 0.4:
                indicators["pupillary_zone"] = "High concentration near center - digestive system focus needed"
        
        if "Ciliary Zone" in zone_counts:
            ciliary_spots = zone_counts["Ciliary Zone"]
            if ciliary_spots > total_spots * 0.4:
                indicators["ciliary_zone"] = "Peripheral spots - elimination organs may need support"
        
        # Organ-specific concerns
        top_organs = sorted(organ_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for organ, count in top_organs:
            if count > 2:  # Multiple spots in same organ area
                indicators[f"organ_{organ.lower().replace(' ', '_')}"] = f"Multiple spots in {organ} area - focused attention recommended"
        
        return indicators
    
    def set_detection_sensitivity(self, sensitivity: str):
        """Set detection parameters based on sensitivity level"""
        sensitivity_presets = {
            'low': {
                'min_area': 50,
                'max_area': 1500,
                'min_circularity': 0.4,
                'brightness_threshold': 30,
                'visibility_threshold': 15,
                'sharpness_threshold': 0.2
            },
            'medium': {
                'min_area': 30,
                'max_area': 2000,
                'min_circularity': 0.3,
                'brightness_threshold': 20,
                'visibility_threshold': 10,
                'sharpness_threshold': 0.1
            },
            'high': {
                'min_area': 15,
                'max_area': 2500,
                'min_circularity': 0.2,
                'brightness_threshold': 15,
                'visibility_threshold': 5,
                'sharpness_threshold': 0.05
            }
        }
        
        if sensitivity in sensitivity_presets:
            self.detection_params.update(sensitivity_presets[sensitivity])
