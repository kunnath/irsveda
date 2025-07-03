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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisZoneAnalyzer:
    """Class for analyzing iris zones and mapping them to ayurvedic concepts."""
    
    def __init__(self, sensitivity="medium"):
        """Initialize the iris zone analyzer."""
        from iris_config import ZONE_CONFIG, SENSITIVITY_PRESETS
        
        self.zones = ZONE_CONFIG
        self.sensitivity = SENSITIVITY_PRESETS[sensitivity]
        self.initialize_zone_detectors()
        
    def initialize_zone_detectors(self):
        """Initialize the zone detection algorithms."""
        self.zone_detectors = {}
        for zone_name, zone_info in self.zones.items():
            self.zone_detectors[zone_name] = {
                'inner_ratio': zone_info['inner_ratio'],
                'outer_ratio': zone_info['outer_ratio'],
                'detector': cv2.createCLAHE(
                    clipLimit=self.sensitivity['contrast_limit'],
                    tileGridSize=(8,8)
                )
            }
            
    def analyze_zones(self, image: np.ndarray, iris_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze iris zones with enhanced accuracy.
        
        Args:
            image: Preprocessed iris image
            iris_data: Dictionary containing iris and pupil information
            
        Returns:
            Dictionary containing zone analysis results
        """
        try:
            # Extract iris boundaries
            iris_center = iris_data.get('iris_center')
            iris_radius = iris_data.get('iris_radius')
            pupil_radius = iris_data.get('pupil_radius')
            
            if not all([iris_center, iris_radius, pupil_radius]):
                raise ValueError("Missing required iris boundary information")
            
            # Initialize results dictionary
            results = {
                'zones': {},
                'analysis_success': True,
                'timestamp': self.get_timestamp()
            }
            
            # Analyze each zone
            for zone_name, zone_info in self.zones.items():
                inner_radius = int(iris_radius * zone_info['inner_ratio'])
                outer_radius = int(iris_radius * zone_info['outer_ratio'])
                
                # Create zone mask
                zone_mask = self.create_zone_mask(
                    image.shape[:2],
                    iris_center,
                    inner_radius,
                    outer_radius
                )
                
                # Extract zone features
                zone_features = self.extract_zone_features(
                    image,
                    zone_mask,
                    self.zone_detectors[zone_name]['detector']
                )
                
                # Store zone results
                results['zones'][zone_name] = {
                    'features': zone_features,
                    'name': zone_info['name'],
                    'systems': zone_info['systems'],
                    'dimensions': {
                        'inner_radius': inner_radius,
                        'outer_radius': outer_radius
                    }
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Zone analysis failed: {str(e)}")
            return {
                'analysis_success': False,
                'error': str(e),
                'timestamp': self.get_timestamp()
            }
            
    def create_zone_mask(self, image_shape: Tuple[int, int], center: Tuple[int, int], 
                        inner_radius: int, outer_radius: int) -> np.ndarray:
        """Create a binary mask for the specified zone."""
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.circle(mask, center, outer_radius, 255, -1)
        cv2.circle(mask, center, inner_radius, 0, -1)
        return mask
        
    def extract_zone_features(self, image: np.ndarray, mask: np.ndarray, 
                            detector: cv2.CLAHE) -> Dict[str, Any]:
        """Extract features from a specific iris zone."""
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast in the zone
        enhanced = detector.apply(gray)
        
        # Detect features
        features = {
            'mean_intensity': float(np.mean(enhanced[mask > 0])),
            'std_intensity': float(np.std(enhanced[mask > 0])),
            'texture_features': self.compute_texture_features(enhanced, mask),
            'spot_count': self.count_spots(enhanced, mask)
        }
        
        return features
        
    def compute_texture_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Compute texture features for the zone."""
        # Basic texture features
        pixels = image[mask > 0]
        return {
            'contrast': float(np.ptp(pixels)),
            'smoothness': 1.0 - (1.0 / (1.0 + np.var(pixels))),
            'uniformity': float(np.sum(np.square(np.histogram(pixels, bins=8)[0]/len(pixels))))
        }
        
    def count_spots(self, image: np.ndarray, mask: np.ndarray) -> int:
        """Count the number of spots in the zone."""
        # Threshold the image
        _, binary = cv2.threshold(
            image, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Apply mask
        binary = cv2.bitwise_and(binary, binary, mask=mask)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size
        valid_spots = [
            cnt for cnt in contours 
            if self.sensitivity['min_spot_size'] < cv2.contourArea(cnt) < self.sensitivity['max_spot_size']
        ]
        
        return len(valid_spots)
        
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
