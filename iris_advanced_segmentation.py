"""
Advanced iris segmentation module for IridoVeda.

This module provides enhanced iris segmentation capabilities using
computer vision techniques like Hough Circle Transform and more
sophisticated boundary detection algorithms.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhance the input image for better iris segmentation.
    
    Args:
        image: The input RGB image array
        
    Returns:
        Tuple containing (enhanced grayscale image, original grayscale image, original image)
    """
    try:
        # Keep a copy of the original
        original = image.copy()
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Apply gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
        
        return blurred, gray, original
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        # Return original in case of error
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray, gray, image

def segment_iris(image: np.ndarray) -> Dict[str, Any]:
    """
    Perform advanced iris segmentation to detect iris and pupil boundaries.
    
    Args:
        image: Preprocessed grayscale image
        
    Returns:
        Dictionary containing segmentation results or None if segmentation fails
    """
    height, width = image.shape[:2]
    
    # Create a copy for visualization
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    try:
        # Apply Hough Circle Transform to detect iris
        # Parameters explanation:
        # - dp: Inverse ratio of accumulator resolution
        # - minDist: Minimum distance between detected centers
        # - param1: Upper threshold for edge detection
        # - param2: Threshold for center detection
        # - minRadius/maxRadius: Min/Max radius constraints
        iris_circles = cv2.HoughCircles(
            image, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=width//2,
            param1=50, 
            param2=30, 
            minRadius=int(width//4), 
            maxRadius=int(width//2)
        )
        
        if iris_circles is None:
            logger.warning("No iris circle detected")
            return None
        
        # Convert to integer coordinates
        iris_circles = np.uint16(np.around(iris_circles))
        
        # Get the most probable iris circle (first one)
        iris_x, iris_y, iris_radius = iris_circles[0][0]
        
        # Create a mask for the iris
        iris_mask = np.zeros_like(image)
        cv2.circle(iris_mask, (iris_x, iris_y), iris_radius, 255, -1)
        
        # Draw iris circle on visualization image
        cv2.circle(vis_image, (iris_x, iris_y), iris_radius, (0, 255, 0), 2)
        
        # Extract the iris region for pupil detection
        iris_region = image.copy()
        # Create a mask where outside iris is black
        outside_iris = np.zeros_like(image)
        cv2.circle(outside_iris, (iris_x, iris_y), iris_radius, 255, -1)
        iris_region[outside_iris == 0] = 255
        
        # Apply thresholding to identify the pupil (darkest part)
        _, binary_iris = cv2.threshold(
            iris_region, 
            0, 
            255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Find the pupil using Hough Circle Transform
        pupil_circles = cv2.HoughCircles(
            binary_iris,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=width//10,
            param1=30,
            param2=15,
            minRadius=int(iris_radius//4),
            maxRadius=int(iris_radius//2)
        )
        
        pupil_x, pupil_y, pupil_radius = None, None, None
        pupil_mask = None
        
        if pupil_circles is not None:
            pupil_circles = np.uint16(np.around(pupil_circles))
            # Get the most probable pupil circle
            pupil_x, pupil_y, pupil_radius = pupil_circles[0][0]
            
            # Create a mask for the pupil
            pupil_mask = np.zeros_like(image)
            cv2.circle(pupil_mask, (pupil_x, pupil_y), pupil_radius, 255, -1)
            
            # Draw pupil circle on visualization image
            cv2.circle(vis_image, (pupil_x, pupil_y), pupil_radius, (255, 0, 0), 2)
            
            # Draw the pupil center
            cv2.circle(vis_image, (pupil_x, pupil_y), 2, (0, 0, 255), -1)
        
        # Create a final segmentation result
        result = {
            "iris_center": (int(iris_x), int(iris_y)),
            "iris_radius": int(iris_radius),
            "pupil_center": (int(pupil_x), int(pupil_y)) if pupil_x is not None else None,
            "pupil_radius": int(pupil_radius) if pupil_radius is not None else None,
            "iris_mask": iris_mask,
            "pupil_mask": pupil_mask if pupil_mask is not None else np.zeros_like(image),
            "segmentation_image": vis_image,
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in iris segmentation: {str(e)}")
        return None

def extract_iris_zones(segmentation_data: Dict[str, Any], num_zones: int = 5) -> Dict[str, Any]:
    """
    Extract iris zones based on segmentation data.
    
    Args:
        segmentation_data: Dictionary containing iris and pupil information
        num_zones: Number of concentric zones to extract
        
    Returns:
        Dictionary with zone masks and visualization
    """
    if not segmentation_data:
        return None
    
    try:
        # Get iris and pupil information
        iris_center = segmentation_data["iris_center"]
        iris_radius = segmentation_data["iris_radius"]
        pupil_center = segmentation_data["pupil_center"]
        pupil_radius = segmentation_data["pupil_radius"] if segmentation_data["pupil_radius"] else int(iris_radius * 0.2)
        
        if not pupil_center:
            pupil_center = iris_center
        
        # Create a visualization image
        vis_image = segmentation_data["segmentation_image"].copy()
        
        # Determine zone widths
        zone_width = (iris_radius - pupil_radius) / num_zones
        
        # Create zone masks and visualizations
        zone_masks = []
        zone_colors = [
            (255, 0, 0),    # Red - innermost zone
            (255, 165, 0),  # Orange
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue - outermost zone
        ]
        
        # Create masks for each zone
        for i in range(num_zones):
            inner_radius = pupil_radius + i * zone_width
            outer_radius = pupil_radius + (i + 1) * zone_width
            
            # Ensure radius values are integers
            inner_radius = int(inner_radius)
            outer_radius = int(outer_radius)
            
            # Create mask for this zone
            zone_mask = np.zeros_like(segmentation_data["iris_mask"])
            cv2.circle(zone_mask, iris_center, outer_radius, 255, -1)
            cv2.circle(zone_mask, iris_center, inner_radius, 0, -1)
            
            # Draw zone on visualization
            cv2.circle(vis_image, iris_center, outer_radius, zone_colors[i % len(zone_colors)], 1)
            
            zone_masks.append({
                "mask": zone_mask,
                "inner_radius": inner_radius,
                "outer_radius": outer_radius,
                "zone_index": i
            })
        
        return {
            "zone_masks": zone_masks,
            "zone_visualization": vis_image,
            "num_zones": num_zones
        }
        
    except Exception as e:
        logger.error(f"Error in zone extraction: {str(e)}")
        return None
