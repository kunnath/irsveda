"""
Advanced iris feature extraction module for IridoVeda.

This module provides functions to extract various features from iris images,
including color analysis, texture patterns, spots/freckles, and more.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.feature import local_binary_pattern
from iris_advanced_segmentation import preprocess_image, segment_iris, extract_iris_zones

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_colors(image: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Extract dominant colors from the iris image using KMeans clustering.
    
    Args:
        image: RGB iris image
        k: Number of color clusters to extract
        
    Returns:
        Array of dominant colors (RGB values)
    """
    try:
        # Ensure image is in RGB
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            # Convert BGR to RGB if needed
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image to be a list of pixels
        pixels = img_rgb.reshape((-1, 3))
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        
        # Get color proportions
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels)
        
        # Sort colors by prevalence
        sorted_indices = np.argsort(-percentages)
        sorted_colors = colors[sorted_indices]
        sorted_percentages = percentages[sorted_indices]
        
        # Return colors with their percentages
        result = []
        for i in range(len(sorted_colors)):
            result.append({
                "color": sorted_colors[i].astype(int),
                "percentage": float(sorted_percentages[i])
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in color extraction: {str(e)}")
        return []

def detect_spots(image: np.ndarray, segmentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect spots/freckles in the iris.
    
    Args:
        image: Grayscale iris image
        segmentation_data: Dictionary with iris and pupil segmentation info
        
    Returns:
        List of detected spots with their properties
    """
    if segmentation_data is None:
        return []
    
    try:
        # Create a mask that includes only the iris area (excluding pupil)
        iris_mask = segmentation_data["iris_mask"].copy()
        pupil_mask = segmentation_data["pupil_mask"].copy()
        
        # Exclude pupil from iris
        iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
        
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=iris_only_mask)
        
        # Use adaptive thresholding to find spots (dark regions)
        binary = cv2.adaptiveThreshold(
            masked_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process each contour to get spot properties
        spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out very small noise
            if area > 5:
                # Get contour properties
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = 0, 0
                
                # Calculate distance from iris center
                iris_center = segmentation_data["iris_center"]
                distance_from_center = np.sqrt(
                    (cx - iris_center[0])**2 + 
                    (cy - iris_center[1])**2
                )
                
                # Calculate relative position (in terms of iris radius)
                relative_position = distance_from_center / segmentation_data["iris_radius"]
                
                # Determine which zone the spot belongs to
                if "zone_masks" in segmentation_data:
                    zone = None
                    for i, zone_data in enumerate(segmentation_data["zone_masks"]):
                        if zone_data["inner_radius"] <= distance_from_center <= zone_data["outer_radius"]:
                            zone = i
                            break
                else:
                    zone = None
                
                # Get spot intensity (average pixel value)
                mask = np.zeros_like(image)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                intensity = np.mean(image[mask == 255])
                
                spots.append({
                    "position": (cx, cy),
                    "area": area,
                    "perimeter": cv2.arcLength(contour, True),
                    "distance_from_center": distance_from_center,
                    "relative_position": relative_position,
                    "intensity": intensity,
                    "zone": zone,
                    "contour": contour
                })
        
        return spots
        
    except Exception as e:
        logger.error(f"Error in spot detection: {str(e)}")
        return []

def extract_texture_features(image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract texture features from the iris using Local Binary Patterns.
    
    Args:
        image: Grayscale iris image
        segmentation_data: Dictionary with iris and pupil segmentation info
        
    Returns:
        Dictionary with texture features
    """
    if segmentation_data is None:
        return {}
    
    try:
        # Create a mask that includes only the iris area (excluding pupil)
        iris_mask = segmentation_data["iris_mask"].copy()
        pupil_mask = segmentation_data["pupil_mask"].copy()
        
        # Exclude pupil from iris
        iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
        
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=iris_only_mask)
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(masked_image, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(
            lbp, 
            density=True, 
            bins=n_bins, 
            range=(0, n_bins)
        )
        
        # Calculate texture properties
        # Contrast measures local variations in the gray-level
        contrast = np.std(masked_image[iris_only_mask > 0])
        
        # Uniformity measures texture uniformity
        uniformity = np.sum(np.square(hist))
        
        # Energy is the squared sum of pixel values
        energy = np.sqrt(np.sum(np.square(masked_image)) / np.sum(iris_only_mask > 0))
        
        # Entropy measures randomness in texture
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            "lbp_histogram": hist.tolist(),
            "contrast": float(contrast),
            "uniformity": float(uniformity),
            "energy": float(energy),
            "entropy": float(entropy)
        }
        
    except Exception as e:
        logger.error(f"Error in texture extraction: {str(e)}")
        return {}

def extract_radial_features(image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features along radial lines from pupil to iris boundary.
    
    Args:
        image: Grayscale iris image
        segmentation_data: Dictionary with iris and pupil segmentation info
        
    Returns:
        Dictionary with radial features
    """
    if segmentation_data is None:
        return {}
    
    try:
        iris_center = segmentation_data["iris_center"]
        iris_radius = segmentation_data["iris_radius"]
        pupil_center = segmentation_data["pupil_center"] or iris_center
        pupil_radius = segmentation_data["pupil_radius"] or int(iris_radius * 0.2)
        
        # Number of radial lines to analyze
        num_lines = 36  # every 10 degrees
        
        # Create a visualization image
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        # Data structure to hold radial features
        radial_data = []
        
        # Analysis for each radial line
        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            
            # Calculate start and end points
            start_x = int(pupil_center[0] + pupil_radius * np.cos(angle))
            start_y = int(pupil_center[1] + pupil_radius * np.sin(angle))
            
            end_x = int(iris_center[0] + iris_radius * np.cos(angle))
            end_y = int(iris_center[1] + iris_radius * np.sin(angle))
            
            # Draw the line on visualization
            cv2.line(vis_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
            
            # Sample points along the line
            num_samples = 20
            x_points = np.linspace(start_x, end_x, num_samples)
            y_points = np.linspace(start_y, end_y, num_samples)
            
            # Extract intensity values along the line
            intensities = []
            for j in range(num_samples):
                x, y = int(x_points[j]), int(y_points[j])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    intensities.append(int(image[y, x]))
            
            # Calculate statistics for this radial line
            if intensities:
                radial_data.append({
                    "angle": angle * 180 / np.pi,  # Convert to degrees
                    "mean_intensity": np.mean(intensities),
                    "std_intensity": np.std(intensities),
                    "min_intensity": np.min(intensities),
                    "max_intensity": np.max(intensities),
                    "intensity_profile": intensities
                })
        
        return {
            "radial_data": radial_data,
            "visualization": vis_image
        }
        
    except Exception as e:
        logger.error(f"Error in radial feature extraction: {str(e)}")
        return {}

def extract_all_features(image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all iris features in one function call.
    
    Args:
        image: Original image (color)
        segmentation_data: Dictionary with iris and pupil segmentation info
        
    Returns:
        Dictionary with all features
    """
    if segmentation_data is None:
        return {}
    
    try:
        # Ensure we have grayscale for texture analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Extract all features
        color_features = extract_colors(image)
        spots = detect_spots(gray, segmentation_data)
        texture_features = extract_texture_features(gray, segmentation_data)
        radial_features = extract_radial_features(gray, segmentation_data)
        
        # Create zones if not already in segmentation data
        if "zone_masks" not in segmentation_data:
            zone_data = extract_iris_zones(segmentation_data)
            if zone_data:
                segmentation_data.update(zone_data)
        
        # Extract color for each zone
        zone_colors = []
        if "zone_masks" in segmentation_data:
            for zone in segmentation_data["zone_masks"]:
                zone_mask = zone["mask"]
                masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
                zone_colors.append(extract_colors(masked_image, k=3))
        
        # Combine all features
        features = {
            "color_features": color_features,
            "spots": spots,
            "texture_features": texture_features,
            "radial_features": radial_features,
            "zone_colors": zone_colors,
            
            # Add statistics
            "num_spots": len(spots),
            "avg_spot_size": np.mean([spot["area"] for spot in spots]) if spots else 0,
            "texture_contrast": texture_features.get("contrast", 0),
            "texture_uniformity": texture_features.get("uniformity", 0)
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting all features: {str(e)}")
        return {}
