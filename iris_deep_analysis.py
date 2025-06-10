"""
Deep Iris Analysis Module for IridoVeda.

This module provides comprehensive analysis of iris features including:
- Advanced spot detection and classification
- Line and pattern recognition
- Deep color analysis with Ayurvedic interpretation
- Texture and structural pattern analysis
- Micro-feature detection and health correlation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.cluster import KMeans, DBSCAN
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from skimage import measure, morphology, feature
from skimage.feature import local_binary_pattern, hog
from skimage.filters import frangi, hessian
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisDeepAnalyzer:
    """
    Advanced iris analysis class for deep feature extraction and health correlation.
    """
    
    def __init__(self):
        """Initialize the deep analyzer with predefined parameters."""
        self.spot_min_size = 5
        self.spot_max_size = 200
        self.line_min_length = 10
        self.color_clusters = 8
        self.texture_radius = 3
        self.texture_n_points = 24
        
        # Ayurvedic color mappings
        self.ayurvedic_colors = {
            'brown_dark': {'range': ([10, 50, 20], [20, 255, 200]), 'dosha': 'Pitta', 'element': 'Fire'},
            'brown_light': {'range': ([10, 20, 50], [20, 100, 255]), 'dosha': 'Pitta-Kapha', 'element': 'Fire-Earth'},
            'blue_light': {'range': ([100, 50, 50], [130, 255, 255]), 'dosha': 'Vata', 'element': 'Air-Ether'},
            'blue_dark': {'range': ([100, 100, 20], [130, 255, 200]), 'dosha': 'Vata-Kapha', 'element': 'Air-Water'},
            'green': {'range': ([40, 50, 50], [80, 255, 255]), 'dosha': 'Kapha-Pitta', 'element': 'Earth-Fire'},
            'gray': {'range': ([0, 0, 50], [180, 30, 200]), 'dosha': 'Kapha', 'element': 'Water-Earth'},
            'hazel': {'range': ([8, 50, 50], [25, 255, 255]), 'dosha': 'Mixed', 'element': 'Tri-Dosha'}
        }
        
        # Iris zone mappings to body systems
        self.zone_mappings = {
            'pupillary_border': {'organs': ['stomach', 'digestive_system'], 'ratio': 0.2},
            'inner_pupillary': {'organs': ['liver', 'gallbladder'], 'ratio': 0.4},
            'middle_ciliary': {'organs': ['kidneys', 'adrenals'], 'ratio': 0.6},
            'outer_ciliary': {'organs': ['lungs', 'lymphatic'], 'ratio': 0.8},
            'peripheral': {'organs': ['circulation', 'extremities'], 'ratio': 1.0}
        }

    def deep_spot_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive spot analysis with classification and health correlation.
        
        Args:
            image: Input iris image (grayscale or RGB)
            segmentation_data: Segmentation information including masks and centers
            
        Returns:
            Comprehensive spot analysis results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Get iris-only region
            iris_mask = segmentation_data.get("iris_mask", np.ones_like(gray))
            pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(gray))
            iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
            
            # Apply mask
            masked_image = cv2.bitwise_and(gray, gray, mask=iris_only_mask)
            
            # Multi-scale spot detection
            spots = self._detect_spots_multiscale(masked_image, iris_only_mask, segmentation_data)
            
            # Classify spots by type
            classified_spots = self._classify_spots(spots, masked_image)
            
            # Analyze spot patterns and distribution
            pattern_analysis = self._analyze_spot_patterns(classified_spots, segmentation_data)
            
            # Health correlation
            health_indicators = self._correlate_spots_to_health(classified_spots, pattern_analysis, segmentation_data)
            
            # Generate visualization
            visualization = self._create_spot_visualization(image, classified_spots, segmentation_data)
            
            return {
                'total_spots': len(classified_spots),
                'spot_types': self._count_spot_types(classified_spots),
                'spots_by_zone': self._group_spots_by_zone(classified_spots, segmentation_data),
                'pattern_analysis': pattern_analysis,
                'health_indicators': health_indicators,
                'visualization': visualization,
                'detailed_spots': classified_spots[:20]  # Limit for performance
            }
            
        except Exception as e:
            logger.error(f"Error in deep spot analysis: {str(e)}")
            return {'error': str(e)}

    def deep_line_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze radial and circumferential lines in the iris.
        
        Args:
            image: Input iris image
            segmentation_data: Segmentation information
            
        Returns:
            Comprehensive line analysis results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Get iris-only region
            iris_mask = segmentation_data.get("iris_mask", np.ones_like(gray))
            pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(gray))
            iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
            
            # Apply mask
            masked_image = cv2.bitwise_and(gray, gray, mask=iris_only_mask)
            
            # Detect different types of lines
            radial_lines = self._detect_radial_lines(masked_image, segmentation_data)
            circumferential_lines = self._detect_circumferential_lines(masked_image, segmentation_data)
            fiber_patterns = self._analyze_fiber_patterns(masked_image, segmentation_data)
            
            # Analyze line characteristics
            line_analysis = self._analyze_line_characteristics(radial_lines, circumferential_lines, fiber_patterns)
            
            # Health correlation
            health_correlation = self._correlate_lines_to_health(line_analysis, segmentation_data)
            
            # Generate visualization
            visualization = self._create_line_visualization(image, radial_lines, circumferential_lines, segmentation_data)
            
            return {
                'radial_lines': len(radial_lines),
                'circumferential_lines': len(circumferential_lines),
                'fiber_density': line_analysis.get('fiber_density', 0),
                'line_analysis': line_analysis,
                'health_correlation': health_correlation,
                'visualization': visualization
            }
            
        except Exception as e:
            logger.error(f"Error in deep line analysis: {str(e)}")
            return {'error': str(e)}

    def deep_color_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive color analysis with Ayurvedic interpretation.
        
        Args:
            image: Input iris image (RGB)
            segmentation_data: Segmentation information
            
        Returns:
            Comprehensive color analysis results
        """
        try:
            # Ensure RGB format
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = image.copy()
            
            # Get iris-only region
            iris_mask = segmentation_data.get("iris_mask", np.ones_like(image[:,:,0]))
            pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(image[:,:,0]))
            iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
            
            # Apply mask to RGB image
            masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=iris_only_mask)
            
            # Convert to different color spaces for analysis
            hsv_image = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2HSV)
            lab_image = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2LAB)
            
            # Detailed color clustering
            color_clusters = self._advanced_color_clustering(masked_rgb, iris_only_mask)
            
            # Ayurvedic color interpretation
            ayurvedic_analysis = self._ayurvedic_color_interpretation(hsv_image, iris_only_mask)
            
            # Color distribution by zones
            zonal_colors = self._analyze_color_by_zones(masked_rgb, segmentation_data)
            
            # Color harmony and balance analysis
            harmony_analysis = self._analyze_color_harmony(color_clusters)
            
            # Health indicators from colors
            health_indicators = self._correlate_colors_to_health(ayurvedic_analysis, zonal_colors)
            
            # Generate visualization
            visualization = self._create_color_visualization(image, color_clusters, segmentation_data)
            
            return {
                'dominant_colors': color_clusters,
                'ayurvedic_interpretation': ayurvedic_analysis,
                'zonal_color_distribution': zonal_colors,
                'color_harmony': harmony_analysis,
                'health_indicators': health_indicators,
                'visualization': visualization
            }
            
        except Exception as e:
            logger.error(f"Error in deep color analysis: {str(e)}")
            return {'error': str(e)}

    def comprehensive_texture_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive texture analysis including micro-patterns and constitution assessment.
        
        Args:
            image: Input iris image
            segmentation_data: Segmentation information
            
        Returns:
            Comprehensive texture analysis results
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Get iris-only region
            iris_mask = segmentation_data.get("iris_mask", np.ones_like(gray))
            pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(gray))
            iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
            
            # Apply mask
            masked_image = cv2.bitwise_and(gray, gray, mask=iris_only_mask)
            
            # Multiple texture analysis methods
            lbp_analysis = self._analyze_local_binary_patterns(masked_image, iris_only_mask)
            gabor_analysis = self._analyze_gabor_features(masked_image, iris_only_mask)
            fractal_analysis = self._analyze_fractal_dimension(masked_image, iris_only_mask)
            wavelet_analysis = self._analyze_wavelet_features(masked_image, iris_only_mask)
            
            # Fiber density and orientation
            fiber_analysis = self._analyze_fiber_structure(masked_image, segmentation_data)
            
            # Constitution assessment
            constitution_assessment = self._assess_constitution_from_texture(
                lbp_analysis, gabor_analysis, fiber_analysis
            )
            
            # Health correlation
            health_correlation = self._correlate_texture_to_health(
                lbp_analysis, fiber_analysis, constitution_assessment
            )
            
            # Generate visualization
            visualization = self._create_texture_visualization(image, fiber_analysis, segmentation_data)
            
            return {
                'lbp_features': lbp_analysis,
                'gabor_features': gabor_analysis,
                'fractal_dimension': fractal_analysis,
                'wavelet_features': wavelet_analysis,
                'fiber_analysis': fiber_analysis,
                'constitution_assessment': constitution_assessment,
                'health_correlation': health_correlation,
                'visualization': visualization
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive texture analysis: {str(e)}")
            return {'error': str(e)}

    def _detect_spots_multiscale(self, image: np.ndarray, mask: np.ndarray, 
                                segmentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect spots using multiple scales and methods."""
        spots = []
        
        # Method 1: Adaptive thresholding
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        binary = cv2.bitwise_and(binary, mask)
        
        # Method 2: Blob detection
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(binary)
        
        # Method 3: Contour-based detection
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.spot_min_size <= area <= self.spot_max_size:
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # Calculate additional properties
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Determine zone
                    zone = self._determine_zone(cx, cy, segmentation_data)
                    
                    spots.append({
                        'center': (cx, cy),
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'zone': zone,
                        'contour': contour
                    })
        
        return spots

    def _classify_spots(self, spots: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        """Classify spots into different types based on characteristics."""
        classified_spots = []
        
        for spot in spots:
            # Extract local features around the spot
            cx, cy = spot['center']
            area = spot['area']
            circularity = spot['circularity']
            
            # Extract local patch for intensity analysis
            patch_size = max(10, int(np.sqrt(area) * 2))
            x1, y1 = max(0, cx - patch_size//2), max(0, cy - patch_size//2)
            x2, y2 = min(image.shape[1], cx + patch_size//2), min(image.shape[0], cy + patch_size//2)
            
            if x2 > x1 and y2 > y1:
                patch = image[y1:y2, x1:x2]
                mean_intensity = np.mean(patch)
                std_intensity = np.std(patch)
            else:
                mean_intensity = std_intensity = 0
            
            # Classify based on characteristics
            if circularity > 0.7 and area < 50:
                spot_type = 'pigment_spot'
            elif circularity < 0.3:
                spot_type = 'lacuna'
            elif mean_intensity < 100:
                spot_type = 'dark_spot'
            elif std_intensity > 30:
                spot_type = 'textural_irregularity'
            else:
                spot_type = 'general_marking'
            
            spot['type'] = spot_type
            spot['intensity'] = mean_intensity
            spot['intensity_variation'] = std_intensity
            classified_spots.append(spot)
        
        return classified_spots

    def _analyze_spot_patterns(self, spots: List[Dict[str, Any]], 
                              segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial patterns in spot distribution."""
        if not spots:
            return {'pattern': 'no_spots', 'density': 0, 'distribution': 'none'}
        
        # Calculate spot density
        iris_center = segmentation_data.get("iris_center", (0, 0))
        iris_radius = segmentation_data.get("iris_radius", 1)
        iris_area = np.pi * iris_radius * iris_radius
        density = len(spots) / iris_area if iris_area > 0 else 0
        
        # Analyze spatial distribution
        positions = np.array([spot['center'] for spot in spots])
        
        # Calculate distances from center
        distances = [np.sqrt((pos[0] - iris_center[0])**2 + (pos[1] - iris_center[1])**2) 
                    for pos in positions]
        
        # Analyze clustering using DBSCAN
        if len(positions) > 2:
            clustering = DBSCAN(eps=20, min_samples=2).fit(positions)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        else:
            n_clusters = 0
        
        # Determine pattern type
        if density > 0.001:
            pattern = 'high_density'
        elif n_clusters > 2:
            pattern = 'clustered'
        elif np.std(distances) > iris_radius * 0.3:
            pattern = 'scattered'
        else:
            pattern = 'uniform'
        
        return {
            'pattern': pattern,
            'density': density,
            'clusters': n_clusters,
            'radial_distribution': np.histogram(distances, bins=5)[0].tolist()
        }

    def _correlate_spots_to_health(self, spots: List[Dict[str, Any]], 
                                  pattern_analysis: Dict[str, Any],
                                  segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate spot patterns to health indicators."""
        health_indicators = {
            'toxin_level': 'low',
            'inflammation_indicators': [],
            'organ_stress_zones': [],
            'recommendations': []
        }
        
        # Analyze by spot count and type
        spot_types = {}
        for spot in spots:
            spot_type = spot['type']
            zone = spot['zone']
            spot_types[spot_type] = spot_types.get(spot_type, 0) + 1
            
            # Map to organ systems
            if zone and zone in self.zone_mappings:
                organs = self.zone_mappings[zone]['organs']
                for organ in organs:
                    if organ not in health_indicators['organ_stress_zones']:
                        health_indicators['organ_stress_zones'].append(organ)
        
        # Determine toxin level
        total_spots = len(spots)
        if total_spots > 20:
            health_indicators['toxin_level'] = 'high'
            health_indicators['recommendations'].append('Consider detoxification protocols')
        elif total_spots > 10:
            health_indicators['toxin_level'] = 'moderate'
            health_indicators['recommendations'].append('Monitor toxin exposure and elimination')
        
        # Analyze inflammation
        dark_spots = spot_types.get('dark_spot', 0)
        if dark_spots > 5:
            health_indicators['inflammation_indicators'].append('Multiple dark spots suggest inflammatory processes')
        
        pattern = pattern_analysis.get('pattern', 'uniform')
        if pattern == 'clustered':
            health_indicators['inflammation_indicators'].append('Clustered spots may indicate localized stress')
        
        return health_indicators

    def _detect_radial_lines(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect radial lines emanating from the pupil."""
        lines = []
        
        # Get center and radius
        iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
        iris_radius = segmentation_data.get("iris_radius", min(image.shape)//4)
        
        # Create polar coordinate transformation
        center_x, center_y = iris_center
        
        # Use Hough Line Transform
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines
        hough_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if hough_lines is not None:
            for line in hough_lines:
                rho, theta = line[0]
                
                # Check if line passes through or near center
                distance_to_center = abs(rho - (center_x * np.cos(theta) + center_y * np.sin(theta)))
                
                if distance_to_center < 20:  # Line passes near center
                    # Calculate line endpoints
                    a, b = np.cos(theta), np.sin(theta)
                    x0, y0 = a * rho, b * rho
                    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
                    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
                    
                    lines.append({
                        'rho': rho,
                        'theta': theta,
                        'endpoints': ((x1, y1), (x2, y2)),
                        'type': 'radial'
                    })
        
        return lines

    def _detect_circumferential_lines(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect circumferential lines around the iris."""
        lines = []
        
        # Get center and radius
        iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
        iris_radius = segmentation_data.get("iris_radius", min(image.shape)//4)
        
        # Create circular masks at different radii
        for radius_ratio in [0.3, 0.5, 0.7, 0.9]:
            radius = int(iris_radius * radius_ratio)
            
            # Create ring mask
            mask = np.zeros_like(image)
            cv2.circle(mask, iris_center, radius + 5, 255, 10)
            cv2.circle(mask, iris_center, radius - 5, 0, 10)
            
            # Apply mask and detect edges
            masked = cv2.bitwise_and(image, mask)
            edges = cv2.Canny(masked, 50, 150)
            
            # Count edge pixels as a measure of circumferential line strength
            edge_count = np.sum(edges > 0)
            
            if edge_count > 100:  # Threshold for significant circumferential line
                lines.append({
                    'radius': radius,
                    'radius_ratio': radius_ratio,
                    'strength': edge_count,
                    'type': 'circumferential'
                })
        
        return lines

    def _analyze_fiber_patterns(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze iris fiber patterns and orientation."""
        # Use Gabor filters to detect fiber-like patterns
        filters = []
        for theta in range(0, 180, 20):  # Different orientations
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 10, 0.5, 0, ktype=cv2.CV_32F)
            filters.append(kernel)
        
        responses = []
        for kernel in filters:
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            responses.append(np.mean(filtered))
        
        # Find dominant fiber orientation
        dominant_angle = np.argmax(responses) * 20
        fiber_strength = max(responses)
        
        return {
            'dominant_orientation': dominant_angle,
            'fiber_strength': fiber_strength,
            'orientation_responses': responses
        }

    def _advanced_color_clustering(self, image: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
        """Perform advanced color clustering with multiple color spaces."""
        # Extract valid pixels
        valid_pixels = image[mask > 0]
        
        if len(valid_pixels) == 0:
            return []
        
        # Reshape for clustering
        pixels = valid_pixels.reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=min(self.color_clusters, len(pixels)), 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Get cluster information
        colors = []
        for i in range(kmeans.n_clusters):
            cluster_mask = labels == i
            cluster_pixels = pixels[cluster_mask]
            
            if len(cluster_pixels) > 0:
                color = kmeans.cluster_centers_[i].astype(int)
                percentage = len(cluster_pixels) / len(pixels)
                
                # Calculate color statistics
                std = np.std(cluster_pixels, axis=0)
                
                colors.append({
                    'color_rgb': color,
                    'percentage': percentage,
                    'pixel_count': len(cluster_pixels),
                    'color_variance': np.mean(std)
                })
        
        # Sort by percentage
        colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return colors

    def _ayurvedic_color_interpretation(self, hsv_image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Interpret colors according to Ayurvedic principles."""
        interpretation = {
            'dominant_dosha': 'Unknown',
            'secondary_dosha': 'Unknown',
            'constitution_type': 'Unknown',
            'color_balance': 'Unknown'
        }
        
        # Extract HSV values
        valid_hsv = hsv_image[mask > 0]
        
        if len(valid_hsv) == 0:
            return interpretation
        
        # Analyze color distribution
        dosha_counts = {'Vata': 0, 'Pitta': 0, 'Kapha': 0, 'Mixed': 0}
        
        for pixel in valid_hsv:
            h, s, v = pixel
            
            # Map to Ayurvedic colors
            for color_name, color_info in self.ayurvedic_colors.items():
                (h_min, s_min, v_min), (h_max, s_max, v_max) = color_info['range']
                
                if h_min <= h <= h_max and s_min <= s <= s_max and v_min <= v <= v_max:
                    dosha = color_info['dosha']
                    if dosha in dosha_counts:
                        dosha_counts[dosha] += 1
                    break
        
        # Determine dominant doshas
        total_pixels = sum(dosha_counts.values())
        if total_pixels > 0:
            dosha_percentages = {k: v/total_pixels for k, v in dosha_counts.items()}
            sorted_doshas = sorted(dosha_percentages.items(), key=lambda x: x[1], reverse=True)
            
            interpretation['dominant_dosha'] = sorted_doshas[0][0]
            if len(sorted_doshas) > 1:
                interpretation['secondary_dosha'] = sorted_doshas[1][0]
            
            # Determine constitution type
            if dosha_percentages['Mixed'] > 0.3:
                interpretation['constitution_type'] = 'Tri-dosha'
            elif dosha_percentages[interpretation['dominant_dosha']] > 0.6:
                interpretation['constitution_type'] = f"Mono-{interpretation['dominant_dosha']}"
            else:
                interpretation['constitution_type'] = f"Bi-dosha ({interpretation['dominant_dosha']}-{interpretation['secondary_dosha']})"
        
        return interpretation

    def _determine_zone(self, x: int, y: int, segmentation_data: Dict[str, Any]) -> str:
        """Determine which iris zone a point belongs to."""
        iris_center = segmentation_data.get("iris_center", (0, 0))
        iris_radius = segmentation_data.get("iris_radius", 1)
        pupil_radius = segmentation_data.get("pupil_radius", iris_radius * 0.2)
        
        # Calculate distance from center
        distance = np.sqrt((x - iris_center[0])**2 + (y - iris_center[1])**2)
        relative_distance = distance / iris_radius
        
        # Map to zones
        for zone_name, zone_info in self.zone_mappings.items():
            if relative_distance <= zone_info['ratio']:
                return zone_name
        
        return 'peripheral'

    def _create_spot_visualization(self, image: np.ndarray, spots: List[Dict[str, Any]], 
                                  segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create visualization of detected spots."""
        vis_image = image.copy()
        
        # Color map for different spot types
        color_map = {
            'pigment_spot': (255, 0, 0),     # Red
            'lacuna': (0, 255, 0),           # Green
            'dark_spot': (0, 0, 255),        # Blue
            'textural_irregularity': (255, 255, 0),  # Yellow
            'general_marking': (255, 0, 255)  # Magenta
        }
        
        for spot in spots:
            center = spot['center']
            spot_type = spot.get('type', 'general_marking')
            color = color_map.get(spot_type, (128, 128, 128))
            
            # Draw spot
            cv2.circle(vis_image, center, 3, color, -1)
            cv2.circle(vis_image, center, 5, color, 1)
        
        return vis_image

    def _create_line_visualization(self, image: np.ndarray, radial_lines: List[Dict[str, Any]], 
                                  circumferential_lines: List[Dict[str, Any]], 
                                  segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create visualization of detected lines."""
        vis_image = image.copy()
        
        # Draw radial lines in green
        for line in radial_lines:
            endpoints = line['endpoints']
            cv2.line(vis_image, endpoints[0], endpoints[1], (0, 255, 0), 1)
        
        # Draw circumferential lines in blue
        iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
        for line in circumferential_lines:
            radius = line['radius']
            cv2.circle(vis_image, iris_center, radius, (255, 0, 0), 1)
        
        return vis_image

    def _create_color_visualization(self, image: np.ndarray, color_clusters: List[Dict[str, Any]], 
                                   segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create color analysis visualization."""
        vis_image = image.copy()
        
        # Create color palette on the side
        palette_width = 100
        palette_height = image.shape[0]
        palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        # Fill palette with dominant colors
        y_step = palette_height // len(color_clusters) if color_clusters else palette_height
        
        for i, color_info in enumerate(color_clusters):
            color = color_info['color_rgb']
            y_start = i * y_step
            y_end = (i + 1) * y_step
            palette[y_start:y_end, :] = color
        
        # Combine image and palette
        combined = np.hstack([vis_image, palette])
        
        return combined

    def _create_texture_visualization(self, image: np.ndarray, fiber_analysis: Dict[str, Any], 
                                     segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create texture analysis visualization."""
        vis_image = image.copy()
        
        # Draw fiber orientation indicators
        iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
        dominant_orientation = fiber_analysis.get('dominant_orientation', 0)
        
        # Draw orientation lines
        angle_rad = np.radians(dominant_orientation)
        line_length = 50
        
        for i in range(8):  # Draw multiple orientation indicators
            angle = angle_rad + i * np.pi / 4
            end_x = int(iris_center[0] + line_length * np.cos(angle))
            end_y = int(iris_center[1] + line_length * np.sin(angle))
            cv2.line(vis_image, iris_center, (end_x, end_y), (0, 255, 255), 2)
        
        return vis_image

    # Additional helper methods for texture analysis, health correlation, etc.
    def _analyze_local_binary_patterns(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze Local Binary Patterns."""
        lbp = local_binary_pattern(image, self.texture_n_points, self.texture_radius, method='uniform')
        lbp_masked = lbp[mask > 0]
        
        if len(lbp_masked) == 0:
            return {}
        
        hist, _ = np.histogram(lbp_masked, bins=self.texture_n_points + 2, 
                              range=(0, self.texture_n_points + 2), density=True)
        
        return {
            'histogram': hist.tolist(),
            'uniformity': np.sum(hist**2),
            'entropy': -np.sum(hist * np.log2(hist + 1e-10))
        }

    def _analyze_gabor_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze Gabor filter responses."""
        responses = []
        
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 10, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            masked_response = filtered[mask > 0]
            
            if len(masked_response) > 0:
                responses.append({
                    'orientation': theta,
                    'mean_response': float(np.mean(masked_response)),
                    'std_response': float(np.std(masked_response))
                })
        
        return {'gabor_responses': responses}

    def _analyze_fractal_dimension(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate fractal dimension of texture."""
        # Simplified box-counting method
        masked_image = image.copy()
        masked_image[mask == 0] = 0
        
        # Convert to binary
        _, binary = cv2.threshold(masked_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            h, w = binary.shape
            scaled_h, scaled_w = h // scale, w // scale
            
            if scaled_h > 0 and scaled_w > 0:
                scaled = cv2.resize(binary, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                count = np.sum(scaled > 0)
                counts.append(count)
        
        if len(counts) > 1:
            # Calculate fractal dimension using linear regression
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(np.array(counts) + 1)
            
            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                return abs(slope)
        
        return 1.5  # Default fractal dimension

    def _analyze_wavelet_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze wavelet features (simplified implementation)."""
        # Apply different frequency filters as a proxy for wavelet analysis
        low_pass = cv2.GaussianBlur(image, (15, 15), 0)
        high_pass = image - low_pass
        
        low_pass_masked = low_pass[mask > 0]
        high_pass_masked = high_pass[mask > 0]
        
        if len(low_pass_masked) > 0 and len(high_pass_masked) > 0:
            return {
                'low_frequency_energy': float(np.mean(low_pass_masked**2)),
                'high_frequency_energy': float(np.mean(high_pass_masked**2)),
                'frequency_ratio': float(np.mean(high_pass_masked**2) / (np.mean(low_pass_masked**2) + 1e-10))
            }
        
        return {}

    def _analyze_fiber_structure(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze iris fiber structure and density."""
        # Use structure tensor to analyze fiber orientation and coherence
        
        # Calculate gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Structure tensor components
        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y
        
        # Apply Gaussian smoothing
        sigma = 2.0
        Ixx = gaussian_filter(Ixx, sigma)
        Iyy = gaussian_filter(Iyy, sigma)
        Ixy = gaussian_filter(Ixy, sigma)
        
        # Calculate coherence and orientation
        coherence = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2) / (Ixx + Iyy + 1e-10)
        orientation = 0.5 * np.arctan2(2*Ixy, Ixx - Iyy)
        
        # Get iris mask
        iris_mask = segmentation_data.get("iris_mask", np.ones_like(image))
        pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(image))
        iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
        
        # Calculate statistics
        coherence_masked = coherence[iris_only_mask > 0]
        orientation_masked = orientation[iris_only_mask > 0]
        
        if len(coherence_masked) > 0:
            return {
                'mean_coherence': float(np.mean(coherence_masked)),
                'fiber_density': float(np.mean(coherence_masked > 0.3)),  # Threshold for significant fibers
                'dominant_orientation': float(np.mean(orientation_masked)),
                'orientation_uniformity': float(1.0 - np.std(orientation_masked) / np.pi)
            }
        
        return {}

    def _assess_constitution_from_texture(self, lbp_analysis: Dict[str, Any], 
                                        gabor_analysis: Dict[str, Any], 
                                        fiber_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess Ayurvedic constitution based on texture features."""
        constitution = {
            'primary_dosha': 'Unknown',
            'constitution_strength': 0.0,
            'texture_characteristics': []
        }
        
        # Analyze fiber density and coherence
        fiber_density = fiber_analysis.get('fiber_density', 0)
        coherence = fiber_analysis.get('mean_coherence', 0)
        
        # LBP uniformity indicates texture regularity
        uniformity = lbp_analysis.get('uniformity', 0)
        
        # Constitution assessment rules
        if fiber_density > 0.7 and coherence > 0.6:
            constitution['primary_dosha'] = 'Kapha'
            constitution['constitution_strength'] = 0.8
            constitution['texture_characteristics'].append('Dense, well-organized fiber structure')
        elif fiber_density < 0.3 and uniformity < 0.1:
            constitution['primary_dosha'] = 'Vata'
            constitution['constitution_strength'] = 0.7
            constitution['texture_characteristics'].append('Loose, irregular fiber patterns')
        elif coherence > 0.4 and uniformity > 0.15:
            constitution['primary_dosha'] = 'Pitta'
            constitution['constitution_strength'] = 0.75
            constitution['texture_characteristics'].append('Moderate density with radial patterns')
        else:
            constitution['primary_dosha'] = 'Mixed'
            constitution['constitution_strength'] = 0.5
            constitution['texture_characteristics'].append('Balanced texture characteristics')
        
        return constitution

    def _correlate_texture_to_health(self, lbp_analysis: Dict[str, Any], 
                                   fiber_analysis: Dict[str, Any], 
                                   constitution: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate texture patterns to health indicators."""
        health_indicators = {
            'constitutional_strength': constitution.get('constitution_strength', 0),
            'stress_indicators': [],
            'recommendations': []
        }
        
        # Analyze fiber coherence for stress indicators
        coherence = fiber_analysis.get('mean_coherence', 0)
        if coherence < 0.2:
            health_indicators['stress_indicators'].append('Low fiber coherence may indicate constitutional weakness')
            health_indicators['recommendations'].append('Focus on strengthening constitutional practices')
        
        # Analyze texture uniformity
        uniformity = lbp_analysis.get('uniformity', 0)
        if uniformity < 0.05:
            health_indicators['stress_indicators'].append('Irregular texture patterns may indicate system imbalance')
            health_indicators['recommendations'].append('Consider lifestyle modifications for better balance')
        
        return health_indicators

    def _count_spot_types(self, spots: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count spots by type."""
        counts = {}
        for spot in spots:
            spot_type = spot.get('type', 'unknown')
            counts[spot_type] = counts.get(spot_type, 0) + 1
        return counts

    def _group_spots_by_zone(self, spots: List[Dict[str, Any]], 
                           segmentation_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group spots by iris zone."""
        zones = {}
        for spot in spots:
            zone = spot.get('zone', 'unknown')
            if zone not in zones:
                zones[zone] = []
            zones[zone].append(spot)
        return zones

    def _analyze_color_by_zones(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze color distribution across iris zones."""
        zonal_colors = {}
        
        iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
        iris_radius = segmentation_data.get("iris_radius", min(image.shape[:2])//4)
        pupil_radius = segmentation_data.get("pupil_radius", iris_radius * 0.2)
        
        for zone_name, zone_info in self.zone_mappings.items():
            # Create zone mask
            outer_radius = int(iris_radius * zone_info['ratio'])
            inner_radius = int(pupil_radius) if zone_name == 'pupillary_border' else int(iris_radius * 0.2)
            
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, iris_center, outer_radius, 255, -1)
            cv2.circle(mask, iris_center, inner_radius, 0, -1)
            
            # Extract colors in this zone
            zone_pixels = image[mask > 0]
            if len(zone_pixels) > 0:
                # Calculate dominant color
                pixels_reshaped = zone_pixels.reshape(-1, 3)
                kmeans = KMeans(n_clusters=min(3, len(pixels_reshaped)), random_state=42, n_init=10)
                kmeans.fit(pixels_reshaped)
                
                dominant_color = kmeans.cluster_centers_[0].astype(int)
                zonal_colors[zone_name] = {
                    'dominant_color': dominant_color.tolist(),
                    'pixel_count': len(zone_pixels)
                }
        
        return zonal_colors

    def _analyze_color_harmony(self, color_clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze color harmony and balance."""
        if len(color_clusters) < 2:
            return {'harmony_score': 1.0, 'balance': 'monochromatic'}
        
        # Calculate color distances
        colors = [cluster['color_rgb'] for cluster in color_clusters]
        distances = []
        
        for i in range(len(colors)):
            for j in range(i+1, len(colors)):
                # Euclidean distance in RGB space
                dist = np.sqrt(np.sum((colors[i] - colors[j])**2))
                distances.append(dist)
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Harmony score based on color distribution
        harmony_score = 1.0 - (std_distance / (mean_distance + 1e-10))
        
        # Determine balance type
        if mean_distance < 50:
            balance = 'monochromatic'
        elif mean_distance < 100:
            balance = 'analogous'
        else:
            balance = 'complementary'
        
        return {
            'harmony_score': float(harmony_score),
            'balance': balance,
            'color_diversity': len(color_clusters)
        }

    def _correlate_colors_to_health(self, ayurvedic_analysis: Dict[str, Any], 
                                  zonal_colors: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate color patterns to health indicators."""
        health_indicators = {
            'constitutional_assessment': ayurvedic_analysis.get('constitution_type', 'Unknown'),
            'dosha_balance': {
                'primary': ayurvedic_analysis.get('dominant_dosha', 'Unknown'),
                'secondary': ayurvedic_analysis.get('secondary_dosha', 'Unknown')
            },
            'health_recommendations': []
        }
        
        # Generate recommendations based on constitution
        constitution = ayurvedic_analysis.get('constitution_type', 'Unknown')
        if 'Vata' in constitution:
            health_indicators['health_recommendations'].append('Focus on grounding practices and warm, nourishing foods')
        elif 'Pitta' in constitution:
            health_indicators['health_recommendations'].append('Emphasize cooling practices and avoid excessive heat')
        elif 'Kapha' in constitution:
            health_indicators['health_recommendations'].append('Incorporate stimulating activities and light, warm foods')
        
        return health_indicators

    def _analyze_line_characteristics(self, radial_lines: List[Dict[str, Any]], 
                                    circumferential_lines: List[Dict[str, Any]], 
                                    fiber_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of detected lines."""
        return {
            'radial_density': len(radial_lines),
            'circumferential_density': len(circumferential_lines),
            'fiber_orientation': fiber_patterns.get('dominant_orientation', 0),
            'fiber_strength': fiber_patterns.get('fiber_strength', 0)
        }

    def _correlate_lines_to_health(self, line_analysis: Dict[str, Any], 
                                 segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate line patterns to health indicators."""
        health_correlation = {
            'constitutional_indicators': [],
            'stress_patterns': [],
            'recommendations': []
        }
        
        radial_density = line_analysis.get('radial_density', 0)
        circumferential_density = line_analysis.get('circumferential_density', 0)
        
        if radial_density > 10:
            health_correlation['constitutional_indicators'].append('High radial line density suggests Vata constitution')
        elif circumferential_density > 5:
            health_correlation['constitutional_indicators'].append('Prominent circumferential lines suggest Kapha constitution')
        
        if radial_density > 15:
            health_correlation['stress_patterns'].append('Excessive radial lines may indicate nervous system stress')
            health_correlation['recommendations'].append('Consider stress reduction techniques')
        
        return health_correlation
