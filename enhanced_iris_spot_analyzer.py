"""
Enhanced Iris Spot Analysis Module for IridoVeda.

This module provides deep analysis of iris features including:
- Advanced spot detection and classification  
- Line and pattern recognition
- Enhanced color analysis with Ayurvedic interpretation
- Micro-feature detection and health correlation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.cluster import KMeans, DBSCAN
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIrisSpotAnalyzer:
    """
    Enhanced iris spot analysis class for deep feature extraction and health correlation.
    """
    
    def __init__(self):
        """Initialize the enhanced analyzer with predefined parameters."""
        self.spot_min_size = 3
        self.spot_max_size = 500
        self.line_min_length = 15
        self.color_clusters = 6
        
        # Enhanced spot classification parameters
        self.circularity_threshold = 0.6
        self.intensity_threshold = 80
        self.size_thresholds = {
            'micro_spot': (3, 15),
            'small_spot': (15, 50),
            'medium_spot': (50, 150),
            'large_spot': (150, 500)
        }
        
        # Ayurvedic interpretations for spot characteristics
        self.spot_health_mapping = {
            'pigment_spots': {
                'zones': {
                    'pupillary_border': 'digestive_toxins',
                    'inner_ciliary': 'liver_congestion', 
                    'middle_ciliary': 'kidney_stress',
                    'outer_ciliary': 'lymphatic_congestion',
                    'peripheral': 'circulation_issues'
                },
                'count_interpretation': {
                    'low': 'minimal_toxin_accumulation',
                    'moderate': 'moderate_ama_buildup',
                    'high': 'significant_toxin_load'
                }
            },
            'dark_spots': {
                'significance': 'inflammatory_processes',
                'recommendations': ['anti_inflammatory_herbs', 'cooling_foods', 'stress_reduction']
            },
            'lacunas': {
                'significance': 'tissue_weakness',
                'recommendations': ['tissue_strengthening', 'constitutional_support']
            }
        }
        
        # Zone mappings for health correlation
        self.zone_health_mapping = {
            'pupillary_border': {
                'organs': ['stomach', 'duodenum', 'pylorus'],
                'systems': ['digestive_fire', 'agni'],
                'radius_ratio': (0.0, 0.3)
            },
            'inner_ciliary': {
                'organs': ['liver', 'gallbladder', 'pancreas'],
                'systems': ['metabolism', 'pitta_processing'],
                'radius_ratio': (0.3, 0.5)
            },
            'middle_ciliary': {
                'organs': ['kidneys', 'adrenals', 'intestines'],
                'systems': ['elimination', 'water_balance'],
                'radius_ratio': (0.5, 0.7)
            },
            'outer_ciliary': {
                'organs': ['lungs', 'heart', 'circulation'],
                'systems': ['oxygenation', 'prana_flow'],
                'radius_ratio': (0.7, 0.9)
            },
            'peripheral': {
                'organs': ['extremities', 'skin', 'lymph'],
                'systems': ['circulation', 'immunity'],
                'radius_ratio': (0.9, 1.0)
            }
        }

    def comprehensive_spot_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive spot analysis with deep classification and health correlation.
        
        Args:
            image: Input iris image (RGB or grayscale)
            segmentation_data: Segmentation information including masks and centers
            
        Returns:
            Comprehensive spot analysis results
        """
        try:
            logger.info("Starting comprehensive spot analysis...")
            
            # Prepare image for analysis
            analysis_image = self._prepare_image_for_analysis(image, segmentation_data)
            
            if analysis_image is None:
                return {'error': 'Failed to prepare image for analysis'}
            
            # Multi-scale spot detection
            detected_spots = self._enhanced_spot_detection(analysis_image, segmentation_data)
            
            # Advanced spot classification
            classified_spots = self._advanced_spot_classification(detected_spots, analysis_image, segmentation_data)
            
            # Pattern analysis
            pattern_analysis = self._analyze_spot_patterns(classified_spots, segmentation_data)
            
            # Zone-based analysis
            zone_analysis = self._analyze_spots_by_zones(classified_spots, segmentation_data)
            
            # Health correlation with Ayurvedic principles
            health_assessment = self._comprehensive_health_correlation(
                classified_spots, pattern_analysis, zone_analysis, segmentation_data
            )
            
            # Generate enhanced visualizations
            visualizations = self._create_enhanced_visualizations(
                image, classified_spots, pattern_analysis, segmentation_data
            )
            
            # Compile comprehensive results
            results = {
                'analysis_summary': {
                    'total_spots': len(classified_spots),
                    'analysis_quality': self._assess_analysis_quality(classified_spots, segmentation_data),
                    'processing_notes': []
                },
                'spot_classification': {
                    'by_type': self._count_spots_by_type(classified_spots),
                    'by_size': self._count_spots_by_size(classified_spots),
                    'by_zone': self._count_spots_by_zone(classified_spots)
                },
                'pattern_analysis': pattern_analysis,
                'zone_analysis': zone_analysis,
                'health_assessment': health_assessment,
                'visualizations': visualizations,
                'detailed_spots': classified_spots[:30]  # Limit for performance
            }
            
            logger.info(f"Spot analysis completed: {len(classified_spots)} spots detected")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive spot analysis: {str(e)}")
            return {'error': str(e)}

    def enhanced_line_and_texture_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze iris lines, fibers, and texture patterns for constitutional assessment.
        
        Args:
            image: Input iris image
            segmentation_data: Segmentation information
            
        Returns:
            Comprehensive line and texture analysis results
        """
        try:
            logger.info("Starting enhanced line and texture analysis...")
            
            # Prepare image
            analysis_image = self._prepare_image_for_analysis(image, segmentation_data)
            
            if analysis_image is None:
                return {'error': 'Failed to prepare image for analysis'}
            
            # Radial pattern analysis
            radial_analysis = self._analyze_radial_patterns(analysis_image, segmentation_data)
            
            # Circumferential pattern analysis
            circumferential_analysis = self._analyze_circumferential_patterns(analysis_image, segmentation_data)
            
            # Fiber density and orientation analysis
            fiber_analysis = self._analyze_fiber_patterns(analysis_image, segmentation_data)
            
            # Texture analysis using multiple methods
            texture_analysis = self._comprehensive_texture_analysis(analysis_image, segmentation_data)
            
            # Constitutional assessment based on patterns
            constitutional_assessment = self._assess_constitution_from_patterns(
                radial_analysis, circumferential_analysis, fiber_analysis, texture_analysis
            )
            
            # Health implications
            health_implications = self._correlate_patterns_to_health(
                radial_analysis, circumferential_analysis, fiber_analysis, constitutional_assessment
            )
            
            # Create visualizations
            visualizations = self._create_pattern_visualizations(
                image, radial_analysis, circumferential_analysis, fiber_analysis, segmentation_data
            )
            
            return {
                'radial_patterns': radial_analysis,
                'circumferential_patterns': circumferential_analysis,
                'fiber_analysis': fiber_analysis,
                'texture_analysis': texture_analysis,
                'constitutional_assessment': constitutional_assessment,
                'health_implications': health_implications,
                'visualizations': visualizations
            }
            
        except Exception as e:
            logger.error(f"Error in line and texture analysis: {str(e)}")
            return {'error': str(e)}

    def deep_color_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deep color analysis with Ayurvedic constitutional interpretation.
        
        Args:
            image: Input iris image (RGB)
            segmentation_data: Segmentation information
            
        Returns:
            Deep color analysis results
        """
        try:
            logger.info("Starting deep color analysis...")
            
            # Ensure RGB format
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = image.copy()
            
            # Get iris-only region
            iris_mask = segmentation_data.get("iris_mask", np.ones_like(image[:,:,0]))
            pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(image[:,:,0]))
            iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
            
            # Advanced color extraction
            color_analysis = self._advanced_color_extraction(rgb_image, iris_only_mask)
            
            # Zone-based color analysis
            zonal_color_analysis = self._analyze_colors_by_zones(rgb_image, segmentation_data)
            
            # Ayurvedic color interpretation
            ayurvedic_interpretation = self._ayurvedic_color_interpretation(rgb_image, iris_only_mask)
            
            # Constitutional assessment from colors
            constitutional_assessment = self._assess_constitution_from_colors(color_analysis, ayurvedic_interpretation)
            
            # Health indicators from color patterns
            health_indicators = self._derive_health_indicators_from_colors(
                color_analysis, zonal_color_analysis, constitutional_assessment
            )
            
            # Create color visualizations
            visualizations = self._create_color_visualizations(
                rgb_image, color_analysis, zonal_color_analysis, segmentation_data
            )
            
            return {
                'color_analysis': color_analysis,
                'zonal_colors': zonal_color_analysis,
                'ayurvedic_interpretation': ayurvedic_interpretation,
                'constitutional_assessment': constitutional_assessment,
                'health_indicators': health_indicators,
                'visualizations': visualizations
            }
            
        except Exception as e:
            logger.error(f"Error in deep color analysis: {str(e)}")
            return {'error': str(e)}

    def _prepare_image_for_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Prepare image for analysis with enhanced preprocessing."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Get masks
            iris_mask = segmentation_data.get("iris_mask", np.ones_like(gray))
            pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(gray))
            
            # Create iris-only mask
            iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
            
            # Apply mask
            masked_image = cv2.bitwise_and(gray, gray, mask=iris_only_mask)
            
            # Enhanced preprocessing
            # 1. Noise reduction
            denoised = cv2.bilateralFilter(masked_image, 9, 75, 75)
            
            # 2. Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Restore mask
            final_image = cv2.bitwise_and(enhanced, enhanced, mask=iris_only_mask)
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error preparing image for analysis: {str(e)}")
            return None

    def _enhanced_spot_detection(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced multi-method spot detection."""
        spots = []
        
        # Get iris mask
        iris_mask = segmentation_data.get("iris_mask", np.ones_like(image))
        pupil_mask = segmentation_data.get("pupil_mask", np.zeros_like(image))
        iris_only_mask = cv2.subtract(iris_mask, pupil_mask)
        
        # Method 1: Adaptive thresholding with multiple block sizes
        for block_size in [7, 11, 15]:
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, 2
            )
            binary = cv2.bitwise_and(binary, iris_only_mask)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.spot_min_size <= area <= self.spot_max_size:
                    spot_data = self._extract_spot_features(contour, image, segmentation_data)
                    if spot_data:
                        spots.append(spot_data)
        
        # Method 2: Top-hat filtering for bright spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        tophat = cv2.bitwise_and(tophat, iris_only_mask)
        
        _, binary_tophat = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_tophat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.spot_min_size <= area <= self.spot_max_size:
                spot_data = self._extract_spot_features(contour, image, segmentation_data)
                if spot_data:
                    spot_data['detection_method'] = 'tophat'
                    spots.append(spot_data)
        
        # Remove duplicate spots (same location detected by different methods)
        unique_spots = self._remove_duplicate_spots(spots)
        
        return unique_spots

    def _extract_spot_features(self, contour: np.ndarray, image: np.ndarray, 
                              segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features for a detected spot."""
        try:
            # Basic geometric features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return None
            
            # Center of mass
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                return None
                
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Shape features
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Position relative to iris center
            iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
            iris_radius = segmentation_data.get("iris_radius", min(image.shape)//4)
            
            distance_from_center = np.sqrt((cx - iris_center[0])**2 + (cy - iris_center[1])**2)
            relative_distance = distance_from_center / iris_radius if iris_radius > 0 else 0
            
            # Angular position
            angle = np.arctan2(cy - iris_center[1], cx - iris_center[0])
            angle_degrees = np.degrees(angle) % 360
            
            # Intensity features
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            spot_pixels = image[mask == 255]
            
            if len(spot_pixels) == 0:
                return None
            
            mean_intensity = np.mean(spot_pixels)
            std_intensity = np.std(spot_pixels)
            min_intensity = np.min(spot_pixels)
            max_intensity = np.max(spot_pixels)
            
            # Local contrast
            # Create a larger mask around the spot
            dilated_mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
            background_mask = cv2.subtract(dilated_mask, mask)
            background_pixels = image[background_mask == 255]
            
            if len(background_pixels) > 0:
                background_intensity = np.mean(background_pixels)
                local_contrast = abs(mean_intensity - background_intensity)
            else:
                local_contrast = 0
            
            # Zone determination
            zone = self._determine_zone_from_distance(relative_distance)
            
            return {
                'center': (cx, cy),
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'distance_from_center': distance_from_center,
                'relative_distance': relative_distance,
                'angle_degrees': angle_degrees,
                'zone': zone,
                'intensity_features': {
                    'mean': mean_intensity,
                    'std': std_intensity,
                    'min': min_intensity,
                    'max': max_intensity,
                    'local_contrast': local_contrast
                },
                'contour': contour,
                'detection_method': 'adaptive_threshold'
            }
            
        except Exception as e:
            logger.error(f"Error extracting spot features: {str(e)}")
            return None

    def _advanced_spot_classification(self, spots: List[Dict[str, Any]], 
                                    image: np.ndarray, 
                                    segmentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced classification of detected spots."""
        classified_spots = []
        
        for spot in spots:
            # Get spot characteristics
            circularity = spot.get('circularity', 0)
            area = spot.get('area', 0)
            intensity_features = spot.get('intensity_features', {})
            mean_intensity = intensity_features.get('mean', 128)
            local_contrast = intensity_features.get('local_contrast', 0)
            solidity = spot.get('solidity', 0)
            
            # Classification logic
            spot_type = 'unknown'
            confidence = 0.0
            characteristics = []
            
            # Pigment spot classification
            if (circularity > 0.7 and 
                area < 100 and 
                mean_intensity < 100 and 
                local_contrast > 20):
                spot_type = 'pigment_spot'
                confidence = 0.8
                characteristics.append('circular')
                characteristics.append('dark')
                characteristics.append('well_defined')
            
            # Lacuna classification (tissue gap)
            elif (circularity < 0.4 and 
                  solidity < 0.7 and 
                  area > 50):
                spot_type = 'lacuna'
                confidence = 0.7
                characteristics.append('irregular_shape')
                characteristics.append('tissue_gap')
            
            # Dark spot classification
            elif (mean_intensity < self.intensity_threshold and 
                  local_contrast > 15):
                spot_type = 'dark_spot'
                confidence = 0.6
                characteristics.append('dark')
                characteristics.append('contrasted')
            
            # Light spot classification
            elif (mean_intensity > 180 and 
                  local_contrast > 15):
                spot_type = 'light_spot'
                confidence = 0.6
                characteristics.append('bright')
                characteristics.append('contrasted')
            
            # Textural irregularity
            elif intensity_features.get('std', 0) > 30:
                spot_type = 'textural_irregularity'
                confidence = 0.5
                characteristics.append('irregular_texture')
            
            # General marking (default)
            else:
                spot_type = 'general_marking'
                confidence = 0.3
                characteristics.append('general')
            
            # Size classification
            size_category = self._classify_spot_size(area)
            
            # Add classification results to spot data
            spot['classification'] = {
                'type': spot_type,
                'confidence': confidence,
                'characteristics': characteristics,
                'size_category': size_category
            }
            
            classified_spots.append(spot)
        
        return classified_spots

    def _analyze_spot_patterns(self, spots: List[Dict[str, Any]], 
                              segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial and statistical patterns in spot distribution."""
        if not spots:
            return {
                'pattern_type': 'no_spots',
                'density': 0,
                'distribution_analysis': {},
                'clustering_analysis': {},
                'symmetry_analysis': {}
            }
        
        # Calculate spot density
        iris_center = segmentation_data.get("iris_center", (0, 0))
        iris_radius = segmentation_data.get("iris_radius", 1)
        iris_area = np.pi * iris_radius * iris_radius
        density = len(spots) / iris_area if iris_area > 0 else 0
        
        # Extract positions
        positions = np.array([spot['center'] for spot in spots])
        distances = [spot['distance_from_center'] for spot in spots]
        angles = [spot['angle_degrees'] for spot in spots]
        
        # Distribution analysis
        distribution_analysis = {
            'radial_distribution': self._analyze_radial_distribution(distances, iris_radius),
            'angular_distribution': self._analyze_angular_distribution(angles),
            'zone_distribution': self._analyze_zone_distribution(spots)
        }
        
        # Clustering analysis
        clustering_analysis = self._analyze_spot_clustering(positions)
        
        # Symmetry analysis
        symmetry_analysis = self._analyze_spot_symmetry(positions, iris_center)
        
        # Determine overall pattern type
        pattern_type = self._determine_overall_pattern(
            density, clustering_analysis, distribution_analysis, symmetry_analysis
        )
        
        return {
            'pattern_type': pattern_type,
            'density': density,
            'total_spots': len(spots),
            'distribution_analysis': distribution_analysis,
            'clustering_analysis': clustering_analysis,
            'symmetry_analysis': symmetry_analysis
        }

    def _analyze_spots_by_zones(self, spots: List[Dict[str, Any]], 
                               segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spots grouped by iris zones."""
        zone_analysis = {}
        
        # Group spots by zone
        spots_by_zone = {}
        for spot in spots:
            zone = spot.get('zone', 'unknown')
            if zone not in spots_by_zone:
                spots_by_zone[zone] = []
            spots_by_zone[zone].append(spot)
        
        # Analyze each zone
        for zone_name, zone_spots in spots_by_zone.items():
            if zone_name in self.zone_health_mapping:
                zone_info = self.zone_health_mapping[zone_name]
                
                # Count by spot type
                type_counts = {}
                for spot in zone_spots:
                    spot_type = spot.get('classification', {}).get('type', 'unknown')
                    type_counts[spot_type] = type_counts.get(spot_type, 0) + 1
                
                # Calculate zone-specific metrics
                total_spots = len(zone_spots)
                zone_density = total_spots  # Simplified density calculation
                
                # Health implications for this zone
                health_implications = self._analyze_zone_health_implications(
                    zone_name, zone_spots, zone_info
                )
                
                zone_analysis[zone_name] = {
                    'spot_count': total_spots,
                    'spot_types': type_counts,
                    'density': zone_density,
                    'associated_organs': zone_info['organs'],
                    'associated_systems': zone_info['systems'],
                    'health_implications': health_implications
                }
        
        return zone_analysis

    def _comprehensive_health_correlation(self, spots: List[Dict[str, Any]], 
                                        pattern_analysis: Dict[str, Any],
                                        zone_analysis: Dict[str, Any],
                                        segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive health correlation based on spot analysis."""
        health_assessment = {
            'overall_assessment': {},
            'toxin_indicators': {},
            'constitutional_indicators': {},
            'organ_system_analysis': {},
            'recommendations': []
        }
        
        # Overall assessment
        total_spots = len(spots)
        density = pattern_analysis.get('density', 0)
        
        # Toxin level assessment
        if total_spots <= 5:
            toxin_level = 'minimal'
            toxin_description = 'Low spot count indicates good elimination and minimal toxin accumulation'
        elif total_spots <= 15:
            toxin_level = 'moderate'
            toxin_description = 'Moderate spot count suggests some toxin accumulation requiring attention'
        else:
            toxin_level = 'significant'
            toxin_description = 'High spot count indicates substantial toxin load requiring detoxification'
        
        health_assessment['toxin_indicators'] = {
            'level': toxin_level,
            'description': toxin_description,
            'spot_count': total_spots,
            'density': density
        }
        
        # Constitutional indicators
        pattern_type = pattern_analysis.get('pattern_type', 'unknown')
        constitutional_indicators = self._derive_constitutional_indicators(
            spots, pattern_analysis, zone_analysis
        )
        
        health_assessment['constitutional_indicators'] = constitutional_indicators
        
        # Organ system analysis
        organ_analysis = {}
        for zone_name, zone_data in zone_analysis.items():
            for organ in zone_data.get('associated_organs', []):
                if organ not in organ_analysis:
                    organ_analysis[organ] = {
                        'stress_level': 'normal',
                        'indicators': [],
                        'recommendations': []
                    }
                
                spot_count = zone_data.get('spot_count', 0)
                if spot_count > 3:
                    organ_analysis[organ]['stress_level'] = 'elevated'
                    organ_analysis[organ]['indicators'].append(f'Multiple spots in {zone_name} zone')
                    organ_analysis[organ]['recommendations'].append(f'Support {organ} function')
        
        health_assessment['organ_system_analysis'] = organ_analysis
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            health_assessment, pattern_analysis, zone_analysis
        )
        
        health_assessment['recommendations'] = recommendations
        
        # Overall assessment summary
        health_assessment['overall_assessment'] = {
            'constitutional_strength': self._assess_constitutional_strength(spots, pattern_analysis),
            'elimination_efficiency': self._assess_elimination_efficiency(spots, zone_analysis),
            'inflammatory_markers': self._assess_inflammatory_markers(spots),
            'priority_areas': self._identify_priority_areas(zone_analysis)
        }
        
        return health_assessment

    def _create_enhanced_visualizations(self, image: np.ndarray, 
                                      spots: List[Dict[str, Any]], 
                                      pattern_analysis: Dict[str, Any],
                                      segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced visualizations for spot analysis."""
        visualizations = {}
        
        # Create spot classification visualization
        spot_vis = self._create_spot_classification_visualization(image, spots, segmentation_data)
        visualizations['spot_classification'] = spot_vis
        
        # Create zone-based visualization
        zone_vis = self._create_zone_visualization(image, spots, segmentation_data)
        visualizations['zone_analysis'] = zone_vis
        
        # Create pattern visualization
        pattern_vis = self._create_pattern_visualization(image, spots, pattern_analysis, segmentation_data)
        visualizations['pattern_analysis'] = pattern_vis
        
        return visualizations

    # Helper methods for analysis
    
    def _determine_zone_from_distance(self, relative_distance: float) -> str:
        """Determine iris zone based on relative distance from center."""
        if relative_distance <= 0.3:
            return 'pupillary_border'
        elif relative_distance <= 0.5:
            return 'inner_ciliary'
        elif relative_distance <= 0.7:
            return 'middle_ciliary'
        elif relative_distance <= 0.9:
            return 'outer_ciliary'
        else:
            return 'peripheral'

    def _classify_spot_size(self, area: float) -> str:
        """Classify spot size category."""
        for size_name, (min_size, max_size) in self.size_thresholds.items():
            if min_size <= area <= max_size:
                return size_name
        return 'unknown'

    def _remove_duplicate_spots(self, spots: List[Dict[str, Any]], 
                               distance_threshold: float = 10.0) -> List[Dict[str, Any]]:
        """Remove duplicate spots detected by different methods."""
        if not spots:
            return spots
        
        unique_spots = []
        
        for spot in spots:
            center = spot['center']
            is_duplicate = False
            
            for unique_spot in unique_spots:
                unique_center = unique_spot['center']
                distance = np.sqrt((center[0] - unique_center[0])**2 + 
                                 (center[1] - unique_center[1])**2)
                
                if distance < distance_threshold:
                    # Keep the spot with higher confidence or larger area
                    current_confidence = spot.get('classification', {}).get('confidence', 0)
                    unique_confidence = unique_spot.get('classification', {}).get('confidence', 0)
                    
                    if current_confidence > unique_confidence:
                        # Replace with current spot
                        unique_spots.remove(unique_spot)
                        unique_spots.append(spot)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_spots.append(spot)
        
        return unique_spots

    def _analyze_radial_distribution(self, distances: List[float], iris_radius: float) -> Dict[str, Any]:
        """Analyze radial distribution of spots."""
        if not distances:
            return {}
        
        # Create histogram of distances
        hist, bin_edges = np.histogram(distances, bins=5, range=(0, iris_radius))
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances))
        }

    def _analyze_angular_distribution(self, angles: List[float]) -> Dict[str, Any]:
        """Analyze angular distribution of spots."""
        if not angles:
            return {}
        
        # Create histogram of angles
        hist, bin_edges = np.histogram(angles, bins=8, range=(0, 360))
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'uniformity': float(np.std(hist) / (np.mean(hist) + 1e-10))
        }

    def _analyze_zone_distribution(self, spots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of spots across zones."""
        zone_counts = {}
        for spot in spots:
            zone = spot.get('zone', 'unknown')
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        return zone_counts

    def _analyze_spot_clustering(self, positions: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering patterns in spot positions."""
        if len(positions) < 2:
            return {'clusters': 0, 'clustering_coefficient': 0}
        
        try:
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=20, min_samples=2).fit(positions)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            # Calculate clustering coefficient
            noise_points = sum(1 for label in clustering.labels_ if label == -1)
            clustering_coefficient = 1 - (noise_points / len(positions))
            
            return {
                'clusters': n_clusters,
                'clustering_coefficient': clustering_coefficient,
                'noise_points': noise_points
            }
        except:
            return {'clusters': 0, 'clustering_coefficient': 0}

    def _analyze_spot_symmetry(self, positions: np.ndarray, iris_center: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze symmetry of spot distribution."""
        if len(positions) < 2:
            return {'symmetry_score': 0}
        
        # Calculate positions relative to center
        relative_positions = positions - np.array(iris_center)
        
        # Analyze bilateral symmetry (left-right)
        left_spots = sum(1 for pos in relative_positions if pos[0] < 0)
        right_spots = sum(1 for pos in relative_positions if pos[0] > 0)
        
        if left_spots + right_spots > 0:
            symmetry_score = 1 - abs(left_spots - right_spots) / (left_spots + right_spots)
        else:
            symmetry_score = 0
        
        return {
            'symmetry_score': symmetry_score,
            'left_spots': left_spots,
            'right_spots': right_spots
        }

    def _determine_overall_pattern(self, density: float, 
                                 clustering_analysis: Dict[str, Any],
                                 distribution_analysis: Dict[str, Any],
                                 symmetry_analysis: Dict[str, Any]) -> str:
        """Determine overall pattern type."""
        clusters = clustering_analysis.get('clusters', 0)
        clustering_coeff = clustering_analysis.get('clustering_coefficient', 0)
        symmetry_score = symmetry_analysis.get('symmetry_score', 0)
        
        if density > 0.001:
            return 'high_density'
        elif clusters > 2:
            return 'clustered'
        elif clustering_coeff < 0.3:
            return 'scattered'
        elif symmetry_score > 0.7:
            return 'symmetric'
        else:
            return 'random'

    def _analyze_zone_health_implications(self, zone_name: str, 
                                        zone_spots: List[Dict[str, Any]], 
                                        zone_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze health implications for a specific zone."""
        implications = {
            'stress_level': 'normal',
            'concerns': [],
            'recommendations': []
        }
        
        spot_count = len(zone_spots)
        organs = zone_info.get('organs', [])
        
        # Assess stress level based on spot count
        if spot_count > 5:
            implications['stress_level'] = 'high'
            implications['concerns'].append(f'High concentration of spots in {zone_name}')
            implications['recommendations'].append(f'Focus on supporting {", ".join(organs)}')
        elif spot_count > 2:
            implications['stress_level'] = 'moderate'
            implications['concerns'].append(f'Moderate spot concentration in {zone_name}')
            implications['recommendations'].append(f'Monitor {", ".join(organs)} function')
        
        # Analyze spot types in this zone
        spot_types = {}
        for spot in zone_spots:
            spot_type = spot.get('classification', {}).get('type', 'unknown')
            spot_types[spot_type] = spot_types.get(spot_type, 0) + 1
        
        # Add specific concerns based on spot types
        if spot_types.get('pigment_spot', 0) > 2:
            implications['concerns'].append('Multiple pigment spots suggest toxin accumulation')
            implications['recommendations'].append('Consider detoxification support')
        
        if spot_types.get('dark_spot', 0) > 1:
            implications['concerns'].append('Dark spots may indicate inflammatory processes')
            implications['recommendations'].append('Anti-inflammatory support recommended')
        
        return implications

    def _count_spots_by_type(self, spots: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count spots by classification type."""
        counts = {}
        for spot in spots:
            spot_type = spot.get('classification', {}).get('type', 'unknown')
            counts[spot_type] = counts.get(spot_type, 0) + 1
        return counts

    def _count_spots_by_size(self, spots: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count spots by size category."""
        counts = {}
        for spot in spots:
            size_category = spot.get('classification', {}).get('size_category', 'unknown')
            counts[size_category] = counts.get(size_category, 0) + 1
        return counts

    def _count_spots_by_zone(self, spots: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count spots by iris zone."""
        counts = {}
        for spot in spots:
            zone = spot.get('zone', 'unknown')
            counts[zone] = counts.get(zone, 0) + 1
        return counts

    def _assess_analysis_quality(self, spots: List[Dict[str, Any]], 
                               segmentation_data: Dict[str, Any]) -> str:
        """Assess the quality of the analysis."""
        iris_radius = segmentation_data.get("iris_radius", 0)
        
        if iris_radius < 50:
            return 'low_resolution'
        elif len(spots) == 0:
            return 'no_features_detected'
        else:
            return 'good'

    def _create_spot_classification_visualization(self, image: np.ndarray, 
                                                spots: List[Dict[str, Any]], 
                                                segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create visualization showing spot classifications."""
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Color map for different spot types
        color_map = {
            'pigment_spot': (255, 0, 0),        # Red
            'lacuna': (0, 255, 0),              # Green  
            'dark_spot': (0, 0, 255),           # Blue
            'light_spot': (255, 255, 0),        # Yellow
            'textural_irregularity': (255, 0, 255),  # Magenta
            'general_marking': (128, 128, 128)   # Gray
        }
        
        for spot in spots:
            center = spot['center']
            spot_type = spot.get('classification', {}).get('type', 'general_marking')
            confidence = spot.get('classification', {}).get('confidence', 0)
            
            color = color_map.get(spot_type, (128, 128, 128))
            
            # Draw spot with color coding
            radius = max(3, int(np.sqrt(spot.get('area', 25)) / 2))
            cv2.circle(vis_image, center, radius, color, -1)
            
            # Draw confidence ring
            ring_thickness = max(1, int(confidence * 3))
            cv2.circle(vis_image, center, radius + 2, color, ring_thickness)
        
        return vis_image

    def _create_zone_visualization(self, image: np.ndarray, 
                                 spots: List[Dict[str, Any]], 
                                 segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create visualization showing spots grouped by zones."""
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Zone color map
        zone_colors = {
            'pupillary_border': (255, 100, 100),    # Light red
            'inner_ciliary': (100, 255, 100),       # Light green
            'middle_ciliary': (100, 100, 255),      # Light blue
            'outer_ciliary': (255, 255, 100),       # Light yellow
            'peripheral': (255, 100, 255)           # Light magenta
        }
        
        # Draw zone boundaries
        iris_center = segmentation_data.get("iris_center", (image.shape[1]//2, image.shape[0]//2))
        iris_radius = segmentation_data.get("iris_radius", min(image.shape)//4)
        
        for zone_name, zone_info in self.zone_health_mapping.items():
            min_ratio, max_ratio = zone_info['radius_ratio']
            inner_radius = int(iris_radius * min_ratio)
            outer_radius = int(iris_radius * max_ratio)
            
            color = zone_colors.get(zone_name, (128, 128, 128))
            cv2.circle(vis_image, iris_center, outer_radius, color, 1)
        
        # Draw spots with zone coloring
        for spot in spots:
            center = spot['center']
            zone = spot.get('zone', 'unknown')
            color = zone_colors.get(zone, (128, 128, 128))
            
            cv2.circle(vis_image, center, 4, color, -1)
            cv2.circle(vis_image, center, 6, color, 1)
        
        return vis_image

    def _create_pattern_visualization(self, image: np.ndarray, 
                                    spots: List[Dict[str, Any]], 
                                    pattern_analysis: Dict[str, Any],
                                    segmentation_data: Dict[str, Any]) -> np.ndarray:
        """Create visualization showing spot patterns."""
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw all spots
        for spot in spots:
            center = spot['center']
            cv2.circle(vis_image, center, 3, (0, 255, 0), -1)
        
        # Add pattern information as text overlay
        pattern_type = pattern_analysis.get('pattern_type', 'unknown')
        density = pattern_analysis.get('density', 0)
        total_spots = pattern_analysis.get('total_spots', 0)
        
        # Add text information
        text_y = 30
        cv2.putText(vis_image, f"Pattern: {pattern_type}", (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
        cv2.putText(vis_image, f"Spots: {total_spots}", (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
        cv2.putText(vis_image, f"Density: {density:.4f}", (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image

    # Additional helper methods for remaining analysis functions...
    
    def _derive_constitutional_indicators(self, spots: List[Dict[str, Any]], 
                                        pattern_analysis: Dict[str, Any],
                                        zone_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Derive constitutional indicators from spot patterns."""
        indicators = {
            'primary_dosha_tendency': 'balanced',
            'constitutional_strength': 'moderate',
            'elimination_capacity': 'normal'
        }
        
        total_spots = len(spots)
        pattern_type = pattern_analysis.get('pattern_type', 'random')
        
        # Constitutional assessment based on spot patterns
        if pattern_type == 'scattered' and total_spots > 15:
            indicators['primary_dosha_tendency'] = 'vata_imbalance'
            indicators['constitutional_strength'] = 'variable'
        elif pattern_type == 'clustered':
            indicators['primary_dosha_tendency'] = 'pitta_concentration'
            indicators['constitutional_strength'] = 'focused'
        elif total_spots < 5:
            indicators['primary_dosha_tendency'] = 'kapha_stable'
            indicators['constitutional_strength'] = 'strong'
        
        return indicators

    def _assess_constitutional_strength(self, spots: List[Dict[str, Any]], 
                                      pattern_analysis: Dict[str, Any]) -> str:
        """Assess overall constitutional strength."""
        total_spots = len(spots)
        pattern_type = pattern_analysis.get('pattern_type', 'random')
        
        if total_spots < 5 and pattern_type != 'clustered':
            return 'strong'
        elif total_spots < 15:
            return 'moderate'
        else:
            return 'needs_support'

    def _assess_elimination_efficiency(self, spots: List[Dict[str, Any]], 
                                     zone_analysis: Dict[str, Any]) -> str:
        """Assess elimination system efficiency."""
        # Check elimination-related zones
        elimination_zones = ['inner_ciliary', 'middle_ciliary']
        elimination_spot_count = 0
        
        for zone in elimination_zones:
            if zone in zone_analysis:
                elimination_spot_count += zone_analysis[zone].get('spot_count', 0)
        
        if elimination_spot_count < 3:
            return 'efficient'
        elif elimination_spot_count < 8:
            return 'moderate'
        else:
            return 'congested'

    def _assess_inflammatory_markers(self, spots: List[Dict[str, Any]]) -> str:
        """Assess inflammatory markers from spot characteristics."""
        inflammatory_spots = 0
        
        for spot in spots:
            spot_type = spot.get('classification', {}).get('type', '')
            if spot_type in ['dark_spot', 'pigment_spot']:
                inflammatory_spots += 1
        
        if inflammatory_spots < 3:
            return 'minimal'
        elif inflammatory_spots < 8:
            return 'moderate'
        else:
            return 'elevated'

    def _identify_priority_areas(self, zone_analysis: Dict[str, Any]) -> List[str]:
        """Identify priority areas for health focus."""
        priority_areas = []
        
        for zone_name, zone_data in zone_analysis.items():
            stress_level = zone_data.get('health_implications', {}).get('stress_level', 'normal')
            if stress_level in ['moderate', 'high']:
                organs = zone_data.get('associated_organs', [])
                priority_areas.extend(organs)
        
        return list(set(priority_areas))  # Remove duplicates

    def _generate_health_recommendations(self, health_assessment: Dict[str, Any], 
                                       pattern_analysis: Dict[str, Any],
                                       zone_analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive health recommendations."""
        recommendations = []
        
        # Toxin-based recommendations
        toxin_level = health_assessment.get('toxin_indicators', {}).get('level', 'minimal')
        if toxin_level == 'significant':
            recommendations.extend([
                'Implement comprehensive detoxification protocol',
                'Increase water intake and support kidney function',
                'Consider Panchakarma therapies for deep cleansing'
            ])
        elif toxin_level == 'moderate':
            recommendations.extend([
                'Support natural elimination pathways',
                'Incorporate detoxifying herbs and foods',
                'Ensure adequate fiber intake for intestinal cleansing'
            ])
        
        # Pattern-based recommendations
        pattern_type = pattern_analysis.get('pattern_type', 'random')
        if pattern_type == 'clustered':
            recommendations.append('Address localized imbalances with targeted therapies')
        elif pattern_type == 'scattered':
            recommendations.append('Focus on overall constitutional balance and grounding practices')
        
        # Zone-specific recommendations
        for zone_name, zone_data in zone_analysis.items():
            stress_level = zone_data.get('health_implications', {}).get('stress_level', 'normal')
            if stress_level != 'normal':
                zone_recommendations = zone_data.get('health_implications', {}).get('recommendations', [])
                recommendations.extend(zone_recommendations)
        
        return list(set(recommendations))  # Remove duplicates

    # Placeholder methods for line and color analysis (can be expanded)
    
    def _analyze_radial_patterns(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze radial patterns in the iris."""
        # Simplified implementation - can be expanded
        return {'radial_lines_detected': 0, 'pattern_strength': 0.0}

    def _analyze_circumferential_patterns(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze circumferential patterns in the iris."""
        # Simplified implementation - can be expanded
        return {'circumferential_lines_detected': 0, 'pattern_strength': 0.0}

    def _analyze_fiber_patterns(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fiber patterns in the iris."""
        # Simplified implementation - can be expanded
        return {'fiber_density': 0.0, 'dominant_orientation': 0.0}

    def _comprehensive_texture_analysis(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive texture analysis."""
        # Simplified implementation - can be expanded
        return {'texture_uniformity': 0.0, 'texture_contrast': 0.0}

    def _assess_constitution_from_patterns(self, radial_analysis: Dict[str, Any], 
                                         circumferential_analysis: Dict[str, Any],
                                         fiber_analysis: Dict[str, Any],
                                         texture_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess constitution from pattern analysis."""
        return {'primary_dosha': 'balanced', 'constitution_type': 'tri_dosha'}

    def _correlate_patterns_to_health(self, radial_analysis: Dict[str, Any],
                                    circumferential_analysis: Dict[str, Any], 
                                    fiber_analysis: Dict[str, Any],
                                    constitutional_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate patterns to health indicators."""
        return {'health_implications': [], 'recommendations': []}

    def _create_pattern_visualizations(self, image: np.ndarray, 
                                     radial_analysis: Dict[str, Any],
                                     circumferential_analysis: Dict[str, Any],
                                     fiber_analysis: Dict[str, Any],
                                     segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create pattern visualization."""
        return {'pattern_visualization': image}

    def _advanced_color_extraction(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Advanced color extraction and analysis."""
        # Simplified implementation - can be expanded
        return {'dominant_colors': [], 'color_distribution': {}}

    def _analyze_colors_by_zones(self, image: np.ndarray, segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze colors by iris zones."""
        # Simplified implementation - can be expanded
        return {'zone_colors': {}}

    def _ayurvedic_color_interpretation(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Ayurvedic color interpretation."""
        # Simplified implementation - can be expanded
        return {'dominant_dosha': 'balanced', 'constitution_indicators': []}

    def _assess_constitution_from_colors(self, color_analysis: Dict[str, Any], 
                                       ayurvedic_interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess constitution from color analysis."""
        return {'constitution_type': 'balanced', 'dosha_distribution': {}}

    def _derive_health_indicators_from_colors(self, color_analysis: Dict[str, Any],
                                            zonal_color_analysis: Dict[str, Any],
                                            constitutional_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Derive health indicators from color patterns."""
        return {'health_indicators': [], 'recommendations': []}

    def _create_color_visualizations(self, image: np.ndarray, 
                                   color_analysis: Dict[str, Any],
                                   zonal_color_analysis: Dict[str, Any],
                                   segmentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create color analysis visualizations."""
        return {'color_visualization': image}
