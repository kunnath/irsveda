"""
Enhanced Iris Analysis Service
Combines WebIrisAnalyzer capabilities with existing IridoVeda analysis system
"""

import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import json
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from PIL import Image

# Import existing components
from iris_predictor import IrisPredictor
from iris_zone_analyzer import IrisZoneAnalyzer
from advanced_iris_analyzer import AdvancedIrisAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIrisAnalysisService:
    """
    Enhanced iris analysis service that combines WebIrisAnalyzer capabilities
    with the existing IridoVeda analysis system.
    """
    
    def __init__(self, output_dir: str = "analysis_output"):
        """
        Initialize the enhanced iris analysis service.
        
        Args:
            output_dir: Directory to store analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.segments_dir = self.output_dir / "segments" 
        self.data_dir = self.output_dir / "data"
        self.reports_dir = self.output_dir / "reports"
        
        # Create directories
        for dir_path in [self.output_dir, self.segments_dir, self.data_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize existing analyzers
        self.iris_predictor = IrisPredictor()
        self.zone_analyzer = IrisZoneAnalyzer()
        
        # Enhanced detection parameters
        self.detection_params = {
            'min_area': 50,
            'max_area': 2000,
            'min_circularity': 0.3,
            'gaussian_blur': 3,
            'canny_low': 50,
            'canny_high': 150,
            'morphology_kernel': 2,
            'brightness_threshold': 20,
            'contrast_enhancement': 2.0,
            'detect_dark_spots': True,
            'detect_light_spots': True,
            'sharpness_threshold': 0.1,
            'visibility_threshold': 10
        }
        
        # Sensitivity presets
        self.sensitivity_presets = {
            'low': {
                'min_area': 100,
                'max_area': 1500,
                'min_circularity': 0.5,
                'gaussian_blur': 5,
                'canny_low': 80,
                'canny_high': 200,
                'brightness_threshold': 30,
                'sharpness_threshold': 0.2,
                'visibility_threshold': 20
            },
            'medium': {
                'min_area': 50,
                'max_area': 2000,
                'min_circularity': 0.3,
                'gaussian_blur': 3,
                'canny_low': 50,
                'canny_high': 150,
                'brightness_threshold': 20,
                'sharpness_threshold': 0.1,
                'visibility_threshold': 10
            },
            'high': {
                'min_area': 20,
                'max_area': 3000,
                'min_circularity': 0.2,
                'gaussian_blur': 1,
                'canny_low': 30,
                'canny_high': 100,
                'brightness_threshold': 10,
                'sharpness_threshold': 0.05,
                'visibility_threshold': 5
            }
        }
        
        # Current analysis session data
        self.current_image_path = None
        self.current_segments_df = pd.DataFrame()
        self.current_analysis_results = {}
        
    def set_sensitivity_preset(self, preset: str) -> Dict[str, Any]:
        """
        Set detection parameters based on sensitivity preset.
        
        Args:
            preset: One of 'low', 'medium', 'high'
            
        Returns:
            Updated detection parameters
        """
        if preset in self.sensitivity_presets:
            self.detection_params.update(self.sensitivity_presets[preset])
            logger.info(f"Set sensitivity preset to: {preset}")
        else:
            logger.warning(f"Unknown preset: {preset}")
        
        return self.detection_params.copy()
    
    def update_detection_params(self, **params) -> Dict[str, Any]:
        """
        Update individual detection parameters.
        
        Args:
            **params: Detection parameters to update
            
        Returns:
            Updated detection parameters
        """
        self.detection_params.update(params)
        logger.info(f"Updated detection parameters: {params}")
        return self.detection_params.copy()
    
    def analyze_iris_comprehensive(self, image_path: str, 
                                 include_zones: bool = True,
                                 include_doshas: bool = True,
                                 include_segmentation: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive iris analysis combining all methods.
        
        Args:
            image_path: Path to iris image
            include_zones: Include iris zone analysis
            include_doshas: Include dosha analysis
            include_segmentation: Include detailed segmentation
            
        Returns:
            Comprehensive analysis results
        """
        self.current_image_path = image_path
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'analysis_params': self.detection_params.copy()
        }
        
        try:
            # Basic iris analysis using existing predictor
            logger.info("Performing basic iris analysis...")
            basic_results = self.iris_predictor.process_iris_image(image_path)
            results['basic_analysis'] = basic_results
            
            # Enhanced segmentation analysis
            if include_segmentation:
                logger.info("Performing enhanced segmentation...")
                segmentation_results = self._perform_enhanced_segmentation(image_path)
                results['segmentation_analysis'] = segmentation_results
            
            # Zone analysis
            if include_zones:
                logger.info("Performing zone analysis...")
                zone_results = self._perform_zone_analysis(image_path)
                results['zone_analysis'] = zone_results
            
            # Dosha analysis
            if include_doshas and 'dosha_analysis' in basic_results:
                results['dosha_analysis'] = basic_results['dosha_analysis']
            
            # Generate comprehensive report
            results['comprehensive_report'] = self._generate_comprehensive_report(results)
            
            # Save results
            self._save_analysis_results(results)
            
            self.current_analysis_results = results
            logger.info("Comprehensive analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _perform_enhanced_segmentation(self, image_path: str) -> Dict[str, Any]:
        """
        Perform enhanced segmentation analysis similar to WebIrisAnalyzer.
        
        Args:
            image_path: Path to iris image
            
        Returns:
            Segmentation analysis results
        """
        try:
            # Load image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect segments using multiple methods
            segments = []
            
            # Method 1: Edge detection
            segments.extend(self._detect_edge_segments(gray_image, rgb_image))
            
            # Method 2: Dark spot detection
            if self.detection_params['detect_dark_spots']:
                segments.extend(self._detect_dark_spots(gray_image, rgb_image))
            
            # Method 3: Light spot detection
            if self.detection_params['detect_light_spots']:
                segments.extend(self._detect_light_spots(gray_image, rgb_image))
            
            # Method 4: Blob detection
            segments.extend(self._detect_blob_segments(gray_image, rgb_image))
            
            # Process and save segments
            processed_segments = self._process_segments(segments, rgb_image)
            
            # Create DataFrame
            if processed_segments:
                self.current_segments_df = pd.DataFrame(processed_segments)
                self._save_segments_data()
            
            # Generate annotated image
            annotated_image = self._create_annotated_image(rgb_image, processed_segments)
            
            return {
                'total_segments': len(processed_segments),
                'segments': processed_segments,
                'detection_methods_used': self._get_detection_methods_used(),
                'annotated_image': self._image_to_base64(annotated_image),
                'segments_saved': len(processed_segments)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced segmentation: {str(e)}")
            return {'error': str(e)}
    
    def _detect_edge_segments(self, gray_image: np.ndarray, rgb_image: np.ndarray) -> List[Dict]:
        """Detect segments using edge detection."""
        segments = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, 
                                  (self.detection_params['gaussian_blur'], 
                                   self.detection_params['gaussian_blur']), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 
                         self.detection_params['canny_low'],
                         self.detection_params['canny_high'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.detection_params['min_area'] <= area <= self.detection_params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity >= self.detection_params['min_circularity']:
                        segments.append({
                            'bounding_box': (x, y, w, h),
                            'area': area,
                            'contour': contour,
                            'circularity': circularity,
                            'detection_method': 'edge',
                            'centroid_x': x + w // 2,
                            'centroid_y': y + h // 2
                        })
        
        return segments
    
    def _detect_dark_spots(self, gray_image: np.ndarray, rgb_image: np.ndarray) -> List[Dict]:
        """Detect dark spots in the iris."""
        segments = []
        
        # Apply morphological operations to enhance dark spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.detection_params['morphology_kernel'],
                                          self.detection_params['morphology_kernel']))
        
        # Apply top-hat transform to detect dark spots
        tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to get dark spots
        _, thresh = cv2.threshold(tophat, self.detection_params['brightness_threshold'], 
                                255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.detection_params['min_area'] <= area <= self.detection_params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                segments.append({
                    'bounding_box': (x, y, w, h),
                    'area': area,
                    'contour': contour,
                    'detection_method': 'dark_spot',
                    'centroid_x': x + w // 2,
                    'centroid_y': y + h // 2
                })
        
        return segments
    
    def _detect_light_spots(self, gray_image: np.ndarray, rgb_image: np.ndarray) -> List[Dict]:
        """Detect light spots in the iris."""
        segments = []
        
        # Apply morphological operations to enhance light spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.detection_params['morphology_kernel'],
                                          self.detection_params['morphology_kernel']))
        
        # Apply black-hat transform to detect light spots
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to get light spots
        _, thresh = cv2.threshold(blackhat, self.detection_params['brightness_threshold'], 
                                255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.detection_params['min_area'] <= area <= self.detection_params['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                segments.append({
                    'bounding_box': (x, y, w, h),
                    'area': area,
                    'contour': contour,
                    'detection_method': 'light_spot',
                    'centroid_x': x + w // 2,
                    'centroid_y': y + h // 2
                })
        
        return segments
    
    def _detect_blob_segments(self, gray_image: np.ndarray, rgb_image: np.ndarray) -> List[Dict]:
        """Detect segments using blob detection."""
        segments = []
        
        # Set up SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = self.detection_params['min_area']
        params.maxArea = self.detection_params['max_area']
        params.filterByCircularity = True
        params.minCircularity = self.detection_params['min_circularity']
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray_image)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            # Create bounding box
            x1, y1 = max(0, x - size//2), max(0, y - size//2)
            x2, y2 = min(gray_image.shape[1], x + size//2), min(gray_image.shape[0], y + size//2)
            w, h = x2 - x1, y2 - y1
            
            segments.append({
                'bounding_box': (x1, y1, w, h),
                'area': kp.size ** 2,
                'contour': None,
                'detection_method': 'blob',
                'centroid_x': x,
                'centroid_y': y
            })
        
        return segments
    
    def _process_segments(self, segments: List[Dict], rgb_image: np.ndarray) -> List[Dict]:
        """Process detected segments and calculate additional properties."""
        processed_segments = []
        
        for i, segment in enumerate(segments):
            x, y, w, h = segment['bounding_box']
            
            # Extract segment region
            segment_region = rgb_image[y:y+h, x:x+w]
            
            if segment_region.size > 0:
                # Calculate color properties
                dominant_color = np.mean(segment_region.reshape(-1, 3), axis=0)
                avg_brightness = np.mean(cv2.cvtColor(segment_region, cv2.COLOR_RGB2GRAY))
                
                # Calculate additional properties
                sharpness = self._calculate_sharpness(segment_region)
                visibility = self._calculate_visibility(segment_region)
                contrast = self._calculate_contrast(segment_region)
                
                # Save segment image
                segment_filename = f"segment_{i+1}_{segment['detection_method']}.png"
                segment_path = self.segments_dir / segment_filename
                
                # Convert RGB to BGR for OpenCV saving
                bgr_segment = cv2.cvtColor(segment_region, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(segment_path), bgr_segment)
                
                processed_segment = {
                    'segment_id': i + 1,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': segment['area'],
                    'dominant_color_r': int(dominant_color[0]),
                    'dominant_color_g': int(dominant_color[1]),
                    'dominant_color_b': int(dominant_color[2]),
                    'avg_brightness': float(avg_brightness),
                    'detection_method': segment['detection_method'],
                    'sharpness': float(sharpness),
                    'visibility': float(visibility),
                    'contrast': float(contrast),
                    'circularity': float(segment.get('circularity', 0)),
                    'centroid_x': segment['centroid_x'],
                    'centroid_y': segment['centroid_y'],
                    'filename': segment_filename,
                    'timestamp': datetime.now().isoformat()
                }
                
                processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _calculate_sharpness(self, image_region: np.ndarray) -> float:
        """Calculate sharpness of image region using Laplacian variance."""
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_region
        
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_visibility(self, image_region: np.ndarray) -> float:
        """Calculate visibility (contrast) of image region."""
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_region
        
        return gray.std()
    
    def _calculate_contrast(self, image_region: np.ndarray) -> float:
        """Calculate contrast of image region."""
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_region
        
        min_val, max_val = gray.min(), gray.max()
        if max_val == min_val:
            return 0.0
        return (max_val - min_val) / (max_val + min_val)
    
    def _create_annotated_image(self, rgb_image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Create annotated image with segment markings."""
        annotated = rgb_image.copy()
        
        # Enhanced color coding for different detection methods
        method_colors = {
            'edge': (0, 255, 0),           # Green
            'dark_spot': (255, 0, 0),      # Red
            'light_spot': (255, 165, 0),   # Orange
            'blob': (255, 255, 0),         # Yellow
            'contour': (255, 0, 255),      # Magenta
            'unknown': (128, 128, 128)     # Gray
        }
        
        # Method symbols
        method_symbols = {
            'edge': 'E',
            'dark_spot': 'D',
            'light_spot': 'L',
            'blob': 'B',
            'contour': 'C',
            'unknown': '?'
        }
        
        for segment in segments:
            x, y, w, h = segment['x'], segment['y'], segment['width'], segment['height']
            method = segment['detection_method']
            color = method_colors.get(method, method_colors['unknown'])
            symbol = method_symbols.get(method, '?')
            
            # Draw bounding rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw center circle with method symbol
            center_x, center_y = segment['centroid_x'], segment['centroid_y']
            cv2.circle(annotated, (center_x, center_y), 8, color, -1)
            cv2.circle(annotated, (center_x, center_y), 8, (255, 255, 255), 1)
            
            # Add method symbol
            cv2.putText(annotated, symbol, (center_x - 4, center_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add segment ID and method label
            label = f"#{segment['segment_id']}-{method.replace('_', ' ').title()}"
            label_y = y - 10 if y > 20 else y + h + 20
            
            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x, label_y - 15), (x + label_size[0] + 10, label_y + 5),
                         (0, 0, 0), -1)
            cv2.rectangle(annotated, (x, label_y - 15), (x + label_size[0] + 10, label_y + 5),
                         color, 2)
            
            cv2.putText(annotated, label, (x + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _perform_zone_analysis(self, image_path: str) -> Dict[str, Any]:
        """Perform iris zone analysis using existing zone analyzer."""
        try:
            # Use existing zone analyzer
            zone_results = self.zone_analyzer.analyze_zones(image_path)
            return zone_results
        except Exception as e:
            logger.error(f"Error in zone analysis: {str(e)}")
            return {'error': str(e)}
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            'summary': {},
            'recommendations': [],
            'detailed_findings': {}
        }
        
        try:
            # Basic analysis summary
            if 'basic_analysis' in results and 'analysis' in results['basic_analysis']:
                basic = results['basic_analysis']['analysis']
                report['summary']['overall_health'] = basic.get('overall_health', 'unknown')
                report['summary']['zones_analyzed'] = len(basic.get('zones', {}))
            
            # Segmentation summary
            if 'segmentation_analysis' in results:
                seg = results['segmentation_analysis']
                report['summary']['segments_detected'] = seg.get('total_segments', 0)
                report['summary']['detection_methods'] = seg.get('detection_methods_used', [])
            
            # Zone analysis summary
            if 'zone_analysis' in results:
                zone = results['zone_analysis']
                report['detailed_findings']['zone_analysis'] = zone
            
            # Generate recommendations based on findings
            report['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            report['error'] = str(e)
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Basic recommendations based on overall health
            if 'basic_analysis' in results and 'analysis' in results['basic_analysis']:
                overall_health = results['basic_analysis']['analysis'].get('overall_health', '')
                
                if 'stressed' in overall_health.lower():
                    recommendations.extend([
                        "Consider stress reduction techniques such as meditation or yoga",
                        "Ensure adequate sleep and rest",
                        "Review dietary habits and consider anti-inflammatory foods"
                    ])
                
                if 'poor' in overall_health.lower():
                    recommendations.extend([
                        "Consult with a healthcare professional for detailed examination",
                        "Consider comprehensive health screening",
                        "Focus on lifestyle modifications for better health"
                    ])
            
            # Recommendations based on segment detection
            if 'segmentation_analysis' in results:
                segments_count = results['segmentation_analysis'].get('total_segments', 0)
                
                if segments_count > 20:
                    recommendations.append("High number of iris markings detected - consider detailed iridology consultation")
                elif segments_count < 5:
                    recommendations.append("Relatively clear iris pattern - maintain current healthy lifestyle")
            
            # Default recommendations if none specific
            if not recommendations:
                recommendations = [
                    "Maintain a balanced diet rich in antioxidants",
                    "Stay hydrated and exercise regularly",
                    "Consider regular health check-ups"
                ]
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations = ["Error generating recommendations - consult healthcare professional"]
        
        return recommendations
    
    def _get_detection_methods_used(self) -> List[str]:
        """Get list of detection methods used in current analysis."""
        methods = ['edge']
        
        if self.detection_params['detect_dark_spots']:
            methods.append('dark_spot')
        
        if self.detection_params['detect_light_spots']:
            methods.append('light_spot')
        
        methods.append('blob')
        
        return methods
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert image array to base64 string."""
        try:
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Encode image
            _, buffer = cv2.imencode('.png', bgr_image)
            img_bytes = buffer.tobytes()
            
            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            return ""
    
    def _save_segments_data(self):
        """Save segments data to CSV."""
        try:
            csv_path = self.data_dir / "iris_segments.csv"
            self.current_segments_df.to_csv(csv_path, index=False)
            logger.info(f"Saved segments data to: {csv_path}")
        except Exception as e:
            logger.error(f"Error saving segments data: {str(e)}")
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save complete analysis results to JSON."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = self.reports_dir / f"iris_analysis_{timestamp}.json"
            
            # Convert numpy arrays and other non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved analysis results to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis session."""
        return {
            'image_path': self.current_image_path,
            'segments_detected': len(self.current_segments_df),
            'detection_params': self.detection_params.copy(),
            'analysis_timestamp': self.current_analysis_results.get('timestamp'),
            'output_directory': str(self.output_dir)
        }
    
    def export_segments_csv(self) -> str:
        """Export segments data to CSV and return file path."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.reports_dir / f"iris_segments_export_{timestamp}.csv"
            self.current_segments_df.to_csv(export_path, index=False)
            logger.info(f"Exported segments to: {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"Error exporting segments: {str(e)}")
            return ""
    
    def get_segments_dataframe(self) -> pd.DataFrame:
        """Get current segments DataFrame."""
        return self.current_segments_df.copy()
