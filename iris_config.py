"""
Configuration file for iris analysis parameters
"""

# Zone analysis configuration
ZONE_CONFIG = {
    "pupillary": {
        "inner_ratio": 0.0,
        "outer_ratio": 0.2,
        "name": "Pupillary Zone",
        "systems": ["digestive", "intestinal"]
    },
    "ciliary": {
        "inner_ratio": 0.2,
        "outer_ratio": 0.5,
        "name": "Ciliary Zone",
        "systems": ["respiratory", "circulatory"]
    },
    "autonomic": {
        "inner_ratio": 0.5,
        "outer_ratio": 0.65,
        "name": "Autonomic Nerve Wreath",
        "systems": ["nervous", "autonomic"]
    },
    "middle": {
        "inner_ratio": 0.65,
        "outer_ratio": 0.8,
        "name": "Middle Zone",
        "systems": ["musculoskeletal", "endocrine"]
    },
    "peripheral": {
        "inner_ratio": 0.8,
        "outer_ratio": 0.95,
        "name": "Peripheral Zone",
        "systems": ["lymphatic", "skin"]
    }
}

# Analysis sensitivity presets
SENSITIVITY_PRESETS = {
    "high": {
        "min_spot_size": 3,
        "max_spot_size": 500,
        "detection_threshold": 0.6,
        "edge_detection_sigma": 1.5,
        "contrast_limit": 3.0
    },
    "medium": {
        "min_spot_size": 5,
        "max_spot_size": 400,
        "detection_threshold": 0.7,
        "edge_detection_sigma": 1.0,
        "contrast_limit": 2.0
    },
    "low": {
        "min_spot_size": 8,
        "max_spot_size": 300,
        "detection_threshold": 0.8,
        "edge_detection_sigma": 0.8,
        "contrast_limit": 1.5
    }
}

# Feature detection parameters
FEATURE_DETECTION = {
    "edge_detection": {
        "canny_low": 50,
        "canny_high": 150,
        "kernel_size": 3
    },
    "circle_detection": {
        "dp": 1,
        "min_dist": 50,
        "param1": 50,
        "param2": 30
    },
    "spot_detection": {
        "blob_min_area": 10,
        "blob_max_area": 1000,
        "blob_min_circularity": 0.3,
        "blob_min_convexity": 0.7,
        "blob_min_inertia": 0.1
    }
}

# Color analysis parameters
COLOR_ANALYSIS = {
    "clustering": {
        "n_clusters": 5,
        "random_state": 42
    },
    "color_spaces": ["RGB", "HSV", "LAB"],
    "histogram_bins": 32
}
