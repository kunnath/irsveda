#!/usr/bin/env python
"""
NLTK Resources Downloader

This script downloads the necessary NLTK resources for the IridoVeda application.
Run this script if you encounter NLTK resource-related errors.
"""

import nltk
import os
import sys

def download_nltk_resources():
    """Download essential NLTK resources for the application."""
    
    # Create a directory for NLTK data if it doesn't exist
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # List of resources to download
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    
    # Download each resource
    for resource_name, resource_path in resources:
        print(f"Checking {resource_name}...")
        try:
            # Check if resource is already downloaded
            nltk.data.find(resource_path)
            print(f"✓ {resource_name} is already downloaded.")
        except LookupError:
            # Download the resource
            print(f"Downloading {resource_name}...")
            nltk.download(resource_name, download_dir=nltk_data_dir)
            print(f"✓ {resource_name} downloaded successfully.")
    
    # Check if punkt_tab is needed
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("✓ punkt_tab is already downloaded.")
    except LookupError:
        print("Downloading all-corpora (includes punkt_tab)...")
        nltk.download('all-corpora', download_dir=nltk_data_dir)
        print("✓ all-corpora downloaded successfully.")
    
    # Verify downloads
    print("\nVerifying NLTK resources...")
    all_verified = True
    
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
            print(f"✓ {resource_name} verified.")
        except LookupError:
            print(f"✗ {resource_name} verification failed.")
            all_verified = False
    
    if all_verified:
        print("\nAll NLTK resources have been downloaded successfully!")
        print(f"NLTK data directory: {nltk_data_dir}")
    else:
        print("\nSome resources could not be verified. You may need to run this script with administrative privileges.")
        print("Alternatively, try running the following in a Python interpreter:")
        print("\nimport nltk")
        for resource_name, _ in resources:
            print(f"nltk.download('{resource_name}')")

if __name__ == "__main__":
    print("==== NLTK Resources Downloader ====")
    download_nltk_resources()
