#!/usr/bin/env python3
# check_nltk_resources.py
# Script to check if all required NLTK resources are correctly installed

import os
import sys
import nltk

def check_nltk_resources():
    """Check if all required NLTK resources are installed."""
    print("Checking NLTK resources...")
    print(f"NLTK data paths: {nltk.data.path}")
    
    # Check critical resources
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab', 
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }
    
    missing_resources = []
    
    for resource_name, resource_path in resources.items():
        resource_found = False
        for path in nltk.data.path:
            if os.path.exists(os.path.join(path, resource_path)):
                print(f"✅ {resource_name} found at {os.path.join(path, resource_path)}")
                resource_found = True
                break
        
        if not resource_found:
            print(f"❌ {resource_name} NOT FOUND")
            missing_resources.append(resource_name)
    
    if missing_resources:
        print("\nMissing resources detected. Attempting to download...")
        if 'punkt_tab' in missing_resources:
            print("Downloading all NLTK resources (required for punkt_tab)...")
            nltk.download('all')
        else:
            for resource in missing_resources:
                print(f"Downloading {resource}...")
                nltk.download(resource)
        
        # Check again after download
        for resource_name in missing_resources:
            resource_path = resources[resource_name]
            resource_found = False
            for path in nltk.data.path:
                if os.path.exists(os.path.join(path, resource_path)):
                    print(f"✅ {resource_name} successfully downloaded")
                    resource_found = True
                    break
            
            if not resource_found:
                print(f"❌ Failed to download {resource_name}")
                return False
    
    # Test tokenization
    try:
        from nltk.tokenize import sent_tokenize
        test_result = sent_tokenize("This is a test. This is another test.")
        print("\nTokenization test successful:", test_result)
        return True
    except Exception as e:
        print(f"\nTokenization test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_nltk_resources()
    if success:
        print("\nAll NLTK resources are correctly installed.")
        sys.exit(0)
    else:
        print("\nSome NLTK resources could not be installed correctly.")
        print("Try manually downloading all resources with: python -c 'import nltk; nltk.download(\"all\")'")
        sys.exit(1)
