#!/usr/bin/env python3
"""
Patch script for fixing uploaded_file_path issue in advanced_app.py
"""
import sys
import re

# Define the pattern to match
pattern = r'tmp_img_path = uploaded_file_path'
replacement = 'tmp_img_path = temp_advanced_path'

def fix_file(filename):
    print(f"Reading file: {filename}")
    with open(filename, 'r') as file:
        content = file.read()
    
    # Check if the pattern exists
    if pattern in content:
        print(f"Found pattern '{pattern}' in {filename}")
        # Replace the pattern
        content = content.replace(pattern, replacement)
        print(f"Replacing with '{replacement}'")
        
        # Write the modified content back to the file
        with open(filename, 'w') as file:
            file.write(content)
        print(f"File updated: {filename}")
    else:
        print(f"Pattern '{pattern}' not found in {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_uploaded_path.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    fix_file(filename)
