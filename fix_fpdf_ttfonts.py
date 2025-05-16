"""
This file patches the ttfonts.py file from the fpdf library to fix the string concatenation issue
"""
import os
import shutil
import sys

# Path to the original file
ttfonts_path = '/Users/kunnath/Projects/irisayush/venv/lib/python3.13/site-packages/fpdf/ttfonts.py'

# Create a backup of the original file
backup_path = ttfonts_path + '.bak'
if not os.path.exists(backup_path):
    print(f"Creating backup of original ttfonts.py as {backup_path}")
    shutil.copy2(ttfonts_path, backup_path)

# Read the file content
with open(ttfonts_path, 'r') as f:
    content = f.read()

# Fix the string concatenation issue by converting version to string
# Replace this line: die("Not a TrueType font: version=" + version)
# With this: die("Not a TrueType font: version=" + str(version))
fixed_content = content.replace(
    'die("Not a TrueType font: version=" + version)',
    'die("Not a TrueType font: version=" + str(version))'
)

# Write the fixed content back to the file
with open(ttfonts_path, 'w') as f:
    f.write(fixed_content)

print("Successfully patched the ttfonts.py file to fix the string concatenation issue.")
print("You can now use the IrisReportGenerator to generate PDF reports.")

# Exit with success code
sys.exit(0)
